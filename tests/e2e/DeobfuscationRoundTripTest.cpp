#include "LiftAndOptFixture.h"
#include "PELoader.h"

#include <llvm/ADT/StringExtras.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/StripDeadPrototypes.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// DLL path macros — injected by CMake.
// ---------------------------------------------------------------------------
#ifndef RT_CLEAN_DLL_PATH
#error "RT_CLEAN_DLL_PATH must be defined"
#endif
#ifndef RT_FLATTEN_DLL_PATH
#error "RT_FLATTEN_DLL_PATH must be defined"
#endif
#ifndef RT_SUBSTITUTE_DLL_PATH
#error "RT_SUBSTITUTE_DLL_PATH must be defined"
#endif
#ifndef RT_BOGUS_CF_DLL_PATH
#error "RT_BOGUS_CF_DLL_PATH must be defined"
#endif
#ifndef RT_OPAQUE_PRED_DLL_PATH
#error "RT_OPAQUE_PRED_DLL_PATH must be defined"
#endif
#ifndef RT_CONST_UNFOLD_DLL_PATH
#error "RT_CONST_UNFOLD_DLL_PATH must be defined"
#endif
#ifndef RT_STRING_ENCRYPT_DLL_PATH
#error "RT_STRING_ENCRYPT_DLL_PATH must be defined"
#endif
#ifndef RT_VECTORIZE_DLL_PATH
#error "RT_VECTORIZE_DLL_PATH must be defined"
#endif
#ifndef RT_ALL_DLL_PATH
#error "RT_ALL_DLL_PATH must be defined"
#endif

// ---------------------------------------------------------------------------
// Win32 forward declarations (no <windows.h> to avoid CHAR conflict).
// ---------------------------------------------------------------------------
extern "C" {
void *__stdcall VirtualAlloc(void *lpAddress, size_t dwSize,
                             unsigned long flAllocationType,
                             unsigned long flProtect);
int __stdcall VirtualFree(void *lpAddress, size_t dwSize,
                          unsigned long dwFreeType);
}

using namespace omill::e2e;

namespace {

static bool wantVerboseLogs() {
  const char *v = std::getenv("OMILL_E2E_VERBOSE");
  return v && (v[0] == '1' || v[0] == 't' || v[0] == 'T') && v[1] == '\0';
}

// =========================================================================
// Null symbol generator — stubs for unresolved externals in JIT.
// =========================================================================
class NullSymbolGenerator : public llvm::orc::DefinitionGenerator {
 public:
  llvm::Error tryToGenerate(
      llvm::orc::LookupState &, llvm::orc::LookupKind,
      llvm::orc::JITDylib &JD, llvm::orc::JITDylibLookupFlags,
      const llvm::orc::SymbolLookupSet &LookupSet) override {
    llvm::orc::SymbolMap symbols;
    for (auto &[name, _] : LookupSet) {
      symbols[name] = {llvm::orc::ExecutorAddr::fromPtr(&stub),
                       llvm::JITSymbolFlags::Exported};
    }
    return JD.define(llvm::orc::absoluteSymbols(std::move(symbols)));
  }

 private:
  static void stub() {}
};

// =========================================================================
// Fixture: one PE per obfuscation variant, shared across all tests.
// =========================================================================
class DeobfuscationRoundTripTest : public LiftAndOptFixture {
 protected:
  // --- Static PE storage (loaded once per suite) ---
  struct Variant {
    const char *dll_path;
    const char *label;
    PEInfo pe;
    bool loaded = false;
  };

  static Variant variants_[];
  static constexpr int kNumVariants = 9;

  enum VariantIdx : int {
    kClean = 0,
    kFlatten,
    kSubstitute,
    kBogusCF,
    kOpaquePred,
    kConstUnfold,
    kStringEncrypt,
    kVectorize,
    kAll,
  };

  static void SetUpTestSuite() {
    for (int i = 0; i < kNumVariants; ++i) {
      auto &v = variants_[i];
      v.loaded = loadPE(v.dll_path, v.pe);
      if (!v.loaded) {
        // Non-fatal; individual tests using this variant will skip.
        llvm::errs() << "[RT] WARNING: Failed to load " << v.dll_path << "\n";
      }
    }
  }

  static void TearDownTestSuite() {
    // PEInfo cleanup is automatic (vector storage).
  }

  void SetUp() override {
    LiftAndOptFixture::SetUp();
  }

  void TearDown() override {
    unmapBinaryForJIT();
  }

  // --- Lift → deobfuscate → JIT pipeline ---

  bool liftDeobfAndJIT(int variant_idx, const std::string &export_name) {
    auto &v = variants_[variant_idx];
    if (!v.loaded) {
      ADD_FAILURE() << "PE not loaded for variant " << v.label;
      return false;
    }

    uint64_t va = 0;
    {
      auto it = v.pe.exports.find(export_name);
      if (it == v.pe.exports.end()) {
        ADD_FAILURE() << "Export '" << export_name << "' not found in "
                      << v.label;
        return false;
      }
      va = it->second;
    }

    // Load .text into trace manager.
    traceManager().clearTraces();
    text_copy_.resize(v.pe.text_size);
    if (!v.pe.memory_map.read(v.pe.text_base, text_copy_.data(),
                              static_cast<unsigned>(v.pe.text_size))) {
      ADD_FAILURE() << "Failed to read .text for " << v.label;
      return false;
    }
    setCode(text_copy_.data(), text_copy_.size(), v.pe.text_base);
    traceManager().setBaseAddr(va);
    traceManager().setMemoryMap(&v.pe.memory_map);

    // Lift.
    auto *M = lift();
    if (!M) {
      ADD_FAILURE() << "Lift failed for " << export_name << " in " << v.label;
      return false;
    }

    // Find the trace function name for this VA.
    auto trace_it = traceManager().traces().find(va);
    if (trace_it != traceManager().traces().end() && trace_it->second)
      expected_native_prefix_ = trace_it->second->getName().str();
    else
      expected_native_prefix_.clear();

    current_export_va_ = va;

    // Deobfuscate.
    omill::PipelineOptions opts;
    opts.recover_abi = true;
    opts.deobfuscate = true;
    optimizeWithMemoryMap(opts, v.pe.memory_map);

    if (!verifyModule()) {
      ADD_FAILURE() << "Module invalid after pipeline for " << export_name
                    << " in " << v.label;
      return false;
    }

    // Map PE sections for JIT (global data references).
    current_variant_idx_ = variant_idx;
    mapBinaryForJIT(v);

    // JIT.
    jit_addr_ = jitLookupNative();
    return jit_addr_ != nullptr;
  }

  // --- Helpers ---

  static std::optional<uint64_t> parseNativeVA(llvm::StringRef name) {
    if (!name.starts_with("sub_") || !name.contains("_native"))
      return std::nullopt;
    llvm::StringRef core = name.drop_front(4);
    size_t hex_len = 0;
    while (hex_len < core.size() && llvm::isHexDigit(core[hex_len]))
      ++hex_len;
    if (hex_len == 0) return std::nullopt;
    uint64_t va = 0;
    if (core.take_front(hex_len).getAsInteger(16, va)) return std::nullopt;
    return va;
  }

  llvm::Function *findNativeFunction() {
    if (!module() || current_export_va_ == 0) return nullptr;

    if (!expected_native_prefix_.empty()) {
      std::string exact = expected_native_prefix_ + "_native";
      if (auto *F = module()->getFunction(exact))
        if (!F->isDeclaration()) return F;
      for (auto &F : *module()) {
        if (F.isDeclaration() || !F.getName().contains("_native")) continue;
        if (F.getName().starts_with(expected_native_prefix_)) return &F;
      }
    }

    llvm::Function *nearest = nullptr;
    uint64_t nearest_dist = UINT64_MAX;
    for (auto &F : *module()) {
      if (F.isDeclaration() || !F.getName().contains("_native")) continue;
      auto va = parseNativeVA(F.getName());
      if (!va) continue;
      if (*va == current_export_va_) return &F;
      uint64_t dist = (*va > current_export_va_)
                          ? (*va - current_export_va_)
                          : (current_export_va_ - *va);
      if (dist < nearest_dist) {
        nearest_dist = dist;
        nearest = &F;
      }
    }
    if (nearest && nearest_dist <= 0x400) return nearest;
    return nullptr;
  }

  unsigned countNativeBasicBlocks() {
    unsigned n = 0;
    for (auto &F : *module()) {
      if (F.isDeclaration() || !F.getName().contains("_native")) continue;
      n += F.size();
    }
    return n;
  }

  unsigned countNativeSwitchInsts() {
    unsigned n = 0;
    for (auto &F : *module()) {
      if (F.isDeclaration() || !F.getName().contains("_native")) continue;
      for (auto &BB : F)
        for (auto &I : BB)
          if (llvm::isa<llvm::SwitchInst>(&I)) n++;
    }
    return n;
  }

  unsigned countNativeInstructions() {
    unsigned n = 0;
    for (auto &F : *module()) {
      if (F.isDeclaration() || !F.getName().contains("_native")) continue;
      for (auto &BB : F) n += BB.size();
    }
    return n;
  }

  bool mapBinaryForJIT(Variant &v) {
    constexpr unsigned long kMemCommit  = 0x1000;
    constexpr unsigned long kMemReserve = 0x2000;
    constexpr unsigned long kPageRW     = 0x04;

    uint64_t min_va = UINT64_MAX, max_va = 0;
    v.pe.memory_map.forEachRegion(
        [&](uint64_t base, const uint8_t *, size_t size) {
          if (base < min_va) min_va = base;
          if (base + size > max_va) max_va = base + size;
        });
    if (min_va >= max_va) return false;

    size_t total_size = static_cast<size_t>(max_va - min_va);
    void *block = VirtualAlloc(reinterpret_cast<void *>(min_va), total_size,
                               kMemCommit | kMemReserve, kPageRW);
    if (!block) {
      if (wantVerboseLogs()) {
        llvm::errs() << "[RT] VirtualAlloc failed for 0x"
                     << llvm::Twine::utohexstr(min_va) << " - 0x"
                     << llvm::Twine::utohexstr(max_va) << "\n";
      }
      return false;
    }
    jit_mapped_regions_.push_back({block, total_size});

    v.pe.memory_map.forEachRegion(
        [&](uint64_t base, const uint8_t *data, size_t size) {
          std::memcpy(reinterpret_cast<void *>(base), data, size);
        });
    return true;
  }

  void unmapBinaryForJIT() {
    constexpr unsigned long kMemRelease = 0x8000;
    for (auto &r : jit_mapped_regions_)
      VirtualFree(r.addr, 0, kMemRelease);
    jit_mapped_regions_.clear();
  }

  void *jitLookupNative() {
    static bool initialized = [] {
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      return true;
    }();
    (void)initialized;

    auto *NF = findNativeFunction();
    if (!NF) {
      ADD_FAILURE() << "No native function found in module";
      return nullptr;
    }

    std::string name = NF->getName().str();
    if (wantVerboseLogs())
      llvm::errs() << "[RT-JIT] Target: " << name << "\n";

    // Strip non-native bodies to reduce serialization size.
    for (auto &F : *module()) {
      if (F.isDeclaration()) continue;
      if (!F.getName().contains("_native")) {
        F.deleteBody();
        F.setLinkage(llvm::GlobalValue::ExternalLinkage);
      }
    }

    // GlobalDCE.
    {
      llvm::PassBuilder PB;
      llvm::LoopAnalysisManager LAM;
      llvm::FunctionAnalysisManager FAM;
      llvm::CGSCCAnalysisManager CGAM;
      llvm::ModuleAnalysisManager MAM;
      PB.registerModuleAnalyses(MAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerLoopAnalyses(LAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
      llvm::ModulePassManager MPM;
      MPM.addPass(llvm::GlobalDCEPass());
      MPM.addPass(llvm::StripDeadPrototypesPass());
      MPM.run(*module(), MAM);
    }

    // Bitcode round-trip into fresh context.
    llvm::SmallString<0> buf;
    {
      llvm::raw_svector_ostream os(buf);
      llvm::WriteBitcodeToFile(*module(), os);
    }

    auto jit_ctx = std::make_unique<llvm::LLVMContext>();
    auto mem_buf = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(buf.data(), buf.size()), "", false);
    auto mod_or = llvm::parseBitcodeFile(mem_buf->getMemBufferRef(), *jit_ctx);
    if (!mod_or) {
      llvm::consumeError(mod_or.takeError());
      ADD_FAILURE() << "Bitcode round-trip parse failed";
      return nullptr;
    }

    auto jit_or = llvm::orc::LLJITBuilder().create();
    if (!jit_or) {
      llvm::consumeError(jit_or.takeError());
      ADD_FAILURE() << "Failed to create LLJIT";
      return nullptr;
    }
    jit_ = std::move(*jit_or);

    auto gen = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
        jit_->getDataLayout().getGlobalPrefix());
    if (gen)
      jit_->getMainJITDylib().addGenerator(std::move(*gen));
    else
      llvm::consumeError(gen.takeError());

    jit_->getMainJITDylib().addGenerator(
        std::make_unique<NullSymbolGenerator>());

    llvm::orc::ThreadSafeContext ts_ctx(std::move(jit_ctx));
    auto err = jit_->addIRModule(
        llvm::orc::ThreadSafeModule(std::move(*mod_or), ts_ctx));
    if (err) {
      llvm::consumeError(std::move(err));
      ADD_FAILURE() << "Failed to add module to JIT";
      return nullptr;
    }

    auto sym_or = jit_->lookup(name);
    if (!sym_or) {
      llvm::consumeError(sym_or.takeError());
      ADD_FAILURE() << "Failed to look up " << name;
      return nullptr;
    }
    return reinterpret_cast<void *>(sym_or->getValue());
  }

  // --- State ---
  struct MappedRegion {
    void *addr;
    size_t size;
  };
  std::vector<MappedRegion> jit_mapped_regions_;
  std::unique_ptr<llvm::orc::LLJIT> jit_;
  std::vector<uint8_t> text_copy_;
  uint64_t current_export_va_ = 0;
  std::string expected_native_prefix_;
  int current_variant_idx_ = -1;
  void *jit_addr_ = nullptr;
};

// Static variant definitions.
DeobfuscationRoundTripTest::Variant
    DeobfuscationRoundTripTest::variants_[kNumVariants] = {
        {RT_CLEAN_DLL_PATH, "clean", {}, false},
        {RT_FLATTEN_DLL_PATH, "flatten", {}, false},
        {RT_SUBSTITUTE_DLL_PATH, "substitute", {}, false},
        {RT_BOGUS_CF_DLL_PATH, "bogus_cf", {}, false},
        {RT_OPAQUE_PRED_DLL_PATH, "opaque_pred", {}, false},
        {RT_CONST_UNFOLD_DLL_PATH, "const_unfold", {}, false},
        {RT_STRING_ENCRYPT_DLL_PATH, "string_encrypt", {}, false},
        {RT_VECTORIZE_DLL_PATH, "vectorize", {}, false},
        {RT_ALL_DLL_PATH, "all", {}, false},
};

// =========================================================================
// Helper macro: generate a test that lifts+deobfuscates+JITs one function
// from one obfuscation variant and verifies semantic correctness.
// =========================================================================
#define RT_JIT_TEST(VariantName, VariantEnum, FuncName, Body)               \
  TEST_F(DeobfuscationRoundTripTest, VariantName##_##FuncName) {            \
    if (!variants_[VariantEnum].loaded) {                                   \
      GTEST_SKIP() << "DLL not loaded for " << variants_[VariantEnum].label;\
    }                                                                       \
    if (!liftDeobfAndJIT(VariantEnum, #FuncName)) {                         \
      GTEST_SKIP() << "Lift/deobf/JIT failed for " #FuncName               \
                      " in " #VariantName;                                  \
    }                                                                       \
    Body                                                                    \
  }

// =========================================================================
// Expected values (computed from the C++ source).
// =========================================================================

// rt_identity(x) = x
// rt_collatz_steps(1)=0, (2)=1, (3)=7, (6)=8, (27)=111
// rt_fibonacci: fib(0)=0, fib(1)=1, fib(10)=55, fib(20)=6765
// rt_gcd: gcd(12,8)=4, gcd(17,13)=1, gcd(100,75)=25
// rt_switch_classify: (-5)=0, (0)=1, (5)=2, (50)=3, (500)=4, (5000)=5
// rt_nested_branch: (5,3)=8, (0,-1)=-1, (-3,2)=5, (-2,-3)=6
// rt_popcount: (0)=0, (1)=1, (0xFF)=8, (0xFFFFFFFF)=32, (0xAAAAAAAA)=16
// rt_bitfield_pack: (0x11,0x22,0x33,0x44) = 0x44332211
// rt_bitfield_unpack: (0x44332211, 0)=0x11, (_, 2)=0x33

// =========================================================================
// CLEAN variant — baseline: no obfuscation, just lift + pipeline + JIT.
// =========================================================================

RT_JIT_TEST(Clean, kClean, rt_identity, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(0), 0);
  EXPECT_EQ(fn(42), 42);
  EXPECT_EQ(fn(-7), -7);
})

RT_JIT_TEST(Clean, kClean, rt_collatz_steps, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(1), 0);
  EXPECT_EQ(fn(2), 1);
  EXPECT_EQ(fn(3), 7);
  EXPECT_EQ(fn(6), 8);
  EXPECT_EQ(fn(27), 111);
})

RT_JIT_TEST(Clean, kClean, rt_fibonacci, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(0), 0);
  EXPECT_EQ(fn(1), 1);
  EXPECT_EQ(fn(10), 55);
  EXPECT_EQ(fn(20), 6765);
})

RT_JIT_TEST(Clean, kClean, rt_gcd, {
  auto fn = reinterpret_cast<int (*)(int, int)>(jit_addr_);
  EXPECT_EQ(fn(12, 8), 4);
  EXPECT_EQ(fn(17, 13), 1);
  EXPECT_EQ(fn(100, 75), 25);
})

RT_JIT_TEST(Clean, kClean, rt_switch_classify, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(-5), 0);
  EXPECT_EQ(fn(0), 1);
  EXPECT_EQ(fn(5), 2);
  EXPECT_EQ(fn(50), 3);
  EXPECT_EQ(fn(500), 4);
  EXPECT_EQ(fn(5000), 5);
})

RT_JIT_TEST(Clean, kClean, rt_nested_branch, {
  auto fn = reinterpret_cast<int (*)(int, int)>(jit_addr_);
  EXPECT_EQ(fn(5, 3), 8);
  EXPECT_EQ(fn(5, 0), 10);
  EXPECT_EQ(fn(5, -2), 7);
  EXPECT_EQ(fn(0, 3), 9);
  EXPECT_EQ(fn(0, -1), -1);
  EXPECT_EQ(fn(-3, 2), 5);
  EXPECT_EQ(fn(-2, -3), 6);
})

RT_JIT_TEST(Clean, kClean, rt_popcount, {
  auto fn = reinterpret_cast<int (*)(uint32_t)>(jit_addr_);
  EXPECT_EQ(fn(0), 0);
  EXPECT_EQ(fn(1), 1);
  EXPECT_EQ(fn(0xFF), 8);
  EXPECT_EQ(fn(0xFFFFFFFF), 32);
  EXPECT_EQ(fn(0xAAAAAAAA), 16);
})

RT_JIT_TEST(Clean, kClean, rt_bitfield_pack, {
  auto fn = reinterpret_cast<uint64_t (*)(uint8_t, uint8_t, uint8_t, uint8_t)>(
      jit_addr_);
  EXPECT_EQ(fn(0x11, 0x22, 0x33, 0x44), 0x44332211ULL);
  EXPECT_EQ(fn(0, 0, 0, 0), 0ULL);
  EXPECT_EQ(fn(0xFF, 0, 0, 0), 0xFFULL);
})

RT_JIT_TEST(Clean, kClean, rt_bitfield_unpack, {
  auto fn = reinterpret_cast<uint8_t (*)(uint64_t, int)>(jit_addr_);
  EXPECT_EQ(fn(0x44332211ULL, 0), 0x11);
  EXPECT_EQ(fn(0x44332211ULL, 1), 0x22);
  EXPECT_EQ(fn(0x44332211ULL, 2), 0x33);
  EXPECT_EQ(fn(0x44332211ULL, 3), 0x44);
})

TEST_F(DeobfuscationRoundTripTest, Clean_rt_array_sum) {
  if (!variants_[kClean].loaded) GTEST_SKIP() << "DLL not loaded";
  if (!liftDeobfAndJIT(kClean, "rt_array_sum"))
    GTEST_SKIP() << "Lift/deobf/JIT failed";
  auto fn = reinterpret_cast<int (*)(const int *, int)>(jit_addr_);
  int arr[] = {1, 2, 3, 4, 5};
  EXPECT_EQ(fn(arr, 5), 15);
  EXPECT_EQ(fn(arr, 0), 0);
  int arr2[] = {-1, 1, -1, 1};
  EXPECT_EQ(fn(arr2, 4), 0);
}

TEST_F(DeobfuscationRoundTripTest, Clean_rt_matrix_trace) {
  if (!variants_[kClean].loaded) GTEST_SKIP() << "DLL not loaded";
  if (!liftDeobfAndJIT(kClean, "rt_matrix_trace"))
    GTEST_SKIP() << "Lift/deobf/JIT failed";
  auto fn = reinterpret_cast<int (*)(const int *, int)>(jit_addr_);
  int mat[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  EXPECT_EQ(fn(mat, 3), 3);
  int mat2[] = {5, 6, 7, 8};
  EXPECT_EQ(fn(mat2, 2), 13);
}

TEST_F(DeobfuscationRoundTripTest, Clean_rt_xor_cipher) {
  if (!variants_[kClean].loaded) GTEST_SKIP() << "DLL not loaded";
  if (!liftDeobfAndJIT(kClean, "rt_xor_cipher"))
    GTEST_SKIP() << "Lift/deobf/JIT failed";
  auto fn = reinterpret_cast<int (*)(uint8_t *, int, uint8_t)>(jit_addr_);
  uint8_t buf[] = {0x41, 0x42, 0x43};
  int sum = fn(buf, 3, 0xFF);
  EXPECT_EQ(sum, 567);
  EXPECT_EQ(buf[0], 0xBE);
  EXPECT_EQ(buf[1], 0xBD);
  EXPECT_EQ(buf[2], 0xBC);
}

// =========================================================================
// FLATTEN variant — CFF obfuscation only.
// =========================================================================

RT_JIT_TEST(Flatten, kFlatten, rt_identity, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(42), 42);
  EXPECT_EQ(fn(-7), -7);
})

RT_JIT_TEST(Flatten, kFlatten, rt_collatz_steps, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(1), 0);
  EXPECT_EQ(fn(3), 7);
  EXPECT_EQ(fn(27), 111);
})

RT_JIT_TEST(Flatten, kFlatten, rt_fibonacci, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(0), 0);
  EXPECT_EQ(fn(1), 1);
  EXPECT_EQ(fn(10), 55);
  EXPECT_EQ(fn(20), 6765);
})

RT_JIT_TEST(Flatten, kFlatten, rt_gcd, {
  auto fn = reinterpret_cast<int (*)(int, int)>(jit_addr_);
  EXPECT_EQ(fn(12, 8), 4);
  EXPECT_EQ(fn(17, 13), 1);
  EXPECT_EQ(fn(100, 75), 25);
})

RT_JIT_TEST(Flatten, kFlatten, rt_switch_classify, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(-5), 0);
  EXPECT_EQ(fn(0), 1);
  EXPECT_EQ(fn(5), 2);
  EXPECT_EQ(fn(50), 3);
  EXPECT_EQ(fn(500), 4);
  EXPECT_EQ(fn(5000), 5);
})

RT_JIT_TEST(Flatten, kFlatten, rt_nested_branch, {
  auto fn = reinterpret_cast<int (*)(int, int)>(jit_addr_);
  EXPECT_EQ(fn(5, 3), 8);
  EXPECT_EQ(fn(0, -1), -1);
  EXPECT_EQ(fn(-3, 2), 5);
})

RT_JIT_TEST(Flatten, kFlatten, rt_popcount, {
  auto fn = reinterpret_cast<int (*)(uint32_t)>(jit_addr_);
  EXPECT_EQ(fn(0), 0);
  EXPECT_EQ(fn(0xFF), 8);
  EXPECT_EQ(fn(0xFFFFFFFF), 32);
})

TEST_F(DeobfuscationRoundTripTest, Flatten_rt_array_sum) {
  if (!variants_[kFlatten].loaded) GTEST_SKIP() << "DLL not loaded";
  if (!liftDeobfAndJIT(kFlatten, "rt_array_sum"))
    GTEST_SKIP() << "Lift/deobf/JIT failed";
  auto fn = reinterpret_cast<int (*)(const int *, int)>(jit_addr_);
  int arr[] = {1, 2, 3, 4, 5};
  EXPECT_EQ(fn(arr, 5), 15);
}

TEST_F(DeobfuscationRoundTripTest, Flatten_rt_xor_cipher) {
  if (!variants_[kFlatten].loaded) GTEST_SKIP() << "DLL not loaded";
  if (!liftDeobfAndJIT(kFlatten, "rt_xor_cipher"))
    GTEST_SKIP() << "Lift/deobf/JIT failed";
  auto fn = reinterpret_cast<int (*)(uint8_t *, int, uint8_t)>(jit_addr_);
  uint8_t buf[] = {0x41, 0x42, 0x43};
  int sum = fn(buf, 3, 0xFF);
  EXPECT_EQ(sum, 567);
}

// =========================================================================
// SUBSTITUTE variant — MBA substitution only.
// =========================================================================

RT_JIT_TEST(Substitute, kSubstitute, rt_identity, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(42), 42);
})

RT_JIT_TEST(Substitute, kSubstitute, rt_fibonacci, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(10), 55);
  EXPECT_EQ(fn(20), 6765);
})

RT_JIT_TEST(Substitute, kSubstitute, rt_popcount, {
  auto fn = reinterpret_cast<int (*)(uint32_t)>(jit_addr_);
  EXPECT_EQ(fn(0xFF), 8);
  EXPECT_EQ(fn(0xAAAAAAAA), 16);
})

RT_JIT_TEST(Substitute, kSubstitute, rt_bitfield_pack, {
  auto fn = reinterpret_cast<uint64_t (*)(uint8_t, uint8_t, uint8_t, uint8_t)>(
      jit_addr_);
  EXPECT_EQ(fn(0x11, 0x22, 0x33, 0x44), 0x44332211ULL);
})

RT_JIT_TEST(Substitute, kSubstitute, rt_gcd, {
  auto fn = reinterpret_cast<int (*)(int, int)>(jit_addr_);
  EXPECT_EQ(fn(12, 8), 4);
  EXPECT_EQ(fn(100, 75), 25);
})

// =========================================================================
// BOGUS_CF variant — bogus control flow only.
// =========================================================================

RT_JIT_TEST(BogusCF, kBogusCF, rt_identity, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(42), 42);
  EXPECT_EQ(fn(-7), -7);
})

RT_JIT_TEST(BogusCF, kBogusCF, rt_fibonacci, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(0), 0);
  EXPECT_EQ(fn(10), 55);
  EXPECT_EQ(fn(20), 6765);
})

RT_JIT_TEST(BogusCF, kBogusCF, rt_nested_branch, {
  auto fn = reinterpret_cast<int (*)(int, int)>(jit_addr_);
  EXPECT_EQ(fn(5, 3), 8);
  EXPECT_EQ(fn(0, -1), -1);
  EXPECT_EQ(fn(-2, -3), 6);
})

RT_JIT_TEST(BogusCF, kBogusCF, rt_collatz_steps, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(1), 0);
  EXPECT_EQ(fn(27), 111);
})

RT_JIT_TEST(BogusCF, kBogusCF, rt_gcd, {
  auto fn = reinterpret_cast<int (*)(int, int)>(jit_addr_);
  EXPECT_EQ(fn(12, 8), 4);
  EXPECT_EQ(fn(17, 13), 1);
})

// =========================================================================
// OPAQUE_PRED variant — opaque predicates only.
// =========================================================================

RT_JIT_TEST(OpaquePred, kOpaquePred, rt_identity, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(42), 42);
})

RT_JIT_TEST(OpaquePred, kOpaquePred, rt_fibonacci, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(10), 55);
  EXPECT_EQ(fn(20), 6765);
})

RT_JIT_TEST(OpaquePred, kOpaquePred, rt_switch_classify, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(-5), 0);
  EXPECT_EQ(fn(0), 1);
  EXPECT_EQ(fn(50), 3);
  EXPECT_EQ(fn(5000), 5);
})

RT_JIT_TEST(OpaquePred, kOpaquePred, rt_nested_branch, {
  auto fn = reinterpret_cast<int (*)(int, int)>(jit_addr_);
  EXPECT_EQ(fn(5, 3), 8);
  EXPECT_EQ(fn(-3, 2), 5);
})

// =========================================================================
// CONST_UNFOLD variant — constant unfolding only.
// =========================================================================

RT_JIT_TEST(ConstUnfold, kConstUnfold, rt_identity, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(42), 42);
})

RT_JIT_TEST(ConstUnfold, kConstUnfold, rt_fibonacci, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(10), 55);
  EXPECT_EQ(fn(20), 6765);
})

RT_JIT_TEST(ConstUnfold, kConstUnfold, rt_collatz_steps, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(1), 0);
  EXPECT_EQ(fn(3), 7);
  EXPECT_EQ(fn(27), 111);
})

RT_JIT_TEST(ConstUnfold, kConstUnfold, rt_popcount, {
  auto fn = reinterpret_cast<int (*)(uint32_t)>(jit_addr_);
  EXPECT_EQ(fn(0xFF), 8);
  EXPECT_EQ(fn(0xFFFFFFFF), 32);
})

RT_JIT_TEST(ConstUnfold, kConstUnfold, rt_bitfield_pack, {
  auto fn = reinterpret_cast<uint64_t (*)(uint8_t, uint8_t, uint8_t, uint8_t)>(
      jit_addr_);
  EXPECT_EQ(fn(0x11, 0x22, 0x33, 0x44), 0x44332211ULL);
})

// =========================================================================
// STRING_ENCRYPT variant — string encryption only.
// =========================================================================

RT_JIT_TEST(StringEncrypt, kStringEncrypt, rt_identity, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(42), 42);
})

RT_JIT_TEST(StringEncrypt, kStringEncrypt, rt_fibonacci, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(10), 55);
})

TEST_F(DeobfuscationRoundTripTest, StringEncrypt_rt_xor_cipher) {
  if (!variants_[kStringEncrypt].loaded) GTEST_SKIP() << "DLL not loaded";
  if (!liftDeobfAndJIT(kStringEncrypt, "rt_xor_cipher"))
    GTEST_SKIP() << "Lift/deobf/JIT failed";
  auto fn = reinterpret_cast<int (*)(uint8_t *, int, uint8_t)>(jit_addr_);
  uint8_t buf[] = {0x41, 0x42, 0x43};
  int sum = fn(buf, 3, 0xFF);
  EXPECT_EQ(sum, 567);
}

// =========================================================================
// VECTORIZE variant — scalar-to-vector only.
// =========================================================================

RT_JIT_TEST(Vectorize, kVectorize, rt_identity, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(42), 42);
})

RT_JIT_TEST(Vectorize, kVectorize, rt_fibonacci, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(10), 55);
  EXPECT_EQ(fn(20), 6765);
})

RT_JIT_TEST(Vectorize, kVectorize, rt_popcount, {
  auto fn = reinterpret_cast<int (*)(uint32_t)>(jit_addr_);
  EXPECT_EQ(fn(0xFF), 8);
  EXPECT_EQ(fn(0xFFFFFFFF), 32);
})

RT_JIT_TEST(Vectorize, kVectorize, rt_bitfield_pack, {
  auto fn = reinterpret_cast<uint64_t (*)(uint8_t, uint8_t, uint8_t, uint8_t)>(
      jit_addr_);
  EXPECT_EQ(fn(0x11, 0x22, 0x33, 0x44), 0x44332211ULL);
})

RT_JIT_TEST(Vectorize, kVectorize, rt_gcd, {
  auto fn = reinterpret_cast<int (*)(int, int)>(jit_addr_);
  EXPECT_EQ(fn(12, 8), 4);
  EXPECT_EQ(fn(100, 75), 25);
})

// =========================================================================
// ALL variant — every obfuscation pass combined.
// =========================================================================

RT_JIT_TEST(All, kAll, rt_identity, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(42), 42);
  EXPECT_EQ(fn(-7), -7);
})

RT_JIT_TEST(All, kAll, rt_collatz_steps, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(1), 0);
  EXPECT_EQ(fn(3), 7);
  EXPECT_EQ(fn(27), 111);
})

RT_JIT_TEST(All, kAll, rt_fibonacci, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(0), 0);
  EXPECT_EQ(fn(1), 1);
  EXPECT_EQ(fn(10), 55);
  EXPECT_EQ(fn(20), 6765);
})

RT_JIT_TEST(All, kAll, rt_gcd, {
  auto fn = reinterpret_cast<int (*)(int, int)>(jit_addr_);
  EXPECT_EQ(fn(12, 8), 4);
  EXPECT_EQ(fn(17, 13), 1);
  EXPECT_EQ(fn(100, 75), 25);
})

RT_JIT_TEST(All, kAll, rt_switch_classify, {
  auto fn = reinterpret_cast<int (*)(int)>(jit_addr_);
  EXPECT_EQ(fn(-5), 0);
  EXPECT_EQ(fn(0), 1);
  EXPECT_EQ(fn(5), 2);
  EXPECT_EQ(fn(50), 3);
  EXPECT_EQ(fn(500), 4);
  EXPECT_EQ(fn(5000), 5);
})

RT_JIT_TEST(All, kAll, rt_nested_branch, {
  auto fn = reinterpret_cast<int (*)(int, int)>(jit_addr_);
  EXPECT_EQ(fn(5, 3), 8);
  EXPECT_EQ(fn(0, -1), -1);
  EXPECT_EQ(fn(-3, 2), 5);
  EXPECT_EQ(fn(-2, -3), 6);
})

RT_JIT_TEST(All, kAll, rt_popcount, {
  auto fn = reinterpret_cast<int (*)(uint32_t)>(jit_addr_);
  EXPECT_EQ(fn(0), 0);
  EXPECT_EQ(fn(0xFF), 8);
  EXPECT_EQ(fn(0xFFFFFFFF), 32);
})

RT_JIT_TEST(All, kAll, rt_bitfield_pack, {
  auto fn = reinterpret_cast<uint64_t (*)(uint8_t, uint8_t, uint8_t, uint8_t)>(
      jit_addr_);
  EXPECT_EQ(fn(0x11, 0x22, 0x33, 0x44), 0x44332211ULL);
})

TEST_F(DeobfuscationRoundTripTest, All_rt_array_sum) {
  if (!variants_[kAll].loaded) GTEST_SKIP() << "DLL not loaded";
  if (!liftDeobfAndJIT(kAll, "rt_array_sum"))
    GTEST_SKIP() << "Lift/deobf/JIT failed";
  auto fn = reinterpret_cast<int (*)(const int *, int)>(jit_addr_);
  int arr[] = {1, 2, 3, 4, 5};
  EXPECT_EQ(fn(arr, 5), 15);
}

TEST_F(DeobfuscationRoundTripTest, All_rt_xor_cipher) {
  if (!variants_[kAll].loaded) GTEST_SKIP() << "DLL not loaded";
  if (!liftDeobfAndJIT(kAll, "rt_xor_cipher"))
    GTEST_SKIP() << "Lift/deobf/JIT failed";
  auto fn = reinterpret_cast<int (*)(uint8_t *, int, uint8_t)>(jit_addr_);
  uint8_t buf[] = {0x41, 0x42, 0x43};
  int sum = fn(buf, 3, 0xFF);
  EXPECT_EQ(sum, 567);
}

TEST_F(DeobfuscationRoundTripTest, All_rt_matrix_trace) {
  if (!variants_[kAll].loaded) GTEST_SKIP() << "DLL not loaded";
  if (!liftDeobfAndJIT(kAll, "rt_matrix_trace"))
    GTEST_SKIP() << "Lift/deobf/JIT failed";
  auto fn = reinterpret_cast<int (*)(const int *, int)>(jit_addr_);
  int mat[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  EXPECT_EQ(fn(mat, 3), 3);
}

// =========================================================================
// Structural quality tests — verify deobfuscation reduces complexity.
// =========================================================================

TEST_F(DeobfuscationRoundTripTest, Flatten_FibonacciStructure) {
  if (!variants_[kFlatten].loaded)
    GTEST_SKIP() << "Flatten DLL not loaded";
  if (!liftDeobfAndJIT(kFlatten, "rt_fibonacci"))
    GTEST_SKIP() << "Lift/deobf failed";
  unsigned bbs = countNativeBasicBlocks();
  EXPECT_LT(bbs, 40u) << "CFF recovery: expected reduced BB count (got "
                       << bbs << ")";
  unsigned switches = countNativeSwitchInsts();
  EXPECT_LE(switches, 2u) << "CFF recovery: expected few residual switches";
}

TEST_F(DeobfuscationRoundTripTest, Flatten_CollatzStructure) {
  if (!variants_[kFlatten].loaded)
    GTEST_SKIP() << "Flatten DLL not loaded";
  if (!liftDeobfAndJIT(kFlatten, "rt_collatz_steps"))
    GTEST_SKIP() << "Lift/deobf failed";
  unsigned bbs = countNativeBasicBlocks();
  EXPECT_LT(bbs, 50u) << "CFF recovery: collatz BBs = " << bbs;
}

TEST_F(DeobfuscationRoundTripTest, Substitute_PopcountInstructionCount) {
  if (!variants_[kSubstitute].loaded)
    GTEST_SKIP() << "Substitute DLL not loaded";
  if (!liftDeobfAndJIT(kSubstitute, "rt_popcount"))
    GTEST_SKIP() << "Lift/deobf failed";
  unsigned insns = countNativeInstructions();
  EXPECT_LT(insns, 200u) << "MBA: popcount should be compact after simplify ("
                          << insns << " insns)";
}

TEST_F(DeobfuscationRoundTripTest, All_IdentityCompact) {
  if (!variants_[kAll].loaded)
    GTEST_SKIP() << "All-passes DLL not loaded";
  if (!liftDeobfAndJIT(kAll, "rt_identity"))
    GTEST_SKIP() << "Lift/deobf failed";
  unsigned insns = countNativeInstructions();
  // Identity should be extremely compact after full deobfuscation.
  EXPECT_LT(insns, 50u)
      << "Full deobf: identity should be tiny (" << insns << " insns)";
}

}  // namespace
