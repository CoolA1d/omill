#include "LiftAndOptFixture.h"
#include "PELoader.h"

#include <llvm/ADT/StringExtras.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifndef OBF_TEST_DLL_PATH
#error "OBF_TEST_DLL_PATH must be defined to point to obf_test.dll"
#endif

extern "C" {
void *__stdcall VirtualAlloc(void *lpAddress, size_t dwSize,
                             unsigned long flAllocationType,
                             unsigned long flProtect);
int __stdcall VirtualFree(void *lpAddress, size_t dwSize,
                          unsigned long dwFreeType);
}

using namespace omill::e2e;

namespace {

class NullSymbolGenerator : public llvm::orc::DefinitionGenerator {
 public:
  llvm::Error tryToGenerate(
      llvm::orc::LookupState &, llvm::orc::LookupKind, llvm::orc::JITDylib &JD,
      llvm::orc::JITDylibLookupFlags,
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

class OLLVMObfAlgorithmsTest : public LiftAndOptFixture {
 protected:
  static PEInfo *shared_pe_;

  static void SetUpTestSuite() {
    shared_pe_ = new PEInfo();
    if (!loadPE(OBF_TEST_DLL_PATH, *shared_pe_)) {
      delete shared_pe_;
      shared_pe_ = nullptr;
      FAIL() << "Failed to load " << OBF_TEST_DLL_PATH;
    }
  }

  static void TearDownTestSuite() {
    delete shared_pe_;
    shared_pe_ = nullptr;
  }

  void SetUp() override {
    LiftAndOptFixture::SetUp();
    ASSERT_NE(shared_pe_, nullptr) << "Shared PE not loaded";
    ASSERT_FALSE(shared_pe_->exports.empty()) << "No exports found in DLL";
    ASSERT_NE(shared_pe_->text_base, 0u) << "No .text section found";
  }

  void TearDown() override {
    unmapBinaryForJIT();
  }

  uint64_t getExportVA(const std::string &name) {
    auto it = shared_pe_->exports.find(name);
    EXPECT_NE(it, shared_pe_->exports.end()) << "Export not found: " << name;
    return (it != shared_pe_->exports.end()) ? it->second : 0;
  }

  llvm::Module *liftExport(uint64_t export_va) {
    traceManager().clearTraces();

    text_copy_.resize(shared_pe_->text_size);
    bool read_ok = shared_pe_->memory_map.read(
        shared_pe_->text_base, text_copy_.data(),
        static_cast<unsigned>(shared_pe_->text_size));
    EXPECT_TRUE(read_ok) << "memory_map.read() failed for .text";
    if (!read_ok)
      return nullptr;

    setCode(text_copy_.data(), text_copy_.size(), shared_pe_->text_base);
    traceManager().setBaseAddr(export_va);
    traceManager().setMemoryMap(&shared_pe_->memory_map);
    return lift();
  }

  bool liftAndOptimize(const std::string &export_name) {
    uint64_t va = getExportVA(export_name);
    if (va == 0)
      return false;

    current_export_va_ = va;
    expected_native_prefix_.clear();

    auto *M = liftExport(va);
    if (!M)
      return false;

    auto trace_it = traceManager().traces().find(va);
    if (trace_it != traceManager().traces().end() && trace_it->second) {
      expected_native_prefix_ = trace_it->second->getName().str();
    }

    omill::PipelineOptions opts;
    opts.recover_abi = true;
    opts.deobfuscate = true;
    optimizeWithMemoryMap(opts, shared_pe_->memory_map);
    return true;
  }

  static std::optional<uint64_t> parseNativeVA(llvm::StringRef name) {
    if (!name.starts_with("sub_") || !name.contains("_native"))
      return std::nullopt;

    llvm::StringRef core = name.drop_front(4);
    size_t hex_len = 0;
    while (hex_len < core.size() && llvm::isHexDigit(core[hex_len]))
      ++hex_len;
    if (hex_len == 0)
      return std::nullopt;

    uint64_t va = 0;
    if (core.take_front(hex_len).getAsInteger(16, va))
      return std::nullopt;
    return va;
  }

  llvm::Function *findNativeForCurrentExport() {
    if (!module() || current_export_va_ == 0)
      return nullptr;

    if (!expected_native_prefix_.empty()) {
      std::string exact_name = expected_native_prefix_ + "_native";
      if (auto *F = module()->getFunction(exact_name)) {
        if (!F->isDeclaration())
          return F;
      }
      for (auto &F : *module()) {
        if (F.isDeclaration() || !F.getName().contains("_native"))
          continue;
        if (F.getName().starts_with(expected_native_prefix_))
          return &F;
      }
    }

    llvm::Function *nearest = nullptr;
    uint64_t nearest_dist = UINT64_MAX;
    for (auto &F : *module()) {
      if (F.isDeclaration() || !F.getName().contains("_native"))
        continue;
      auto va = parseNativeVA(F.getName());
      if (!va)
        continue;
      if (*va == current_export_va_)
        return &F;
      uint64_t dist = (*va > current_export_va_) ? (*va - current_export_va_)
                                                 : (current_export_va_ - *va);
      if (dist < nearest_dist) {
        nearest_dist = dist;
        nearest = &F;
      }
    }
    if (nearest && nearest_dist <= 0x400)
      return nearest;
    return nullptr;
  }

  llvm::Function *findNativeFunction() {
    if (current_export_va_ != 0) {
      return findNativeForCurrentExport();
    }
    for (auto &F : *module()) {
      if (!F.isDeclaration() && F.getName().contains("_native"))
        return &F;
    }
    return nullptr;
  }

  unsigned countFunctionInstructions(const llvm::Function &F) {
    unsigned n = 0;
    for (const auto &BB : F)
      n += BB.size();
    return n;
  }

  unsigned countTargetNativeInstructions() {
    auto *NF = findNativeFunction();
    if (!NF)
      return 0;
    return countFunctionInstructions(*NF);
  }

  bool mapBinaryForJIT() {
    constexpr unsigned long kMemCommit = 0x1000;
    constexpr unsigned long kMemReserve = 0x2000;
    constexpr unsigned long kPageRW = 0x04;

    uint64_t min_va = UINT64_MAX, max_va = 0;
    shared_pe_->memory_map.forEachRegion(
        [&](uint64_t base, const uint8_t *, size_t size) {
          if (base < min_va)
            min_va = base;
          if (base + size > max_va)
            max_va = base + size;
        });
    if (min_va >= max_va)
      return false;

    size_t total_size = static_cast<size_t>(max_va - min_va);
    void *block = VirtualAlloc(reinterpret_cast<void *>(min_va), total_size,
                               kMemCommit | kMemReserve, kPageRW);
    if (!block)
      return false;

    jit_mapped_regions_.push_back({block, total_size});
    shared_pe_->memory_map.forEachRegion(
        [&](uint64_t base, const uint8_t *data, size_t size) {
          std::memcpy(reinterpret_cast<void *>(base), data, size);
        });
    return true;
  }

  void unmapBinaryForJIT() {
    constexpr unsigned long kMemRelease = 0x8000;
    for (auto &r : jit_mapped_regions_) {
      VirtualFree(r.addr, 0, kMemRelease);
    }
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
    EXPECT_NE(NF, nullptr) << "No native function found in module";
    if (!NF)
      return nullptr;
    std::string name = NF->getName().str();

    for (auto &F : *module()) {
      if (F.isDeclaration())
        continue;
      if (!F.getName().contains("_native")) {
        F.deleteBody();
        F.setLinkage(llvm::GlobalValue::ExternalLinkage);
      }
    }

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

    llvm::SmallString<0> buf;
    {
      llvm::raw_svector_ostream os(buf);
      llvm::WriteBitcodeToFile(*module(), os);
    }

    auto jit_ctx = std::make_unique<llvm::LLVMContext>();
    auto mem_buf = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(buf.data(), buf.size()), "", false);
    auto mod_or = llvm::parseBitcodeFile(mem_buf->getMemBufferRef(), *jit_ctx);
    EXPECT_TRUE(!!mod_or) << "Failed to parse bitcode for JIT";
    if (!mod_or) {
      llvm::consumeError(mod_or.takeError());
      return nullptr;
    }

    auto jit_or = llvm::orc::LLJITBuilder().create();
    EXPECT_TRUE(!!jit_or) << "Failed to create LLJIT";
    if (!jit_or) {
      llvm::consumeError(jit_or.takeError());
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
    EXPECT_FALSE(!!err) << "Failed to add module to JIT";
    if (err) {
      llvm::consumeError(std::move(err));
      return nullptr;
    }

    auto sym_or = jit_->lookup(name);
    EXPECT_TRUE(!!sym_or) << "Failed to look up " << name << " in JIT";
    if (!sym_or) {
      llvm::consumeError(sym_or.takeError());
      return nullptr;
    }
    return reinterpret_cast<void *>(sym_or->getValue());
  }

  struct MappedRegion {
    void *addr;
    size_t size;
  };
  std::vector<MappedRegion> jit_mapped_regions_;
  std::unique_ptr<llvm::orc::LLJIT> jit_;
  std::vector<uint8_t> text_copy_;
  uint64_t current_export_va_ = 0;
  std::string expected_native_prefix_;
};

PEInfo *OLLVMObfAlgorithmsTest::shared_pe_ = nullptr;

TEST_F(OLLVMObfAlgorithmsTest, CRC32JIT) {
  ASSERT_TRUE(liftAndOptimize("obf_algo_crc32"));
  ASSERT_TRUE(verifyModule()) << "Module invalid after optimization";

  auto *NF = findNativeFunction();
  ASSERT_NE(NF, nullptr);
  EXPECT_EQ(NF->arg_size(), 2u)
      << "Expected CRC32 signature to recover 2 params (ptr, len)";
  EXPECT_LT(countTargetNativeInstructions(), 2500u)
      << "Expected CRC32 to recover to a reasonable instruction count";

  mapBinaryForJIT();
  void *addr = jitLookupNative();
  if (!addr) {
    GTEST_SKIP() << "JIT compilation failed";
    return;
  }

  auto fn = reinterpret_cast<uint64_t (*)(const uint8_t *, uint64_t)>(addr);
  const uint8_t input[] = "Hello, World!";
  uint64_t result = fn(input, 13);
  EXPECT_EQ(static_cast<uint32_t>(result), 0xEC4AC3D0u)
      << "CRC32 mismatch: got 0x" << std::hex << static_cast<uint32_t>(result)
      << ", expected 0xEC4AC3D0";
}

TEST_F(OLLVMObfAlgorithmsTest, SHA256JIT) {
  ASSERT_TRUE(liftAndOptimize("obf_algo_sha256"));
  ASSERT_TRUE(verifyModule()) << "Module invalid after optimization";

  auto *NF = findNativeFunction();
  ASSERT_NE(NF, nullptr);
  EXPECT_GE(NF->arg_size(), 3u)
      << "Expected SHA256 to recover at least 3 params";
  EXPECT_LE(NF->arg_size(), 4u)
      << "Expected SHA256 to recover at most 4 params";
  EXPECT_LT(countTargetNativeInstructions(), 18000u)
      << "Expected SHA256 to recover to a reasonable instruction count";

  mapBinaryForJIT();
  void *addr = jitLookupNative();
  if (!addr) {
    GTEST_SKIP() << "JIT compilation failed";
    return;
  }

  auto fn =
      reinterpret_cast<uint64_t (*)(const uint8_t *, uint64_t, uint8_t *)>(
          addr);
  const uint8_t input[] = "Hello, World!";
  uint8_t digest[32] = {};
  fn(input, 13, digest);

  const uint8_t expected[32] = {
      0xdf, 0xfd, 0x60, 0x21, 0xbb, 0x2b, 0xd5, 0xb0,
      0xaf, 0x67, 0x62, 0x90, 0x80, 0x9e, 0xc3, 0xa5,
      0x31, 0x91, 0xdd, 0x81, 0xc7, 0xf7, 0x0a, 0x4b,
      0x28, 0x68, 0x8a, 0x36, 0x21, 0x82, 0x98, 0x6f};
  EXPECT_EQ(std::memcmp(digest, expected, 32), 0) << "SHA256 digest mismatch";
}

}  // namespace
