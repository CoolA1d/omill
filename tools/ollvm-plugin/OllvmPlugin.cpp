/// @file OllvmPlugin.cpp
/// LLVM pass plugin for per-function OLLVM obfuscation via annotations.
///
/// Load with: clang -fpass-plugin=OllvmPlugin.so ...
/// Or:        opt -load-pass-plugin=OllvmPlugin.so -passes=ollvm-obfuscate ...
///
/// Annotate functions in source with OLLVM_FLATTEN, OLLVM_SUBSTITUTE, etc.

#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "BogusControlFlow.h"
#include "ConstantUnfolding.h"
#include "Flattening.h"
#include "OpaquePredicates.h"
#include "StringEncryption.h"
#include "Substitution.h"
#include "Vectorize.h"

namespace {

// ---------------------------------------------------------------------------
// Annotation helpers
// ---------------------------------------------------------------------------

/// Collect all annotation strings attached to @p F via llvm.global.annotations.
static std::vector<llvm::StringRef>
collectAnnotations(llvm::Function &F) {
  std::vector<llvm::StringRef> result;
  auto *GA = F.getParent()->getNamedGlobal("llvm.global.annotations");
  if (!GA || !GA->hasInitializer())
    return result;
  auto *CA = llvm::dyn_cast<llvm::ConstantArray>(GA->getInitializer());
  if (!CA)
    return result;
  for (unsigned i = 0, e = CA->getNumOperands(); i < e; ++i) {
    auto *CS = llvm::dyn_cast<llvm::ConstantStruct>(CA->getOperand(i));
    if (!CS || CS->getNumOperands() < 2)
      continue;
    auto *annotatedFn = llvm::dyn_cast<llvm::Function>(
        CS->getOperand(0)->stripPointerCasts());
    if (annotatedFn != &F)
      continue;
    auto *annoGV = llvm::dyn_cast<llvm::GlobalVariable>(
        CS->getOperand(1)->stripPointerCasts());
    if (!annoGV || !annoGV->hasInitializer())
      continue;
    if (auto *cda =
            llvm::dyn_cast<llvm::ConstantDataArray>(annoGV->getInitializer())) {
      result.push_back(cda->getAsString().rtrim('\0'));
    }
  }
  return result;
}

/// Return true when @p annos contains @p key.
static bool hasAnno(const std::vector<llvm::StringRef> &annos,
                    llvm::StringRef key) {
  for (auto &a : annos)
    if (a == key)
      return true;
  return false;
}

// ---------------------------------------------------------------------------
// Seed derivation (same splitmix as ollvm-obf/main.cpp)
// ---------------------------------------------------------------------------

static uint32_t mixSeed(uint32_t base, uint32_t salt) {
  uint32_t x = base ^ salt;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ---------------------------------------------------------------------------
// The module pass
// ---------------------------------------------------------------------------

struct OllvmObfuscatePass : llvm::PassInfoMixin<OllvmObfuscatePass> {
  uint32_t seed;
  explicit OllvmObfuscatePass(uint32_t s = 0xB16B00B5u) : seed(s) {}

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager & /*MAM*/) {
    // Scan annotations once.
    struct FnFlags {
      bool flatten = false;
      bool substitute = false;
      bool stringEncrypt = false;
      bool constUnfold = false;
      bool vectorize = false;
      bool opaquePred = false;
      bool bogusCf = false;
    };

    bool anyAnnotated = false;
    bool wantStringEncrypt = false;
    std::vector<std::pair<llvm::Function *, FnFlags>> plan;

    for (auto &F : M) {
      if (F.isDeclaration())
        continue;
      auto annos = collectAnnotations(F);
      if (annos.empty())
        continue;
      if (hasAnno(annos, "ollvm_none"))
        continue;

      FnFlags flags;
      bool all = hasAnno(annos, "ollvm_all");
      flags.flatten = all || hasAnno(annos, "ollvm_flatten");
      flags.substitute = all || hasAnno(annos, "ollvm_substitute");
      flags.stringEncrypt = all || hasAnno(annos, "ollvm_string_encrypt");
      flags.constUnfold = all || hasAnno(annos, "ollvm_const_unfold");
      flags.vectorize = all || hasAnno(annos, "ollvm_vectorize");
      flags.opaquePred = all || hasAnno(annos, "ollvm_opaque_predicates");
      flags.bogusCf = all || hasAnno(annos, "ollvm_bogus_cf");

      if (flags.stringEncrypt)
        wantStringEncrypt = true;

      anyAnnotated = true;
      plan.emplace_back(&F, flags);
    }

    if (!anyAnnotated)
      return llvm::PreservedAnalyses::all();

    // -----------------------------------------------------------------------
    // Module-level passes first.
    // -----------------------------------------------------------------------
    if (wantStringEncrypt)
      ollvm::encryptStringsModule(M, mixSeed(seed, 0x11A48D53u));

    // -----------------------------------------------------------------------
    // Per-function passes in fixed order matching ollvm-obf.
    // We run each transform across all requesting functions before the next
    // transform to maintain deterministic ordering.
    // -----------------------------------------------------------------------

    // 1. Substitution
    {
      std::mt19937 rng(mixSeed(seed, 0x5B2E6D4Fu));
      for (auto &[fn, flags] : plan)
        if (flags.substitute)
          ollvm::substituteModule(M, mixSeed(seed, 0x5B2E6D4Fu));
    }

    // Opaque predicates (before CFF so flattening sees the injected branches).
    for (auto &[fn, flags] : plan) {
      if (flags.opaquePred) {
        ollvm::insertOpaquePredicatesModule(M, mixSeed(seed, 0xD4E5F6A7u));
        break; // module-level — run once
      }
    }

    // Bogus control flow (before CFF).
    for (auto &[fn, flags] : plan) {
      if (flags.bogusCf) {
        ollvm::insertBogusControlFlowModule(M, mixSeed(seed, 0xE7F8A9B0u));
        break; // module-level — run once
      }
    }

    // Flattening.
    for (auto &[fn, flags] : plan) {
      if (flags.flatten) {
        ollvm::flattenModule(M, mixSeed(seed, 0xA1F3707Bu));
        break; // module-level — run once
      }
    }

    // Constant unfolding.
    for (auto &[fn, flags] : plan) {
      if (flags.constUnfold) {
        ollvm::unfoldConstantsModule(M, mixSeed(seed, 0xC93A1E27u));
        break; // module-level — run once
      }
    }

    // Vectorization (must run last).
    for (auto &[fn, flags] : plan) {
      if (flags.vectorize) {
        ollvm::vectorizeModule(M, mixSeed(seed, 0x3D7C9A61u));
        break; // module-level — run once
      }
    }

    return llvm::PreservedAnalyses::none();
  }
};

// ---------------------------------------------------------------------------
// Seed parsing: "ollvm-obfuscate" or "ollvm-obfuscate<seed=12345>"
// ---------------------------------------------------------------------------

static bool parseSeedParam(llvm::StringRef params, uint32_t &outSeed) {
  outSeed = 0xB16B00B5u;
  if (params.empty())
    return true;
  if (params.consume_front("seed=")) {
    unsigned long long v;
    if (params.getAsInteger(0, v))
      return false;
    outSeed = static_cast<uint32_t>(v);
    return true;
  }
  return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// Plugin entry point
// ---------------------------------------------------------------------------

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {
      LLVM_PLUGIN_API_VERSION, "OllvmPlugin", LLVM_VERSION_STRING,
      [](llvm::PassBuilder &PB) {
        // Allow explicit invocation: -passes=ollvm-obfuscate
        PB.registerPipelineParsingCallback(
            [](llvm::StringRef Name, llvm::ModulePassManager &MPM,
               llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
              if (Name.consume_front("ollvm-obfuscate")) {
                uint32_t s;
                // Name now holds the remainder inside <...> if any.
                if (!parseSeedParam(Name, s))
                  return false;
                MPM.addPass(OllvmObfuscatePass(s));
                return true;
              }
              return false;
            });

        // Also run automatically at the end of -O1/-O2/-O3 pipelines.
        PB.registerOptimizerLastEPCallback(
            [](llvm::ModulePassManager &MPM, llvm::OptimizationLevel,
               llvm::ThinOrFullLTOPhase) {
              MPM.addPass(OllvmObfuscatePass());
            });
      }};
}
