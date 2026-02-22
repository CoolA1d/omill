#pragma once

#include <llvm/IR/PassManager.h>

namespace omill {

/// Replaces indirect memory loads via GEP with a computed index (where the
/// index has few possible values determined by computeKnownBits) with an
/// explicit select/switch cascade of concrete loads.
///
/// VMProtect uses `getelementptr @table, %masked_index` for flag-based branch
/// selection.  When the mask is known (e.g., `%idx = and %x, 1`), this pass
/// turns opaque memory into clean `select` that enables constant propagation.
class KnownIndexSelectPass
    : public llvm::PassInfoMixin<KnownIndexSelectPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  static llvm::StringRef name() { return "KnownIndexSelectPass"; }
};

}  // namespace omill
