#pragma once

#include <llvm/IR/PassManager.h>

namespace omill {

/// Detect and reverse OLLVM-style control flow flattening.
///
/// Identifies switch-on-PHI dispatcher blocks where case blocks set a
/// next-state value and branch back.  Replaces dispatcher back-edges with
/// direct branches, reconstructing the original CFG.
///
/// Handles:
///   - Constant next-state values (unconditional redirect)
///   - Select of two constants (conditional redirect)
///   - Multiple PHIs in the dispatcher (values threaded through proxy PHIs)
///   - Nested/multiple dispatchers (iterates to fixpoint)
///
/// Requires ConstantMemoryFolding + InstCombine + SimplifyMBA to have run
/// first so that XOR-encrypted state values and MBA-protected selectors are
/// resolved to constants.
class ControlFlowUnflattenerPass
    : public llvm::PassInfoMixin<ControlFlowUnflattenerPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

}  // namespace omill
