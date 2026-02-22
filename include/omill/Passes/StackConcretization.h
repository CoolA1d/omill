#pragma once

#include <llvm/IR/PassManager.h>

namespace omill {

/// Concretizes remaining RSP-relative stack accesses that RecoverStackFrame
/// missed, producing allocas that SROA can decompose into SSA values.
///
/// RecoverStackFrame only detects stack bases that have at least one
/// NEGATIVE-offset inttoptr user.  This misses:
///   - Leaf functions accessing shadow space / args at positive RSP offsets
///   - Home register stores at [rsp+0x8..0x20] in Win64 leaf functions
///   - Stack accesses through alignment-masked RSP (and rsp, -16)
///   - Accesses through RBP when RBP is a copy of RSP with no negative uses
///
/// This pass complements RecoverStackFrame by:
///   1. Detecting stack bases from ANY constant-offset inttoptr (pos or neg)
///   2. Handling alignment masking: and(base, -pow2) as a stack-derived value
///   3. Skipping inttoptr that RecoverStackFrame already replaced
///   4. Creating [N x i8] allocas for each remaining offset cluster
///
/// Runs AFTER RecoverStackFrame and BEFORE SROA.
class StackConcretizationPass
    : public llvm::PassInfoMixin<StackConcretizationPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  static llvm::StringRef name() { return "StackConcretizationPass"; }
};

}  // namespace omill
