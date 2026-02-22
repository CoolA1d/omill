#pragma once

#include <llvm/IR/PassManager.h>

namespace omill {

/// Synthesize high-level comparison operations from remill's explicit flag
/// modeling.
///
/// After remill lifting + Phase 1 lowering + SROA/mem2reg, flag computations
/// become individual icmp/shift/xor chains in SSA form.  This pass recognizes
/// combined flag patterns and folds them into direct icmp instructions.
///
/// Primary pattern:
///   xor(SF, OF) after SUB  →  icmp slt lhs, rhs
///
/// Where SF = icmp slt (sub lhs, rhs), 0  and OF is the overflow flag
/// formula derived from the same subtraction.
///
/// Compound patterns (JGE, JLE, JG) are then simplified by a follow-up
/// InstCombine pass:
///   not(icmp slt L, R)              →  icmp sge L, R
///   or(icmp eq L, R, icmp slt L, R) →  icmp sle L, R
///   and(icmp ne L, R, icmp sge L, R) → icmp sgt L, R
class SynthesizeFlagsPass
    : public llvm::PassInfoMixin<SynthesizeFlagsPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

}  // namespace omill
