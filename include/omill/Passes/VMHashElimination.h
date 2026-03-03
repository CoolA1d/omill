#pragma once

#include <llvm/IR/PassManager.h>

namespace omill {

/// Eliminates branchless hash integrity checks from VM handler functions.
///
/// EAC's VM uses murmur-style hash chains for handler integrity verification.
/// The hash result is used as a branchless multiplier (0 or 1) — when the
/// check passes, the result is 1, preserving computed values.  When it fails
/// (tampered), the result is 0, corrupting computations.
///
/// The hash pattern:
///   %xor   = xor i64 %input, <K>
///   %mul   = mul i64 %xor, <K>
///   %inner = lshr i64 %mul, 32
///   %shift = lshr i64 %mul, 60
///   %var   = lshr i64 %inner, %shift
///   %fold  = xor i64 %var, %mul
///   %result = mul i64 %fold, <K>
///
/// Multiple rounds are combined with OR, then compared to a magic constant:
///   %flag = icmp eq i64 (or chain), <magic>
///   %mul_flag = zext i1 %flag to i64
///
/// This pass replaces the integrity flag with `i64 1`, making the subsequent
/// multiplications identity operations.  InstCombine + ADCE then clean up
/// the dead hash computations.
///
/// Also handles the hash token range check at handler entry:
///   %cmp1 = icmp ugt i64 %token, <const>
///   %cmp2 = icmp ult i64 %token, <const>
///   %flag = or i1 %cmp1, %cmp2
///   %r10  = zext i1 %flag to i64
/// Replaced with i64 1.
class VMHashEliminationPass
    : public llvm::PassInfoMixin<VMHashEliminationPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  static llvm::StringRef name() { return "VMHashEliminationPass"; }
};

}  // namespace omill
