#pragma once

#include <llvm/IR/PassManager.h>

namespace omill {

/// Eliminate dead stores to volatile (caller-saved) State struct fields at
/// call boundaries.  After RewriteLiftedCallsToNative + AlwaysInliner, the
/// State alloca is local to each _native wrapper.  Native calls don't access
/// State — their params are explicit.  So stores to volatile fields (RAX, RCX,
/// RDX, R8-R11, XMM0-5) that aren't read before the next call or function
/// exit are provably dead.
///
/// This pass runs BEFORE SROA to make decomposition more likely to succeed.
/// Cases where SROA succeeds don't benefit (ADCE handles the rest), but
/// partially-resolved functions with residual dispatch calls do benefit.
class DeadStateStoreDSEPass
    : public llvm::PassInfoMixin<DeadStateStoreDSEPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);
};

}  // namespace omill
