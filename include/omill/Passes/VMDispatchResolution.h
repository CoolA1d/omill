#pragma once

#include <llvm/IR/PassManager.h>

namespace omill {

/// Resolves VM handler dispatch targets using binary-extracted handler graph.
///
/// EAC-style VMs compute dispatch targets as `image_base + RVA` where the
/// image base is loaded from an opaque VM context.  Standard optimization
/// cannot resolve these because the image base is runtime-variable in the IR.
///
/// This pass bypasses IR analysis entirely: it uses the VMHandlerGraph
/// (extracted from binary byte pattern scanning) to directly replace opaque
/// dispatch targets with known constant VAs.
///
/// For each function tagged with "omill.vm_handler":
///   1. Extract the handler's entry VA from its name (sub_<hex>)
///   2. Look up successor target VAs in the handler graph
///   3. Find __omill_dispatch_jump / __omill_dispatch_call calls
///   4. Replace the target operand with the resolved constant VA
///
/// After this pass, LowerResolvedDispatchCalls can convert the now-constant
/// dispatch calls into direct function calls.
class VMDispatchResolutionPass
    : public llvm::PassInfoMixin<VMDispatchResolutionPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

  static llvm::StringRef name() { return "VMDispatchResolutionPass"; }
};

}  // namespace omill
