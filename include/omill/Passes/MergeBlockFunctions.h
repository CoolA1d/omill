#pragma once

#include <llvm/IR/PassManager.h>

namespace omill {

/// Module pass that merges per-block functions (produced by BlockLifter)
/// into multi-block trace functions.
///
/// Each group of block-functions sharing a common entry point is merged
/// into a single function by inlining musttail calls between blocks.
/// The resulting functions look identical to what TraceLifter would
/// produce — a single function with one BasicBlock per original block,
/// connected by branches.
///
/// The merge is cycle-aware: back-edges (loops) are handled by inlining
/// in reverse-post-order and converting remaining musttail self/back-edge
/// calls into branches after all forward edges have been inlined.
///
/// After merging, the individual block-functions are internalized and
/// removed by GlobalDCE.
class MergeBlockFunctionsPass
    : public llvm::PassInfoMixin<MergeBlockFunctionsPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &AM);

  static llvm::StringRef name() { return "MergeBlockFunctionsPass"; }
};

}  // namespace omill
