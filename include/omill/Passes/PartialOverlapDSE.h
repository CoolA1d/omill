#pragma once

#include <llvm/IR/PassManager.h>

namespace omill {

/// Extends dead store elimination to kill stores whose byte ranges are fully
/// covered by subsequent stores before any load of the same region.
///
/// Standard LLVM DSE only kills stores fully overwritten by a single later
/// store. This pass handles partial redefinition: e.g., a 64-bit store killed
/// by two adjacent 32-bit stores covering the same bytes.
class PartialOverlapDSEPass
    : public llvm::PassInfoMixin<PartialOverlapDSEPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  static llvm::StringRef name() { return "PartialOverlapDSEPass"; }
};

}  // namespace omill
