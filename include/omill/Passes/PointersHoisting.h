#pragma once

#include <llvm/IR/PassManager.h>

namespace omill {

/// Hoists loop-invariant pointer computation instructions (inttoptr, GEP with
/// constant indices, bitcast, addrspacecast, ptrtoint) from loop bodies to
/// loop preheaders.
///
/// Stock LICM is conservative about pointer computations in obfuscated code.
/// This pass specifically targets `inttoptr(add %base, const)` and similar
/// patterns that prevent downstream DSE/ADCE from eliminating dead switch
/// arms and stores.
class PointersHoistingPass
    : public llvm::PassInfoMixin<PointersHoistingPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  static llvm::StringRef name() { return "PointersHoistingPass"; }
};

}  // namespace omill
