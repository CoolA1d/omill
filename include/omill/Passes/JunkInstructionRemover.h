#pragma once

#include <llvm/IR/PassManager.h>

namespace omill {

/// Removes VM-specific junk instruction patterns that survive generic LLVM
/// dead code elimination: identity operations, self-canceling pairs,
/// dead store-load pairs, unused allocas, and type roundtrip casts.
class JunkInstructionRemoverPass
    : public llvm::PassInfoMixin<JunkInstructionRemoverPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  static llvm::StringRef name() { return "JunkInstructionRemoverPass"; }
};

}  // namespace omill
