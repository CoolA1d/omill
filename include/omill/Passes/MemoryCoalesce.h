#pragma once

#include <llvm/IR/PassManager.h>

namespace omill {

/// Merges fragmented byte-wise stores that fully define a subsequent wider
/// load. VM obfuscators scatter 64-bit values across 8/16/32-bit stores
/// (byte-wise bswap, struct init); this pass reassembles them into constants.
///
/// Operates within a single basic block. Tracks stores to known pointer
/// bases (alloca/inttoptr + GEP constant offsets) and replaces loads whose
/// byte ranges are fully covered by prior constant stores.
class MemoryCoalescePass : public llvm::PassInfoMixin<MemoryCoalescePass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  static llvm::StringRef name() { return "MemoryCoalescePass"; }
};

}  // namespace omill
