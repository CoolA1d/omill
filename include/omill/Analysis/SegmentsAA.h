#pragma once

#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/IR/PassManager.h>

#include <cstdint>

namespace omill {

/// Classifies a pointer into a memory segment.
///
/// Segments:
///   Stack  — derived from alloca
///   Global — derived from GlobalVariable
///   Image  — inttoptr of constant in [imageBase, imageBase + imageSize)
///   Unknown — cannot classify (may alias anything)
///
/// Two pointers in different non-Unknown segments are guaranteed NoAlias.
class SegmentClassifier {
 public:
  enum class Segment { Stack, Global, Image, Unknown };

  SegmentClassifier() = default;
  SegmentClassifier(uint64_t image_base, uint64_t image_size)
      : image_base_(image_base), image_size_(image_size) {}

  /// Classify a pointer value into a memory segment.
  Segment classify(const llvm::Value *ptr) const;

  /// Returns true if two pointers are guaranteed not to alias.
  bool isNoAlias(const llvm::Value *A, const llvm::Value *B) const;

  bool invalidate(llvm::Function &, const llvm::PreservedAnalyses &,
                  llvm::FunctionAnalysisManager::Invalidator &) {
    return false;
  }

 private:
  uint64_t image_base_ = 0;
  uint64_t image_size_ = 0;
};

/// AAResultBase implementation using segment classification.
class SegmentsAAResult : public llvm::AAResultBase {
 public:
  explicit SegmentsAAResult(SegmentClassifier classifier)
      : classifier_(std::move(classifier)) {}

  llvm::AliasResult alias(const llvm::MemoryLocation &LocA,
                          const llvm::MemoryLocation &LocB,
                          llvm::AAQueryInfo &AAQI,
                          const llvm::Instruction *CtxI = nullptr);

  bool invalidate(llvm::Function &, const llvm::PreservedAnalyses &,
                  llvm::FunctionAnalysisManager::Invalidator &) {
    return false;
  }

 private:
  SegmentClassifier classifier_;
};

/// Function analysis returning a SegmentsAAResult.
///
/// Reads the BinaryMemoryMap (via module proxy) to determine image bounds.
/// If unavailable, falls back to a conservative classifier (Image segment
/// never matches, only Stack/Global still proven NoAlias).
class SegmentsAA : public llvm::AnalysisInfoMixin<SegmentsAA> {
  friend llvm::AnalysisInfoMixin<SegmentsAA>;
  static llvm::AnalysisKey Key;

 public:
  using Result = SegmentsAAResult;
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};

}  // namespace omill
