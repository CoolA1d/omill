#include "omill/Analysis/SegmentsAA.h"

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/PassManager.h>

#include "omill/Analysis/BinaryMemoryMap.h"

namespace omill {

// ===----------------------------------------------------------------------===
// SegmentClassifier
// ===----------------------------------------------------------------------===

SegmentClassifier::Segment SegmentClassifier::classify(
    const llvm::Value *ptr) const {
  if (!ptr)
    return Segment::Unknown;

  // Strip pointer casts and GEPs to find the underlying object.
  const llvm::Value *obj = llvm::getUnderlyingObject(ptr);

  // Stack: derived from alloca.
  if (llvm::isa<llvm::AllocaInst>(obj))
    return Segment::Stack;

  // Global: derived from GlobalVariable.
  if (llvm::isa<llvm::GlobalVariable>(obj))
    return Segment::Global;

  // Image: inttoptr of a constant within the binary image range.
  if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(obj)) {
    if (CE->getOpcode() == llvm::Instruction::IntToPtr) {
      if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(CE->getOperand(0))) {
        uint64_t addr = CI->getZExtValue();
        if (image_size_ > 0 && addr >= image_base_ &&
            addr < image_base_ + image_size_)
          return Segment::Image;
      }
    }
  }

  // inttoptr instruction (not constant expression).
  if (auto *I2P = llvm::dyn_cast<llvm::IntToPtrInst>(obj)) {
    if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(I2P->getOperand(0))) {
      uint64_t addr = CI->getZExtValue();
      if (image_size_ > 0 && addr >= image_base_ &&
          addr < image_base_ + image_size_)
        return Segment::Image;
    }
  }

  return Segment::Unknown;
}

bool SegmentClassifier::isNoAlias(const llvm::Value *A,
                                  const llvm::Value *B) const {
  auto segA = classify(A);
  auto segB = classify(B);
  if (segA == Segment::Unknown || segB == Segment::Unknown)
    return false;
  return segA != segB;
}

// ===----------------------------------------------------------------------===
// SegmentsAAResult
// ===----------------------------------------------------------------------===

llvm::AliasResult SegmentsAAResult::alias(const llvm::MemoryLocation &LocA,
                                          const llvm::MemoryLocation &LocB,
                                          llvm::AAQueryInfo &AAQI,
                                          const llvm::Instruction *CtxI) {
  if (classifier_.isNoAlias(LocA.Ptr, LocB.Ptr))
    return llvm::AliasResult::NoAlias;
  return AAResultBase::alias(LocA, LocB, AAQI, CtxI);
}

// ===----------------------------------------------------------------------===
// SegmentsAA (analysis)
// ===----------------------------------------------------------------------===

llvm::AnalysisKey SegmentsAA::Key;

SegmentsAAResult SegmentsAA::run(llvm::Function &F,
                                 llvm::FunctionAnalysisManager &FAM) {
  uint64_t image_base = 0;
  uint64_t image_size = 0;

  // Try to get BinaryMemoryMap from the module analysis manager.
  auto &proxy = FAM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  if (auto *map =
          proxy.getCachedResult<BinaryMemoryAnalysis>(*F.getParent())) {
    image_base = map->imageBase();
    image_size = map->imageSize();
  }

  return SegmentsAAResult(SegmentClassifier(image_base, image_size));
}

}  // namespace omill
