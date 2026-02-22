#include "omill/Passes/MemoryCoalesce.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Debug.h>

namespace omill {

namespace {

/// Decompose a pointer into a base pointer and a constant byte offset.
/// Handles alloca, inttoptr, and chains of GEPs with constant indices.
/// Returns false if the pointer cannot be decomposed.
bool decomposePointer(llvm::Value *Ptr, const llvm::DataLayout &DL,
                      llvm::Value *&Base, int64_t &ByteOffset) {
  ByteOffset = 0;
  Ptr = Ptr->stripPointerCasts();

  while (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(Ptr)) {
    llvm::APInt GEPOffset(DL.getPointerSizeInBits(), 0);
    if (!GEP->accumulateConstantOffset(DL, GEPOffset))
      return false;
    ByteOffset += GEPOffset.getSExtValue();
    Ptr = GEP->getPointerOperand()->stripPointerCasts();
  }

  // Accept alloca or inttoptr as base pointers.
  if (llvm::isa<llvm::AllocaInst>(Ptr) ||
      llvm::isa<llvm::IntToPtrInst>(Ptr) ||
      llvm::isa<llvm::Argument>(Ptr)) {
    Base = Ptr;
    return true;
  }

  return false;
}

/// Key for the per-byte store tracking map.
struct ByteKey {
  llvm::Value *Base;
  int64_t Offset;

  bool operator==(const ByteKey &Other) const {
    return Base == Other.Base && Offset == Other.Offset;
  }
};

struct ByteKeyInfo {
  static ByteKey getEmptyKey() {
    return {llvm::DenseMapInfo<llvm::Value *>::getEmptyKey(), 0};
  }
  static ByteKey getTombstoneKey() {
    return {llvm::DenseMapInfo<llvm::Value *>::getTombstoneKey(), 0};
  }
  static unsigned getHashValue(const ByteKey &K) {
    return llvm::DenseMapInfo<llvm::Value *>::getHashValue(K.Base) ^
           llvm::DenseMapInfo<int64_t>::getHashValue(K.Offset);
  }
  static bool isEqual(const ByteKey &A, const ByteKey &B) { return A == B; }
};

/// Information about a single store contributing byte data.
struct StoreInfo {
  uint8_t ConstByte;     ///< The constant value stored at this byte.
  bool IsConst = false;  ///< Whether this is a known constant byte.
};

/// Attempt to extract constant bytes from a store instruction.
/// Returns true if all bytes of the store are constant.
bool extractConstantBytes(llvm::StoreInst *SI, const llvm::DataLayout &DL,
                          llvm::SmallVectorImpl<uint8_t> &Bytes) {
  auto *Val = SI->getValueOperand();
  auto *CI = llvm::dyn_cast<llvm::ConstantInt>(Val);
  if (!CI)
    return false;

  uint64_t StoreSizeBytes = DL.getTypeStoreSize(Val->getType());
  llvm::APInt V = CI->getValue();

  Bytes.clear();
  Bytes.reserve(StoreSizeBytes);
  for (uint64_t i = 0; i < StoreSizeBytes; ++i) {
    Bytes.push_back(static_cast<uint8_t>(V.extractBitsAsZExtValue(8, i * 8)));
  }
  return true;
}

/// Check whether a load is fully covered by tracked stores, and if so
/// return the assembled constant. Returns nullptr if not fully covered.
llvm::Constant *tryAssembleConstant(
    llvm::LoadInst *LI, llvm::Value *Base, int64_t LoadOffset,
    const llvm::DataLayout &DL,
    const llvm::DenseMap<ByteKey, StoreInfo, ByteKeyInfo> &ByteMap) {
  auto *LoadTy = LI->getType();
  uint64_t LoadSize = DL.getTypeStoreSize(LoadTy);

  // Check all bytes are covered.
  llvm::APInt Assembled(LoadSize * 8, 0);
  for (uint64_t i = 0; i < LoadSize; ++i) {
    ByteKey Key{Base, LoadOffset + static_cast<int64_t>(i)};
    auto It = ByteMap.find(Key);
    if (It == ByteMap.end() || !It->second.IsConst)
      return nullptr;

    llvm::APInt ByteVal(LoadSize * 8, It->second.ConstByte);
    ByteVal <<= (i * 8);
    Assembled |= ByteVal;
  }

  // Build the constant of the appropriate integer type.
  auto *IntTy = llvm::IntegerType::get(LI->getContext(), LoadSize * 8);
  auto *Result = llvm::ConstantInt::get(IntTy, Assembled);

  // If the load type is a pointer, wrap in inttoptr constant expr.
  if (LoadTy->isPointerTy())
    return llvm::ConstantExpr::getIntToPtr(Result, LoadTy);

  // If load type matches, return directly.
  if (LoadTy == IntTy)
    return Result;

  return nullptr;
}

/// Returns true if an instruction may clobber memory in ways we can't track.
bool isClobberingInstruction(llvm::Instruction &I) {
  if (llvm::isa<llvm::CallBase>(I))
    return true;
  // fence, atomicrmw, cmpxchg also clobber.
  if (llvm::isa<llvm::FenceInst>(I) || llvm::isa<llvm::AtomicRMWInst>(I) ||
      llvm::isa<llvm::AtomicCmpXchgInst>(I))
    return true;
  return false;
}

/// Process a single basic block, returning true if any changes were made.
bool processBlock(llvm::BasicBlock &BB, const llvm::DataLayout &DL) {
  llvm::DenseMap<ByteKey, StoreInfo, ByteKeyInfo> ByteMap;
  llvm::SmallVector<llvm::Instruction *, 8> ToErase;
  bool Changed = false;

  for (auto &I : BB) {
    // Memory-clobbering instructions clear all tracking.
    if (isClobberingInstruction(I)) {
      ByteMap.clear();
      continue;
    }

    // Track stores.
    if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(&I)) {
      if (SI->isVolatile())
        continue;

      llvm::Value *Base = nullptr;
      int64_t Offset = 0;
      if (!decomposePointer(SI->getPointerOperand(), DL, Base, Offset))
        continue;

      llvm::SmallVector<uint8_t, 8> Bytes;
      if (extractConstantBytes(SI, DL, Bytes)) {
        for (uint64_t j = 0; j < Bytes.size(); ++j) {
          ByteKey Key{Base, Offset + static_cast<int64_t>(j)};
          ByteMap[Key] = StoreInfo{Bytes[j], true};
        }
      } else {
        // Non-constant store: mark the affected bytes as unknown.
        uint64_t StoreSize =
            DL.getTypeStoreSize(SI->getValueOperand()->getType());
        for (uint64_t j = 0; j < StoreSize; ++j) {
          ByteKey Key{Base, Offset + static_cast<int64_t>(j)};
          ByteMap[Key] = StoreInfo{0, false};
        }
      }
      continue;
    }

    // Try to fold loads.
    if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
      if (LI->isVolatile())
        continue;

      // Only fold integer loads for now.
      if (!LI->getType()->isIntegerTy())
        continue;

      llvm::Value *Base = nullptr;
      int64_t Offset = 0;
      if (!decomposePointer(LI->getPointerOperand(), DL, Base, Offset))
        continue;

      if (auto *C = tryAssembleConstant(LI, Base, Offset, DL, ByteMap)) {
        LI->replaceAllUsesWith(C);
        ToErase.push_back(LI);
        Changed = true;
      }
    }
  }

  for (auto *I : ToErase)
    I->eraseFromParent();

  return Changed;
}

}  // namespace

llvm::PreservedAnalyses MemoryCoalescePass::run(
    llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  if (F.isDeclaration())
    return llvm::PreservedAnalyses::all();

  const auto &DL = F.getDataLayout();
  bool Changed = false;

  for (auto &BB : F)
    Changed |= processBlock(BB, DL);

  return Changed ? llvm::PreservedAnalyses::none()
                 : llvm::PreservedAnalyses::all();
}

}  // namespace omill
