#include "omill/Passes/PartialOverlapDSE.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Operator.h>
#include <llvm/Support/Debug.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace omill {

namespace {

/// A half-open byte interval [start, end).
struct Interval {
  int64_t start;
  int64_t end;

  bool contains(const Interval &other) const {
    return start <= other.start && other.end <= end;
  }
};

/// Sorted, non-overlapping set of byte intervals.
/// Supports union, subtraction, and subset queries.
class IntervalSet {
 public:
  /// Check if the given interval is fully covered by intervals in this set.
  bool covers(Interval query) const {
    for (const auto &iv : intervals_) {
      if (iv.start > query.start)
        break;
      if (iv.contains(query))
        return true;
    }
    return false;
  }

  /// Add an interval, merging with overlapping/adjacent intervals.
  void add(Interval iv) {
    std::vector<Interval> merged;
    bool inserted = false;
    for (const auto &existing : intervals_) {
      if (inserted || existing.end < iv.start) {
        merged.push_back(existing);
        continue;
      }
      if (existing.start > iv.end) {
        if (!inserted) {
          merged.push_back(iv);
          inserted = true;
        }
        merged.push_back(existing);
        continue;
      }
      // Overlapping or adjacent: merge into iv.
      iv.start = std::min(iv.start, existing.start);
      iv.end = std::max(iv.end, existing.end);
    }
    if (!inserted)
      merged.push_back(iv);
    intervals_ = std::move(merged);
  }

  /// Remove an interval from the set (punch a hole).
  void remove(Interval hole) {
    std::vector<Interval> result;
    for (const auto &iv : intervals_) {
      if (iv.end <= hole.start || iv.start >= hole.end) {
        result.push_back(iv);
        continue;
      }
      // Partial overlap: keep non-overlapping portions.
      if (iv.start < hole.start)
        result.push_back({iv.start, hole.start});
      if (iv.end > hole.end)
        result.push_back({hole.end, iv.end});
    }
    intervals_ = std::move(result);
  }

  void clear() { intervals_.clear(); }

 private:
  std::vector<Interval> intervals_;
};

/// Resolve a pointer to (base, byte offset).
/// Returns nullopt for non-constant offsets.
std::optional<std::pair<llvm::Value *, int64_t>> resolvePointer(
    llvm::Value *ptr, const llvm::DataLayout &DL) {
  ptr = ptr->stripPointerCasts();

  int64_t totalOffset = 0;

  // Walk through GEP chains accumulating constant offsets.
  while (auto *GEP = llvm::dyn_cast<llvm::GEPOperator>(ptr)) {
    llvm::APInt gepOff(DL.getPointerSizeInBits(), 0);
    if (!GEP->accumulateConstantOffset(DL, gepOff))
      return std::nullopt;
    totalOffset += gepOff.getSExtValue();
    ptr = GEP->getPointerOperand()->stripPointerCasts();
  }

  return std::make_pair(ptr, totalOffset);
}

/// Get the byte width of a store/load type. Returns 0 for unsized types.
uint64_t getTypeByteWidth(llvm::Type *ty, const llvm::DataLayout &DL) {
  if (!ty->isSized())
    return 0;
  return DL.getTypeStoreSize(ty);
}

/// Returns true if the instruction may read or write memory in a way
/// that should invalidate tracking.
bool isClobberingCall(const llvm::Instruction &I) {
  if (auto *CI = llvm::dyn_cast<llvm::CallBase>(&I)) {
    if (CI->onlyReadsMemory())
      return false;
    return true;
  }
  return false;
}

}  // namespace

llvm::PreservedAnalyses PartialOverlapDSEPass::run(
    llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  if (F.isDeclaration())
    return llvm::PreservedAnalyses::all();

  const auto &DL = F.getDataLayout();
  bool changed = false;

  for (auto &BB : F) {
    // Per-base interval sets tracking bytes that are "covered" by later stores
    // (walking backward). A store is dead if its range is fully covered.
    llvm::DenseMap<llvm::Value *, IntervalSet> coveredBytes;
    llvm::SmallVector<llvm::Instruction *, 8> deadStores;

    // Walk backward through the basic block.
    for (auto it = BB.rbegin(); it != BB.rend(); ++it) {
      llvm::Instruction &I = *it;

      // Memory-clobbering calls invalidate all tracking.
      if (isClobberingCall(I)) {
        coveredBytes.clear();
        continue;
      }

      // Handle loads: remove loaded byte range from covered set (those bytes
      // are needed by this load, so earlier stores to them are live).
      if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
        if (LI->isVolatile())
          continue;
        auto resolved = resolvePointer(LI->getPointerOperand(), DL);
        if (!resolved)
          continue;
        auto [base, offset] = *resolved;
        uint64_t width = getTypeByteWidth(LI->getType(), DL);
        if (width == 0)
          continue;
        auto mapIt = coveredBytes.find(base);
        if (mapIt != coveredBytes.end())
          mapIt->second.remove(
              {offset, static_cast<int64_t>(offset + width)});
        continue;
      }

      // Handle stores.
      auto *SI = llvm::dyn_cast<llvm::StoreInst>(&I);
      if (!SI)
        continue;

      // Never eliminate volatile stores.
      if (SI->isVolatile())
        continue;

      auto resolved = resolvePointer(SI->getPointerOperand(), DL);
      if (!resolved) {
        // Unknown store destination: conservatively clear all tracking.
        coveredBytes.clear();
        continue;
      }

      auto [base, offset] = *resolved;
      uint64_t width = getTypeByteWidth(SI->getValueOperand()->getType(), DL);
      if (width == 0)
        continue;

      Interval storeRange{offset, static_cast<int64_t>(offset + width)};

      // Check if this store's entire range is already covered by later stores.
      auto &ivSet = coveredBytes[base];
      if (ivSet.covers(storeRange)) {
        deadStores.push_back(SI);
      }

      // Union this store's range into the covered set.
      ivSet.add(storeRange);
    }

    // Erase dead stores.
    for (auto *dead : deadStores) {
      dead->eraseFromParent();
      changed = true;
    }
  }

  return changed ? llvm::PreservedAnalyses::none()
                 : llvm::PreservedAnalyses::all();
}

}  // namespace omill
