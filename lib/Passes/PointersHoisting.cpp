#include "omill/Passes/PointersHoisting.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

namespace omill {

namespace {

/// Returns true if the instruction is a pointer computation or supporting
/// integer arithmetic that we want to hoist as part of a pointer computation
/// chain (e.g., add %base, 16 feeding inttoptr).
bool isHoistableCandidate(llvm::Instruction *I) {
  // Never hoist instructions with side effects or that read/write memory.
  if (I->mayHaveSideEffects() || I->mayReadOrWriteMemory())
    return false;

  switch (I->getOpcode()) {
    // Pointer computations.
    case llvm::Instruction::IntToPtr:
    case llvm::Instruction::PtrToInt:
    case llvm::Instruction::BitCast:
    case llvm::Instruction::AddrSpaceCast:
    case llvm::Instruction::GetElementPtr:
    // Integer arithmetic commonly used in address computation.
    case llvm::Instruction::Add:
    case llvm::Instruction::Sub:
    case llvm::Instruction::Mul:
    case llvm::Instruction::Shl:
    case llvm::Instruction::LShr:
    case llvm::Instruction::AShr:
    case llvm::Instruction::And:
    case llvm::Instruction::Or:
    case llvm::Instruction::Xor:
    case llvm::Instruction::ZExt:
    case llvm::Instruction::SExt:
    case llvm::Instruction::Trunc:
      return true;
    default:
      return false;
  }
}

/// Returns true if the instruction is used by a PHI in the loop header.
bool isUsedByHeaderPHI(llvm::Instruction *I, llvm::Loop *L) {
  auto *Header = L->getHeader();
  for (auto *User : I->users()) {
    if (auto *PHI = llvm::dyn_cast<llvm::PHINode>(User)) {
      if (PHI->getParent() == Header)
        return true;
    }
  }
  return false;
}

/// Returns true if all operands of I are either defined outside the loop
/// or are themselves in the hoistable set.
bool canHoist(llvm::Instruction *I, llvm::Loop *L,
              llvm::DenseSet<llvm::Instruction *> &hoistable) {
  for (auto &Op : I->operands()) {
    if (auto *OpI = llvm::dyn_cast<llvm::Instruction>(Op.get())) {
      if (L->contains(OpI) && !hoistable.count(OpI))
        return false;
    }
  }
  return true;
}

/// Hoist instructions from a single loop. Returns true if anything changed.
bool hoistFromLoop(llvm::Loop *L) {
  auto *Preheader = L->getLoopPreheader();
  if (!Preheader)
    return false;

  // First pass: identify all hoistable instructions.
  // Iterate to fixed point to handle chains of hoistable instructions.
  llvm::DenseSet<llvm::Instruction *> hoistable;
  bool added = true;
  while (added) {
    added = false;
    for (auto *BB : L->blocks()) {
      for (auto &I : *BB) {
        if (hoistable.count(&I))
          continue;
        if (!isHoistableCandidate(&I))
          continue;
        if (isUsedByHeaderPHI(&I, L))
          continue;
        if (!canHoist(&I, L, hoistable))
          continue;
        hoistable.insert(&I);
        added = true;
      }
    }
  }

  if (hoistable.empty())
    return false;

  // Collect in block order (topological w.r.t. dominance within the loop).
  llvm::SmallVector<llvm::Instruction *, 16> ordered;
  for (auto *BB : L->blocks()) {
    for (auto &I : *BB) {
      if (hoistable.count(&I))
        ordered.push_back(&I);
    }
  }

  auto *InsertPt = Preheader->getTerminator();
  for (auto *I : ordered) {
    I->moveBefore(InsertPt->getIterator());
  }

  return true;
}

}  // namespace

llvm::PreservedAnalyses PointersHoistingPass::run(
    llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  if (F.isDeclaration())
    return llvm::PreservedAnalyses::all();

  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);

  bool changed = false;

  // Process innermost loops first (post-order over the loop tree).
  llvm::SmallVector<llvm::Loop *, 8> worklist;
  for (auto *TopLoop : LI) {
    llvm::SmallVector<llvm::Loop *, 8> stack;
    stack.push_back(TopLoop);
    while (!stack.empty()) {
      auto *L = stack.pop_back_val();
      worklist.push_back(L);
      for (auto *Sub : *L)
        stack.push_back(Sub);
    }
  }
  // Reverse so innermost loops come first.
  std::reverse(worklist.begin(), worklist.end());

  for (auto *L : worklist) {
    changed |= hoistFromLoop(L);
  }

  if (!changed)
    return llvm::PreservedAnalyses::all();

  // We moved instructions but didn't change CFG structure.
  llvm::PreservedAnalyses PA;
  PA.preserveSet<llvm::CFGAnalyses>();
  return PA;
}

}  // namespace omill
