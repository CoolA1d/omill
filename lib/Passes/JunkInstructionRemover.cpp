#include "omill/Passes/JunkInstructionRemover.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/Transforms/Utils/Local.h>

namespace omill {

namespace {

using namespace llvm::PatternMatch;

/// Replace all uses of \p Old with \p New, then mark \p Old for removal.
void replaceAndMark(llvm::Instruction *Old, llvm::Value *New,
                    llvm::SmallVectorImpl<llvm::Instruction *> &Dead) {
  Old->replaceAllUsesWith(New);
  Dead.push_back(Old);
}

/// Match identity binary operations and replace with the non-constant operand.
/// Patterns: add %x,0  sub %x,0  mul %x,1  xor %x,0  or %x,0
///           and %x,-1  shl %x,0  lshr %x,0  ashr %x,0
bool foldIdentityOps(llvm::Function &F,
                     llvm::SmallVectorImpl<llvm::Instruction *> &Dead) {
  bool changed = false;
  for (auto it = llvm::inst_begin(F), end = llvm::inst_end(F); it != end;
       ++it) {
    auto *I = &*it;
    auto *BO = llvm::dyn_cast<llvm::BinaryOperator>(I);
    if (!BO)
      continue;

    llvm::Value *X = nullptr;
    switch (BO->getOpcode()) {
    case llvm::Instruction::Add:
    case llvm::Instruction::Xor:
    case llvm::Instruction::Or:
    case llvm::Instruction::Shl:
    case llvm::Instruction::LShr:
    case llvm::Instruction::AShr:
      // op %x, 0 → %x  or  op 0, %x → %x (commutative for add/xor/or)
      if (match(BO->getOperand(1), m_Zero()))
        X = BO->getOperand(0);
      else if (match(BO->getOperand(0), m_Zero()) &&
               (BO->getOpcode() == llvm::Instruction::Add ||
                BO->getOpcode() == llvm::Instruction::Xor ||
                BO->getOpcode() == llvm::Instruction::Or))
        X = BO->getOperand(1);
      break;
    case llvm::Instruction::Sub:
      // sub %x, 0 → %x
      if (match(BO->getOperand(1), m_Zero()))
        X = BO->getOperand(0);
      break;
    case llvm::Instruction::Mul:
      // mul %x, 1 → %x  or  mul 1, %x → %x
      if (match(BO->getOperand(1), m_One()))
        X = BO->getOperand(0);
      else if (match(BO->getOperand(0), m_One()))
        X = BO->getOperand(1);
      break;
    case llvm::Instruction::And:
      // and %x, -1 → %x  or  and -1, %x → %x
      if (match(BO->getOperand(1), m_AllOnes()))
        X = BO->getOperand(0);
      else if (match(BO->getOperand(0), m_AllOnes()))
        X = BO->getOperand(1);
      break;
    default:
      break;
    }

    if (X) {
      replaceAndMark(BO, X, Dead);
      changed = true;
    }
  }
  return changed;
}

/// Match self-canceling pairs: not(not(x)) → x, neg(neg(x)) → x.
bool foldSelfCancelingPairs(llvm::Function &F,
                            llvm::SmallVectorImpl<llvm::Instruction *> &Dead) {
  bool changed = false;
  for (auto it = llvm::inst_begin(F), end = llvm::inst_end(F); it != end;
       ++it) {
    auto *I = &*it;

    // not(not(x)): xor(xor(x, -1), -1) → x
    {
      llvm::Value *X;
      if (match(I, m_Not(m_Not(m_Value(X))))) {
        // Verify the inner not has no other uses (safe to remove pair).
        auto *innerNot = llvm::cast<llvm::Instruction>(I->getOperand(0));
        if (innerNot->hasOneUse()) {
          replaceAndMark(I, X, Dead);
          Dead.push_back(innerNot);
          changed = true;
          continue;
        }
      }
    }

    // neg(neg(x)): sub(0, sub(0, x)) → x
    {
      llvm::Value *X;
      if (match(I, m_Neg(m_Neg(m_Value(X))))) {
        auto *innerNeg = llvm::cast<llvm::Instruction>(I->getOperand(1));
        if (innerNeg->hasOneUse()) {
          replaceAndMark(I, X, Dead);
          Dead.push_back(innerNeg);
          changed = true;
          continue;
        }
      }
    }
  }
  return changed;
}

/// Remove allocas that have no load users (only stores).
bool removeUnusedAllocas(llvm::Function &F,
                         llvm::SmallVectorImpl<llvm::Instruction *> &Dead) {
  bool changed = false;
  for (auto it = llvm::inst_begin(F), end = llvm::inst_end(F); it != end;
       ++it) {
    auto *AI = llvm::dyn_cast<llvm::AllocaInst>(&*it);
    if (!AI)
      continue;

    bool hasLoad = false;
    bool allUsesAreStoresOrLifetime = true;
    llvm::SmallVector<llvm::Instruction *, 4> users;

    for (auto *U : AI->users()) {
      auto *UI = llvm::dyn_cast<llvm::Instruction>(U);
      if (!UI) {
        allUsesAreStoresOrLifetime = false;
        break;
      }
      if (llvm::isa<llvm::LoadInst>(UI)) {
        hasLoad = true;
        break;
      }
      if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(UI)) {
        // Only count stores TO the alloca (pointer operand), not stores OF
        // the alloca value.
        if (SI->getPointerOperand() == AI) {
          users.push_back(SI);
          continue;
        }
      }
      // GEP, bitcast, or other use — conservatively keep.
      allUsesAreStoresOrLifetime = false;
      break;
    }

    if (!hasLoad && allUsesAreStoresOrLifetime) {
      for (auto *UI : users)
        Dead.push_back(UI);
      Dead.push_back(AI);
      changed = true;
    }
  }
  return changed;
}

/// Remove pointer roundtrips: inttoptr(ptrtoint %p) → %p
/// when the result type matches the original pointer type.
bool foldPointerRoundtrips(
    llvm::Function &F, llvm::SmallVectorImpl<llvm::Instruction *> &Dead) {
  bool changed = false;
  for (auto it = llvm::inst_begin(F), end = llvm::inst_end(F); it != end;
       ++it) {
    auto *I = &*it;
    auto *ITP = llvm::dyn_cast<llvm::IntToPtrInst>(I);
    if (!ITP)
      continue;

    auto *PTI = llvm::dyn_cast<llvm::PtrToIntInst>(ITP->getOperand(0));
    if (!PTI)
      continue;

    auto *origPtr = PTI->getPointerOperand();
    // In opaque-pointer LLVM, all pointer types in the same address space are
    // identical, so inttoptr(ptrtoint(%p)) always roundtrips.
    if (origPtr->getType() == ITP->getType()) {
      replaceAndMark(ITP, origPtr, Dead);
      if (PTI->hasOneUse())
        Dead.push_back(PTI);
      changed = true;
    }
  }
  return changed;
}

}  // namespace

llvm::PreservedAnalyses JunkInstructionRemoverPass::run(
    llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  if (F.isDeclaration())
    return llvm::PreservedAnalyses::all();

  bool changed = false;
  llvm::SmallVector<llvm::Instruction *, 16> Dead;

  changed |= foldIdentityOps(F, Dead);
  changed |= foldSelfCancelingPairs(F, Dead);
  changed |= removeUnusedAllocas(F, Dead);
  changed |= foldPointerRoundtrips(F, Dead);

  // Batch removal — erase in reverse order to avoid use-before-def issues.
  for (auto it = Dead.rbegin(); it != Dead.rend(); ++it) {
    auto *I = *it;
    if (!I->use_empty())
      I->replaceAllUsesWith(llvm::UndefValue::get(I->getType()));
    I->eraseFromParent();
  }

  // Let LLVM clean up any transitively dead instructions.
  if (changed) {
    llvm::SmallVector<llvm::Instruction *, 8> worklist;
    for (auto it = llvm::inst_begin(F), end = llvm::inst_end(F); it != end;
         ++it)
      worklist.push_back(&*it);
    for (auto *I : worklist)
      llvm::RecursivelyDeleteTriviallyDeadInstructions(I);
  }

  return changed ? llvm::PreservedAnalyses::none()
                 : llvm::PreservedAnalyses::all();
}

}  // namespace omill
