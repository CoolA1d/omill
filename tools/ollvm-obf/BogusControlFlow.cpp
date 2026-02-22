#include "BogusControlFlow.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include <cstdint>
#include <random>
#include <vector>

namespace ollvm {

/// Build an opaque predicate that always evaluates to true:
///   (x * (x + 1)) % 2 == 0
/// This holds for all integers since x*(x+1) is always even.
static llvm::Value *buildOpaqueTruePredicate(llvm::IRBuilder<> &builder,
                                             llvm::Value *x) {
  auto *ty = x->getType();
  auto *one = llvm::ConstantInt::get(ty, 1);
  auto *two = llvm::ConstantInt::get(ty, 2);
  auto *zero = llvm::ConstantInt::get(ty, 0);
  auto *xPlus1 = builder.CreateAdd(x, one, "opq.xp1");
  auto *mul = builder.CreateMul(x, xPlus1, "opq.mul");
  auto *rem = builder.CreateURem(mul, two, "opq.rem");
  return builder.CreateICmpEQ(rem, zero, "opq.cond");
}

/// Determine whether a basic block is eligible for BCF.
/// The block must end with an unconditional branch (not entry, not return,
/// not invoke/resume, not trivial).
static bool isEligibleBlock(llvm::BasicBlock &BB) {
  if (&BB == &BB.getParent()->getEntryBlock())
    return false;
  if (BB.size() <= 1)
    return false;

  auto *term = BB.getTerminator();
  auto *br = llvm::dyn_cast<llvm::BranchInst>(term);
  if (!br || !br->isUnconditional())
    return false;

  return true;
}

/// Mutate a cloned instruction to look structurally different.
static void mutateClonedInstruction(llvm::Instruction *I,
                                    std::mt19937 &rng) {
  if (auto *bin = llvm::dyn_cast<llvm::BinaryOperator>(I)) {
    std::uniform_int_distribution<int> coin(0, 1);
    if (coin(rng) == 0)
      return;

    if (bin->getOpcode() == llvm::Instruction::Add) {
      auto *newBin = llvm::BinaryOperator::Create(
          llvm::Instruction::Sub, bin->getOperand(0), bin->getOperand(1),
          bin->getName(), bin->getIterator());
      bin->replaceAllUsesWith(newBin);
      bin->eraseFromParent();
    } else if (bin->getOpcode() == llvm::Instruction::Sub) {
      auto *newBin = llvm::BinaryOperator::Create(
          llvm::Instruction::Add, bin->getOperand(0), bin->getOperand(1),
          bin->getName(), bin->getIterator());
      bin->replaceAllUsesWith(newBin);
      bin->eraseFromParent();
    }
    return;
  }

  for (unsigned i = 0, e = I->getNumOperands(); i < e; ++i) {
    if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(I->getOperand(i))) {
      if (ci->getBitWidth() >= 8 && !ci->isZero()) {
        std::uniform_int_distribution<int> delta(1, 3);
        auto newVal = ci->getValue() + delta(rng);
        I->setOperand(i, llvm::ConstantInt::get(ci->getType(), newVal));
        break;
      }
    }
  }
}

static void insertBogusControlFlowFunction(llvm::Function &F,
                                           std::mt19937 &rng) {
  if (F.isDeclaration() || F.size() <= 1)
    return;

  std::vector<llvm::BasicBlock *> candidates;
  for (auto &BB : F) {
    if (isEligibleBlock(BB))
      candidates.push_back(&BB);
  }

  if (candidates.empty())
    return;

  // Select ~30% of eligible blocks.
  std::uniform_int_distribution<int> percent(0, 99);
  std::vector<llvm::BasicBlock *> selected;
  for (auto *BB : candidates) {
    if (percent(rng) < 30)
      selected.push_back(BB);
  }

  if (selected.empty())
    return;

  // Obtain an integer value for the opaque predicate.
  auto &ctx = F.getContext();
  auto *i32Ty = llvm::Type::getInt32Ty(ctx);
  llvm::Value *opaqueInput = nullptr;

  if (!F.arg_empty()) {
    for (auto &arg : F.args()) {
      if (arg.getType()->isIntegerTy()) {
        opaqueInput = &arg;
        break;
      }
    }
    if (!opaqueInput) {
      auto &entry = F.getEntryBlock();
      llvm::IRBuilder<> entryBuilder(&entry, entry.getFirstInsertionPt());
      opaqueInput =
          entryBuilder.CreatePtrToInt(&*F.arg_begin(), i32Ty, "opq.ptr2int");
    }
  } else {
    auto *M = F.getParent();
    auto *gv = new llvm::GlobalVariable(
        *M, i32Ty, /*isConstant=*/false, llvm::GlobalValue::PrivateLinkage,
        llvm::ConstantInt::get(i32Ty, 0), "bcf_opaque_var");
    auto &entry = F.getEntryBlock();
    llvm::IRBuilder<> entryBuilder(&entry, entry.getFirstInsertionPt());
    auto *load = entryBuilder.CreateLoad(i32Ty, gv, /*isVolatile=*/true,
                                         "opq.load");
    opaqueInput = load;
  }

  // Strategy: for each selected block ending with `br successor`:
  //   1. Create a junk block with mutated copies of origBB's instructions
  //      that branches to `successor`.
  //   2. Replace origBB's unconditional `br successor` with
  //      `br opaque_true, successor, junkBB`.
  //   3. Update PHI nodes in successor for the new junkBB predecessor.
  //
  // This preserves domination since origBB is never moved or split.

  for (auto *origBB : selected) {
    auto *br = llvm::cast<llvm::BranchInst>(origBB->getTerminator());
    auto *successor = br->getSuccessor(0);

    // Create the junk (bogus) block.
    auto *junkBB = llvm::BasicBlock::Create(
        ctx, origBB->getName() + ".bcf_bogus", &F);

    // Clone non-terminator instructions from origBB into junkBB.
    llvm::ValueToValueMapTy vmap;
    for (auto &I : *origBB) {
      if (I.isTerminator()) break;
      // Skip PHI nodes in the clone — they reference predecessors of origBB
      // that don't apply to junkBB.
      if (llvm::isa<llvm::PHINode>(&I)) continue;
      auto *cloned = I.clone();
      cloned->setName(I.getName() + ".bogus");
      cloned->insertInto(junkBB, junkBB->end());
      vmap[&I] = cloned;
    }

    // Remap operands within the cloned block.
    for (auto &I : *junkBB) {
      for (unsigned i = 0, e = I.getNumOperands(); i < e; ++i) {
        auto it = vmap.find(I.getOperand(i));
        if (it != vmap.end())
          I.setOperand(i, it->second);
      }
    }

    // Mutate some cloned instructions.
    std::vector<llvm::Instruction *> clonedInsts;
    for (auto &I : *junkBB)
      clonedInsts.push_back(&I);
    for (auto *I : clonedInsts)
      mutateClonedInstruction(I, rng);

    // junkBB branches unconditionally to successor.
    llvm::IRBuilder<> junkBuilder(junkBB);
    junkBuilder.CreateBr(successor);

    // Update PHI nodes in successor: junkBB is a new predecessor.
    // Use undef since the junk path is always dead.
    for (auto &inst : *successor) {
      auto *phi = llvm::dyn_cast<llvm::PHINode>(&inst);
      if (!phi) break;
      phi->addIncoming(llvm::UndefValue::get(phi->getType()), junkBB);
    }

    // Replace origBB's unconditional branch with conditional:
    // true (always taken) → successor, false → junkBB.
    llvm::IRBuilder<> builder(br);
    auto *cond = buildOpaqueTruePredicate(builder, opaqueInput);
    br->eraseFromParent();
    llvm::BranchInst::Create(successor, junkBB, cond, origBB);
  }
}

void insertBogusControlFlowModule(llvm::Module &M, uint32_t seed) {
  std::mt19937 rng(seed);
  for (auto &F : M) {
    insertBogusControlFlowFunction(F, rng);
  }
}

}  // namespace ollvm
