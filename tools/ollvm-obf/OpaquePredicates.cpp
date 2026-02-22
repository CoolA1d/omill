#include "OpaquePredicates.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>

#include <cstdint>
#include <random>
#include <vector>

namespace ollvm {

/// Obtain an integer value to use as the opaque predicate variable.
/// Prefers the first integer argument; falls back to loading from an alloca.
static llvm::Value *getPredicateInput(llvm::Function &F,
                                       llvm::IRBuilder<> &builder,
                                       llvm::AllocaInst *&fallbackAlloca) {
  // Prefer the first integer argument.
  for (auto &arg : F.args()) {
    if (arg.getType()->isIntegerTy() && arg.getType()->getIntegerBitWidth() >= 8)
      return &arg;
  }

  // No suitable argument — create a dedicated alloca in the entry block.
  if (!fallbackAlloca) {
    auto *i64Ty = llvm::Type::getInt64Ty(F.getContext());
    llvm::IRBuilder<> entryBuilder(&F.getEntryBlock(),
                                   F.getEntryBlock().begin());
    fallbackAlloca = entryBuilder.CreateAlloca(i64Ty, nullptr, "opq_var");
    entryBuilder.CreateStore(llvm::ConstantInt::get(i64Ty, 0), fallbackAlloca);
  }

  return builder.CreateLoad(fallbackAlloca->getAllocatedType(), fallbackAlloca,
                            "opq_load");
}

/// Build an always-true condition based on the selected variant.
/// Returns an i1 value that is guaranteed to be true for all inputs.
static llvm::Value *buildOpaquePredicate(llvm::IRBuilder<> &builder,
                                          llvm::Value *x, int variant,
                                          std::mt19937 &rng) {
  auto *ty = x->getType();
  auto *negOne = llvm::ConstantInt::getSigned(ty, -1);
  auto *zero = llvm::ConstantInt::get(ty, 0);

  switch (variant) {
  case 0: {
    // x | ~x == -1  (always true)
    auto *notX = builder.CreateXor(x, negOne, "opq_not");
    auto *orVal = builder.CreateOr(x, notX, "opq_or");
    return builder.CreateICmpEQ(orVal, negOne, "opq_cmp");
  }
  case 1: {
    // (x * (x + 1)) & 1 == 0  (product of consecutive ints is even)
    auto *one = llvm::ConstantInt::get(ty, 1);
    auto *xPlus1 = builder.CreateAdd(x, one, "opq_inc");
    auto *prod = builder.CreateMul(x, xPlus1, "opq_prod");
    auto *bit = builder.CreateAnd(prod, one, "opq_bit");
    return builder.CreateICmpEQ(bit, zero, "opq_cmp");
  }
  case 2: {
    // (x * x) >=u 0  (always true for unsigned comparison)
    auto *sq = builder.CreateMul(x, x, "opq_sq");
    return builder.CreateICmpUGE(sq, zero, "opq_cmp");
  }
  case 3: {
    // x & (x - 1) != x when x is a known non-power-of-2 constant.
    // We substitute x with a random odd constant >= 3 to guarantee truth.
    std::uniform_int_distribution<uint64_t> dist(1, 0x7FFFFFFE);
    uint64_t c = dist(rng) | 3;  // ensure at least bits 0 and 1 set
    auto *constVal = llvm::ConstantInt::get(ty, c);
    auto *one = llvm::ConstantInt::get(ty, 1);
    auto *sub = builder.CreateSub(constVal, one, "opq_dec");
    auto *andVal = builder.CreateAnd(constVal, sub, "opq_and");
    return builder.CreateICmpNE(andVal, zero, "opq_cmp");
  }
  default:
    llvm_unreachable("invalid opaque predicate variant");
  }
}

/// Create a junk basic block that does a few harmless operations and then
/// branches unconditionally to the given target.
static llvm::BasicBlock *createJunkBlock(llvm::Function &F,
                                          llvm::BasicBlock *target,
                                          std::mt19937 &rng) {
  auto &ctx = F.getContext();
  auto *junkBB = llvm::BasicBlock::Create(ctx, "opq_junk", &F);
  llvm::IRBuilder<> builder(junkBB);

  // Insert a few dummy arithmetic operations so the block isn't empty.
  auto *i32Ty = llvm::Type::getInt32Ty(ctx);
  std::uniform_int_distribution<uint32_t> valDist(1, 0xFFFF);
  auto *a = llvm::ConstantInt::get(i32Ty, valDist(rng));
  auto *b = llvm::ConstantInt::get(i32Ty, valDist(rng));
  auto *dummy = builder.CreateAdd(a, b, "junk_add");
  builder.CreateMul(dummy, a, "junk_mul");

  builder.CreateBr(target);
  return junkBB;
}

static void insertOpaquePredicatesFunction(llvm::Function &F,
                                            std::mt19937 &rng) {
  if (F.isDeclaration())
    return;

  // Skip single-block functions.
  if (F.size() <= 1)
    return;

  // Collect candidate blocks: those ending with an unconditional branch
  // and that are NOT return blocks.
  std::vector<llvm::BasicBlock *> candidates;
  for (auto &BB : F) {
    auto *term = BB.getTerminator();
    if (!term)
      continue;
    if (llvm::isa<llvm::ReturnInst>(term))
      continue;
    auto *br = llvm::dyn_cast<llvm::BranchInst>(term);
    if (br && br->isUnconditional())
      candidates.push_back(&BB);
  }

  if (candidates.empty())
    return;

  std::uniform_int_distribution<int> coinFlip(0, 99);
  std::uniform_int_distribution<int> variantDist(0, 3);
  llvm::AllocaInst *fallbackAlloca = nullptr;

  for (auto *BB : candidates) {
    // ~40% probability of insertion.
    if (coinFlip(rng) >= 40)
      continue;

    auto *br = llvm::cast<llvm::BranchInst>(BB->getTerminator());
    auto *originalTarget = br->getSuccessor(0);

    // Build the opaque predicate just before the branch.
    llvm::IRBuilder<> builder(br);
    auto *x = getPredicateInput(F, builder, fallbackAlloca);
    int variant = variantDist(rng);
    auto *cond = buildOpaquePredicate(builder, x, variant, rng);

    // Create a junk block that also branches to the original target.
    auto *junkBB = createJunkBlock(F, originalTarget, rng);

    // Replace the unconditional branch with a conditional one.
    // True → original target, False → junk block.
    br->eraseFromParent();
    llvm::BranchInst::Create(originalTarget, junkBB, cond, BB);

    // Update PHI nodes in originalTarget: junkBB is a new predecessor
    // that carries the same incoming value as BB (junk path is dead).
    for (auto &inst : *originalTarget) {
      auto *phi = llvm::dyn_cast<llvm::PHINode>(&inst);
      if (!phi) break;
      auto *val = phi->getIncomingValueForBlock(BB);
      phi->addIncoming(val, junkBB);
    }
  }
}

void insertOpaquePredicatesModule(llvm::Module &M, uint32_t seed) {
  std::mt19937 rng(seed);
  for (auto &F : M) {
    insertOpaquePredicatesFunction(F, rng);
  }
}

}  // namespace ollvm
