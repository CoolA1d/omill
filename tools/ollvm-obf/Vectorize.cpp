#include "Vectorize.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Instructions.h>

#include <algorithm>
#include <map>
#include <random>
#include <vector>

namespace ollvm {

namespace {

bool isSupportedOpcode(unsigned op) {
  switch (op) {
  case llvm::Instruction::Add:
  case llvm::Instruction::Sub:
  case llvm::Instruction::Xor:
  case llvm::Instruction::And:
  case llvm::Instruction::Or:
  case llvm::Instruction::Mul:
    return true;
  default:
    return false;
  }
}

bool isZeroLane(llvm::Value *idx) {
  auto *ci = llvm::dyn_cast<llvm::ConstantInt>(idx);
  return ci && ci->isZero();
}

llvm::Value *createInlineSplat(llvm::IRBuilder<> &builder, llvm::Value *scalar,
                               llvm::FixedVectorType *vec_ty,
                               llvm::ConstantInt *idx_zero) {
  auto *seed =
      builder.CreateInsertElement(llvm::UndefValue::get(vec_ty), scalar, idx_zero);
  return builder.CreateShuffleVector(seed, llvm::UndefValue::get(vec_ty),
                                     {0, 0, 0, 0});
}

llvm::Value *liftLane0ToVector(llvm::IRBuilder<> &builder, llvm::Value *v,
                               llvm::FixedVectorType *vec_ty,
                               llvm::Constant *zero_vec,
                               llvm::ConstantInt *idx_zero,
                               bool reuse_vector_data,
                               llvm::DenseMap<llvm::Value *, llvm::Value *> &scalar_to_vector) {
  if (reuse_vector_data) {
    if (auto it = scalar_to_vector.find(v); it != scalar_to_vector.end()) {
      if (it->second->getType() == vec_ty)
        return it->second;
    }
  }

  // Data-aware mode: if this scalar came from extractelement %vec, 0,
  // reuse the vector payload directly to keep data in vector space.
  if (reuse_vector_data) {
    if (auto *ee = llvm::dyn_cast<llvm::ExtractElementInst>(v)) {
      if (ee->getVectorOperand()->getType() == vec_ty && isZeroLane(ee->getIndexOperand()))
        return ee->getVectorOperand();
    }
  }

  return builder.CreateInsertElement(zero_vec, v, idx_zero);
}

llvm::Value *buildVectorBitwiseAdd(llvm::IRBuilder<> &builder,
                                   llvm::Value *lhs, llvm::Value *rhs,
                                   llvm::Value *one_vec) {
  // a + b == (a ^ b) + ((a & b) << 1)
  auto *sum = builder.CreateXor(lhs, rhs);
  auto *carry = builder.CreateAnd(lhs, rhs);
  auto *carry2 = builder.CreateShl(carry, one_vec);
  return builder.CreateAdd(sum, carry2);
}

llvm::Value *buildVectorBitwiseSub(llvm::IRBuilder<> &builder,
                                   llvm::Value *lhs, llvm::Value *rhs,
                                   llvm::Value *one_vec,
                                   llvm::Value *all_ones_vec) {
  // a - b == (a ^ b) - (((~a) & b) << 1)
  auto *diff = builder.CreateXor(lhs, rhs);
  auto *not_lhs = builder.CreateXor(lhs, all_ones_vec);
  auto *borrow = builder.CreateAnd(not_lhs, rhs);
  auto *borrow2 = builder.CreateShl(borrow, one_vec);
  return builder.CreateSub(diff, borrow2);
}

llvm::Value *buildVectorBinOp(llvm::IRBuilder<> &builder, unsigned opcode,
                              llvm::Value *va, llvm::Value *vb,
                              const VectorizeOptions &opts,
                              llvm::Value *one_vec,
                              llvm::Value *all_ones_vec) {
  if (!opts.vectorize_bitwise) {
    return builder.CreateBinOp(
        static_cast<llvm::Instruction::BinaryOps>(opcode), va, vb);
  }

  if (opcode == llvm::Instruction::Add) {
    return buildVectorBitwiseAdd(builder, va, vb, one_vec);
  }

  if (opcode == llvm::Instruction::Sub) {
    return buildVectorBitwiseSub(builder, va, vb, one_vec, all_ones_vec);
  }

  return builder.CreateBinOp(
      static_cast<llvm::Instruction::BinaryOps>(opcode), va, vb);
}

void vectorizeI32StackData(llvm::Function &F, llvm::FixedVectorType *vec_ty,
                           llvm::Constant *zero_vec,
                           llvm::ConstantInt *idx_zero) {
  if (F.isDeclaration())
    return;

  llvm::SmallVector<llvm::AllocaInst *, 16> candidates;
  llvm::BasicBlock &entry = F.getEntryBlock();

  for (auto &I : entry) {
    auto *alloca = llvm::dyn_cast<llvm::AllocaInst>(&I);
    if (!alloca)
      continue;
    if (alloca->isArrayAllocation())
      continue;
    if (!alloca->getAllocatedType()->isIntegerTy(32))
      continue;

    bool supported = true;
    for (llvm::User *U : alloca->users()) {
      if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(U)) {
        if (LI->isVolatile() || LI->isAtomic() || LI->getPointerOperand() != alloca) {
          supported = false;
          break;
        }
        continue;
      }
      if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(U)) {
        if (SI->isVolatile() || SI->isAtomic() || SI->getPointerOperand() != alloca ||
            !SI->getValueOperand()->getType()->isIntegerTy(32)) {
          supported = false;
          break;
        }
        continue;
      }
      if (auto *II = llvm::dyn_cast<llvm::IntrinsicInst>(U)) {
        switch (II->getIntrinsicID()) {
        case llvm::Intrinsic::lifetime_start:
        case llvm::Intrinsic::lifetime_end:
        case llvm::Intrinsic::dbg_declare:
        case llvm::Intrinsic::dbg_value:
          continue;
        default:
          break;
        }
      }

      supported = false;
      break;
    }

    if (supported)
      candidates.push_back(alloca);
  }

  for (auto *alloca : candidates) {
    llvm::Instruction *insert_pt = alloca->getNextNode();
    if (!insert_pt)
      insert_pt = entry.getTerminator();

    llvm::IRBuilder<> alloca_builder(insert_pt);
    auto *vec_alloca =
        alloca_builder.CreateAlloca(vec_ty, nullptr, alloca->getName() + ".vec");
    vec_alloca->setAlignment(alloca->getAlign());

    llvm::SmallVector<llvm::Instruction *, 16> users;
    for (llvm::User *U : alloca->users()) {
      if (auto *I = llvm::dyn_cast<llvm::Instruction>(U))
        users.push_back(I);
    }

    for (auto *U : users) {
      if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(U)) {
        llvm::IRBuilder<> b(LI);
        auto *vload = b.CreateLoad(vec_ty, vec_alloca, LI->getName() + ".v");
        vload->setAlignment(LI->getAlign());
        auto *lane0 = b.CreateExtractElement(vload, idx_zero, LI->getName() + ".lane0");
        LI->replaceAllUsesWith(lane0);
        LI->eraseFromParent();
        continue;
      }

      if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(U)) {
        llvm::IRBuilder<> b(SI);
        auto *packed = b.CreateInsertElement(zero_vec, SI->getValueOperand(), idx_zero);
        auto *vstore = b.CreateStore(packed, vec_alloca);
        vstore->setAlignment(SI->getAlign());
        SI->eraseFromParent();
      }
    }
  }
}

void vectorizeFunction(llvm::Function &F, std::mt19937 &rng,
                       const VectorizeOptions &opts) {
  if (F.isDeclaration())
    return;

  std::uniform_int_distribution<int> percent(0, 99);

  // Collect eligible instructions.
  std::vector<llvm::BinaryOperator *> work;
  for (auto &BB : F) {
    for (auto &I : BB) {
      auto *bin = llvm::dyn_cast<llvm::BinaryOperator>(&I);
      if (!bin)
        continue;
      // Only i32 for clean SSE2 mapping.
      if (!bin->getType()->isIntegerTy(32))
        continue;
      if (!isSupportedOpcode(bin->getOpcode()))
        continue;
      // Skip constant-only expressions.
      if (llvm::isa<llvm::Constant>(bin->getOperand(0)) &&
          llvm::isa<llvm::Constant>(bin->getOperand(1)))
        continue;
      work.push_back(bin);
    }
  }

  auto &ctx = F.getContext();
  auto *vec_ty = llvm::FixedVectorType::get(llvm::Type::getInt32Ty(ctx), 4);
  auto *i32_ty = llvm::Type::getInt32Ty(ctx);
  auto *zero_vec = llvm::ConstantAggregateZero::get(vec_ty);
  auto *idx_zero = llvm::ConstantInt::get(i32_ty, 0);

  // Materialize splats in-function to avoid spilling vector constants to
  // global constant pools.
  llvm::IRBuilder<> entry_builder(&*F.getEntryBlock().getFirstInsertionPt());
  auto *one_vec = createInlineSplat(
      entry_builder, llvm::ConstantInt::get(i32_ty, 1), vec_ty, idx_zero);
  auto *all_ones_vec = createInlineSplat(
      entry_builder, llvm::ConstantInt::getSigned(i32_ty, -1), vec_ty, idx_zero);

  std::map<uint64_t, llvm::Value *> const_splats;
  auto getConstSplat = [&](llvm::ConstantInt *ci) -> llvm::Value * {
    uint64_t key = ci->getValue().getZExtValue();
    auto it = const_splats.find(key);
    if (it != const_splats.end())
      return it->second;
    auto *v = createInlineSplat(entry_builder, ci, vec_ty, idx_zero);
    const_splats[key] = v;
    return v;
  };

  if (opts.vectorize_data) {
    vectorizeI32StackData(F, vec_ty, zero_vec, idx_zero);
  }

  const unsigned threshold = std::min(opts.transform_percent, 100u);
  llvm::DenseMap<llvm::Value *, llvm::Value *> scalar_to_vector;

  for (auto *bin : work) {
    // Apply to configured percent of eligible ops.
    if (threshold == 0 || percent(rng) >= static_cast<int>(threshold))
      continue;

    llvm::IRBuilder<> builder(bin);
    auto *a = bin->getOperand(0);
    auto *b = bin->getOperand(1);

    llvm::Value *va = nullptr;
    llvm::Value *vb = nullptr;
    if (auto *ca = llvm::dyn_cast<llvm::ConstantInt>(a)) {
      va = getConstSplat(ca);
    } else {
      va = liftLane0ToVector(builder, a, vec_ty, zero_vec, idx_zero,
                             opts.vectorize_data, scalar_to_vector);
    }
    if (auto *cb = llvm::dyn_cast<llvm::ConstantInt>(b)) {
      vb = getConstSplat(cb);
    } else {
      vb = liftLane0ToVector(builder, b, vec_ty, zero_vec, idx_zero,
                             opts.vectorize_data, scalar_to_vector);
    }
    auto *vr = buildVectorBinOp(builder, bin->getOpcode(), va, vb, opts,
                                one_vec, all_ones_vec);

    auto *result = builder.CreateExtractElement(vr, idx_zero);
    scalar_to_vector[result] = vr;

    bin->replaceAllUsesWith(result);
    bin->eraseFromParent();
  }

  // Drop now-unused lane-0 extracts created as temporary scalar adapters.
  llvm::SmallVector<llvm::Instruction *, 16> dead_extracts;
  for (auto &BB : F) {
    for (auto &I : BB) {
      auto *EE = llvm::dyn_cast<llvm::ExtractElementInst>(&I);
      if (!EE || !EE->use_empty())
        continue;
      if (EE->getVectorOperandType() == vec_ty && isZeroLane(EE->getIndexOperand()))
        dead_extracts.push_back(EE);
    }
  }
  for (auto *I : dead_extracts)
    I->eraseFromParent();
}

}  // namespace

void vectorizeModule(llvm::Module &M, uint32_t seed,
                     const VectorizeOptions &opts) {
  std::mt19937 rng(seed);
  for (auto &F : M) {
    vectorizeFunction(F, rng, opts);
  }
}

}  // namespace ollvm
