#include "omill/Passes/StackConcretization.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>

#include <gtest/gtest.h>

namespace {

static const char *kDataLayout =
    "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-"
    "f80:128-n8:16:32:64-S128";

/// RSP and RBP byte offsets in the remill x86_64 State struct.
static constexpr int64_t kRSPOffset = 48;
static constexpr int64_t kRBPOffset = 56;

class StackConcretizationTest : public ::testing::Test {
 protected:
  llvm::LLVMContext Ctx;

  std::unique_ptr<llvm::Module> createModule() {
    auto M = std::make_unique<llvm::Module>("test", Ctx);
    M->setDataLayout(kDataLayout);
    return M;
  }

  void runPass(llvm::Function *F) {
    auto *M = F->getParent();

    llvm::PassBuilder PB;
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    llvm::FunctionPassManager FPM;
    FPM.addPass(omill::StackConcretizationPass());
    FPM.run(*F, FAM);
  }

  /// The standard lifted function type: (State*, i64, Memory*) -> Memory*
  llvm::FunctionType *getLiftedFnType() {
    auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
    auto *ptrTy = llvm::PointerType::get(Ctx, 0);
    return llvm::FunctionType::get(ptrTy, {ptrTy, i64Ty, ptrTy}, false);
  }

  /// Create a GEP into State (arg0) at a given byte offset.
  llvm::Value *buildStateGEP(llvm::IRBuilder<> &B, llvm::Value *state,
                             int64_t offset) {
    auto *i8Ty = llvm::Type::getInt8Ty(Ctx);
    return B.CreateGEP(i8Ty, state, B.getInt32(offset), "state_gep");
  }

  /// Load an i64 from State at a given byte offset.
  llvm::LoadInst *loadFromState(llvm::IRBuilder<> &B, llvm::Value *state,
                                int64_t offset) {
    auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
    auto *gep = buildStateGEP(B, state, offset);
    return B.CreateLoad(i64Ty, gep, "state_val");
  }

  /// Count alloca instructions in a function.
  unsigned countAllocas(llvm::Function *F) {
    unsigned n = 0;
    for (auto &BB : *F)
      for (auto &I : BB)
        if (llvm::isa<llvm::AllocaInst>(&I))
          ++n;
    return n;
  }

  /// Count inttoptr instructions in a function.
  unsigned countIntToPtr(llvm::Function *F) {
    unsigned n = 0;
    for (auto &BB : *F)
      for (auto &I : BB)
        if (llvm::isa<llvm::IntToPtrInst>(&I))
          ++n;
    return n;
  }

  /// Count GEP instructions with a given name prefix.
  unsigned countGEPsNamed(llvm::Function *F, llvm::StringRef prefix) {
    unsigned n = 0;
    for (auto &BB : *F)
      for (auto &I : BB)
        if (auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(&I))
          if (gep->getName().starts_with(prefix))
            ++n;
    return n;
  }

  /// Find first AllocaInst named with a given prefix.
  llvm::AllocaInst *findAlloca(llvm::Function *F, llvm::StringRef prefix) {
    for (auto &BB : *F)
      for (auto &I : BB)
        if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(&I))
          if (AI->getName().starts_with(prefix))
            return AI;
    return nullptr;
  }
};

// ===----------------------------------------------------------------------===
// Test 1: Positive-offset stack access (leaf function shadow space)
// ===----------------------------------------------------------------------===
TEST_F(StackConcretizationTest, PositiveOffset_LeafShadowSpace) {
  auto M = createModule();
  auto *fnTy = getLiftedFnType();
  auto *fn = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                    "sub_140001000", *M);
  auto *BB = llvm::BasicBlock::Create(Ctx, "entry", fn);
  llvm::IRBuilder<> B(BB);

  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);

  // Load RSP from State.
  auto *rsp = loadFromState(B, fn->getArg(0), kRSPOffset);

  // Access shadow space: [rsp+0x8], [rsp+0x10], [rsp+0x18], [rsp+0x20]
  auto *addr1 = B.CreateAdd(rsp, llvm::ConstantInt::get(i64Ty, 0x08));
  auto *ptr1 = B.CreateIntToPtr(addr1, ptrTy);
  auto *v1 = B.CreateLoad(i64Ty, ptr1);

  auto *addr2 = B.CreateAdd(rsp, llvm::ConstantInt::get(i64Ty, 0x10));
  auto *ptr2 = B.CreateIntToPtr(addr2, ptrTy);
  auto *v2 = B.CreateLoad(i64Ty, ptr2);

  auto *addr3 = B.CreateAdd(rsp, llvm::ConstantInt::get(i64Ty, 0x18));
  auto *ptr3 = B.CreateIntToPtr(addr3, ptrTy);
  B.CreateStore(v1, ptr3);

  auto *addr4 = B.CreateAdd(rsp, llvm::ConstantInt::get(i64Ty, 0x20));
  auto *ptr4 = B.CreateIntToPtr(addr4, ptrTy);
  B.CreateStore(v2, ptr4);

  B.CreateRet(fn->getArg(2));

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));

  // Before: 4 inttoptr, 0 concrete_stack allocas.
  EXPECT_EQ(countIntToPtr(fn), 4u);
  EXPECT_EQ(findAlloca(fn, "concrete_stack"), nullptr);

  runPass(fn);

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));

  // After: 0 inttoptr, 1+ concrete_stack alloca.
  EXPECT_EQ(countIntToPtr(fn), 0u);
  auto *alloca = findAlloca(fn, "concrete_stack");
  ASSERT_NE(alloca, nullptr);

  // GEPs should have stack_ptr names.
  EXPECT_GE(countGEPsNamed(fn, "stack_ptr"), 4u);
}

// ===----------------------------------------------------------------------===
// Test 2: Negative-offset (complement RecoverStackFrame)
// ===----------------------------------------------------------------------===
TEST_F(StackConcretizationTest, NegativeOffset_LocalFrame) {
  auto M = createModule();
  auto *fnTy = getLiftedFnType();
  auto *fn = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                    "sub_140001000", *M);
  auto *BB = llvm::BasicBlock::Create(Ctx, "entry", fn);
  llvm::IRBuilder<> B(BB);

  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);

  // Load RSP, sub 0x20 (frame allocation), store to local vars.
  auto *rsp = loadFromState(B, fn->getArg(0), kRSPOffset);
  auto *rsp_adj = B.CreateSub(rsp, llvm::ConstantInt::get(i64Ty, 0x20));

  // [rsp-0x20+0x00] and [rsp-0x20+0x08]
  auto *ptr1 = B.CreateIntToPtr(rsp_adj, ptrTy);
  B.CreateStore(llvm::ConstantInt::get(i64Ty, 42), ptr1);

  auto *addr2 = B.CreateAdd(rsp_adj, llvm::ConstantInt::get(i64Ty, 0x08));
  auto *ptr2 = B.CreateIntToPtr(addr2, ptrTy);
  B.CreateStore(llvm::ConstantInt::get(i64Ty, 43), ptr2);

  B.CreateRet(fn->getArg(2));

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  EXPECT_EQ(countIntToPtr(fn), 2u);

  runPass(fn);

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  EXPECT_EQ(countIntToPtr(fn), 0u);
  ASSERT_NE(findAlloca(fn, "concrete_stack"), nullptr);
}

// ===----------------------------------------------------------------------===
// Test 3: Mixed positive and negative offsets — unified frame
// ===----------------------------------------------------------------------===
TEST_F(StackConcretizationTest, MixedPositiveNegative_UnifiedFrame) {
  auto M = createModule();
  auto *fnTy = getLiftedFnType();
  auto *fn = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                    "sub_140001000", *M);
  auto *BB = llvm::BasicBlock::Create(Ctx, "entry", fn);
  llvm::IRBuilder<> B(BB);

  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);

  auto *rsp = loadFromState(B, fn->getArg(0), kRSPOffset);

  // Negative: [rsp-0x08]
  auto *addr_neg = B.CreateSub(rsp, llvm::ConstantInt::get(i64Ty, 0x08));
  auto *ptr_neg = B.CreateIntToPtr(addr_neg, ptrTy);
  B.CreateStore(llvm::ConstantInt::get(i64Ty, 1), ptr_neg);

  // Positive: [rsp+0x08]
  auto *addr_pos = B.CreateAdd(rsp, llvm::ConstantInt::get(i64Ty, 0x08));
  auto *ptr_pos = B.CreateIntToPtr(addr_pos, ptrTy);
  auto *val = B.CreateLoad(i64Ty, ptr_pos);

  B.CreateRet(fn->getArg(2));

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  EXPECT_EQ(countIntToPtr(fn), 2u);

  runPass(fn);

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  EXPECT_EQ(countIntToPtr(fn), 0u);

  // Both accesses should be in the same alloca (gap = 16, within tolerance).
  auto *alloca = findAlloca(fn, "concrete_stack");
  ASSERT_NE(alloca, nullptr);

  // Frame should cover from -0x08 to +0x08, size = 0x08 - (-0x08) + 8 = 24.
  auto *arr_ty = llvm::dyn_cast<llvm::ArrayType>(alloca->getAllocatedType());
  ASSERT_NE(arr_ty, nullptr);
  EXPECT_EQ(arr_ty->getNumElements(), 24u);
}

// ===----------------------------------------------------------------------===
// Test 4: Alignment masking — and(rsp, -16)
// ===----------------------------------------------------------------------===
TEST_F(StackConcretizationTest, AlignmentMask_AndNeg16) {
  auto M = createModule();
  auto *fnTy = getLiftedFnType();
  auto *fn = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                    "sub_140001000", *M);
  auto *BB = llvm::BasicBlock::Create(Ctx, "entry", fn);
  llvm::IRBuilder<> B(BB);

  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);

  auto *rsp = loadFromState(B, fn->getArg(0), kRSPOffset);

  // sub rsp, 0x100; and rsp, -16 (align to 16 bytes)
  auto *rsp_sub = B.CreateSub(rsp, llvm::ConstantInt::get(i64Ty, 0x100));
  auto *rsp_aligned = B.CreateAnd(rsp_sub, llvm::ConstantInt::get(i64Ty, -16));

  // [aligned_rsp + 0x00]
  auto *ptr1 = B.CreateIntToPtr(rsp_aligned, ptrTy);
  B.CreateStore(llvm::ConstantInt::get(i64Ty, 0xDEAD), ptr1);

  // [aligned_rsp + 0x08]
  auto *addr2 = B.CreateAdd(rsp_aligned, llvm::ConstantInt::get(i64Ty, 0x08));
  auto *ptr2 = B.CreateIntToPtr(addr2, ptrTy);
  B.CreateStore(llvm::ConstantInt::get(i64Ty, 0xBEEF), ptr2);

  B.CreateRet(fn->getArg(2));

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  EXPECT_EQ(countIntToPtr(fn), 2u);

  runPass(fn);

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  EXPECT_EQ(countIntToPtr(fn), 0u);
  ASSERT_NE(findAlloca(fn, "concrete_stack"), nullptr);
}

// ===----------------------------------------------------------------------===
// Test 5: RBP-based accesses
// ===----------------------------------------------------------------------===
TEST_F(StackConcretizationTest, RBPBasedAccess) {
  auto M = createModule();
  auto *fnTy = getLiftedFnType();
  auto *fn = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                    "sub_140001000", *M);
  auto *BB = llvm::BasicBlock::Create(Ctx, "entry", fn);
  llvm::IRBuilder<> B(BB);

  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);

  // Load RBP from State (offset 56).
  auto *rbp = loadFromState(B, fn->getArg(0), kRBPOffset);

  // [rbp+0x10], [rbp+0x18] — positive offset accesses from RBP.
  auto *addr1 = B.CreateAdd(rbp, llvm::ConstantInt::get(i64Ty, 0x10));
  auto *ptr1 = B.CreateIntToPtr(addr1, ptrTy);
  auto *v1 = B.CreateLoad(i64Ty, ptr1);

  auto *addr2 = B.CreateAdd(rbp, llvm::ConstantInt::get(i64Ty, 0x18));
  auto *ptr2 = B.CreateIntToPtr(addr2, ptrTy);
  B.CreateStore(v1, ptr2);

  B.CreateRet(fn->getArg(2));

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  EXPECT_EQ(countIntToPtr(fn), 2u);

  runPass(fn);

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  EXPECT_EQ(countIntToPtr(fn), 0u);
  ASSERT_NE(findAlloca(fn, "concrete_stack"), nullptr);
}

// ===----------------------------------------------------------------------===
// Test 6: Non-stack State load is not concretized
// ===----------------------------------------------------------------------===
TEST_F(StackConcretizationTest, NonStackRegister_LeftAlone) {
  auto M = createModule();
  auto *fnTy = getLiftedFnType();
  auto *fn = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                    "sub_140001000", *M);
  auto *BB = llvm::BasicBlock::Create(Ctx, "entry", fn);
  llvm::IRBuilder<> B(BB);

  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);

  // Load RAX from State (offset 2208 — not RSP or RBP).
  auto *rax = loadFromState(B, fn->getArg(0), 2208);
  auto *ptr1 = B.CreateIntToPtr(rax, ptrTy);
  auto *val = B.CreateLoad(i64Ty, ptr1);

  B.CreateRet(fn->getArg(2));

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  EXPECT_EQ(countIntToPtr(fn), 1u);

  runPass(fn);

  // Should be untouched — RAX is not a stack register.
  EXPECT_EQ(countIntToPtr(fn), 1u);
  EXPECT_EQ(findAlloca(fn, "concrete_stack"), nullptr);
}

// ===----------------------------------------------------------------------===
// Test 7: Direct inttoptr from RSP (no arithmetic)
// ===----------------------------------------------------------------------===
TEST_F(StackConcretizationTest, DirectIntToPtr_NoArithmetic) {
  auto M = createModule();
  auto *fnTy = getLiftedFnType();
  auto *fn = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                    "sub_140001000", *M);
  auto *BB = llvm::BasicBlock::Create(Ctx, "entry", fn);
  llvm::IRBuilder<> B(BB);

  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);

  // Load RSP and directly inttoptr (access [rsp+0] = return address).
  auto *rsp = loadFromState(B, fn->getArg(0), kRSPOffset);
  auto *ptr = B.CreateIntToPtr(rsp, ptrTy);
  auto *ret_addr = B.CreateLoad(i64Ty, ptr);

  B.CreateRet(fn->getArg(2));

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  EXPECT_EQ(countIntToPtr(fn), 1u);

  runPass(fn);

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  EXPECT_EQ(countIntToPtr(fn), 0u);
  ASSERT_NE(findAlloca(fn, "concrete_stack"), nullptr);
}

// ===----------------------------------------------------------------------===
// Test 8: Or-based offset (aligned RSP + or offset)
// ===----------------------------------------------------------------------===
TEST_F(StackConcretizationTest, OrBasedOffset) {
  auto M = createModule();
  auto *fnTy = getLiftedFnType();
  auto *fn = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                    "sub_140001000", *M);
  auto *BB = llvm::BasicBlock::Create(Ctx, "entry", fn);
  llvm::IRBuilder<> B(BB);

  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);

  auto *rsp = loadFromState(B, fn->getArg(0), kRSPOffset);
  auto *rsp_sub = B.CreateSub(rsp, llvm::ConstantInt::get(i64Ty, 0x40));
  auto *rsp_aligned = B.CreateAnd(rsp_sub, llvm::ConstantInt::get(i64Ty, -32));

  // or(aligned_rsp, 8) — used instead of add when base is known aligned.
  auto *addr = B.CreateOr(rsp_aligned, llvm::ConstantInt::get(i64Ty, 8));
  auto *ptr = B.CreateIntToPtr(addr, ptrTy);
  B.CreateStore(llvm::ConstantInt::get(i64Ty, 0x1234), ptr);

  B.CreateRet(fn->getArg(2));

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  EXPECT_EQ(countIntToPtr(fn), 1u);

  runPass(fn);

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  EXPECT_EQ(countIntToPtr(fn), 0u);
}

// ===----------------------------------------------------------------------===
// Test 9: No State access — nothing to do
// ===----------------------------------------------------------------------===
TEST_F(StackConcretizationTest, NoStateAccess_NothingToDo) {
  auto M = createModule();
  auto *fnTy = getLiftedFnType();
  auto *fn = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                    "sub_140001000", *M);
  auto *BB = llvm::BasicBlock::Create(Ctx, "entry", fn);
  llvm::IRBuilder<> B(BB);

  // No State loads, just return.
  B.CreateRet(fn->getArg(2));

  runPass(fn);

  EXPECT_EQ(countAllocas(fn), 0u);
}

// ===----------------------------------------------------------------------===
// Test 10: Multiple distinct regions create separate allocas
// ===----------------------------------------------------------------------===
TEST_F(StackConcretizationTest, DisjointRegions_SeparateAllocas) {
  auto M = createModule();
  auto *fnTy = getLiftedFnType();
  auto *fn = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                    "sub_140001000", *M);
  auto *BB = llvm::BasicBlock::Create(Ctx, "entry", fn);
  llvm::IRBuilder<> B(BB);

  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);

  auto *rsp = loadFromState(B, fn->getArg(0), kRSPOffset);

  // Region 1: offset -0x10
  auto *addr1 = B.CreateSub(rsp, llvm::ConstantInt::get(i64Ty, 0x10));
  auto *ptr1 = B.CreateIntToPtr(addr1, ptrTy);
  B.CreateStore(llvm::ConstantInt::get(i64Ty, 1), ptr1);

  // Region 2: offset +0x100 (far enough to be a separate region, gap > 16)
  auto *addr2 = B.CreateAdd(rsp, llvm::ConstantInt::get(i64Ty, 0x100));
  auto *ptr2 = B.CreateIntToPtr(addr2, ptrTy);
  B.CreateStore(llvm::ConstantInt::get(i64Ty, 2), ptr2);

  B.CreateRet(fn->getArg(2));

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  EXPECT_EQ(countIntToPtr(fn), 2u);

  runPass(fn);

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  EXPECT_EQ(countIntToPtr(fn), 0u);

  // Two disjoint regions should create two allocas.
  unsigned alloca_count = 0;
  for (auto &I : fn->getEntryBlock())
    if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(&I))
      if (AI->getName().starts_with("concrete_stack"))
        ++alloca_count;
  EXPECT_EQ(alloca_count, 2u);
}

}  // namespace
