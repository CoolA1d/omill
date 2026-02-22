#include "omill/Passes/PointersHoisting.h"

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

class PointersHoistingTest : public ::testing::Test {
 protected:
  llvm::LLVMContext Ctx;

  std::unique_ptr<llvm::Module> createModule() {
    auto M = std::make_unique<llvm::Module>("test", Ctx);
    M->setDataLayout(kDataLayout);
    return M;
  }

  void runPass(llvm::Function *F) {
    llvm::FunctionPassManager FPM;
    FPM.addPass(omill::PointersHoistingPass());

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

    FPM.run(*F, FAM);
  }

  /// Create a simple loop: preheader -> header -> body -> latch -> header,
  /// with an exit from header. Returns {preheader, header, body, latch, exit}.
  struct LoopBlocks {
    llvm::BasicBlock *Preheader;
    llvm::BasicBlock *Header;
    llvm::BasicBlock *Body;
    llvm::BasicBlock *Latch;
    llvm::BasicBlock *Exit;
  };

  /// Build a canonical loop with an induction variable.
  /// Loop iterates while i < limit.
  LoopBlocks buildLoop(llvm::Function *F, llvm::IRBuilder<> &B,
                       llvm::Value *Limit, const llvm::Twine &Prefix = "") {
    auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
    auto prefix = Prefix.str();

    auto *Preheader =
        llvm::BasicBlock::Create(Ctx, prefix + "preheader", F);
    auto *Header = llvm::BasicBlock::Create(Ctx, prefix + "header", F);
    auto *Body = llvm::BasicBlock::Create(Ctx, prefix + "body", F);
    auto *Latch = llvm::BasicBlock::Create(Ctx, prefix + "latch", F);
    auto *Exit = llvm::BasicBlock::Create(Ctx, prefix + "exit", F);

    // preheader -> header
    B.SetInsertPoint(Preheader);
    B.CreateBr(Header);

    // header: phi i = [0, preheader], [i.next, latch]; if i < limit -> body
    // else exit
    B.SetInsertPoint(Header);
    auto *IV = B.CreatePHI(i64Ty, 2, prefix + "iv");
    IV->addIncoming(llvm::ConstantInt::get(i64Ty, 0), Preheader);
    auto *Cmp = B.CreateICmpULT(IV, Limit, prefix + "cmp");
    B.CreateCondBr(Cmp, Body, Exit);

    // body -> latch
    B.SetInsertPoint(Body);
    B.CreateBr(Latch);

    // latch: i.next = i + 1; br header
    B.SetInsertPoint(Latch);
    auto *IVNext =
        B.CreateAdd(IV, llvm::ConstantInt::get(i64Ty, 1), prefix + "iv.next");
    IV->addIncoming(IVNext, Latch);
    B.CreateBr(Header);

    return {Preheader, Header, Body, Latch, Exit};
  }
};

// Test 1: inttoptr with loop-invariant base is hoisted to preheader.
TEST_F(PointersHoistingTest, HoistIntToPtrFromLoop) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);
  auto *voidTy = llvm::Type::getVoidTy(Ctx);

  auto *FTy = llvm::FunctionType::get(voidTy, {i64Ty, i64Ty}, false);
  auto *F = llvm::Function::Create(FTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);
  auto *Base = F->getArg(0);
  Base->setName("base");
  auto *Limit = F->getArg(1);
  Limit->setName("limit");

  auto *Entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(Entry);

  auto Loop = buildLoop(F, B, Limit);
  // entry -> preheader
  B.SetInsertPoint(Entry);
  B.CreateBr(Loop.Preheader);

  // In the loop body: inttoptr(base + 16)
  B.SetInsertPoint(
      &*Loop.Body->getFirstInsertionPt());  // before the br to latch
  auto *Addr = B.CreateAdd(Base, llvm::ConstantInt::get(i64Ty, 16), "addr");
  auto *Ptr = B.CreateIntToPtr(Addr, ptrTy, "ptr");
  // Use the ptr so it's not dead (store something).
  auto *LoadedVal = B.CreateLoad(i64Ty, Ptr, "loaded");
  (void)LoadedVal;
  // Remove the branch to latch, add our use, then re-branch.
  // Actually the buildLoop already placed a br to latch at Body.
  // The IRBuilder insertions go before the terminator since we used
  // getFirstInsertionPt. Let me restructure: remove body terminator, re-add.
  Loop.Body->getTerminator()->eraseFromParent();
  B.SetInsertPoint(Loop.Body);
  B.CreateBr(Loop.Latch);

  // exit: ret void
  B.SetInsertPoint(Loop.Exit);
  B.CreateRetVoid();

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Verify inttoptr is in loop body before pass.
  ASSERT_EQ(llvm::cast<llvm::Instruction>(Ptr)->getParent(), Loop.Body);

  runPass(F);

  // inttoptr and the add should be hoisted to the preheader.
  EXPECT_EQ(llvm::cast<llvm::Instruction>(Ptr)->getParent(), Loop.Preheader);
  EXPECT_EQ(llvm::cast<llvm::Instruction>(Addr)->getParent(), Loop.Preheader);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
}

// Test 2: GEP with loop-invariant base is hoisted.
TEST_F(PointersHoistingTest, HoistGEPFromLoop) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);
  auto *voidTy = llvm::Type::getVoidTy(Ctx);

  // Struct type: { i32, i32, i32, i64 }
  auto *StructTy =
      llvm::StructType::create(Ctx, {i32Ty, i32Ty, i32Ty, i64Ty}, "S");

  auto *FTy = llvm::FunctionType::get(voidTy, {ptrTy, i64Ty}, false);
  auto *F = llvm::Function::Create(FTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);
  auto *BasePtr = F->getArg(0);
  BasePtr->setName("base_ptr");
  auto *Limit = F->getArg(1);
  Limit->setName("limit");

  auto *Entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(Entry);

  auto Loop = buildLoop(F, B, Limit);
  B.SetInsertPoint(Entry);
  B.CreateBr(Loop.Preheader);

  // In loop body: gep %base_ptr, 0, 3 (constant indices, invariant base)
  Loop.Body->getTerminator()->eraseFromParent();
  B.SetInsertPoint(Loop.Body);
  auto *GEP = B.CreateStructGEP(StructTy, BasePtr, 3, "field3");
  auto *Val = B.CreateLoad(i64Ty, GEP, "val");
  (void)Val;
  B.CreateBr(Loop.Latch);

  B.SetInsertPoint(Loop.Exit);
  B.CreateRetVoid();

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  ASSERT_EQ(llvm::cast<llvm::Instruction>(GEP)->getParent(), Loop.Body);

  runPass(F);

  EXPECT_EQ(llvm::cast<llvm::Instruction>(GEP)->getParent(), Loop.Preheader);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
}

// Test 3: GEP with loop-variant index must NOT be hoisted.
TEST_F(PointersHoistingTest, NoHoistVariantOperand) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);
  auto *voidTy = llvm::Type::getVoidTy(Ctx);

  auto *FTy = llvm::FunctionType::get(voidTy, {ptrTy, i64Ty}, false);
  auto *F = llvm::Function::Create(FTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);
  auto *BasePtr = F->getArg(0);
  BasePtr->setName("base_ptr");
  auto *Limit = F->getArg(1);
  Limit->setName("limit");

  auto *Entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(Entry);

  auto Loop = buildLoop(F, B, Limit);
  B.SetInsertPoint(Entry);
  B.CreateBr(Loop.Preheader);

  // In loop body: gep %base_ptr, %iv (loop-variant index)
  // Get the IV from the header PHI.
  auto *IV = &*Loop.Header->begin();  // The PHI node
  Loop.Body->getTerminator()->eraseFromParent();
  B.SetInsertPoint(Loop.Body);
  auto *GEP = B.CreateGEP(i64Ty, BasePtr, IV, "variant_gep");
  auto *Val = B.CreateLoad(i64Ty, GEP, "val");
  (void)Val;
  B.CreateBr(Loop.Latch);

  B.SetInsertPoint(Loop.Exit);
  B.CreateRetVoid();

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  ASSERT_EQ(llvm::cast<llvm::Instruction>(GEP)->getParent(), Loop.Body);

  runPass(F);

  // GEP should NOT have been hoisted since index is loop-variant.
  EXPECT_EQ(llvm::cast<llvm::Instruction>(GEP)->getParent(), Loop.Body);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
}

// Test 4: Load is not hoisted (has memory side effects).
TEST_F(PointersHoistingTest, NoHoistLoadFromLoop) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);
  auto *voidTy = llvm::Type::getVoidTy(Ctx);

  auto *FTy = llvm::FunctionType::get(voidTy, {ptrTy, i64Ty}, false);
  auto *F = llvm::Function::Create(FTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);
  auto *Ptr = F->getArg(0);
  Ptr->setName("ptr");
  auto *Limit = F->getArg(1);
  Limit->setName("limit");

  auto *Entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(Entry);

  auto Loop = buildLoop(F, B, Limit);
  B.SetInsertPoint(Entry);
  B.CreateBr(Loop.Preheader);

  // In loop body: load from loop-invariant pointer
  Loop.Body->getTerminator()->eraseFromParent();
  B.SetInsertPoint(Loop.Body);
  auto *Load = B.CreateLoad(i64Ty, Ptr, "loaded");
  B.CreateBr(Loop.Latch);

  B.SetInsertPoint(Loop.Exit);
  B.CreateRetVoid();

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  ASSERT_EQ(llvm::cast<llvm::Instruction>(Load)->getParent(), Loop.Body);

  runPass(F);

  // Load must NOT be hoisted — it's not a pointer computation.
  EXPECT_EQ(llvm::cast<llvm::Instruction>(Load)->getParent(), Loop.Body);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
}

// Test 5: Function without loops passes through unchanged.
TEST_F(PointersHoistingTest, NoLoopNoChange) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);

  auto *FTy = llvm::FunctionType::get(ptrTy, {i64Ty}, false);
  auto *F = llvm::Function::Create(FTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);

  auto *Entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(Entry);
  auto *Ptr = B.CreateIntToPtr(F->getArg(0), ptrTy, "ptr");
  B.CreateRet(Ptr);

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Count instructions before.
  unsigned instrCountBefore = 0;
  for (auto &BB : *F)
    for (auto &I : BB)
      ++instrCountBefore;

  runPass(F);

  // Same instruction count, same block.
  unsigned instrCountAfter = 0;
  for (auto &BB : *F)
    for (auto &I : BB)
      ++instrCountAfter;

  EXPECT_EQ(instrCountBefore, instrCountAfter);
  EXPECT_EQ(llvm::cast<llvm::Instruction>(Ptr)->getParent(), Entry);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
}

// Test 6: Nested loops — instruction with outer-invariant operands hoisted to
// inner preheader.
TEST_F(PointersHoistingTest, NestedLoopHoist) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);
  auto *voidTy = llvm::Type::getVoidTy(Ctx);

  auto *FTy = llvm::FunctionType::get(voidTy, {i64Ty, i64Ty}, false);
  auto *F = llvm::Function::Create(FTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);
  auto *Base = F->getArg(0);
  Base->setName("base");
  auto *Limit = F->getArg(1);
  Limit->setName("limit");

  auto *Entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(Entry);

  // Build outer loop.
  auto Outer = buildLoop(F, B, Limit, "outer.");
  B.SetInsertPoint(Entry);
  B.CreateBr(Outer.Preheader);

  // Build inner loop inside outer body.
  // Replace outer body terminator.
  Outer.Body->getTerminator()->eraseFromParent();
  B.SetInsertPoint(Outer.Body);

  auto Inner = buildLoop(F, B, Limit, "inner.");
  B.SetInsertPoint(Outer.Body);
  B.CreateBr(Inner.Preheader);

  // Inner body: inttoptr(base + 16) — base is invariant to both loops.
  Inner.Body->getTerminator()->eraseFromParent();
  B.SetInsertPoint(Inner.Body);
  auto *Addr = B.CreateAdd(Base, llvm::ConstantInt::get(i64Ty, 16), "addr");
  auto *Ptr = B.CreateIntToPtr(Addr, ptrTy, "ptr");
  auto *Val = B.CreateLoad(i64Ty, Ptr, "val");
  (void)Val;
  B.CreateBr(Inner.Latch);

  // Inner exit -> outer latch.
  B.SetInsertPoint(Inner.Exit);
  B.CreateBr(Outer.Latch);

  // Outer exit: ret void.
  B.SetInsertPoint(Outer.Exit);
  B.CreateRetVoid();

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  ASSERT_EQ(llvm::cast<llvm::Instruction>(Ptr)->getParent(), Inner.Body);

  runPass(F);

  // The inttoptr should be hoisted out of the inner loop.
  // Since base is also outer-invariant, it could go to inner preheader
  // (which is outer body) first, then outer preheader on the outer pass.
  // The pass processes innermost first, so it goes to inner preheader first.
  // Then the outer loop pass sees it in the outer body and hoists to outer
  // preheader.
  EXPECT_NE(llvm::cast<llvm::Instruction>(Ptr)->getParent(), Inner.Body);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
}

// Test 7: After hoisting, function still verifies.
TEST_F(PointersHoistingTest, PreservesCorrectness) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);
  auto *voidTy = llvm::Type::getVoidTy(Ctx);

  auto *FTy = llvm::FunctionType::get(voidTy, {i64Ty, i64Ty}, false);
  auto *F = llvm::Function::Create(FTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);
  auto *Base = F->getArg(0);
  Base->setName("base");
  auto *Limit = F->getArg(1);
  Limit->setName("limit");

  auto *Entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(Entry);

  auto Loop = buildLoop(F, B, Limit);
  B.SetInsertPoint(Entry);
  B.CreateBr(Loop.Preheader);

  // Multiple hoistable instructions: inttoptr, bitcast chain.
  Loop.Body->getTerminator()->eraseFromParent();
  B.SetInsertPoint(Loop.Body);
  auto *Addr = B.CreateAdd(Base, llvm::ConstantInt::get(i64Ty, 32), "addr");
  auto *Ptr = B.CreateIntToPtr(Addr, ptrTy, "ptr");
  auto *Val = B.CreateLoad(i64Ty, Ptr, "val");
  B.CreateBr(Loop.Latch);

  B.SetInsertPoint(Loop.Exit);
  B.CreateRetVoid();

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(F);

  // Function must verify after hoisting.
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // The hoisted instructions should be in the preheader.
  EXPECT_EQ(llvm::cast<llvm::Instruction>(Ptr)->getParent(), Loop.Preheader);
  EXPECT_EQ(llvm::cast<llvm::Instruction>(Addr)->getParent(), Loop.Preheader);
  // The load should stay in the loop body.
  EXPECT_EQ(llvm::cast<llvm::Instruction>(Val)->getParent(), Loop.Body);
}

}  // namespace
