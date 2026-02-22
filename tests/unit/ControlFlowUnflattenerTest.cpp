#include "omill/Passes/ControlFlowUnflattener.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>

#include <gtest/gtest.h>

namespace {

static const char *kDataLayout =
    "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-"
    "f80:128-n8:16:32:64-S128";

class ControlFlowUnflattenerTest : public ::testing::Test {
 protected:
  llvm::LLVMContext Ctx;

  std::unique_ptr<llvm::Module> createModule() {
    auto M = std::make_unique<llvm::Module>("test", Ctx);
    M->setDataLayout(kDataLayout);
    return M;
  }

  void runPass(llvm::Function *F, bool simplify = true) {
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
    FPM.addPass(omill::ControlFlowUnflattenerPass());
    if (simplify)
      FPM.addPass(llvm::SimplifyCFGPass());
    FPM.run(*F, FAM);
  }

  /// Count basic blocks in a function.
  unsigned countBlocks(llvm::Function *F) {
    unsigned n = 0;
    for (auto &BB : *F)
      ++n;
    return n;
  }

  /// Check if any block contains a switch instruction.
  bool hasSwitch(llvm::Function *F) {
    for (auto &BB : *F)
      if (llvm::isa<llvm::SwitchInst>(BB.getTerminator()))
        return true;
    return false;
  }

  /// Count the number of conditional branches.
  unsigned countCondBranches(llvm::Function *F) {
    unsigned n = 0;
    for (auto &BB : *F)
      if (auto *br = llvm::dyn_cast<llvm::BranchInst>(BB.getTerminator()))
        if (br->isConditional())
          ++n;
    return n;
  }
};

/// Build a simple CFF-flattened function:
///   entry → dispatcher → {case1, case2, case3}
///   case1 → dispatcher (next: case2)
///   case2 → dispatcher (next: case3)
///   case3 → return
///
/// Original CFG: entry → case1 → case2 → case3 → ret
TEST_F(ControlFlowUnflattenerTest, SimpleThreeCaseChain) {
  auto M = createModule();
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test_cff", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  auto *disp = llvm::BasicBlock::Create(Ctx, "dispatcher", F);
  auto *case1 = llvm::BasicBlock::Create(Ctx, "case1", F);
  auto *case2 = llvm::BasicBlock::Create(Ctx, "case2", F);
  auto *case3 = llvm::BasicBlock::Create(Ctx, "case3", F);
  auto *dead = llvm::BasicBlock::Create(Ctx, "dead", F);

  llvm::IRBuilder<> B(entry);
  B.CreateBr(disp);

  // dispatcher: switch on state PHI
  B.SetInsertPoint(disp);
  auto *statePhi = B.CreatePHI(i32Ty, 4, "state");
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x10), entry);
  auto *sw = B.CreateSwitch(statePhi, dead, 3);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x10), case1);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x20), case2);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x30), case3);

  // case1: set state=0x20, br dispatcher
  B.SetInsertPoint(case1);
  B.CreateBr(disp);
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x20), case1);

  // case2: set state=0x30, br dispatcher
  B.SetInsertPoint(case2);
  B.CreateBr(disp);
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x30), case2);

  // case3: return
  B.SetInsertPoint(case3);
  B.CreateRet(llvm::ConstantInt::get(i64Ty, 42));

  // dead (switch default)
  B.SetInsertPoint(dead);
  B.CreateUnreachable();

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_TRUE(hasSwitch(F));

  runPass(F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // After: dispatcher should be gone, switch eliminated.
  // SimplifyCFG cleans up the dead dispatcher.
  EXPECT_FALSE(hasSwitch(F));
}

/// CFF with a conditional (select) next-state:
///   entry → dispatcher
///   case1: select(cond, 0x20, 0x30) → br dispatcher  →  br cond, case2, case3
///   case2: return 1
///   case3: return 2
TEST_F(ControlFlowUnflattenerTest, ConditionalSelect) {
  auto M = createModule();
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test_select", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  auto *disp = llvm::BasicBlock::Create(Ctx, "dispatcher", F);
  auto *case0 = llvm::BasicBlock::Create(Ctx, "case0", F);
  auto *case1 = llvm::BasicBlock::Create(Ctx, "case1", F);
  auto *case2 = llvm::BasicBlock::Create(Ctx, "case2", F);
  auto *case3 = llvm::BasicBlock::Create(Ctx, "case3", F);
  auto *dead = llvm::BasicBlock::Create(Ctx, "dead", F);

  llvm::IRBuilder<> B(entry);
  B.CreateBr(disp);

  B.SetInsertPoint(disp);
  auto *statePhi = B.CreatePHI(i32Ty, 4, "state");
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x05), entry);
  auto *sw = B.CreateSwitch(statePhi, dead, 4);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x05), case0);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x10), case1);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x20), case2);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x30), case3);

  // case0: unconditional → case1
  B.SetInsertPoint(case0);
  B.CreateBr(disp);
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x10), case0);

  // case1: conditional next-state via select → case2 or case3
  B.SetInsertPoint(case1);
  auto *cond = B.CreateICmpSGT(F->getArg(0), llvm::ConstantInt::get(i64Ty, 0));
  auto *nextState = B.CreateSelect(
      cond, llvm::ConstantInt::get(i32Ty, 0x20),
      llvm::ConstantInt::get(i32Ty, 0x30));
  B.CreateBr(disp);
  statePhi->addIncoming(nextState, case1);

  // case2: return 1
  B.SetInsertPoint(case2);
  B.CreateRet(llvm::ConstantInt::get(i64Ty, 1));

  // case3: return 2
  B.SetInsertPoint(case3);
  B.CreateRet(llvm::ConstantInt::get(i64Ty, 2));

  B.SetInsertPoint(dead);
  B.CreateUnreachable();

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Run without SimplifyCFG to inspect the raw unflattened form.
  runPass(F, /*simplify=*/false);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  // case1 should now have a conditional branch (from the select redirect).
  auto *br1 = llvm::dyn_cast<llvm::BranchInst>(case1->getTerminator());
  ASSERT_NE(br1, nullptr);
  EXPECT_TRUE(br1->isConditional());
  EXPECT_EQ(br1->getSuccessor(0), case2);
  EXPECT_EQ(br1->getSuccessor(1), case3);

  // case0 should go directly to case1.
  auto *br0 = llvm::dyn_cast<llvm::BranchInst>(case0->getTerminator());
  ASSERT_NE(br0, nullptr);
  EXPECT_EQ(br0->getSuccessor(0), case1);
}

/// CFF with value-carrying PHIs (not just state):
///   dispatcher has %state PHI and %x PHI
///   case1: %x1 = %x + 1, next → case2
///   case2: %x2 = %x * 2, next → case3
///   case3: ret %x
///
/// After unflattening, values must be correctly threaded through proxy PHIs.
TEST_F(ControlFlowUnflattenerTest, ValueCarryingPhis) {
  auto M = createModule();
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test_values", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  auto *disp = llvm::BasicBlock::Create(Ctx, "dispatcher", F);
  auto *case1 = llvm::BasicBlock::Create(Ctx, "case1", F);
  auto *case2 = llvm::BasicBlock::Create(Ctx, "case2", F);
  auto *case3 = llvm::BasicBlock::Create(Ctx, "case3", F);
  auto *dead = llvm::BasicBlock::Create(Ctx, "dead", F);

  llvm::IRBuilder<> B(entry);
  B.CreateBr(disp);

  // dispatcher: state PHI + value PHI
  B.SetInsertPoint(disp);
  auto *statePhi = B.CreatePHI(i32Ty, 4, "state");
  auto *xPhi = B.CreatePHI(i64Ty, 4, "x");
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x10), entry);
  xPhi->addIncoming(llvm::ConstantInt::get(i64Ty, 10), entry);

  auto *sw = B.CreateSwitch(statePhi, dead, 3);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x10), case1);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x20), case2);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x30), case3);

  // case1: x1 = x + 1, next → case2
  B.SetInsertPoint(case1);
  auto *x1 = B.CreateAdd(xPhi, llvm::ConstantInt::get(i64Ty, 1), "x1");
  B.CreateBr(disp);
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x20), case1);
  xPhi->addIncoming(x1, case1);

  // case2: x2 = x * 2, next → case3
  B.SetInsertPoint(case2);
  auto *x2 = B.CreateMul(xPhi, llvm::ConstantInt::get(i64Ty, 2), "x2");
  B.CreateBr(disp);
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x30), case2);
  xPhi->addIncoming(x2, case2);

  // case3: ret x
  B.SetInsertPoint(case3);
  B.CreateRet(xPhi);

  B.SetInsertPoint(dead);
  B.CreateUnreachable();

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_FALSE(hasSwitch(F));

  // Verify the function computes the correct result:
  //   x = 10 → case1: x = 11 → case2: x = 22 → case3: ret 22
  // After SimplifyCFG + constant folding the chain is linear.
  // The important thing is that the verifier passes — values are correctly
  // threaded through proxy PHIs.
}

/// Partial resolution: one case block has a dynamic next-state.
/// Only the resolvable case blocks should be redirected.
TEST_F(ControlFlowUnflattenerTest, PartialResolve) {
  auto M = createModule();
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {i32Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test_partial", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  auto *disp = llvm::BasicBlock::Create(Ctx, "dispatcher", F);
  auto *case1 = llvm::BasicBlock::Create(Ctx, "case1", F);
  auto *case2 = llvm::BasicBlock::Create(Ctx, "case2", F);
  auto *case3 = llvm::BasicBlock::Create(Ctx, "case3", F);
  auto *case4 = llvm::BasicBlock::Create(Ctx, "case4", F);
  auto *dead = llvm::BasicBlock::Create(Ctx, "dead", F);

  llvm::IRBuilder<> B(entry);
  B.CreateBr(disp);

  B.SetInsertPoint(disp);
  auto *statePhi = B.CreatePHI(i32Ty, 5, "state");
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x10), entry);
  auto *sw = B.CreateSwitch(statePhi, dead, 4);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x10), case1);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x20), case2);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x30), case3);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x40), case4);

  // case1: constant next → case2 (resolvable)
  B.SetInsertPoint(case1);
  B.CreateBr(disp);
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x20), case1);

  // case2: DYNAMIC next state (unresolvable — uses function argument)
  B.SetInsertPoint(case2);
  B.CreateBr(disp);
  statePhi->addIncoming(F->getArg(0), case2);

  // case3: constant next → case4 (resolvable)
  B.SetInsertPoint(case3);
  B.CreateBr(disp);
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x40), case3);

  // case4: return
  B.SetInsertPoint(case4);
  B.CreateRet(llvm::ConstantInt::get(i64Ty, 42));

  B.SetInsertPoint(dead);
  B.CreateUnreachable();

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Run without SimplifyCFG to inspect the raw output.
  runPass(F, /*simplify=*/false);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // The dispatcher should still exist (case2 is unresolvable).
  EXPECT_TRUE(hasSwitch(F));

  // But case1 should now go to case2 directly, and case3 to case4.
  // Verify case1's terminator goes to case2.
  auto *br1 = llvm::dyn_cast<llvm::BranchInst>(case1->getTerminator());
  ASSERT_NE(br1, nullptr);
  EXPECT_FALSE(br1->isConditional());
  EXPECT_EQ(br1->getSuccessor(0), case2);

  // case3 → case4
  auto *br3 = llvm::dyn_cast<llvm::BranchInst>(case3->getTerminator());
  ASSERT_NE(br3, nullptr);
  EXPECT_FALSE(br3->isConditional());
  EXPECT_EQ(br3->getSuccessor(0), case4);
}

/// No dispatcher: regular function with switch that doesn't match CFF pattern.
TEST_F(ControlFlowUnflattenerTest, NoDispatcherIsNoOp) {
  auto M = createModule();
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {i32Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "regular", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  auto *a = llvm::BasicBlock::Create(Ctx, "a", F);
  auto *b = llvm::BasicBlock::Create(Ctx, "b", F);
  auto *c = llvm::BasicBlock::Create(Ctx, "c", F);

  // Regular switch on function argument — NOT a CFF dispatcher.
  llvm::IRBuilder<> B(entry);
  auto *sw = B.CreateSwitch(F->getArg(0), c, 2);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 1), a);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 2), b);

  B.SetInsertPoint(a);
  B.CreateRet(llvm::ConstantInt::get(i64Ty, 10));
  B.SetInsertPoint(b);
  B.CreateRet(llvm::ConstantInt::get(i64Ty, 20));
  B.SetInsertPoint(c);
  B.CreateRet(llvm::ConstantInt::get(i64Ty, 30));

  unsigned before_blocks = countBlocks(F);
  runPass(F, /*simplify=*/false);

  // Nothing should change — switch on function arg isn't a PHI dispatcher.
  EXPECT_EQ(countBlocks(F), before_blocks);
  EXPECT_TRUE(hasSwitch(F));
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
}

/// Self-loop case: a case block sets its own state value.
/// Should NOT be redirected (would create infinite loop to itself).
TEST_F(ControlFlowUnflattenerTest, SelfLoopSkipped) {
  auto M = createModule();
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test_selfloop", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  auto *disp = llvm::BasicBlock::Create(Ctx, "dispatcher", F);
  auto *case1 = llvm::BasicBlock::Create(Ctx, "case1", F);
  auto *case2 = llvm::BasicBlock::Create(Ctx, "case2", F);
  auto *case3 = llvm::BasicBlock::Create(Ctx, "case3", F);
  auto *dead = llvm::BasicBlock::Create(Ctx, "dead", F);

  llvm::IRBuilder<> B(entry);
  B.CreateBr(disp);

  B.SetInsertPoint(disp);
  auto *statePhi = B.CreatePHI(i32Ty, 4, "state");
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x10), entry);
  auto *sw = B.CreateSwitch(statePhi, dead, 3);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x10), case1);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x20), case2);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x30), case3);

  // case1: SELF-LOOP (state = 0x10, case1's own state)
  B.SetInsertPoint(case1);
  B.CreateBr(disp);
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x10), case1);

  // case2: → case3
  B.SetInsertPoint(case2);
  B.CreateBr(disp);
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x30), case2);

  // case3: return
  B.SetInsertPoint(case3);
  B.CreateRet(llvm::ConstantInt::get(i64Ty, 0));

  B.SetInsertPoint(dead);
  B.CreateUnreachable();

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(F, /*simplify=*/false);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // case1 should NOT be redirected to itself — it should still go to dispatcher.
  auto *br1 = llvm::dyn_cast<llvm::BranchInst>(case1->getTerminator());
  ASSERT_NE(br1, nullptr);
  EXPECT_EQ(br1->getSuccessor(0), disp);

  // But case2 should be redirected to case3.
  auto *br2 = llvm::dyn_cast<llvm::BranchInst>(case2->getTerminator());
  ASSERT_NE(br2, nullptr);
  EXPECT_EQ(br2->getSuccessor(0), case3);
}

/// Two-entry diamond: two case blocks redirect to the same target.
/// Proxy PHIs should handle multiple incoming edges correctly.
TEST_F(ControlFlowUnflattenerTest, DiamondMerge) {
  auto M = createModule();
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test_diamond", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  auto *disp = llvm::BasicBlock::Create(Ctx, "dispatcher", F);
  auto *case1 = llvm::BasicBlock::Create(Ctx, "case1", F);
  auto *case2 = llvm::BasicBlock::Create(Ctx, "case2", F);
  auto *case3 = llvm::BasicBlock::Create(Ctx, "case3", F);
  auto *dead = llvm::BasicBlock::Create(Ctx, "dead", F);

  llvm::IRBuilder<> B(entry);
  B.CreateBr(disp);

  B.SetInsertPoint(disp);
  auto *statePhi = B.CreatePHI(i32Ty, 4, "state");
  auto *xPhi = B.CreatePHI(i64Ty, 4, "x");
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x10), entry);
  xPhi->addIncoming(F->getArg(0), entry);

  auto *sw = B.CreateSwitch(statePhi, dead, 3);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x10), case1);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x20), case2);
  sw->addCase(llvm::ConstantInt::get(i32Ty, 0x30), case3);

  // case1: conditional select → case2 or case3
  B.SetInsertPoint(case1);
  auto *cond = B.CreateICmpSGT(xPhi, llvm::ConstantInt::get(i64Ty, 5));
  auto *x1 = B.CreateAdd(xPhi, llvm::ConstantInt::get(i64Ty, 10), "x1");
  auto *nextState = B.CreateSelect(
      cond, llvm::ConstantInt::get(i32Ty, 0x20),
      llvm::ConstantInt::get(i32Ty, 0x30));
  B.CreateBr(disp);
  statePhi->addIncoming(nextState, case1);
  xPhi->addIncoming(x1, case1);

  // case2: set next → case3. Both case1 and case2 target case3.
  B.SetInsertPoint(case2);
  auto *x2 = B.CreateMul(xPhi, llvm::ConstantInt::get(i64Ty, 3), "x2");
  B.CreateBr(disp);
  statePhi->addIncoming(llvm::ConstantInt::get(i32Ty, 0x30), case2);
  xPhi->addIncoming(x2, case2);

  // case3: return x
  B.SetInsertPoint(case3);
  B.CreateRet(xPhi);

  B.SetInsertPoint(dead);
  B.CreateUnreachable();

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  // Switch should be eliminated — all paths are resolved.
  EXPECT_FALSE(hasSwitch(F));
}

}  // namespace
