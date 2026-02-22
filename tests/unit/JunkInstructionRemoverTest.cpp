#include "omill/Passes/JunkInstructionRemover.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
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

class JunkInstructionRemoverTest : public ::testing::Test {
 protected:
  llvm::LLVMContext Ctx;

  std::unique_ptr<llvm::Module> createModule() {
    auto M = std::make_unique<llvm::Module>("test", Ctx);
    M->setDataLayout(kDataLayout);
    return M;
  }

  void runPass(llvm::Function &F) {
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
    FPM.addPass(omill::JunkInstructionRemoverPass());
    FPM.run(F, FAM);
  }

  /// Count instructions of a given opcode in a function.
  unsigned countOpcode(llvm::Function &F, unsigned Opcode) {
    unsigned count = 0;
    for (auto it = llvm::inst_begin(F), end = llvm::inst_end(F); it != end;
         ++it)
      if (it->getOpcode() == Opcode)
        ++count;
    return count;
  }

  /// Count total instructions in a function.
  unsigned countInstructions(llvm::Function &F) {
    return std::distance(llvm::inst_begin(F), llvm::inst_end(F));
  }

  /// Check if any instruction uses the given value.
  bool hasUserInstruction(llvm::Value *V, unsigned Opcode) {
    for (auto *U : V->users())
      if (auto *I = llvm::dyn_cast<llvm::Instruction>(U))
        if (I->getOpcode() == Opcode)
          return true;
    return false;
  }
};

// ---------- Identity operation tests ----------

TEST_F(JunkInstructionRemoverTest, RemoveIdentityAdd) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *x = F->getArg(0);
  auto *add = B.CreateAdd(x, llvm::ConstantInt::get(i64Ty, 0), "junk_add");
  B.CreateRet(add);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  runPass(*F);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // The add should be gone; ret should directly use the argument.
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Add), 0u);
  auto *retInst = llvm::cast<llvm::ReturnInst>(entry->getTerminator());
  EXPECT_EQ(retInst->getReturnValue(), x);
}

TEST_F(JunkInstructionRemoverTest, RemoveIdentityMul) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *x = F->getArg(0);
  auto *mul = B.CreateMul(x, llvm::ConstantInt::get(i64Ty, 1), "junk_mul");
  B.CreateRet(mul);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  runPass(*F);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Mul), 0u);
  auto *retInst = llvm::cast<llvm::ReturnInst>(entry->getTerminator());
  EXPECT_EQ(retInst->getReturnValue(), x);
}

TEST_F(JunkInstructionRemoverTest, RemoveIdentityXor) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *x = F->getArg(0);
  auto *xorI = B.CreateXor(x, llvm::ConstantInt::get(i64Ty, 0), "junk_xor");
  B.CreateRet(xorI);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  runPass(*F);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Xor), 0u);
  auto *retInst = llvm::cast<llvm::ReturnInst>(entry->getTerminator());
  EXPECT_EQ(retInst->getReturnValue(), x);
}

TEST_F(JunkInstructionRemoverTest, RemoveIdentityAnd) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *x = F->getArg(0);
  // and %x, -1 → %x
  auto *andI = B.CreateAnd(x, llvm::ConstantInt::getSigned(i64Ty, -1),
                           "junk_and");
  B.CreateRet(andI);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  runPass(*F);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  EXPECT_EQ(countOpcode(*F, llvm::Instruction::And), 0u);
  auto *retInst = llvm::cast<llvm::ReturnInst>(entry->getTerminator());
  EXPECT_EQ(retInst->getReturnValue(), x);
}

// ---------- Self-canceling pair tests ----------

TEST_F(JunkInstructionRemoverTest, RemoveDoubleNot) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *x = F->getArg(0);
  auto *not1 = B.CreateNot(x, "not1");
  auto *not2 = B.CreateNot(not1, "not2");
  B.CreateRet(not2);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  runPass(*F);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Both xors (not instructions) should be removed.
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Xor), 0u);
  auto *retInst = llvm::cast<llvm::ReturnInst>(entry->getTerminator());
  EXPECT_EQ(retInst->getReturnValue(), x);
}

TEST_F(JunkInstructionRemoverTest, RemoveDoubleNeg) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *x = F->getArg(0);
  auto *neg1 = B.CreateNeg(x, "neg1");
  auto *neg2 = B.CreateNeg(neg1, "neg2");
  B.CreateRet(neg2);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  runPass(*F);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Both sub instructions (neg) should be removed.
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Sub), 0u);
  auto *retInst = llvm::cast<llvm::ReturnInst>(entry->getTerminator());
  EXPECT_EQ(retInst->getReturnValue(), x);
}

// ---------- Alloca tests ----------

TEST_F(JunkInstructionRemoverTest, RemoveUnusedAlloca) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *voidTy = llvm::Type::getVoidTy(Ctx);
  auto *fnTy = llvm::FunctionType::get(voidTy, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *alloca = B.CreateAlloca(i64Ty, nullptr, "unused_slot");
  B.CreateStore(F->getArg(0), alloca);
  B.CreateStore(llvm::ConstantInt::get(i64Ty, 42), alloca);
  B.CreateRetVoid();

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  unsigned before = countInstructions(*F);
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Alloca), 1u);
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Store), 2u);

  runPass(*F);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Alloca and both stores should be removed.
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Alloca), 0u);
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Store), 0u);
  EXPECT_LT(countInstructions(*F), before);
}

TEST_F(JunkInstructionRemoverTest, PreserveUsedAlloca) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *alloca = B.CreateAlloca(i64Ty, nullptr, "used_slot");
  B.CreateStore(F->getArg(0), alloca);
  auto *loaded = B.CreateLoad(i64Ty, alloca, "loaded");
  B.CreateRet(loaded);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(*F);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Alloca, store, and load should all be preserved.
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Alloca), 1u);
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Store), 1u);
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Load), 1u);
}

// ---------- Pointer roundtrip test ----------

TEST_F(JunkInstructionRemoverTest, RemovePointerRoundtrip) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);
  auto *fnTy = llvm::FunctionType::get(ptrTy, {ptrTy}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *p = F->getArg(0);
  auto *asInt = B.CreatePtrToInt(p, i64Ty, "as_int");
  auto *asPtr = B.CreateIntToPtr(asInt, ptrTy, "as_ptr");
  B.CreateRet(asPtr);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  runPass(*F);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Both ptrtoint and inttoptr should be removed.
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::PtrToInt), 0u);
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::IntToPtr), 0u);
  auto *retInst = llvm::cast<llvm::ReturnInst>(entry->getTerminator());
  EXPECT_EQ(retInst->getReturnValue(), p);
}

// ---------- No-change / preservation test ----------

TEST_F(JunkInstructionRemoverTest, NoChangeCleanFunction) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty, i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  // Legitimate add of two unknowns — not an identity op.
  auto *sum = B.CreateAdd(F->getArg(0), F->getArg(1), "sum");
  B.CreateRet(sum);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  unsigned before = countInstructions(*F);

  runPass(*F);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Nothing should change.
  EXPECT_EQ(countInstructions(*F), before);
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Add), 1u);
}

// ---------- Combined patterns test ----------

TEST_F(JunkInstructionRemoverTest, CombinedPatterns) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty, ptrTy}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *x = F->getArg(0);
  auto *p = F->getArg(1);

  // Junk 1: identity add
  auto *junkAdd = B.CreateAdd(x, llvm::ConstantInt::get(i64Ty, 0), "junk_add");
  // Junk 2: double not on top of junk add result
  auto *not1 = B.CreateNot(junkAdd, "not1");
  auto *not2 = B.CreateNot(not1, "not2");
  // Junk 3: unused alloca with store
  auto *junkAlloca = B.CreateAlloca(i64Ty, nullptr, "junk_slot");
  B.CreateStore(x, junkAlloca);
  // Junk 4: pointer roundtrip (unused, just for removal)
  auto *asInt = B.CreatePtrToInt(p, i64Ty, "as_int");
  auto *asPtr = B.CreateIntToPtr(asInt, ptrTy, "as_ptr");

  // Use only not2 (which should collapse to x) in the return.
  // asPtr is unused — will be cleaned up.
  (void)asPtr;
  B.CreateRet(not2);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(*F);
  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // All junk should be gone.
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Add), 0u);
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Xor), 0u);
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Alloca), 0u);
  EXPECT_EQ(countOpcode(*F, llvm::Instruction::Store), 0u);

  // Return should use the original argument directly.
  auto *retInst = llvm::cast<llvm::ReturnInst>(entry->getTerminator());
  EXPECT_EQ(retInst->getReturnValue(), x);
}

}  // namespace
