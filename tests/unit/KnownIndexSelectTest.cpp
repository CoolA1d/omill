#include "omill/Passes/KnownIndexSelect.h"

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

class KnownIndexSelectTest : public ::testing::Test {
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
    FPM.addPass(omill::KnownIndexSelectPass());
    FPM.run(F, FAM);
  }

  /// Create a global i64 array with the given number of elements.
  llvm::GlobalVariable *createGlobalArray(llvm::Module &M, unsigned numElems,
                                          llvm::StringRef name = "arr") {
    auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
    auto *arrTy = llvm::ArrayType::get(i64Ty, numElems);
    llvm::SmallVector<llvm::Constant *, 16> elems;
    for (unsigned i = 0; i < numElems; ++i)
      elems.push_back(llvm::ConstantInt::get(i64Ty, (i + 1) * 10));
    auto *init = llvm::ConstantArray::get(arrTy, elems);
    return new llvm::GlobalVariable(M, arrTy, /*isConstant=*/true,
                                    llvm::GlobalValue::InternalLinkage, init,
                                    name);
  }

  /// Count select instructions in a function.
  unsigned countSelects(llvm::Function &F) {
    unsigned count = 0;
    for (auto &BB : F)
      for (auto &I : BB)
        if (llvm::isa<llvm::SelectInst>(&I))
          ++count;
    return count;
  }

  /// Count load instructions in a function.
  unsigned countLoads(llvm::Function &F) {
    unsigned count = 0;
    for (auto &BB : F)
      for (auto &I : BB)
        if (llvm::isa<llvm::LoadInst>(&I))
          ++count;
    return count;
  }

  /// Count GEP instructions in a function.
  unsigned countGEPs(llvm::Function &F) {
    unsigned count = 0;
    for (auto &BB : F)
      for (auto &I : BB)
        if (llvm::isa<llvm::GetElementPtrInst>(&I))
          ++count;
    return count;
  }
};

// Test 1: gep @arr, (and %x, 1) → two concrete loads + select
TEST_F(KnownIndexSelectTest, SingleBitIndex) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *arrGV = createGlobalArray(*M, 4);
  auto *arrTy = llvm::ArrayType::get(i64Ty, 4);

  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);

  auto *x = F->getArg(0);
  auto *idx = B.CreateAnd(x, llvm::ConstantInt::get(i64Ty, 1), "idx");
  auto *gep = B.CreateGEP(arrTy, arrGV,
                           {llvm::ConstantInt::get(i64Ty, 0), idx}, "ptr");
  auto *load = B.CreateLoad(i64Ty, gep, "val");
  B.CreateRet(load);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  // Should have 2 concrete loads and 1 select.
  EXPECT_EQ(countLoads(*F), 2u);
  EXPECT_EQ(countSelects(*F), 1u);
}

// Test 2: gep @arr, (and %x, 3) → 4 concrete loads + select chain
TEST_F(KnownIndexSelectTest, TwoBitIndex) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *arrGV = createGlobalArray(*M, 4);
  auto *arrTy = llvm::ArrayType::get(i64Ty, 4);

  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);

  auto *x = F->getArg(0);
  auto *idx = B.CreateAnd(x, llvm::ConstantInt::get(i64Ty, 3), "idx");
  auto *gep = B.CreateGEP(arrTy, arrGV,
                           {llvm::ConstantInt::get(i64Ty, 0), idx}, "ptr");
  auto *load = B.CreateLoad(i64Ty, gep, "val");
  B.CreateRet(load);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  // 4 concrete loads and 3 selects (chain of 4 values).
  EXPECT_EQ(countLoads(*F), 4u);
  EXPECT_EQ(countSelects(*F), 3u);
}

// Test 3: gep @arr, 5 (constant index) → no transformation
TEST_F(KnownIndexSelectTest, FullyKnownIndex) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *arrGV = createGlobalArray(*M, 8);
  auto *arrTy = llvm::ArrayType::get(i64Ty, 8);

  auto *fnTy = llvm::FunctionType::get(i64Ty, {}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);

  auto *gep = B.CreateGEP(
      arrTy, arrGV,
      {llvm::ConstantInt::get(i64Ty, 0), llvm::ConstantInt::get(i64Ty, 5)},
      "ptr");
  auto *load = B.CreateLoad(i64Ty, gep, "val");
  B.CreateRet(load);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  unsigned loadsBefore = countLoads(*F);
  unsigned gepsBefore = countGEPs(*F);
  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  // No change — already constant indices.
  EXPECT_EQ(countLoads(*F), loadsBefore);
  EXPECT_EQ(countGEPs(*F), gepsBefore);
  EXPECT_EQ(countSelects(*F), 0u);
}

// Test 4: index with too many unknown bits → no transformation
TEST_F(KnownIndexSelectTest, TooManyValues) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *arrGV = createGlobalArray(*M, 16);
  auto *arrTy = llvm::ArrayType::get(i64Ty, 16);

  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);

  auto *x = F->getArg(0);
  // AND with 0x3FF → 10 unknown bits → 1024 possible values, way over limit.
  auto *idx = B.CreateAnd(x, llvm::ConstantInt::get(i64Ty, 0x3FF), "idx");
  auto *gep = B.CreateGEP(arrTy, arrGV,
                           {llvm::ConstantInt::get(i64Ty, 0), idx}, "ptr");
  auto *load = B.CreateLoad(i64Ty, gep, "val");
  B.CreateRet(load);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  unsigned loadsBefore = countLoads(*F);
  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  // No transformation: too many possible values.
  EXPECT_EQ(countLoads(*F), loadsBefore);
  EXPECT_EQ(countSelects(*F), 0u);
}

// Test 5: Index that resolves to a single value → direct load, no select
TEST_F(KnownIndexSelectTest, SinglePossibleValue) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *arrGV = createGlobalArray(*M, 8);
  auto *arrTy = llvm::ArrayType::get(i64Ty, 8);

  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);

  auto *x = F->getArg(0);
  // (x & 0) | 3 → always 3. The AND zeros all bits, OR sets bits 0 and 1.
  auto *zeroed = B.CreateAnd(x, llvm::ConstantInt::get(i64Ty, 0), "zeroed");
  auto *idx = B.CreateOr(zeroed, llvm::ConstantInt::get(i64Ty, 3), "idx");
  auto *gep = B.CreateGEP(arrTy, arrGV,
                           {llvm::ConstantInt::get(i64Ty, 0), idx}, "ptr");
  auto *load = B.CreateLoad(i64Ty, gep, "val");
  B.CreateRet(load);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  // Single possible value → one load, no select.
  EXPECT_EQ(countLoads(*F), 1u);
  EXPECT_EQ(countSelects(*F), 0u);
}

// Test 6: load not through GEP → unchanged
TEST_F(KnownIndexSelectTest, PreservesNonGEPLoad) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);

  auto *fnTy = llvm::FunctionType::get(i64Ty, {ptrTy}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);

  auto *load = B.CreateLoad(i64Ty, F->getArg(0), "val");
  B.CreateRet(load);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  unsigned loadsBefore = countLoads(*F);
  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countLoads(*F), loadsBefore);
  EXPECT_EQ(countSelects(*F), 0u);
}

// Test 7: volatile load through GEP with known index → not transformed
TEST_F(KnownIndexSelectTest, VolatileLoadSkipped) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *arrGV = createGlobalArray(*M, 4);
  auto *arrTy = llvm::ArrayType::get(i64Ty, 4);

  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);

  auto *x = F->getArg(0);
  auto *idx = B.CreateAnd(x, llvm::ConstantInt::get(i64Ty, 1), "idx");
  auto *gep = B.CreateGEP(arrTy, arrGV,
                           {llvm::ConstantInt::get(i64Ty, 0), idx}, "ptr");
  // Create volatile load.
  auto *load = B.CreateLoad(i64Ty, gep, /*isVolatile=*/true, "val");
  B.CreateRet(load);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  unsigned loadsBefore = countLoads(*F);
  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  // Volatile load must not be transformed.
  EXPECT_EQ(countLoads(*F), loadsBefore);
  EXPECT_EQ(countSelects(*F), 0u);
}

// Test 8: function with no GEP loads → passes through unchanged
TEST_F(KnownIndexSelectTest, FunctionWithNoGEPs) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);

  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);

  auto *x = F->getArg(0);
  auto *result = B.CreateAdd(x, llvm::ConstantInt::get(i64Ty, 42), "result");
  B.CreateRet(result);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countLoads(*F), 0u);
  EXPECT_EQ(countSelects(*F), 0u);
  EXPECT_EQ(countGEPs(*F), 0u);
}

}  // namespace
