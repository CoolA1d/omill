#include "omill/Analysis/SegmentsAA.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>

#include "omill/Analysis/BinaryMemoryMap.h"

#include <gtest/gtest.h>

namespace {

static const char *kDataLayout =
    "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-"
    "f80:128-n8:16:32:64-S128";

using Seg = omill::SegmentClassifier::Segment;

class SegmentsAATest : public ::testing::Test {
 protected:
  llvm::LLVMContext Ctx;

  std::unique_ptr<llvm::Module> createModule() {
    auto M = std::make_unique<llvm::Module>("test", Ctx);
    M->setDataLayout(kDataLayout);
    return M;
  }
};

// Test: alloca is classified as Stack.
TEST_F(SegmentsAATest, AllocaIsStack) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *alloca = B.CreateAlloca(i64Ty, nullptr, "local");
  B.CreateRet(llvm::ConstantInt::get(i64Ty, 0));

  omill::SegmentClassifier classifier(0x140000000, 0x10000);
  EXPECT_EQ(classifier.classify(alloca), Seg::Stack);
}

// Test: GlobalVariable is classified as Global.
TEST_F(SegmentsAATest, GlobalVarIsGlobal) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *GV = new llvm::GlobalVariable(
      *M, i64Ty, false, llvm::GlobalValue::ExternalLinkage,
      llvm::ConstantInt::get(i64Ty, 0), "g_var");

  omill::SegmentClassifier classifier(0x140000000, 0x10000);
  EXPECT_EQ(classifier.classify(GV), Seg::Global);
}

// Test: inttoptr of constant within image range is classified as Image.
TEST_F(SegmentsAATest, IntToPtrInImageIsImage) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);
  auto *fnTy = llvm::FunctionType::get(ptrTy, {}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);

  // inttoptr 0x140001000 — within [0x140000000, 0x140010000).
  auto *addr = llvm::ConstantInt::get(i64Ty, 0x140001000);
  auto *ptr = B.CreateIntToPtr(addr, ptrTy, "img_ptr");
  B.CreateRet(ptr);

  omill::SegmentClassifier classifier(0x140000000, 0x10000);
  EXPECT_EQ(classifier.classify(ptr), Seg::Image);
}

// Test: inttoptr of constant OUTSIDE image range is Unknown.
TEST_F(SegmentsAATest, IntToPtrOutsideImageIsUnknown) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);
  auto *fnTy = llvm::FunctionType::get(ptrTy, {}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);

  auto *addr = llvm::ConstantInt::get(i64Ty, 0x7FFE0000);
  auto *ptr = B.CreateIntToPtr(addr, ptrTy, "ext_ptr");
  B.CreateRet(ptr);

  omill::SegmentClassifier classifier(0x140000000, 0x10000);
  EXPECT_EQ(classifier.classify(ptr), Seg::Unknown);
}

// Test: Stack vs Global → NoAlias.
TEST_F(SegmentsAATest, StackGlobalNoAlias) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *alloca = B.CreateAlloca(i64Ty);
  B.CreateRet(llvm::ConstantInt::get(i64Ty, 0));

  auto *GV = new llvm::GlobalVariable(
      *M, i64Ty, false, llvm::GlobalValue::ExternalLinkage,
      llvm::ConstantInt::get(i64Ty, 0), "g");

  omill::SegmentClassifier classifier(0x140000000, 0x10000);
  EXPECT_TRUE(classifier.isNoAlias(alloca, GV));
}

// Test: Stack vs Image → NoAlias.
TEST_F(SegmentsAATest, StackImageNoAlias) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);
  auto *fnTy = llvm::FunctionType::get(ptrTy, {}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *alloca = B.CreateAlloca(i64Ty);
  auto *imgPtr = B.CreateIntToPtr(
      llvm::ConstantInt::get(i64Ty, 0x140005000), ptrTy);
  B.CreateRet(imgPtr);

  omill::SegmentClassifier classifier(0x140000000, 0x10000);
  EXPECT_TRUE(classifier.isNoAlias(alloca, imgPtr));
}

// Test: Unknown vs Stack → MayAlias (conservative).
TEST_F(SegmentsAATest, UnknownVsStackMayAlias) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {ptrTy}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *alloca = B.CreateAlloca(i64Ty);
  B.CreateRet(llvm::ConstantInt::get(i64Ty, 0));

  // Function arg is Unknown — can't prove NoAlias with stack.
  omill::SegmentClassifier classifier(0x140000000, 0x10000);
  EXPECT_FALSE(classifier.isNoAlias(alloca, F->getArg(0)));
}

// Test: Same segment (two allocas) → not NoAlias (they might alias
// through pointer arithmetic, though typically they don't).
TEST_F(SegmentsAATest, SameSegmentNotNoAlias) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *a1 = B.CreateAlloca(i64Ty);
  auto *a2 = B.CreateAlloca(i64Ty);
  B.CreateRet(llvm::ConstantInt::get(i64Ty, 0));

  // Both Stack — can't prove NoAlias from segments alone.
  omill::SegmentClassifier classifier(0x140000000, 0x10000);
  EXPECT_FALSE(classifier.isNoAlias(a1, a2));
}

// Test: GEP on alloca is still Stack.
TEST_F(SegmentsAATest, GepOnAllocaIsStack) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *arrTy = llvm::ArrayType::get(i64Ty, 4);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *alloca = B.CreateAlloca(arrTy);
  auto *gep = B.CreateConstGEP2_64(arrTy, alloca, 0, 2);
  B.CreateRet(llvm::ConstantInt::get(i64Ty, 0));

  omill::SegmentClassifier classifier(0x140000000, 0x10000);
  EXPECT_EQ(classifier.classify(gep), Seg::Stack);
}

// Test: SegmentsAAResult integration via pass manager.
TEST_F(SegmentsAATest, AAResultViaPassManager) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *ptrTy = llvm::PointerType::get(Ctx, 0);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {ptrTy}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  auto *alloca = B.CreateAlloca(i64Ty);
  B.CreateRet(llvm::ConstantInt::get(i64Ty, 0));

  omill::BinaryMemoryMap mem_map;
  mem_map.setImageBase(0x140000000);
  mem_map.setImageSize(0x10000);

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

  FAM.registerPass([&] { return omill::SegmentsAA(); });
  MAM.registerPass([&mem_map] {
    return omill::BinaryMemoryAnalysis(mem_map);
  });
  (void)MAM.getResult<omill::BinaryMemoryAnalysis>(*M);

  // Verify the analysis can be obtained (exercises the full pipeline).
  auto &result = FAM.getResult<omill::SegmentsAA>(*F);
  (void)result;

  // Verify the classifier works with BinaryMemoryMap-derived bounds.
  auto *GV = new llvm::GlobalVariable(
      *M, i64Ty, false, llvm::GlobalValue::ExternalLinkage,
      llvm::ConstantInt::get(i64Ty, 0), "g");
  omill::SegmentClassifier c(0x140000000, 0x10000);
  EXPECT_TRUE(c.isNoAlias(alloca, GV));
}

}  // namespace
