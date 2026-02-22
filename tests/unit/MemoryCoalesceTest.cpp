#include "omill/Passes/MemoryCoalesce.h"

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

class MemoryCoalesceTest : public ::testing::Test {
 protected:
  llvm::LLVMContext Ctx;

  std::unique_ptr<llvm::Module> createModule() {
    auto M = std::make_unique<llvm::Module>("test", Ctx);
    M->setDataLayout(kDataLayout);
    return M;
  }

  void runPass(llvm::Function *F) {
    llvm::FunctionPassManager FPM;
    FPM.addPass(omill::MemoryCoalescePass());

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

  /// Create a function with a single alloca of the given size, returning the
  /// function and alloca pointer.
  std::pair<llvm::Function *, llvm::AllocaInst *> createFuncWithAlloca(
      llvm::Module &M, const char *Name, unsigned AllocaBytes) {
    auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
    auto *fnTy = llvm::FunctionType::get(i64Ty, {}, false);
    auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                     Name, M);
    auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
    llvm::IRBuilder<> B(entry);
    auto *allocaTy = llvm::ArrayType::get(llvm::Type::getInt8Ty(Ctx),
                                          AllocaBytes);
    auto *alloca = B.CreateAlloca(allocaTy, nullptr, "buf");
    return {F, alloca};
  }
};

// Test 1: 4 i8 stores to consecutive offsets → single i32 load folds to
// constant.
TEST_F(MemoryCoalesceTest, FourByteStoresToI32Load) {
  auto M = createModule();
  auto [F, alloca] = createFuncWithAlloca(*M, "test_fn", 4);

  llvm::IRBuilder<> B(&F->getEntryBlock());

  auto *i8Ty = llvm::Type::getInt8Ty(Ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);

  // Store bytes: 0x78, 0x56, 0x34, 0x12 → little-endian i32 = 0x12345678
  for (unsigned i = 0; i < 4; ++i) {
    uint8_t byteVal = (0x12345678 >> (i * 8)) & 0xFF;
    auto *ptr = B.CreateConstGEP1_32(i8Ty, alloca, i);
    B.CreateStore(llvm::ConstantInt::get(i8Ty, byteVal), ptr);
  }

  auto *loadPtr = B.CreateBitCast(alloca, llvm::PointerType::getUnqual(i32Ty));
  auto *load = B.CreateLoad(i32Ty, loadPtr, "result");
  B.CreateRet(B.CreateZExt(load, llvm::Type::getInt64Ty(Ctx)));

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(F);

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // The load should have been replaced. Check that the ret operand is a
  // constant (or derived from one via zext).
  auto *retInst = llvm::dyn_cast<llvm::ReturnInst>(
      F->getEntryBlock().getTerminator());
  ASSERT_NE(retInst, nullptr);
  // Walk through potential zext to find the constant.
  auto *retVal = retInst->getReturnValue();
  if (auto *zext = llvm::dyn_cast<llvm::ZExtInst>(retVal))
    retVal = zext->getOperand(0);
  auto *CI = llvm::dyn_cast<llvm::ConstantInt>(retVal);
  ASSERT_NE(CI, nullptr);
  EXPECT_EQ(CI->getZExtValue(), 0x12345678u);
}

// Test 2: 8 i8 stores → i64 load folds.
TEST_F(MemoryCoalesceTest, EightByteStoresToI64Load) {
  auto M = createModule();
  auto [F, alloca] = createFuncWithAlloca(*M, "test_fn", 8);

  llvm::IRBuilder<> B(&F->getEntryBlock());

  auto *i8Ty = llvm::Type::getInt8Ty(Ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);

  uint64_t expected = 0xDEADBEEFCAFEBABEULL;
  for (unsigned i = 0; i < 8; ++i) {
    uint8_t byteVal = (expected >> (i * 8)) & 0xFF;
    auto *ptr = B.CreateConstGEP1_32(i8Ty, alloca, i);
    B.CreateStore(llvm::ConstantInt::get(i8Ty, byteVal), ptr);
  }

  auto *loadPtr = B.CreateBitCast(alloca, llvm::PointerType::getUnqual(i64Ty));
  auto *load = B.CreateLoad(i64Ty, loadPtr, "result");
  B.CreateRet(load);

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(F);

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  auto *retInst = llvm::dyn_cast<llvm::ReturnInst>(
      F->getEntryBlock().getTerminator());
  ASSERT_NE(retInst, nullptr);
  auto *CI = llvm::dyn_cast<llvm::ConstantInt>(retInst->getReturnValue());
  ASSERT_NE(CI, nullptr);
  EXPECT_EQ(CI->getZExtValue(), expected);
}

// Test 3: 2 i16 stores → i32 load folds.
TEST_F(MemoryCoalesceTest, TwoI16StoresToI32Load) {
  auto M = createModule();
  auto [F, alloca] = createFuncWithAlloca(*M, "test_fn", 4);

  llvm::IRBuilder<> B(&F->getEntryBlock());

  auto *i8Ty = llvm::Type::getInt8Ty(Ctx);
  auto *i16Ty = llvm::Type::getInt16Ty(Ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);

  // Store 0xBEEF at offset 0, 0xDEAD at offset 2 → i32 = 0xDEADBEEF (LE)
  auto *ptr0 = B.CreateBitCast(alloca, llvm::PointerType::getUnqual(i16Ty));
  B.CreateStore(llvm::ConstantInt::get(i16Ty, 0xBEEF), ptr0);

  auto *ptr2Bytes = B.CreateConstGEP1_32(i8Ty, alloca, 2);
  auto *ptr2 =
      B.CreateBitCast(ptr2Bytes, llvm::PointerType::getUnqual(i16Ty));
  B.CreateStore(llvm::ConstantInt::get(i16Ty, 0xDEAD), ptr2);

  auto *loadPtr = B.CreateBitCast(alloca, llvm::PointerType::getUnqual(i32Ty));
  auto *load = B.CreateLoad(i32Ty, loadPtr, "result");
  B.CreateRet(B.CreateZExt(load, llvm::Type::getInt64Ty(Ctx)));

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(F);

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  auto *retInst = llvm::dyn_cast<llvm::ReturnInst>(
      F->getEntryBlock().getTerminator());
  ASSERT_NE(retInst, nullptr);
  auto *retVal = retInst->getReturnValue();
  if (auto *zext = llvm::dyn_cast<llvm::ZExtInst>(retVal))
    retVal = zext->getOperand(0);
  auto *CI = llvm::dyn_cast<llvm::ConstantInt>(retVal);
  ASSERT_NE(CI, nullptr);
  EXPECT_EQ(CI->getZExtValue(), 0xDEADBEEFu);
}

// Test 4: Partial coverage leaves load unchanged.
TEST_F(MemoryCoalesceTest, PartialCoverageLeavesLoad) {
  auto M = createModule();
  auto [F, alloca] = createFuncWithAlloca(*M, "test_fn", 4);

  llvm::IRBuilder<> B(&F->getEntryBlock());

  auto *i8Ty = llvm::Type::getInt8Ty(Ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);

  // Only store 3 of 4 bytes.
  for (unsigned i = 0; i < 3; ++i) {
    auto *ptr = B.CreateConstGEP1_32(i8Ty, alloca, i);
    B.CreateStore(llvm::ConstantInt::get(i8Ty, 0xAA), ptr);
  }

  auto *loadPtr = B.CreateBitCast(alloca, llvm::PointerType::getUnqual(i32Ty));
  auto *load = B.CreateLoad(i32Ty, loadPtr, "result");
  B.CreateRet(B.CreateZExt(load, llvm::Type::getInt64Ty(Ctx)));

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(F);

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // The load should still be present (not folded).
  bool hasLoad = false;
  for (auto &I : F->getEntryBlock()) {
    if (llvm::isa<llvm::LoadInst>(&I)) {
      hasLoad = true;
      break;
    }
  }
  EXPECT_TRUE(hasLoad);
}

// Test 5: Intervening call clears tracking.
TEST_F(MemoryCoalesceTest, InterveningCallClearsTracking) {
  auto M = createModule();
  auto [F, alloca] = createFuncWithAlloca(*M, "test_fn", 4);

  // Declare an external void function.
  auto *voidTy = llvm::Type::getVoidTy(Ctx);
  auto *calleeTy = llvm::FunctionType::get(voidTy, {}, false);
  auto *callee = llvm::Function::Create(calleeTy,
                                        llvm::Function::ExternalLinkage,
                                        "clobber", *M);

  llvm::IRBuilder<> B(&F->getEntryBlock());

  auto *i8Ty = llvm::Type::getInt8Ty(Ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);

  // Store all 4 bytes.
  for (unsigned i = 0; i < 4; ++i) {
    auto *ptr = B.CreateConstGEP1_32(i8Ty, alloca, i);
    B.CreateStore(llvm::ConstantInt::get(i8Ty, 0xBB), ptr);
  }

  // Intervening call clobbers memory.
  B.CreateCall(callee);

  auto *loadPtr = B.CreateBitCast(alloca, llvm::PointerType::getUnqual(i32Ty));
  auto *load = B.CreateLoad(i32Ty, loadPtr, "result");
  B.CreateRet(B.CreateZExt(load, llvm::Type::getInt64Ty(Ctx)));

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(F);

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Load should NOT have been folded due to the intervening call.
  bool hasLoad = false;
  for (auto &I : F->getEntryBlock()) {
    if (llvm::isa<llvm::LoadInst>(&I)) {
      hasLoad = true;
      break;
    }
  }
  EXPECT_TRUE(hasLoad);
}

// Test 6: Overwritten store uses the latest value.
TEST_F(MemoryCoalesceTest, OverwrittenStoreUsesLatest) {
  auto M = createModule();
  auto [F, alloca] = createFuncWithAlloca(*M, "test_fn", 4);

  llvm::IRBuilder<> B(&F->getEntryBlock());

  auto *i8Ty = llvm::Type::getInt8Ty(Ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);

  // Store all 4 bytes with value 0x11111111.
  for (unsigned i = 0; i < 4; ++i) {
    auto *ptr = B.CreateConstGEP1_32(i8Ty, alloca, i);
    B.CreateStore(llvm::ConstantInt::get(i8Ty, 0x11), ptr);
  }

  // Overwrite byte 0 with 0xFF.
  auto *ptr0 = B.CreateConstGEP1_32(i8Ty, alloca, 0);
  B.CreateStore(llvm::ConstantInt::get(i8Ty, 0xFF), ptr0);

  auto *loadPtr = B.CreateBitCast(alloca, llvm::PointerType::getUnqual(i32Ty));
  auto *load = B.CreateLoad(i32Ty, loadPtr, "result");
  B.CreateRet(B.CreateZExt(load, llvm::Type::getInt64Ty(Ctx)));

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(F);

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  auto *retInst = llvm::dyn_cast<llvm::ReturnInst>(
      F->getEntryBlock().getTerminator());
  ASSERT_NE(retInst, nullptr);
  auto *retVal = retInst->getReturnValue();
  if (auto *zext = llvm::dyn_cast<llvm::ZExtInst>(retVal))
    retVal = zext->getOperand(0);
  auto *CI = llvm::dyn_cast<llvm::ConstantInt>(retVal);
  ASSERT_NE(CI, nullptr);
  // Little-endian: byte 0 = 0xFF, bytes 1-3 = 0x11 → 0x111111FF
  EXPECT_EQ(CI->getZExtValue(), 0x111111FFu);
}

// Test 7: Volatile load is not folded.
TEST_F(MemoryCoalesceTest, VolatileLoadSkipped) {
  auto M = createModule();
  auto [F, alloca] = createFuncWithAlloca(*M, "test_fn", 4);

  llvm::IRBuilder<> B(&F->getEntryBlock());

  auto *i8Ty = llvm::Type::getInt8Ty(Ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);

  for (unsigned i = 0; i < 4; ++i) {
    auto *ptr = B.CreateConstGEP1_32(i8Ty, alloca, i);
    B.CreateStore(llvm::ConstantInt::get(i8Ty, 0xCC), ptr);
  }

  auto *loadPtr = B.CreateBitCast(alloca, llvm::PointerType::getUnqual(i32Ty));
  // Volatile load.
  auto *load = B.CreateLoad(i32Ty, loadPtr, "result");
  load->setVolatile(true);
  B.CreateRet(B.CreateZExt(load, llvm::Type::getInt64Ty(Ctx)));

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(F);

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Load must still exist.
  bool hasLoad = false;
  for (auto &I : F->getEntryBlock()) {
    if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
      if (LI->isVolatile()) {
        hasLoad = true;
        break;
      }
    }
  }
  EXPECT_TRUE(hasLoad);
}

// Test 8: No stores, no change.
TEST_F(MemoryCoalesceTest, NoStoresNoChange) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_fn", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  B.CreateRet(F->getArg(0));

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Count instructions before.
  unsigned instCountBefore = 0;
  for (auto &I : F->getEntryBlock())
    ++instCountBefore;

  runPass(F);

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Count instructions after — should be unchanged.
  unsigned instCountAfter = 0;
  for (auto &I : F->getEntryBlock())
    ++instCountAfter;

  EXPECT_EQ(instCountBefore, instCountAfter);
}

}  // namespace
