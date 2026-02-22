#include "omill/Passes/PartialOverlapDSE.h"

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

class PartialOverlapDSETest : public ::testing::Test {
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
    FPM.addPass(omill::PartialOverlapDSEPass());
    FPM.run(F, FAM);
  }

  /// Count store instructions in a function.
  unsigned countStores(llvm::Function &F) {
    unsigned count = 0;
    for (auto &BB : F)
      for (auto &I : BB)
        if (llvm::isa<llvm::StoreInst>(&I))
          ++count;
    return count;
  }

  /// Create a function with a single alloca of the given type, returning
  /// {function, alloca_ptr}.
  std::pair<llvm::Function *, llvm::AllocaInst *> createFuncWithAlloca(
      llvm::Module &M, llvm::Type *allocaTy, const llvm::Twine &name = "f") {
    auto *voidTy = llvm::Type::getVoidTy(Ctx);
    auto *fnTy = llvm::FunctionType::get(voidTy, false);
    auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                     name, M);
    auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
    llvm::IRBuilder<> B(entry);
    auto *alloca = B.CreateAlloca(allocaTy, nullptr, "buf");
    return {F, alloca};
  }
};

// 1. Two i32 stores at offsets 0 and 4 kill an earlier i64 store at offset 0.
TEST_F(PartialOverlapDSETest, TwoI32StoresKillI64Store) {
  auto M = createModule();
  auto *i8Ty = llvm::Type::getInt8Ty(Ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);

  auto [F, alloca] = createFuncWithAlloca(*M, llvm::ArrayType::get(i8Ty, 8));
  auto *entry = &F->getEntryBlock();
  llvm::IRBuilder<> B(entry);

  // i64 store at offset 0 (the one that should be killed).
  auto *ptr64 = B.CreateBitCast(alloca, llvm::PointerType::getUnqual(i64Ty));
  B.CreateStore(llvm::ConstantInt::get(i64Ty, 0xDEADBEEFCAFEBABE), ptr64);

  // i32 store at offset 0.
  auto *ptr32_0 = B.CreateBitCast(alloca, llvm::PointerType::getUnqual(i32Ty));
  B.CreateStore(llvm::ConstantInt::get(i32Ty, 0x11111111), ptr32_0);

  // i32 store at offset 4.
  auto *gep4 = B.CreateConstGEP1_64(i8Ty, alloca, 4, "off4");
  auto *ptr32_4 = B.CreateBitCast(gep4, llvm::PointerType::getUnqual(i32Ty));
  B.CreateStore(llvm::ConstantInt::get(i32Ty, 0x22222222), ptr32_4);

  B.CreateRetVoid();

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countStores(*F), 3u);

  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countStores(*F), 2u);  // i64 store eliminated
}

// 2. Four i8 stores covering [0,4) kill an earlier i32 store at offset 0.
TEST_F(PartialOverlapDSETest, FourI8StoresKillI32Store) {
  auto M = createModule();
  auto *i8Ty = llvm::Type::getInt8Ty(Ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);

  auto [F, alloca] = createFuncWithAlloca(*M, llvm::ArrayType::get(i8Ty, 4));
  auto *entry = &F->getEntryBlock();
  llvm::IRBuilder<> B(entry);

  // i32 store at offset 0 (should be killed).
  auto *ptr32 = B.CreateBitCast(alloca, llvm::PointerType::getUnqual(i32Ty));
  B.CreateStore(llvm::ConstantInt::get(i32Ty, 0xAAAAAAAA), ptr32);

  // Four i8 stores at offsets 0, 1, 2, 3.
  for (unsigned i = 0; i < 4; ++i) {
    auto *gep = B.CreateConstGEP1_64(i8Ty, alloca, i);
    B.CreateStore(llvm::ConstantInt::get(i8Ty, i + 1), gep);
  }

  B.CreateRetVoid();

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countStores(*F), 5u);

  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countStores(*F), 4u);  // i32 store eliminated
}

// 3. Partial overwrite does NOT kill: i64 store then only i32 at offset 0.
TEST_F(PartialOverlapDSETest, PartialOverwriteKeepsStore) {
  auto M = createModule();
  auto *i8Ty = llvm::Type::getInt8Ty(Ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);

  auto [F, alloca] = createFuncWithAlloca(*M, llvm::ArrayType::get(i8Ty, 8));
  auto *entry = &F->getEntryBlock();
  llvm::IRBuilder<> B(entry);

  // i64 store at offset 0 (should NOT be killed — only half overwritten).
  auto *ptr64 = B.CreateBitCast(alloca, llvm::PointerType::getUnqual(i64Ty));
  B.CreateStore(llvm::ConstantInt::get(i64Ty, 0xDEADBEEFCAFEBABE), ptr64);

  // Only i32 store at offset 0 (covers [0,4) but not [4,8)).
  auto *ptr32 = B.CreateBitCast(alloca, llvm::PointerType::getUnqual(i32Ty));
  B.CreateStore(llvm::ConstantInt::get(i32Ty, 0x11111111), ptr32);

  B.CreateRetVoid();

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countStores(*F), 2u);

  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countStores(*F), 2u);  // Both stores preserved
}

// 4. Intervening load prevents kill even if bytes are later fully covered.
TEST_F(PartialOverlapDSETest, InterveningLoadPreventsKill) {
  auto M = createModule();
  auto *i8Ty = llvm::Type::getInt8Ty(Ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);

  auto [F, alloca] = createFuncWithAlloca(*M, llvm::ArrayType::get(i8Ty, 8));
  auto *entry = &F->getEntryBlock();
  llvm::IRBuilder<> B(entry);

  // i64 store at offset 0.
  auto *ptr64 = B.CreateBitCast(alloca, llvm::PointerType::getUnqual(i64Ty));
  B.CreateStore(llvm::ConstantInt::get(i64Ty, 0xDEADBEEFCAFEBABE), ptr64);

  // Load of first 4 bytes (makes bytes [0,4) live).
  auto *ptr32_0 = B.CreateBitCast(alloca, llvm::PointerType::getUnqual(i32Ty));
  auto *loaded = B.CreateLoad(i32Ty, ptr32_0, "loaded");
  // Use the loaded value to prevent it from being optimized away.
  (void)loaded;

  // i32 stores at offsets 0 and 4 (fully covering [0,8)).
  B.CreateStore(llvm::ConstantInt::get(i32Ty, 0x11111111), ptr32_0);
  auto *gep4 = B.CreateConstGEP1_64(i8Ty, alloca, 4, "off4");
  auto *ptr32_4 = B.CreateBitCast(gep4, llvm::PointerType::getUnqual(i32Ty));
  B.CreateStore(llvm::ConstantInt::get(i32Ty, 0x22222222), ptr32_4);

  B.CreateRetVoid();

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countStores(*F), 3u);

  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  // The i64 store must survive: the load read bytes [0,4) from it.
  EXPECT_EQ(countStores(*F), 3u);
}

// 5. A memory-clobbering call between stores prevents elimination.
TEST_F(PartialOverlapDSETest, CallClearsTracking) {
  auto M = createModule();
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *voidTy = llvm::Type::getVoidTy(Ctx);

  auto *externFnTy = llvm::FunctionType::get(voidTy, false);
  auto *externFn = llvm::Function::Create(
      externFnTy, llvm::Function::ExternalLinkage, "side_effect", *M);

  auto [F, alloca] = createFuncWithAlloca(*M, i32Ty, "test_call");
  auto *entry = &F->getEntryBlock();
  llvm::IRBuilder<> B(entry);

  // First i32 store.
  B.CreateStore(llvm::ConstantInt::get(i32Ty, 42), alloca);

  // Call that may clobber memory.
  B.CreateCall(externFn);

  // Second i32 store at same address.
  B.CreateStore(llvm::ConstantInt::get(i32Ty, 99), alloca);

  B.CreateRetVoid();

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countStores(*F), 2u);

  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countStores(*F), 2u);  // Both stores preserved (call between)
}

// 6. Full overwrite by a single same-sized store kills the first (standard DSE).
TEST_F(PartialOverlapDSETest, FullOverwriteSingleStore) {
  auto M = createModule();
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);

  auto [F, alloca] = createFuncWithAlloca(*M, i32Ty, "test_full");
  auto *entry = &F->getEntryBlock();
  llvm::IRBuilder<> B(entry);

  B.CreateStore(llvm::ConstantInt::get(i32Ty, 42), alloca);
  B.CreateStore(llvm::ConstantInt::get(i32Ty, 99), alloca);

  B.CreateRetVoid();

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countStores(*F), 2u);

  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countStores(*F), 1u);  // First store dead
}

// 7. Stores to different base pointers don't interfere.
TEST_F(PartialOverlapDSETest, IndependentBasesNoInterference) {
  auto M = createModule();
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);

  auto *voidTy = llvm::Type::getVoidTy(Ctx);
  auto *fnTy = llvm::FunctionType::get(voidTy, false);
  auto *F = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                   "test_bases", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);

  auto *alloca1 = B.CreateAlloca(i32Ty, nullptr, "buf1");
  auto *alloca2 = B.CreateAlloca(i32Ty, nullptr, "buf2");

  // Store to buf1, then store to buf2 — these are independent.
  B.CreateStore(llvm::ConstantInt::get(i32Ty, 42), alloca1);
  B.CreateStore(llvm::ConstantInt::get(i32Ty, 99), alloca2);

  B.CreateRetVoid();

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countStores(*F), 2u);

  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countStores(*F), 2u);  // Both stores preserved
}

// 8. Volatile stores must never be eliminated.
TEST_F(PartialOverlapDSETest, VolatileStorePreserved) {
  auto M = createModule();
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);

  auto [F, alloca] = createFuncWithAlloca(*M, i32Ty, "test_volatile");
  auto *entry = &F->getEntryBlock();
  llvm::IRBuilder<> B(entry);

  // Volatile store followed by non-volatile store at same address.
  B.CreateStore(llvm::ConstantInt::get(i32Ty, 42), alloca, /*isVolatile=*/true);
  B.CreateStore(llvm::ConstantInt::get(i32Ty, 99), alloca);

  B.CreateRetVoid();

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countStores(*F), 2u);

  runPass(*F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countStores(*F), 2u);  // Volatile store must survive
}

}  // namespace
