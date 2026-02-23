#include "omill/Analysis/StateFieldAccessAnalysis.h"

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>

#include <gtest/gtest.h>

namespace {

class StateFieldAccessAnalysisTest : public ::testing::Test {
 protected:
  llvm::LLVMContext Ctx;

  /// Remill lifted function type: (ptr, i64, ptr) -> ptr
  llvm::FunctionType *liftedFnTy() {
    auto *ptr_ty = llvm::PointerType::get(Ctx, 0);
    auto *i64_ty = llvm::Type::getInt64Ty(Ctx);
    return llvm::FunctionType::get(ptr_ty, {ptr_ty, i64_ty, ptr_ty}, false);
  }

  /// Create a lifted function with an entry block.
  llvm::Function *createLifted(llvm::Module &M, const char *name = "sub_1000") {
    auto *F = llvm::Function::Create(
        liftedFnTy(), llvm::Function::ExternalLinkage, name, M);
    llvm::BasicBlock::Create(Ctx, "entry", F);
    return F;
  }

  /// Run StateFieldAccessAnalysis on F and return the result.
  omill::StateFieldAccessInfo analyze(llvm::Function &F) {
    llvm::FunctionAnalysisManager FAM;
    llvm::LoopAnalysisManager LAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;
    llvm::PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    FAM.registerPass([&] { return omill::StateFieldAccessAnalysis(); });
    return FAM.getResult<omill::StateFieldAccessAnalysis>(F);
  }
};

// ===----------------------------------------------------------------------===
// Test 1: Single load from State+48 → live_in
// ===----------------------------------------------------------------------===

TEST_F(StateFieldAccessAnalysisTest, SingleLoad) {
  auto M = std::make_unique<llvm::Module>("test", Ctx);
  auto *F = createLifted(*M);
  llvm::IRBuilder<> B(&F->getEntryBlock());

  auto *state_ptr = F->getArg(0);
  auto *i8_ty = llvm::Type::getInt8Ty(Ctx);
  auto *i64_ty = llvm::Type::getInt64Ty(Ctx);
  auto *gep = B.CreateGEP(i8_ty, state_ptr, B.getInt64(48));
  B.CreateLoad(i64_ty, gep);
  B.CreateRet(F->getArg(2));

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  auto result = analyze(*F);

  EXPECT_TRUE(result.hasAccesses());
  ASSERT_EQ(result.fields.count(48u), 1u);

  auto &info = result.fields[48u];
  EXPECT_EQ(info.offset, 48u);
  EXPECT_EQ(info.size, 8u);
  EXPECT_EQ(info.loads.size(), 1u);
  EXPECT_TRUE(info.stores.empty());
  EXPECT_TRUE(info.is_live_in);
  EXPECT_FALSE(info.is_live_out);
  EXPECT_TRUE(result.live_in_offsets.count(48u));
  EXPECT_FALSE(result.live_out_offsets.count(48u));
}

// ===----------------------------------------------------------------------===
// Test 2: Single store to State+2216 → live_out
// ===----------------------------------------------------------------------===

TEST_F(StateFieldAccessAnalysisTest, SingleStore) {
  auto M = std::make_unique<llvm::Module>("test", Ctx);
  auto *F = createLifted(*M);
  llvm::IRBuilder<> B(&F->getEntryBlock());

  auto *state_ptr = F->getArg(0);
  auto *i8_ty = llvm::Type::getInt8Ty(Ctx);
  auto *i64_ty = llvm::Type::getInt64Ty(Ctx);
  auto *gep = B.CreateGEP(i8_ty, state_ptr, B.getInt64(2216));
  B.CreateStore(llvm::ConstantInt::get(i64_ty, 42), gep);
  B.CreateRet(F->getArg(2));

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  auto result = analyze(*F);

  EXPECT_TRUE(result.hasAccesses());
  ASSERT_EQ(result.fields.count(2216u), 1u);

  auto &info = result.fields[2216u];
  EXPECT_EQ(info.offset, 2216u);
  EXPECT_EQ(info.size, 8u);
  EXPECT_TRUE(info.loads.empty());
  EXPECT_EQ(info.stores.size(), 1u);
  EXPECT_FALSE(info.is_live_in);
  EXPECT_TRUE(info.is_live_out);
  EXPECT_FALSE(result.live_in_offsets.count(2216u));
  EXPECT_TRUE(result.live_out_offsets.count(2216u));
}

// ===----------------------------------------------------------------------===
// Test 3: Store then load same field → NOT live_in (written first)
// ===----------------------------------------------------------------------===

TEST_F(StateFieldAccessAnalysisTest, StoreThenLoad) {
  auto M = std::make_unique<llvm::Module>("test", Ctx);
  auto *F = createLifted(*M);
  llvm::IRBuilder<> B(&F->getEntryBlock());

  auto *state_ptr = F->getArg(0);
  auto *i8_ty = llvm::Type::getInt8Ty(Ctx);
  auto *i64_ty = llvm::Type::getInt64Ty(Ctx);
  auto *gep = B.CreateGEP(i8_ty, state_ptr, B.getInt64(48));

  // Store first, then load.
  B.CreateStore(llvm::ConstantInt::get(i64_ty, 99), gep);
  B.CreateLoad(i64_ty, gep);
  B.CreateRet(F->getArg(2));

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  auto result = analyze(*F);

  ASSERT_EQ(result.fields.count(48u), 1u);
  auto &info = result.fields[48u];
  EXPECT_FALSE(info.is_live_in) << "Store before load → not live_in";
  EXPECT_TRUE(info.is_live_out);
  EXPECT_EQ(info.loads.size(), 1u);
  EXPECT_EQ(info.stores.size(), 1u);
  EXPECT_FALSE(result.live_in_offsets.count(48u));
}

// ===----------------------------------------------------------------------===
// Test 4: Load then store same field → live_in AND live_out
// ===----------------------------------------------------------------------===

TEST_F(StateFieldAccessAnalysisTest, LoadThenStore) {
  auto M = std::make_unique<llvm::Module>("test", Ctx);
  auto *F = createLifted(*M);
  llvm::IRBuilder<> B(&F->getEntryBlock());

  auto *state_ptr = F->getArg(0);
  auto *i8_ty = llvm::Type::getInt8Ty(Ctx);
  auto *i64_ty = llvm::Type::getInt64Ty(Ctx);
  auto *gep = B.CreateGEP(i8_ty, state_ptr, B.getInt64(48));

  // Load first, then store.
  auto *val = B.CreateLoad(i64_ty, gep);
  auto *inc = B.CreateAdd(val, llvm::ConstantInt::get(i64_ty, 1));
  B.CreateStore(inc, gep);
  B.CreateRet(F->getArg(2));

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  auto result = analyze(*F);

  ASSERT_EQ(result.fields.count(48u), 1u);
  auto &info = result.fields[48u];
  EXPECT_TRUE(info.is_live_in);
  EXPECT_TRUE(info.is_live_out);
  EXPECT_EQ(info.loads.size(), 1u);
  EXPECT_EQ(info.stores.size(), 1u);
  EXPECT_TRUE(result.live_in_offsets.count(48u));
  EXPECT_TRUE(result.live_out_offsets.count(48u));
}

// ===----------------------------------------------------------------------===
// Test 5: Multiple fields at different offsets
// ===----------------------------------------------------------------------===

TEST_F(StateFieldAccessAnalysisTest, MultipleFields) {
  auto M = std::make_unique<llvm::Module>("test", Ctx);
  auto *F = createLifted(*M);
  llvm::IRBuilder<> B(&F->getEntryBlock());

  auto *state_ptr = F->getArg(0);
  auto *i8_ty = llvm::Type::getInt8Ty(Ctx);
  auto *i64_ty = llvm::Type::getInt64Ty(Ctx);
  auto *i32_ty = llvm::Type::getInt32Ty(Ctx);

  // Load from offset 0 (8 bytes)
  auto *gep0 = B.CreateGEP(i8_ty, state_ptr, B.getInt64(0));
  B.CreateLoad(i64_ty, gep0);

  // Store to offset 48 (8 bytes)
  auto *gep48 = B.CreateGEP(i8_ty, state_ptr, B.getInt64(48));
  B.CreateStore(llvm::ConstantInt::get(i64_ty, 0), gep48);

  // Load from offset 100 (4 bytes)
  auto *gep100 = B.CreateGEP(i8_ty, state_ptr, B.getInt64(100));
  B.CreateLoad(i32_ty, gep100);

  B.CreateRet(F->getArg(2));

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  auto result = analyze(*F);

  EXPECT_EQ(result.fields.size(), 3u);
  EXPECT_TRUE(result.fields.count(0u));
  EXPECT_TRUE(result.fields.count(48u));
  EXPECT_TRUE(result.fields.count(100u));

  EXPECT_EQ(result.fields[0u].size, 8u);
  EXPECT_EQ(result.fields[48u].size, 8u);
  EXPECT_EQ(result.fields[100u].size, 4u);

  EXPECT_EQ(result.all_state_loads.size(), 2u);
  EXPECT_EQ(result.all_state_stores.size(), 1u);
}

// ===----------------------------------------------------------------------===
// Test 6: No State access → empty results
// ===----------------------------------------------------------------------===

TEST_F(StateFieldAccessAnalysisTest, NoStateAccess) {
  auto M = std::make_unique<llvm::Module>("test", Ctx);
  auto *F = createLifted(*M);
  llvm::IRBuilder<> B(&F->getEntryBlock());

  // Only use arg(1) and arg(2), never touch arg(0) (State).
  B.CreateRet(F->getArg(2));

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  auto result = analyze(*F);

  EXPECT_FALSE(result.hasAccesses());
  EXPECT_TRUE(result.fields.empty());
  EXPECT_TRUE(result.live_in_offsets.empty());
  EXPECT_TRUE(result.live_out_offsets.empty());
}

// ===----------------------------------------------------------------------===
// Test 7: Declaration (no body) → empty results
// ===----------------------------------------------------------------------===

TEST_F(StateFieldAccessAnalysisTest, DeclarationSkip) {
  auto M = std::make_unique<llvm::Module>("test", Ctx);
  // Create declaration only (no basic blocks).
  auto *F = llvm::Function::Create(
      liftedFnTy(), llvm::Function::ExternalLinkage, "sub_decl", *M);
  ASSERT_TRUE(F->isDeclaration());

  auto result = analyze(*F);

  EXPECT_FALSE(result.hasAccesses());
  EXPECT_TRUE(result.fields.empty());
}

// ===----------------------------------------------------------------------===
// Test 8: Chained GEP resolves to correct total offset
// ===----------------------------------------------------------------------===

TEST_F(StateFieldAccessAnalysisTest, ChainedGEP) {
  auto M = std::make_unique<llvm::Module>("test", Ctx);
  auto *F = createLifted(*M);
  llvm::IRBuilder<> B(&F->getEntryBlock());

  auto *state_ptr = F->getArg(0);
  auto *i8_ty = llvm::Type::getInt8Ty(Ctx);
  auto *i64_ty = llvm::Type::getInt64Ty(Ctx);

  // GEP chain: base+32, then +16 → total offset 48
  auto *gep1 = B.CreateGEP(i8_ty, state_ptr, B.getInt64(32));
  auto *gep2 = B.CreateGEP(i8_ty, gep1, B.getInt64(16));
  B.CreateLoad(i64_ty, gep2);
  B.CreateRet(F->getArg(2));

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  auto result = analyze(*F);

  EXPECT_TRUE(result.hasAccesses());
  ASSERT_EQ(result.fields.count(48u), 1u);
  EXPECT_EQ(result.fields[48u].offset, 48u);
  EXPECT_TRUE(result.fields[48u].is_live_in);
}

// ===----------------------------------------------------------------------===
// Test 9: Non-State pointer GEP is ignored
// ===----------------------------------------------------------------------===

TEST_F(StateFieldAccessAnalysisTest, NonStatePtrIgnored) {
  auto M = std::make_unique<llvm::Module>("test", Ctx);
  auto *F = createLifted(*M);
  llvm::IRBuilder<> B(&F->getEntryBlock());

  // Access arg(2) (Memory*), not arg(0) (State*).
  auto *mem_ptr = F->getArg(2);
  auto *i8_ty = llvm::Type::getInt8Ty(Ctx);
  auto *i64_ty = llvm::Type::getInt64Ty(Ctx);
  auto *gep = B.CreateGEP(i8_ty, mem_ptr, B.getInt64(48));
  B.CreateLoad(i64_ty, gep);
  B.CreateRet(F->getArg(2));

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  auto result = analyze(*F);

  EXPECT_FALSE(result.hasAccesses());
  EXPECT_TRUE(result.fields.empty());
}

}  // namespace
