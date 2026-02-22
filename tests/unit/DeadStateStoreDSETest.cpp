#include "omill/Passes/DeadStateStoreDSE.h"

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

class DeadStateStoreDSETest : public ::testing::Test {
 protected:
  llvm::LLVMContext Ctx;

  /// Build a module with struct.State, __remill_basic_block, and a test
  /// _native function with inlined State alloca.
  struct TestModule {
    std::unique_ptr<llvm::Module> M;
    llvm::StructType *state_ty = nullptr;
    llvm::Function *test_fn = nullptr;
    llvm::AllocaInst *state_alloca = nullptr;
    llvm::IRBuilder<> *builder = nullptr;  // set after entry BB is created

    // Register offsets (simplified, matching __remill_basic_block GEPs).
    static constexpr unsigned kRAX = 0;
    static constexpr unsigned kRBX = 8;
    static constexpr unsigned kRCX = 16;
    static constexpr unsigned kRDX = 24;
    static constexpr unsigned kRSI = 32;
    static constexpr unsigned kRDI = 40;
    static constexpr unsigned kRSP = 48;
    static constexpr unsigned kRBP = 56;
    static constexpr unsigned kR8 = 64;
    static constexpr unsigned kR9 = 72;
    static constexpr unsigned kR10 = 80;
    static constexpr unsigned kR11 = 88;
  };

  TestModule createTestModule() {
    TestModule tm;
    tm.M = std::make_unique<llvm::Module>("test", Ctx);
    tm.M->setDataLayout(
        "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-"
        "n8:16:32:64-S128");

    auto *i64_ty = llvm::Type::getInt64Ty(Ctx);
    auto *i8_ty = llvm::Type::getInt8Ty(Ctx);
    auto *ptr_ty = llvm::PointerType::get(Ctx, 0);

    // Create struct.State (3504 bytes).
    tm.state_ty = llvm::StructType::create(Ctx, "struct.State");
    auto *arr_ty = llvm::ArrayType::get(i8_ty, 3504);
    tm.state_ty->setBody({arr_ty});

    // __remill_basic_block with named register GEPs.
    auto *bb_fn_ty =
        llvm::FunctionType::get(ptr_ty, {ptr_ty, i64_ty, ptr_ty}, false);
    auto *bb_fn = llvm::Function::Create(
        bb_fn_ty, llvm::Function::ExternalLinkage, "__remill_basic_block",
        *tm.M);
    auto *bb_entry = llvm::BasicBlock::Create(Ctx, "entry", bb_fn);
    llvm::IRBuilder<> BBB(bb_entry);
    auto *bb_state = bb_fn->getArg(0);

    struct RegDef {
      const char *name;
      unsigned offset;
    };
    RegDef regs[] = {
        {"RAX", 0},   {"RBX", 8},  {"RCX", 16},  {"RDX", 24},
        {"RSI", 32},  {"RDI", 40}, {"RSP", 48},   {"RBP", 56},
        {"R8", 64},   {"R9", 72},  {"R10", 80},   {"R11", 88},
        {"R12", 96},  {"R13", 104}, {"R14", 112}, {"R15", 120},
        {"RIP", 128},
    };
    for (auto &reg : regs) {
      auto *gep = BBB.CreateGEP(BBB.getInt64Ty(), bb_state,
                                BBB.getInt64(reg.offset / 8));
      gep->setName(reg.name);
    }
    BBB.CreateRet(bb_fn->getArg(2));

    // Test _native function with State alloca.
    auto *fn_ty = llvm::FunctionType::get(i64_ty, {i64_ty, i64_ty}, false);
    tm.test_fn = llvm::Function::Create(
        fn_ty, llvm::Function::ExternalLinkage, "sub_401000_native", *tm.M);

    auto *entry = llvm::BasicBlock::Create(Ctx, "entry", tm.test_fn);
    llvm::IRBuilder<> B(entry);

    // Alloca for State (simulating post-inlining).
    tm.state_alloca = B.CreateAlloca(tm.state_ty, nullptr, "state");
    B.CreateMemSet(tm.state_alloca, B.getInt8(0), 3504, llvm::MaybeAlign(16));

    return tm;
  }

  /// Helper: create a GEP into the State alloca at a byte offset.
  llvm::Value *stateGEP(llvm::IRBuilder<> &B, llvm::AllocaInst *state,
                         unsigned offset) {
    return B.CreateConstGEP1_64(B.getInt8Ty(), state, offset);
  }

  /// Helper: count stores in a function.
  unsigned countStores(llvm::Function *F) {
    unsigned count = 0;
    for (auto &BB : *F)
      for (auto &I : BB)
        if (llvm::isa<llvm::StoreInst>(&I))
          ++count;
    return count;
  }

  /// Run the pass on a function.
  void runPass(llvm::Function *F) {
    llvm::FunctionPassManager FPM;
    FPM.addPass(omill::DeadStateStoreDSEPass());
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
    FPM.run(*F, FAM);
  }
};

// ===----------------------------------------------------------------------===
// Test 1: Store to volatile field followed by non-State call → dead
// ===----------------------------------------------------------------------===

TEST_F(DeadStateStoreDSETest, VolatileStoreBeforeNativeCall_Eliminated) {
  auto tm = createTestModule();
  auto *F = tm.test_fn;
  auto *state = tm.state_alloca;
  auto &entry = F->getEntryBlock();
  llvm::IRBuilder<> B(&entry);
  // Insert at end of entry block (after memset, no terminator yet).
  B.SetInsertPoint(&entry, entry.end());

  auto *i64_ty = B.getInt64Ty();

  // Store to RAX (volatile) — should be dead.
  B.CreateStore(B.getInt64(42), stateGEP(B, state, TestModule::kRAX));
  // Store to R10 (volatile) — should be dead.
  B.CreateStore(B.getInt64(99), stateGEP(B, state, TestModule::kR10));

  // Call a native function that doesn't take State.
  auto *callee_ty = llvm::FunctionType::get(i64_ty, {i64_ty}, false);
  auto *callee = llvm::Function::Create(
      callee_ty, llvm::Function::ExternalLinkage, "other_native", *tm.M);
  B.CreateCall(callee, {B.getInt64(1)});

  B.CreateRet(B.getInt64(0));

  ASSERT_FALSE(llvm::verifyModule(*tm.M, &llvm::errs()));
  // memset is a call, not a StoreInst.  Only 2 volatile stores.
  unsigned stores_before = countStores(F);
  EXPECT_EQ(stores_before, 2u);

  runPass(F);

  ASSERT_FALSE(llvm::verifyModule(*tm.M, &llvm::errs()));
  // Both volatile stores should be eliminated.
  unsigned stores_after = countStores(F);
  EXPECT_EQ(stores_after, stores_before - 2);
}

// ===----------------------------------------------------------------------===
// Test 2: Store to callee-saved field before call → NOT dead
// ===----------------------------------------------------------------------===

TEST_F(DeadStateStoreDSETest, CalleeSavedStoreBeforeCall_Preserved) {
  auto tm = createTestModule();
  auto *F = tm.test_fn;
  auto *state = tm.state_alloca;
  auto &entry = F->getEntryBlock();
  llvm::IRBuilder<> B(&entry);
  // Insert at end of entry block (after memset, no terminator yet).
  B.SetInsertPoint(&entry, entry.end());

  auto *i64_ty = B.getInt64Ty();

  // Store to RBX (callee-saved) — should NOT be eliminated.
  B.CreateStore(B.getInt64(42), stateGEP(B, state, TestModule::kRBX));

  auto *callee_ty = llvm::FunctionType::get(i64_ty, {i64_ty}, false);
  auto *callee = llvm::Function::Create(
      callee_ty, llvm::Function::ExternalLinkage, "other_native", *tm.M);
  B.CreateCall(callee, {B.getInt64(1)});

  B.CreateRet(B.getInt64(0));

  ASSERT_FALSE(llvm::verifyModule(*tm.M, &llvm::errs()));
  unsigned stores_before = countStores(F);

  runPass(F);

  ASSERT_FALSE(llvm::verifyModule(*tm.M, &llvm::errs()));
  // RBX store should be preserved.
  EXPECT_EQ(countStores(F), stores_before);
}

// ===----------------------------------------------------------------------===
// Test 3: Store to volatile field that IS read before call → NOT dead
// ===----------------------------------------------------------------------===

TEST_F(DeadStateStoreDSETest, VolatileStoreReadBeforeCall_Preserved) {
  auto tm = createTestModule();
  auto *F = tm.test_fn;
  auto *state = tm.state_alloca;
  auto &entry = F->getEntryBlock();
  llvm::IRBuilder<> B(&entry);
  // Insert at end of entry block (after memset, no terminator yet).
  B.SetInsertPoint(&entry, entry.end());

  auto *i64_ty = B.getInt64Ty();

  // Store to RAX (volatile).
  B.CreateStore(B.getInt64(42), stateGEP(B, state, TestModule::kRAX));
  // Read RAX before the call — store is needed.
  auto *val = B.CreateLoad(i64_ty, stateGEP(B, state, TestModule::kRAX));

  auto *callee_ty = llvm::FunctionType::get(i64_ty, {i64_ty}, false);
  auto *callee = llvm::Function::Create(
      callee_ty, llvm::Function::ExternalLinkage, "other_native", *tm.M);
  B.CreateCall(callee, {val});

  B.CreateRet(B.getInt64(0));

  ASSERT_FALSE(llvm::verifyModule(*tm.M, &llvm::errs()));
  unsigned stores_before = countStores(F);

  runPass(F);

  ASSERT_FALSE(llvm::verifyModule(*tm.M, &llvm::errs()));
  // RAX store should be preserved (it's read).
  EXPECT_EQ(countStores(F), stores_before);
}

// ===----------------------------------------------------------------------===
// Test 4: Store-store to same volatile field → first store dead
// ===----------------------------------------------------------------------===

TEST_F(DeadStateStoreDSETest, StoreStoreToSameVolatile_FirstEliminated) {
  auto tm = createTestModule();
  auto *F = tm.test_fn;
  auto *state = tm.state_alloca;
  auto &entry = F->getEntryBlock();
  llvm::IRBuilder<> B(&entry);
  // Insert at end of entry block (after memset, no terminator yet).
  B.SetInsertPoint(&entry, entry.end());

  // Two stores to RAX without read between them.
  B.CreateStore(B.getInt64(1), stateGEP(B, state, TestModule::kRAX));
  B.CreateStore(B.getInt64(2), stateGEP(B, state, TestModule::kRAX));

  B.CreateRet(B.getInt64(0));

  ASSERT_FALSE(llvm::verifyModule(*tm.M, &llvm::errs()));
  unsigned stores_before = countStores(F);

  runPass(F);

  ASSERT_FALSE(llvm::verifyModule(*tm.M, &llvm::errs()));
  // First RAX store is dead (overwritten), second is dead (at return).
  // Both eliminated.
  EXPECT_EQ(countStores(F), stores_before - 2);
}

// ===----------------------------------------------------------------------===
// Test 5: Volatile store before return → dead
// ===----------------------------------------------------------------------===

TEST_F(DeadStateStoreDSETest, VolatileStoreBeforeReturn_Eliminated) {
  auto tm = createTestModule();
  auto *F = tm.test_fn;
  auto *state = tm.state_alloca;
  auto &entry = F->getEntryBlock();
  llvm::IRBuilder<> B(&entry);
  // Insert at end of entry block (after memset, no terminator yet).
  B.SetInsertPoint(&entry, entry.end());

  // Store to R11 (volatile) right before return — dead.
  B.CreateStore(B.getInt64(42), stateGEP(B, state, TestModule::kR11));
  B.CreateRet(B.getInt64(0));

  ASSERT_FALSE(llvm::verifyModule(*tm.M, &llvm::errs()));
  unsigned stores_before = countStores(F);

  runPass(F);

  ASSERT_FALSE(llvm::verifyModule(*tm.M, &llvm::errs()));
  EXPECT_EQ(countStores(F), stores_before - 1);
}

// ===----------------------------------------------------------------------===
// Test 6: State-escaping call → stores NOT eliminated
// ===----------------------------------------------------------------------===

TEST_F(DeadStateStoreDSETest, StateEscapingCall_StoresPreserved) {
  auto tm = createTestModule();
  auto *F = tm.test_fn;
  auto *state = tm.state_alloca;
  auto &entry = F->getEntryBlock();
  llvm::IRBuilder<> B(&entry);
  // Insert at end of entry block (after memset, no terminator yet).
  B.SetInsertPoint(&entry, entry.end());

  auto *i64_ty = B.getInt64Ty();
  auto *ptr_ty = llvm::PointerType::get(Ctx, 0);

  // Store to RAX (volatile).
  B.CreateStore(B.getInt64(42), stateGEP(B, state, TestModule::kRAX));

  // Call that takes State* as argument → State escapes.
  auto *callee_ty = llvm::FunctionType::get(ptr_ty, {ptr_ty, i64_ty, ptr_ty}, false);
  auto *callee = llvm::Function::Create(
      callee_ty, llvm::Function::ExternalLinkage, "__omill_dispatch_call", *tm.M);
  B.CreateCall(callee, {state, B.getInt64(0x402000),
                        llvm::Constant::getNullValue(ptr_ty)});

  B.CreateRet(B.getInt64(0));

  ASSERT_FALSE(llvm::verifyModule(*tm.M, &llvm::errs()));
  unsigned stores_before = countStores(F);

  runPass(F);

  ASSERT_FALSE(llvm::verifyModule(*tm.M, &llvm::errs()));
  // Store should be preserved — call reads State.
  EXPECT_EQ(countStores(F), stores_before);
}

// ===----------------------------------------------------------------------===
// Test 7: Non-_native function → pass is no-op
// ===----------------------------------------------------------------------===

TEST_F(DeadStateStoreDSETest, NonNativeFunction_Skipped) {
  auto M = std::make_unique<llvm::Module>("test", Ctx);
  M->setDataLayout(
      "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-"
      "n8:16:32:64-S128");

  auto *i64_ty = llvm::Type::getInt64Ty(Ctx);
  auto *fn_ty = llvm::FunctionType::get(i64_ty, {i64_ty}, false);
  auto *F = llvm::Function::Create(
      fn_ty, llvm::Function::ExternalLinkage, "sub_401000", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  B.CreateRet(B.getInt64(0));

  ASSERT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
  unsigned stores_before = countStores(F);

  runPass(F);

  // No changes — function doesn't end with _native.
  EXPECT_EQ(countStores(F), stores_before);
}

// ===----------------------------------------------------------------------===
// Test 8: Multiple volatile stores before call — all eliminated
// ===----------------------------------------------------------------------===

TEST_F(DeadStateStoreDSETest, MultipleVolatileStoresBeforeCall_AllEliminated) {
  auto tm = createTestModule();
  auto *F = tm.test_fn;
  auto *state = tm.state_alloca;
  auto &entry = F->getEntryBlock();
  llvm::IRBuilder<> B(&entry);
  // Insert at end of entry block (after memset, no terminator yet).
  B.SetInsertPoint(&entry, entry.end());

  auto *i64_ty = B.getInt64Ty();

  // Store to RAX, RCX, RDX, R8, R9, R10, R11 (all volatile).
  B.CreateStore(B.getInt64(1), stateGEP(B, state, TestModule::kRAX));
  B.CreateStore(B.getInt64(2), stateGEP(B, state, TestModule::kRCX));
  B.CreateStore(B.getInt64(3), stateGEP(B, state, TestModule::kRDX));
  B.CreateStore(B.getInt64(4), stateGEP(B, state, TestModule::kR8));
  B.CreateStore(B.getInt64(5), stateGEP(B, state, TestModule::kR9));
  B.CreateStore(B.getInt64(6), stateGEP(B, state, TestModule::kR10));
  B.CreateStore(B.getInt64(7), stateGEP(B, state, TestModule::kR11));

  // Non-State call.
  auto *callee_ty = llvm::FunctionType::get(i64_ty, {i64_ty}, false);
  auto *callee = llvm::Function::Create(
      callee_ty, llvm::Function::ExternalLinkage, "foo_native", *tm.M);
  B.CreateCall(callee, {B.getInt64(0)});

  B.CreateRet(B.getInt64(0));

  ASSERT_FALSE(llvm::verifyModule(*tm.M, &llvm::errs()));
  unsigned stores_before = countStores(F);

  runPass(F);

  ASSERT_FALSE(llvm::verifyModule(*tm.M, &llvm::errs()));
  // All 7 volatile stores should be eliminated.
  EXPECT_EQ(countStores(F), stores_before - 7);
}

}  // namespace
