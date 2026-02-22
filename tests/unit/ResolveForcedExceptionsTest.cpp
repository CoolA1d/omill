#include "omill/Passes/ResolveForcedExceptions.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>

#include "omill/Analysis/ExceptionInfo.h"
#include "omill/Analysis/LiftedFunctionMap.h"

#include <gtest/gtest.h>
#include <optional>

namespace {

class ResolveForcedExceptionsTest : public ::testing::Test {
 protected:
  llvm::LLVMContext Ctx;

  llvm::FunctionType *liftedFnTy() {
    auto *ptr_ty = llvm::PointerType::get(Ctx, 0);
    auto *i64_ty = llvm::Type::getInt64Ty(Ctx);
    return llvm::FunctionType::get(ptr_ty, {ptr_ty, i64_ty, ptr_ty}, false);
  }

  struct TestSetup {
    std::unique_ptr<llvm::Module> M;
    llvm::Function *test_fn;
    llvm::Function *handler_fn;
    llvm::Function *error_fn;
  };

  TestSetup createTestModule(bool with_exception_info = true,
                              unsigned num_error_calls = 1) {
    auto M = std::make_unique<llvm::Module>("test", Ctx);
    M->setDataLayout(
        "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-"
        "n8:16:32:64-S128");

    // Create __remill_error declaration.
    auto *error_fn = llvm::Function::Create(
        liftedFnTy(), llvm::Function::ExternalLinkage, "__remill_error", *M);

    // Create the handler function (lifted).
    auto *handler_fn = llvm::Function::Create(
        liftedFnTy(), llvm::Function::ExternalLinkage, "sub_403000", *M);
    {
      auto *entry = llvm::BasicBlock::Create(Ctx, "entry", handler_fn);
      llvm::IRBuilder<> B(entry);
      B.CreateRet(handler_fn->getArg(2));
    }

    // Create test function sub_401000 with __remill_error call(s).
    auto *test_fn = llvm::Function::Create(
        liftedFnTy(), llvm::Function::ExternalLinkage, "sub_401000", *M);

    auto *entry = llvm::BasicBlock::Create(Ctx, "entry", test_fn);
    llvm::IRBuilder<> B(entry);

    for (unsigned i = 0; i < num_error_calls; ++i) {
      auto *result = B.CreateCall(error_fn,
          {test_fn->getArg(0), B.getInt64(0x401050 + i * 0x10),
           test_fn->getArg(2)});
      if (i == num_error_calls - 1) {
        B.CreateRet(result);
      }
    }

    return {std::move(M), test_fn, handler_fn, error_fn};
  }

  void runPassWithExceptionInfo(llvm::Function &F, bool with_info) {
    auto &M = *F.getParent();

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

    MAM.registerPass([&] { return omill::LiftedFunctionAnalysis(); });

    if (with_info) {
      omill::ExceptionInfo excinfo;
      excinfo.addEntry({
          /*begin_va=*/0x401000,
          /*end_va=*/0x402000,
          /*handler_va=*/0x403000,
          /*handler_data_va=*/0,
          /*dc_synthetic_va=*/0x500000,
          /*ctx_synthetic_va=*/0x600000,
      });

      MAM.registerPass(
          [&] { return omill::ExceptionInfoAnalysis(std::move(excinfo)); });
    } else {
      MAM.registerPass([&] { return omill::ExceptionInfoAnalysis(); });
    }

    // Pre-compute module analyses so getCachedResult() finds them.
    MAM.getResult<omill::ExceptionInfoAnalysis>(M);
    MAM.getResult<omill::LiftedFunctionAnalysis>(M);

    llvm::FunctionPassManager FPM;
    FPM.addPass(omill::ResolveForcedExceptionsPass());
    FPM.run(F, FAM);
  }
};

TEST_F(ResolveForcedExceptionsTest, ErrorCallReplacedWithHandler) {
  auto [M, test_fn, handler_fn, error_fn] = createTestModule(true, 1);

  // Before: should have __remill_error call.
  bool has_error_call = false;
  for (auto &BB : *test_fn)
    for (auto &I : BB)
      if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I))
        if (CI->getCalledFunction() == error_fn)
          has_error_call = true;
  ASSERT_TRUE(has_error_call);

  runPassWithExceptionInfo(*test_fn, true);

  // After: error call replaced with handler call.
  bool has_error_after = false;
  bool calls_handler = false;
  for (auto &BB : *test_fn)
    for (auto &I : BB) {
      if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I)) {
        if (CI->getCalledFunction() == error_fn)
          has_error_after = true;
        if (CI->getCalledFunction() == handler_fn)
          calls_handler = true;
      }
    }

  EXPECT_FALSE(has_error_after);
  EXPECT_TRUE(calls_handler);
  EXPECT_FALSE(llvm::verifyFunction(*test_fn, &llvm::errs()));
}

TEST_F(ResolveForcedExceptionsTest, HandlerMarkedInline) {
  auto [M, test_fn, handler_fn, error_fn] = createTestModule(true, 1);

  EXPECT_EQ(handler_fn->getLinkage(), llvm::GlobalValue::ExternalLinkage);
  EXPECT_FALSE(handler_fn->hasFnAttribute(llvm::Attribute::AlwaysInline));

  runPassWithExceptionInfo(*test_fn, true);

  EXPECT_TRUE(handler_fn->hasFnAttribute(llvm::Attribute::AlwaysInline));
  EXPECT_EQ(handler_fn->getLinkage(), llvm::GlobalValue::InternalLinkage);
}

TEST_F(ResolveForcedExceptionsTest, NoExceptionInfoSkipped) {
  auto [M, test_fn, handler_fn, error_fn] = createTestModule(false, 1);

  // Run without exception info — function should be unchanged.
  unsigned call_count_before = 0;
  for (auto &BB : *test_fn)
    for (auto &I : BB)
      if (llvm::isa<llvm::CallInst>(&I))
        call_count_before++;

  runPassWithExceptionInfo(*test_fn, false);

  unsigned call_count_after = 0;
  for (auto &BB : *test_fn)
    for (auto &I : BB)
      if (llvm::isa<llvm::CallInst>(&I))
        call_count_after++;

  EXPECT_EQ(call_count_before, call_count_after);
}

TEST_F(ResolveForcedExceptionsTest, MultipleErrorCallsResolved) {
  auto [M, test_fn, handler_fn, error_fn] = createTestModule(true, 2);

  // Count error calls before.
  unsigned error_count = 0;
  for (auto &BB : *test_fn)
    for (auto &I : BB)
      if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I))
        if (CI->getCalledFunction() == error_fn)
          error_count++;
  ASSERT_EQ(error_count, 2u);

  runPassWithExceptionInfo(*test_fn, true);

  // All error calls should be replaced.
  unsigned error_count_after = 0;
  unsigned handler_count = 0;
  for (auto &BB : *test_fn)
    for (auto &I : BB)
      if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I)) {
        if (CI->getCalledFunction() == error_fn)
          error_count_after++;
        if (CI->getCalledFunction() == handler_fn)
          handler_count++;
      }

  EXPECT_EQ(error_count_after, 0u);
  // At least one handler call (the second error call may be erased as dead code
  // after the first error→ret transformation).
  EXPECT_GE(handler_count, 1u);
}

TEST_F(ResolveForcedExceptionsTest, SeedsExpectedStateRegistersOnly) {
  auto [M, test_fn, handler_fn, error_fn] = createTestModule(true, 1);

  runPassWithExceptionInfo(*test_fn, true);

  auto getStateOffset = [](llvm::Value *ptr) -> std::optional<uint64_t> {
    auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(ptr);
    if (!gep || gep->getNumIndices() != 1)
      return std::nullopt;
    auto *idx = llvm::dyn_cast<llvm::ConstantInt>(gep->getOperand(1));
    if (!idx)
      return std::nullopt;
    return idx->getZExtValue();
  };

  // Win64 State byte offsets used by the pass.
  static constexpr uint64_t kRCXOffset = 2248;
  static constexpr uint64_t kRDXOffset = 2264;
  static constexpr uint64_t kR8Offset = 2344;
  static constexpr uint64_t kR9Offset = 2360;

  bool saw_rdx_seed = false;
  bool saw_r8_seed = false;
  bool saw_r9_seed = false;
  bool saw_rcx_clobber = false;

  for (auto &BB : *test_fn) {
    for (auto &I : BB) {
      auto *SI = llvm::dyn_cast<llvm::StoreInst>(&I);
      if (!SI)
        continue;
      auto off = getStateOffset(SI->getPointerOperand());
      if (!off.has_value())
        continue;

      auto *val_ci = llvm::dyn_cast<llvm::ConstantInt>(SI->getValueOperand());
      if (!val_ci)
        continue;
      uint64_t v = val_ci->getZExtValue();

      if (*off == kRDXOffset && v == 0x401000)
        saw_rdx_seed = true;      // begin_va
      if (*off == kR9Offset && v == 0x500000)
        saw_r9_seed = true;       // synthetic DC in test setup
      if (*off == kRCXOffset)
        saw_rcx_clobber = true;
      if (*off == kR8Offset && v == 0x600000)
        saw_r8_seed = true;       // ctx_synthetic_va
    }
  }

  EXPECT_TRUE(saw_rdx_seed);
  EXPECT_TRUE(saw_r8_seed);
  EXPECT_TRUE(saw_r9_seed);
  EXPECT_FALSE(saw_rcx_clobber);
}

}  // namespace
