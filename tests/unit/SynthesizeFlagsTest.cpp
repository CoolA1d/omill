#include "omill/Passes/SynthesizeFlags.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>

#include <gtest/gtest.h>

namespace {

static const char *kDataLayout =
    "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-"
    "f80:128-n8:16:32:64-S128";

class SynthesizeFlagsTest : public ::testing::Test {
 protected:
  llvm::LLVMContext Ctx;

  std::unique_ptr<llvm::Module> createModule() {
    auto M = std::make_unique<llvm::Module>("test", Ctx);
    M->setDataLayout(kDataLayout);
    return M;
  }

  void runPass(llvm::Function *F, bool instcombine = false) {
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
    FPM.addPass(omill::SynthesizeFlagsPass());
    if (instcombine)
      FPM.addPass(llvm::InstCombinePass());
    FPM.run(*F, FAM);
  }

  /// Count icmp instructions with a specific predicate.
  unsigned countICmp(llvm::Function *F, llvm::CmpInst::Predicate pred) {
    unsigned n = 0;
    for (auto &BB : *F)
      for (auto &I : BB)
        if (auto *cmp = llvm::dyn_cast<llvm::ICmpInst>(&I))
          if (cmp->getPredicate() == pred)
            ++n;
    return n;
  }

  /// Count XOR i1 instructions.
  unsigned countXorI1(llvm::Function *F) {
    unsigned n = 0;
    for (auto &BB : *F)
      for (auto &I : BB)
        if (auto *bin = llvm::dyn_cast<llvm::BinaryOperator>(&I))
          if (bin->getOpcode() == llvm::Instruction::Xor &&
              bin->getType()->isIntegerTy(1))
            ++n;
    return n;
  }

  /// Build the remill-style overflow flag formula for subtraction.
  /// OF = (2 == ((sign_lhs ^ sign_rhs) + (sign_lhs ^ sign_res)))
  llvm::Value *buildSubOverflow(llvm::IRBuilder<> &B, llvm::Value *lhs,
                                llvm::Value *rhs, llvm::Value *res) {
    auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
    auto *shift = llvm::ConstantInt::get(i64Ty, 63);
    auto *sign_lhs = B.CreateLShr(lhs, shift, "sign.lhs");
    auto *sign_rhs = B.CreateLShr(rhs, shift, "sign.rhs");
    auto *sign_res = B.CreateLShr(res, shift, "sign.res");
    auto *xor1 = B.CreateXor(sign_lhs, sign_rhs, "xor1");
    auto *xor2 = B.CreateXor(sign_lhs, sign_res, "xor2");
    auto *sum = B.CreateAdd(xor1, xor2, "of.sum");
    return B.CreateICmpEQ(sum, llvm::ConstantInt::get(i64Ty, 2), "of");
  }

  /// Create a function with the JL pattern: xor(SF, OF) after SUB.
  /// Returns a pair of (function, xor_instruction).
  llvm::Function *createJLFunction(llvm::Module &M, const char *name) {
    auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
    auto *i1Ty = llvm::Type::getInt1Ty(Ctx);
    auto *fnTy = llvm::FunctionType::get(i1Ty, {i64Ty, i64Ty}, false);
    auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                     name, &M);
    auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
    llvm::IRBuilder<> B(entry);

    auto *lhs = F->getArg(0);
    auto *rhs = F->getArg(1);
    auto *res = B.CreateSub(lhs, rhs, "res");

    // SF = icmp slt res, 0
    auto *sf = B.CreateICmpSLT(res, llvm::ConstantInt::get(i64Ty, 0), "sf");

    // OF = overflow formula
    auto *of = buildSubOverflow(B, lhs, rhs, res);

    // JL condition = xor(SF, OF)
    auto *jl = B.CreateXor(sf, of, "jl");

    B.CreateRet(jl);
    return F;
  }
};

/// Basic test: xor(SF, OF) → icmp slt
TEST_F(SynthesizeFlagsTest, XorSfOfBecomesIcmpSlt) {
  auto M = createModule();
  auto *F = createJLFunction(*M, "test_jl");

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Before: should have xor i1 and NO icmp slt on lhs, rhs directly.
  EXPECT_GE(countXorI1(F), 1u);

  runPass(F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // After: xor(SF, OF) should be replaced with icmp slt.
  EXPECT_EQ(countXorI1(F), 0u);
  EXPECT_GE(countICmp(F, llvm::CmpInst::ICMP_SLT), 1u);
}

/// Reversed operand ordering: xor(OF, SF) should also be matched.
TEST_F(SynthesizeFlagsTest, ReversedOperandOrder) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *i1Ty = llvm::Type::getInt1Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i1Ty, {i64Ty, i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test_reversed", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);

  auto *lhs = F->getArg(0);
  auto *rhs = F->getArg(1);
  auto *res = B.CreateSub(lhs, rhs, "res");
  auto *sf = B.CreateICmpSLT(res, llvm::ConstantInt::get(i64Ty, 0), "sf");
  auto *of = buildSubOverflow(B, lhs, rhs, res);

  // Note: OF first, SF second (reversed order).
  auto *jl = B.CreateXor(of, sf, "jl");
  B.CreateRet(jl);

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countXorI1(F), 0u);
  EXPECT_GE(countICmp(F, llvm::CmpInst::ICMP_SLT), 1u);
}

/// XOR i1 that does NOT involve sign flag should be left alone.
TEST_F(SynthesizeFlagsTest, NonFlagXorUnchanged) {
  auto M = createModule();
  auto *i1Ty = llvm::Type::getInt1Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i1Ty, {i1Ty, i1Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test_nonflag", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);

  // Plain xor of two function args — not a flag pattern.
  auto *x = B.CreateXor(F->getArg(0), F->getArg(1), "plain_xor");
  B.CreateRet(x);

  runPass(F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  // The xor should remain.
  EXPECT_EQ(countXorI1(F), 1u);
}

/// No xor instructions at all — pass is a no-op.
TEST_F(SynthesizeFlagsTest, NoXorIsNoOp) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i64Ty, {i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test_noop", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);
  B.CreateRet(F->getArg(0));

  unsigned blocks_before = 0;
  for (auto &BB : *F)
    ++blocks_before;

  runPass(F);

  unsigned blocks_after = 0;
  for (auto &BB : *F)
    ++blocks_after;

  EXPECT_EQ(blocks_before, blocks_after);
}

/// JGE pattern: not(xor(SF, OF)) → icmp sge after SynthesizeFlags + InstCombine.
TEST_F(SynthesizeFlagsTest, JGEWithInstCombine) {
  auto M = createModule();
  auto *i64Ty = llvm::Type::getInt64Ty(Ctx);
  auto *i1Ty = llvm::Type::getInt1Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i1Ty, {i64Ty, i64Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test_jge", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);

  auto *lhs = F->getArg(0);
  auto *rhs = F->getArg(1);
  auto *res = B.CreateSub(lhs, rhs, "res");
  auto *sf = B.CreateICmpSLT(res, llvm::ConstantInt::get(i64Ty, 0), "sf");
  auto *of = buildSubOverflow(B, lhs, rhs, res);

  // JGE = not(xor(SF, OF)) = SF == OF
  auto *jl = B.CreateXor(sf, of, "sf_ne_of");
  auto *jge = B.CreateXor(jl, llvm::ConstantInt::getTrue(Ctx), "jge");
  B.CreateRet(jge);

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  // Run with InstCombine to fold not(icmp slt) → icmp sge.
  runPass(F, /*instcombine=*/true);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  // After: should have icmp sge (InstCombine folds not(slt) → sge).
  EXPECT_GE(countICmp(F, llvm::CmpInst::ICMP_SGE), 1u);
}

/// 32-bit operands: pattern should work with different integer widths.
TEST_F(SynthesizeFlagsTest, Width32) {
  auto M = createModule();
  auto *i32Ty = llvm::Type::getInt32Ty(Ctx);
  auto *i1Ty = llvm::Type::getInt1Ty(Ctx);
  auto *fnTy = llvm::FunctionType::get(i1Ty, {i32Ty, i32Ty}, false);
  auto *F = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage,
                                   "test_32", *M);
  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", F);
  llvm::IRBuilder<> B(entry);

  auto *lhs = F->getArg(0);
  auto *rhs = F->getArg(1);
  auto *res = B.CreateSub(lhs, rhs, "res");
  auto *sf = B.CreateICmpSLT(res, llvm::ConstantInt::get(i32Ty, 0), "sf");

  // OF for 32-bit sub
  auto *shift = llvm::ConstantInt::get(i32Ty, 31);
  auto *sign_lhs = B.CreateLShr(lhs, shift);
  auto *sign_rhs = B.CreateLShr(rhs, shift);
  auto *sign_res = B.CreateLShr(res, shift);
  auto *xor1 = B.CreateXor(sign_lhs, sign_rhs);
  auto *xor2 = B.CreateXor(sign_lhs, sign_res);
  auto *sum = B.CreateAdd(xor1, xor2);
  auto *of = B.CreateICmpEQ(sum, llvm::ConstantInt::get(i32Ty, 2), "of");

  auto *jl = B.CreateXor(sf, of, "jl");
  B.CreateRet(jl);

  ASSERT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));

  runPass(F);

  EXPECT_FALSE(llvm::verifyFunction(*F, &llvm::errs()));
  EXPECT_EQ(countXorI1(F), 0u);
  EXPECT_GE(countICmp(F, llvm::CmpInst::ICMP_SLT), 1u);
}

}  // namespace
