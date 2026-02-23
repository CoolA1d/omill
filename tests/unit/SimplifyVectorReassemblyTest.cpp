#include "omill/Passes/SimplifyVectorReassembly.h"

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

class SimplifyVectorReassemblyTest : public ::testing::Test {
 protected:
  llvm::LLVMContext Ctx;

  static constexpr const char *kDataLayout =
      "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-"
      "n8:16:32:64-S128";

  void runPass(llvm::Function &F) {
    llvm::FunctionPassManager FPM;
    FPM.addPass(omill::SimplifyVectorReassemblyPass());

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

    FPM.run(F, FAM);
  }

  std::unique_ptr<llvm::Module> makeModule() {
    auto M = std::make_unique<llvm::Module>("test", Ctx);
    M->setDataLayout(kDataLayout);
    return M;
  }
};

// 1. insertelement chain from poison with all constants → ConstantVector.
TEST_F(SimplifyVectorReassemblyTest, ConstantVectorChain_Folded) {
  auto M = makeModule();

  auto *i32_ty = llvm::Type::getInt32Ty(Ctx);
  auto *v4i32_ty = llvm::FixedVectorType::get(i32_ty, 4);

  auto *fn_ty = llvm::FunctionType::get(v4i32_ty, {}, false);
  auto *test_fn = llvm::Function::Create(
      fn_ty, llvm::Function::ExternalLinkage, "test_func", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", test_fn);
  llvm::IRBuilder<> B(entry);

  // Start with poison, insert constants at indices 0,1,2,3.
  llvm::Value *vec = llvm::PoisonValue::get(v4i32_ty);
  vec = B.CreateInsertElement(vec, llvm::ConstantInt::get(i32_ty, 10),
                              B.getInt64(0), "ins0");
  vec = B.CreateInsertElement(vec, llvm::ConstantInt::get(i32_ty, 20),
                              B.getInt64(1), "ins1");
  vec = B.CreateInsertElement(vec, llvm::ConstantInt::get(i32_ty, 30),
                              B.getInt64(2), "ins2");
  vec = B.CreateInsertElement(vec, llvm::ConstantInt::get(i32_ty, 40),
                              B.getInt64(3), "ins3");
  B.CreateRet(vec);

  runPass(*test_fn);

  auto *ret = llvm::dyn_cast<llvm::ReturnInst>(
      test_fn->getEntryBlock().getTerminator());
  ASSERT_NE(ret, nullptr);
  EXPECT_TRUE(llvm::isa<llvm::Constant>(ret->getReturnValue()))
      << "insertelement chain of constants should fold to ConstantVector";

  EXPECT_FALSE(llvm::verifyFunction(*test_fn, &llvm::errs()));
}

// 2. extractelement(shufflevector(a, b, mask), idx) → direct extract.
TEST_F(SimplifyVectorReassemblyTest, ExtractFromShuffle_Simplified) {
  auto M = makeModule();

  auto *i32_ty = llvm::Type::getInt32Ty(Ctx);
  auto *v4i32_ty = llvm::FixedVectorType::get(i32_ty, 4);

  auto *fn_ty =
      llvm::FunctionType::get(i32_ty, {v4i32_ty, v4i32_ty}, false);
  auto *test_fn = llvm::Function::Create(
      fn_ty, llvm::Function::ExternalLinkage, "test_func", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", test_fn);
  llvm::IRBuilder<> B(entry);

  auto *a = test_fn->getArg(0);
  auto *b = test_fn->getArg(1);

  // Blend: {0, 5, 2, 7} means lanes 0,2 from a; lanes 1,3 from b.
  auto *shuffled = B.CreateShuffleVector(a, b, {0, 5, 2, 7}, "shuffled");

  // Extract lane 1 → mask[1]=5 → b[1].
  auto *extracted =
      B.CreateExtractElement(shuffled, B.getInt64(1), "extracted");
  B.CreateRet(extracted);

  runPass(*test_fn);

  // After: should extract directly from b at lane 1.
  bool found_direct_extract = false;
  for (auto &I : test_fn->getEntryBlock()) {
    if (auto *EE = llvm::dyn_cast<llvm::ExtractElementInst>(&I)) {
      if (EE->getVectorOperand() == b) {
        if (auto *ci =
                llvm::dyn_cast<llvm::ConstantInt>(EE->getIndexOperand()))
          if (ci->getZExtValue() == 1)
            found_direct_extract = true;
      }
    }
  }
  EXPECT_TRUE(found_direct_extract)
      << "extractelement through shuffle should resolve to b[1]";

  EXPECT_FALSE(llvm::verifyFunction(*test_fn, &llvm::errs()));
}

// 3. Byte-level OR tree from <16 x i8> → coalesced to wider extract.
TEST_F(SimplifyVectorReassemblyTest, ByteReassembly_Coalesced) {
  auto M = makeModule();

  auto *i64_ty = llvm::Type::getInt64Ty(Ctx);
  auto *i8_ty = llvm::Type::getInt8Ty(Ctx);
  auto *v16i8_ty = llvm::FixedVectorType::get(i8_ty, 16);

  auto *fn_ty = llvm::FunctionType::get(i64_ty, {v16i8_ty}, false);
  auto *test_fn = llvm::Function::Create(
      fn_ty, llvm::Function::ExternalLinkage, "test_func", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", test_fn);
  llvm::IRBuilder<> B(entry);

  auto *src = test_fn->getArg(0);  // <16 x i8>

  // Build OR tree: extract bytes 0-7, zext to i64, shift to position, OR.
  // byte[0] << 0 | byte[1] << 8 | ... | byte[7] << 56
  llvm::Value *accum = nullptr;
  for (unsigned i = 0; i < 8; ++i) {
    auto *byte_val =
        B.CreateExtractElement(src, B.getInt64(i), "b" + llvm::Twine(i));
    auto *ext = B.CreateZExt(byte_val, i64_ty, "z" + llvm::Twine(i));
    llvm::Value *shifted = ext;
    if (i > 0) {
      shifted = B.CreateShl(ext, B.getInt64(i * 8),
                             "s" + llvm::Twine(i));
    }
    if (!accum) {
      accum = shifted;
    } else {
      accum = B.CreateOr(accum, shifted, "or" + llvm::Twine(i));
      if (auto *BO = llvm::dyn_cast<llvm::PossiblyDisjointInst>(accum))
        BO->setIsDisjoint(true);
    }
  }

  B.CreateRet(accum);

  // Before: should have OR instructions.
  unsigned or_count_before = 0;
  for (auto &I : test_fn->getEntryBlock())
    if (auto *BO = llvm::dyn_cast<llvm::BinaryOperator>(&I))
      if (BO->getOpcode() == llvm::Instruction::Or)
        or_count_before++;
  ASSERT_GT(or_count_before, 0u);

  runPass(*test_fn);

  // The pass should coalesce the root OR into a wide extract.
  // Inner ORs may remain as dead code (no DCE in this pass).
  // Check that the return value is NOT an OR (it should be the coalesced extract),
  // or that at least a wide extractelement<i64> was produced.
  auto *term = test_fn->getEntryBlock().getTerminator();
  auto *ret = llvm::dyn_cast<llvm::ReturnInst>(term);
  ASSERT_NE(ret, nullptr);

  bool root_or_removed = !llvm::isa<llvm::BinaryOperator>(ret->getReturnValue());
  bool has_wide_extract = false;
  for (auto &I : test_fn->getEntryBlock()) {
    if (auto *EE = llvm::dyn_cast<llvm::ExtractElementInst>(&I)) {
      if (auto *VTy = llvm::dyn_cast<llvm::FixedVectorType>(
              EE->getVectorOperand()->getType())) {
        if (VTy->getElementType()->isIntegerTy(64))
          has_wide_extract = true;
      }
    }
  }

  // Either the root OR was replaced, or a wide extract was created.
  EXPECT_TRUE(root_or_removed || has_wide_extract)
      << "Expected root OR coalesced or wide extract created";

  EXPECT_FALSE(llvm::verifyFunction(*test_fn, &llvm::errs()));
}

// 4. insertelement with non-constant values → preserved.
TEST_F(SimplifyVectorReassemblyTest, NonConstantChain_Preserved) {
  auto M = makeModule();

  auto *i32_ty = llvm::Type::getInt32Ty(Ctx);
  auto *v4i32_ty = llvm::FixedVectorType::get(i32_ty, 4);

  auto *fn_ty =
      llvm::FunctionType::get(v4i32_ty, {i32_ty, i32_ty}, false);
  auto *test_fn = llvm::Function::Create(
      fn_ty, llvm::Function::ExternalLinkage, "test_func", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", test_fn);
  llvm::IRBuilder<> B(entry);

  auto *arg0 = test_fn->getArg(0);
  auto *arg1 = test_fn->getArg(1);

  llvm::Value *vec = llvm::PoisonValue::get(v4i32_ty);
  vec = B.CreateInsertElement(vec, arg0, B.getInt64(0), "ins0");
  vec = B.CreateInsertElement(vec, arg1, B.getInt64(1), "ins1");
  vec = B.CreateInsertElement(vec, llvm::ConstantInt::get(i32_ty, 99),
                              B.getInt64(2), "ins2");
  vec = B.CreateInsertElement(vec, arg0, B.getInt64(3), "ins3");
  B.CreateRet(vec);

  // Count insertelements before.
  unsigned insert_count_before = 0;
  for (auto &I : test_fn->getEntryBlock())
    if (llvm::isa<llvm::InsertElementInst>(&I))
      insert_count_before++;

  runPass(*test_fn);

  // Non-constant chain should not be folded to ConstantVector.
  auto *ret = llvm::dyn_cast<llvm::ReturnInst>(
      test_fn->getEntryBlock().getTerminator());
  ASSERT_NE(ret, nullptr);
  EXPECT_FALSE(llvm::isa<llvm::ConstantVector>(ret->getReturnValue()))
      << "Non-constant insertelement chain should not fold";

  EXPECT_FALSE(llvm::verifyFunction(*test_fn, &llvm::errs()));
}

// 5. Declaration (no body) → pass is a no-op.
TEST_F(SimplifyVectorReassemblyTest, DeclarationFunction_Skipped) {
  auto M = makeModule();

  auto *i32_ty = llvm::Type::getInt32Ty(Ctx);
  auto *fn_ty = llvm::FunctionType::get(i32_ty, {}, false);
  auto *test_fn = llvm::Function::Create(
      fn_ty, llvm::Function::ExternalLinkage, "test_decl", *M);

  // No body — just a declaration.
  ASSERT_TRUE(test_fn->isDeclaration());

  // Should not crash.
  runPass(*test_fn);

  EXPECT_TRUE(test_fn->isDeclaration());
}

// 6. Function with just a ret → no change.
TEST_F(SimplifyVectorReassemblyTest, EmptyFunction_NoOp) {
  auto M = makeModule();

  auto *void_ty = llvm::Type::getVoidTy(Ctx);
  auto *fn_ty = llvm::FunctionType::get(void_ty, {}, false);
  auto *test_fn = llvm::Function::Create(
      fn_ty, llvm::Function::ExternalLinkage, "test_empty", *M);

  auto *entry = llvm::BasicBlock::Create(Ctx, "entry", test_fn);
  llvm::IRBuilder<> B(entry);
  B.CreateRetVoid();

  unsigned inst_count_before = 0;
  for (auto &I : test_fn->getEntryBlock()) inst_count_before++;

  runPass(*test_fn);

  unsigned inst_count_after = 0;
  for (auto &I : test_fn->getEntryBlock()) inst_count_after++;

  EXPECT_EQ(inst_count_before, inst_count_after);
  EXPECT_FALSE(llvm::verifyFunction(*test_fn, &llvm::errs()));
}

}  // namespace
