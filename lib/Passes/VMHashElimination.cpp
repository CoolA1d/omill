#include "omill/Passes/VMHashElimination.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm::PatternMatch;

namespace omill {

namespace {

/// Check if an instruction is a murmur-style hash round.
///
/// Pattern:
///   %xor   = xor i64 %input, <K>
///   %mul   = mul i64 %xor, <K>      (same K or different)
///   %inner = lshr i64 %mul, 32
///   %shift = lshr i64 %mul, 60
///   %var   = lshr i64 %inner, %shift
///   %fold  = xor i64 %var, %mul
///   %result = mul i64 %fold, <K>
///
/// We detect this by looking for the signature: lshr(X, 60) where X is a mul.
/// Then verify the full chain.
struct MurmurRound {
  llvm::Instruction *shift60 = nullptr;  // lshr %mul, 60
  llvm::Instruction *mul_input = nullptr;  // the mul instruction
  llvm::Instruction *result_mul = nullptr;  // final mul(fold, K)
};

/// Try to match a murmur hash round starting from an lshr by 60.
static std::optional<MurmurRound> matchMurmurRound(llvm::Instruction *I) {
  // Match: %shift = lshr i64 %mul, 60
  llvm::Value *mul_val = nullptr;
  if (!match(I, m_LShr(m_Value(mul_val), m_SpecificInt(60))))
    return std::nullopt;

  // %mul must be a mul instruction.
  auto *mul_inst = llvm::dyn_cast<llvm::Instruction>(mul_val);
  if (!mul_inst || mul_inst->getOpcode() != llvm::Instruction::Mul)
    return std::nullopt;

  // Look for lshr(%mul, 32) among mul's users.
  llvm::Instruction *inner_shift = nullptr;
  for (auto *user : mul_inst->users()) {
    auto *u_inst = llvm::dyn_cast<llvm::Instruction>(user);
    if (!u_inst)
      continue;
    if (match(u_inst, m_LShr(m_Specific(mul_inst), m_SpecificInt(32)))) {
      inner_shift = u_inst;
      break;
    }
  }
  if (!inner_shift)
    return std::nullopt;

  // Look for the variable shift: lshr(%inner, %shift60) among inner's users.
  llvm::Instruction *var_shift = nullptr;
  for (auto *user : inner_shift->users()) {
    auto *u_inst = llvm::dyn_cast<llvm::Instruction>(user);
    if (!u_inst)
      continue;
    if (match(u_inst, m_LShr(m_Specific(inner_shift), m_Specific(I)))) {
      var_shift = u_inst;
      break;
    }
  }
  if (!var_shift)
    return std::nullopt;

  // Look for xor(%var_shift, %mul) among var_shift's users.
  llvm::Instruction *fold_xor = nullptr;
  for (auto *user : var_shift->users()) {
    auto *u_inst = llvm::dyn_cast<llvm::Instruction>(user);
    if (!u_inst)
      continue;
    if (match(u_inst, m_Xor(m_Specific(var_shift), m_Specific(mul_inst))) ||
        match(u_inst, m_Xor(m_Specific(mul_inst), m_Specific(var_shift)))) {
      fold_xor = u_inst;
      break;
    }
  }
  if (!fold_xor)
    return std::nullopt;

  // Look for the final mul(%fold, <K>) among fold's users.
  llvm::Instruction *result_mul = nullptr;
  for (auto *user : fold_xor->users()) {
    auto *u_inst = llvm::dyn_cast<llvm::Instruction>(user);
    if (!u_inst)
      continue;
    if (u_inst->getOpcode() == llvm::Instruction::Mul) {
      // One operand should be fold_xor, the other a constant.
      if (u_inst->getOperand(0) == fold_xor ||
          u_inst->getOperand(1) == fold_xor) {
        result_mul = u_inst;
        break;
      }
    }
  }
  if (!result_mul)
    return std::nullopt;

  return MurmurRound{I, mul_inst, result_mul};
}

/// Walk forward from hash round results to find the combined integrity flag.
///
/// The rounds are combined: or(or(round1_result, round2_result), ...)
/// Then compared: icmp eq <combined>, <magic>
/// Then: zext i1 to i64
///
/// We look for zext(icmp eq(or-chain, const)) patterns where the or-chain
/// includes at least one murmur round result.
static llvm::SmallVector<llvm::Instruction *, 4>
findIntegrityFlags(llvm::Function &F,
                   const llvm::SmallPtrSetImpl<llvm::Instruction *> &round_results) {
  llvm::SmallVector<llvm::Instruction *, 4> flags;

  // Track all values derived from round results through OR chains.
  llvm::SmallPtrSet<llvm::Value *, 32> derived;
  for (auto *result : round_results)
    derived.insert(result);

  // BFS through OR chains.
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto *val : derived) {
      for (auto *user : val->users()) {
        auto *inst = llvm::dyn_cast<llvm::Instruction>(user);
        if (!inst)
          continue;
        if (inst->getOpcode() == llvm::Instruction::Or ||
            inst->getOpcode() == llvm::Instruction::Xor) {
          if (derived.insert(inst).second)
            changed = true;
        }
      }
    }
  }

  // Look for icmp eq(derived_val, const) → zext patterns.
  for (auto *val : derived) {
    for (auto *user : val->users()) {
      auto *cmp = llvm::dyn_cast<llvm::ICmpInst>(user);
      if (!cmp || cmp->getPredicate() != llvm::ICmpInst::ICMP_EQ)
        continue;
      // One operand should be in the derived set, the other a constant.
      if (!llvm::isa<llvm::ConstantInt>(cmp->getOperand(0)) &&
          !llvm::isa<llvm::ConstantInt>(cmp->getOperand(1)))
        continue;

      // Look for zext of the comparison result.
      for (auto *cmp_user : cmp->users()) {
        auto *zext = llvm::dyn_cast<llvm::ZExtInst>(cmp_user);
        if (zext && zext->getType()->isIntegerTy(64)) {
          flags.push_back(zext);
        }
      }
    }
  }

  return flags;
}

/// Find hash token range checks at handler entry.
///
/// Pattern:
///   %cmp1 = icmp ugt i64 %token, <const>
///   %cmp2 = icmp ult i64 %token, <const>
///   %flag = or i1 %cmp1, %cmp2
///   %r10  = zext i1 %flag to i64
///
/// These produce a 0-or-1 flag used as a branchless multiplier.
static llvm::SmallVector<llvm::Instruction *, 4>
findTokenRangeChecks(llvm::Function &F) {
  llvm::SmallVector<llvm::Instruction *, 4> flags;

  for (auto &BB : F) {
    for (auto &I : BB) {
      auto *zext = llvm::dyn_cast<llvm::ZExtInst>(&I);
      if (!zext || !zext->getType()->isIntegerTy(64))
        continue;
      auto *src = zext->getOperand(0);
      if (!src->getType()->isIntegerTy(1))
        continue;

      // Check if src is an OR of two icmps.
      auto *or_inst = llvm::dyn_cast<llvm::BinaryOperator>(src);
      if (!or_inst || or_inst->getOpcode() != llvm::Instruction::Or)
        continue;

      auto *cmp1 = llvm::dyn_cast<llvm::ICmpInst>(or_inst->getOperand(0));
      auto *cmp2 = llvm::dyn_cast<llvm::ICmpInst>(or_inst->getOperand(1));
      if (!cmp1 || !cmp2)
        continue;

      // Check if one is ugt and the other is ult, comparing the same
      // non-constant value against constants.
      bool is_range_check = false;
      auto pred1 = cmp1->getPredicate();
      auto pred2 = cmp2->getPredicate();

      if (((pred1 == llvm::ICmpInst::ICMP_UGT &&
            pred2 == llvm::ICmpInst::ICMP_ULT) ||
           (pred1 == llvm::ICmpInst::ICMP_ULT &&
            pred2 == llvm::ICmpInst::ICMP_UGT)) &&
          (llvm::isa<llvm::ConstantInt>(cmp1->getOperand(1)) ||
           llvm::isa<llvm::ConstantInt>(cmp1->getOperand(0))) &&
          (llvm::isa<llvm::ConstantInt>(cmp2->getOperand(1)) ||
           llvm::isa<llvm::ConstantInt>(cmp2->getOperand(0)))) {
        is_range_check = true;
      }

      if (is_range_check)
        flags.push_back(zext);
    }
  }

  return flags;
}

}  // namespace

llvm::PreservedAnalyses VMHashEliminationPass::run(
    llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  if (F.isDeclaration())
    return llvm::PreservedAnalyses::all();

  // Only process VM handler functions.
  if (!F.hasFnAttribute("omill.vm_handler"))
    return llvm::PreservedAnalyses::all();

  bool changed = false;
  unsigned eliminated_count = 0;

  // Phase 1: Find murmur hash rounds by scanning for lshr(X, 60).
  llvm::SmallPtrSet<llvm::Instruction *, 16> round_results;
  llvm::SmallVector<MurmurRound, 16> rounds;

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto round = matchMurmurRound(&I)) {
        rounds.push_back(*round);
        round_results.insert(round->result_mul);
      }
    }
  }

  // Phase 2: Find combined integrity flags (zext(icmp eq(or-chain, magic))).
  if (!round_results.empty()) {
    auto integrity_flags = findIntegrityFlags(F, round_results);
    for (auto *flag : integrity_flags) {
      auto *one = llvm::ConstantInt::get(flag->getType(), 1);
      flag->replaceAllUsesWith(one);
      changed = true;
      ++eliminated_count;
    }
  }

  // Phase 3: Find and eliminate hash token range checks.
  auto range_checks = findTokenRangeChecks(F);
  for (auto *check : range_checks) {
    auto *one = llvm::ConstantInt::get(check->getType(), 1);
    check->replaceAllUsesWith(one);
    changed = true;
    ++eliminated_count;
  }

  if (eliminated_count > 0) {
    llvm::errs() << "VMHashElimination[" << F.getName()
                 << "]: eliminated " << eliminated_count
                 << " integrity checks (" << rounds.size()
                 << " hash rounds found)\n";
  }

  if (!changed)
    return llvm::PreservedAnalyses::all();
  return llvm::PreservedAnalyses::none();
}

}  // namespace omill
