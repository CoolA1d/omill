#include "omill/Passes/VMDispatchResolution.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include "omill/Analysis/VMHandlerGraph.h"

namespace omill {

namespace {

/// Extract a handler VA from a function name like "sub_1402A1000".
static std::optional<uint64_t> extractVAFromName(llvm::StringRef name) {
  if (!name.starts_with("sub_"))
    return std::nullopt;
  uint64_t va = 0;
  if (name.drop_front(4).getAsInteger(16, va))
    return std::nullopt;
  return va;
}

/// Collect all calls to __omill_dispatch_jump or __omill_dispatch_call
/// in a function.
static void collectDispatchCalls(
    llvm::Function &F,
    llvm::SmallVectorImpl<llvm::CallInst *> &dispatch_jumps,
    llvm::SmallVectorImpl<llvm::CallInst *> &dispatch_calls) {
  for (auto &BB : F) {
    for (auto &I : BB) {
      auto *call = llvm::dyn_cast<llvm::CallInst>(&I);
      if (!call)
        continue;
      auto *callee = call->getCalledFunction();
      if (!callee || call->arg_size() < 3)
        continue;

      auto name = callee->getName();
      if (name == "__omill_dispatch_jump")
        dispatch_jumps.push_back(call);
      else if (name == "__omill_dispatch_call")
        dispatch_calls.push_back(call);
    }
  }
}

/// Check if a constant looks like it could be an RVA (relative to image base).
/// RVAs for EAC handlers are typically in the range 0x20000..0x3FFFFFFF.
static bool isPlausibleRVA(uint64_t val) {
  return val >= 0x1000 && val <= 0x7FFFFFFF;
}

/// Recursively try to resolve a dispatch target value to a constant VA.
///
/// Handles these patterns:
///   1. ConstantInt: already a constant VA
///   2. add i64 %X, <const>: image_base + RVA
///   3. add i64 %X, %Y: recursively check if one operand resolves
///   4. select i1 %cond, %a, %b: resolve both branches (returns nullopt,
///      but populates resolved_select if both branches resolve)
///
/// Returns the resolved constant VA, or nullopt if unresolvable.
///
/// \param depth  Recursion depth limit.
static std::optional<uint64_t>
resolveTargetValue(llvm::Value *V, const VMHandlerGraph &graph,
                   llvm::Module &M, uint64_t handler_va, unsigned depth = 0) {
  if (depth > 8)
    return std::nullopt;

  // Already a constant.
  if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(V))
    return ci->getZExtValue();

  auto *inst = llvm::dyn_cast<llvm::Instruction>(V);
  if (!inst)
    return std::nullopt;

  // Pattern: add i64 %X, <const> or add i64 <const>, %X
  if (inst->getOpcode() == llvm::Instruction::Add) {
    llvm::Value *op0 = inst->getOperand(0);
    llvm::Value *op1 = inst->getOperand(1);

    // Check each operand for a constant.
    for (auto [const_op, other_op] :
         {std::pair{op0, op1}, std::pair{op1, op0}}) {
      auto *ci = llvm::dyn_cast<llvm::ConstantInt>(const_op);
      if (!ci)
        continue;

      uint64_t val = ci->getZExtValue();

      // Strategy A: The constant is an RVA, compute image_base + RVA.
      if (isPlausibleRVA(val)) {
        // First try the binary scan map (known patterns).
        if (auto resolved = graph.resolveRVA(static_cast<uint32_t>(val)))
          return *resolved;

        // Direct computation: image_base + const.
        uint64_t target_va = graph.imageBase() + val;
        return target_va;
      }

      // Strategy B: The other operand resolves to a constant.
      if (auto base = resolveTargetValue(other_op, graph, M, handler_va,
                                         depth + 1)) {
        return *base + val;
      }
    }

    // Strategy C: Both operands are variables, try to resolve each.
    // This handles nested add chains like: add(add(%pc, const1), %base)
    auto resolved0 =
        resolveTargetValue(op0, graph, M, handler_va, depth + 1);
    auto resolved1 =
        resolveTargetValue(op1, graph, M, handler_va, depth + 1);
    if (resolved0 && resolved1)
      return *resolved0 + *resolved1;
  }

  // Pattern: %program_counter argument (arg1 of lifted function).
  // In remill-lifted functions, arg1 is the initial program counter = handler VA.
  if (auto *arg = llvm::dyn_cast<llvm::Argument>(V)) {
    if (arg->getArgNo() == 1)
      return handler_va;
  }

  // Pattern: zext i32 %X to i64 or zext i1 %X to i64
  if (auto *zext = llvm::dyn_cast<llvm::ZExtInst>(inst))
    return resolveTargetValue(zext->getOperand(0), graph, M, handler_va,
                              depth + 1);

  // Pattern: trunc i64 %X to i32 (RVA truncation)
  if (auto *trunc = llvm::dyn_cast<llvm::TruncInst>(inst))
    return resolveTargetValue(trunc->getOperand(0), graph, M, handler_va,
                              depth + 1);

  // Pattern: and i64 %X, 0xFFFFFFFF (32-bit mask = RVA extraction)
  if (inst->getOpcode() == llvm::Instruction::And) {
    if (auto *mask = llvm::dyn_cast<llvm::ConstantInt>(inst->getOperand(1))) {
      if (mask->getZExtValue() == 0xFFFFFFFF) {
        return resolveTargetValue(inst->getOperand(0), graph, M, handler_va,
                                  depth + 1);
      }
    }
  }

  return std::nullopt;
}

/// Try to resolve a select instruction's dispatch target.
/// Returns true if the select was replaced with a new select of constants.
static bool resolveSelectTarget(llvm::CallInst *call, llvm::SelectInst *sel,
                                const VMHandlerGraph &graph, llvm::Module &M,
                                uint64_t handler_va) {
  auto resolved_true =
      resolveTargetValue(sel->getTrueValue(), graph, M, handler_va);
  auto resolved_false =
      resolveTargetValue(sel->getFalseValue(), graph, M, handler_va);

  if (!resolved_true || !resolved_false)
    return false;

  auto &ctx = M.getContext();
  auto *i64_ty = llvm::Type::getInt64Ty(ctx);
  auto *const_true = llvm::ConstantInt::get(i64_ty, *resolved_true);
  auto *const_false = llvm::ConstantInt::get(i64_ty, *resolved_false);

  // Build: select %cond, const_true, const_false
  llvm::IRBuilder<> builder(call);
  auto *new_sel =
      builder.CreateSelect(sel->getCondition(), const_true, const_false);
  call->setArgOperand(1, new_sel);
  return true;
}

}  // namespace

llvm::PreservedAnalyses VMDispatchResolutionPass::run(
    llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  auto &graph = MAM.getResult<VMHandlerGraphAnalysis>(M);
  if (graph.empty())
    return llvm::PreservedAnalyses::all();

  unsigned resolved_count = 0;
  unsigned select_count = 0;
  unsigned skipped_count = 0;
  unsigned discovery_count = 0;
  auto &ctx = M.getContext();
  auto *i64_ty = llvm::Type::getInt64Ty(ctx);

  // Collect newly-discovered target VAs (image_base + RVA for VAs not yet
  // in the module as function definitions).
  llvm::DenseSet<uint64_t> discovered_targets;

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    if (!F.hasFnAttribute("omill.vm_handler"))
      continue;

    auto va_opt = extractVAFromName(F.getName());
    if (!va_opt)
      continue;
    uint64_t handler_va = *va_opt;

    // Collect dispatch calls in this function.
    llvm::SmallVector<llvm::CallInst *, 4> dispatch_jumps, dispatch_calls;
    collectDispatchCalls(F, dispatch_jumps, dispatch_calls);

    auto resolve_calls = [&](llvm::SmallVectorImpl<llvm::CallInst *> &calls) {
      for (auto *call : calls) {
        auto *target_val = call->getArgOperand(1);

        // Skip already-resolved (constant) targets.
        if (llvm::isa<llvm::ConstantInt>(target_val))
          continue;

        // Priority 1: chain-solved targets from VMHandlerChainSolver.
        // The chain solver concretely emulates handlers and knows the exact
        // successor VA.  This takes priority over IR pattern matching because
        // the IR-level `image_base + RVA` formula is wrong for EAC-style VMs
        // where the dispatch base is `delta` (not image_base).
        {
          auto chain_succs = graph.getChainTargets(handler_va);
          if (chain_succs.size() == 1) {
            auto *const_target =
                llvm::ConstantInt::get(i64_ty, chain_succs[0]);
            call->setArgOperand(1, const_target);
            ++resolved_count;

            std::string target_name =
                "sub_" + llvm::Twine::utohexstr(chain_succs[0]).str();
            if (!M.getFunction(target_name))
              discovered_targets.insert(chain_succs[0]);
            continue;
          }
          if (chain_succs.size() == 2) {
            // Two successors — can't determine which branch without
            // analyzing the condition. For now, skip.
            // TODO: match the IR-level branch condition to pick the right
            // successor, or emit a select of both.
          }
        }

        // Priority 2: recursive IR pattern resolution.
        if (auto resolved =
                resolveTargetValue(target_val, graph, M, handler_va)) {
          auto *const_target = llvm::ConstantInt::get(i64_ty, *resolved);
          call->setArgOperand(1, const_target);
          ++resolved_count;

          std::string target_name =
              "sub_" + llvm::Twine::utohexstr(*resolved).str();
          if (!M.getFunction(target_name)) {
            discovered_targets.insert(*resolved);
          }
          continue;
        }

        // Priority 3: select-specific resolution.
        if (auto *sel = llvm::dyn_cast<llvm::SelectInst>(target_val)) {
          if (resolveSelectTarget(call, sel, graph, M, handler_va)) {
            ++select_count;

            if (auto *new_sel =
                    llvm::dyn_cast<llvm::SelectInst>(call->getArgOperand(1))) {
              for (auto *op :
                   {new_sel->getTrueValue(), new_sel->getFalseValue()}) {
                if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(op)) {
                  std::string name =
                      "sub_" +
                      llvm::Twine::utohexstr(ci->getZExtValue()).str();
                  if (!M.getFunction(name))
                    discovered_targets.insert(ci->getZExtValue());
                }
              }
            }
            continue;
          }
        }

        ++skipped_count;
      }
    };

    // Two passes: second pass may benefit from earlier resolutions.
    resolve_calls(dispatch_jumps);
    resolve_calls(dispatch_calls);
    resolve_calls(dispatch_jumps);
    resolve_calls(dispatch_calls);
  }

  // Store discovered targets as named metadata so the tool can re-lift them.
  if (!discovered_targets.empty()) {
    // Clear any previous discovered targets.
    if (auto *old_md = M.getNamedMetadata("omill.vm_discovered_targets"))
      M.eraseNamedMetadata(old_md);

    auto *named_md =
        M.getOrInsertNamedMetadata("omill.vm_discovered_targets");
    for (auto va : discovered_targets) {
      auto *ci_md = llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(i64_ty, va));
      named_md->addOperand(llvm::MDTuple::get(ctx, {ci_md}));
    }
    discovery_count = discovered_targets.size();
  }

  llvm::errs() << "VMDispatchResolution: resolved " << resolved_count;
  if (select_count > 0)
    llvm::errs() << " + " << select_count << " selects";
  if (skipped_count > 0)
    llvm::errs() << ", skipped " << skipped_count;
  if (discovery_count > 0)
    llvm::errs() << ", discovered " << discovery_count << " new targets";
  llvm::errs() << "\n";

  if (resolved_count == 0 && select_count == 0)
    return llvm::PreservedAnalyses::all();
  return llvm::PreservedAnalyses::none();
}

}  // namespace omill
