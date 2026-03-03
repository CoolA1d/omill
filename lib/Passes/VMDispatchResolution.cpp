#include "omill/Passes/VMDispatchResolution.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
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

/// Try to extract an RVA constant from a dispatch target value.
///
/// The dispatch target computation in the IR is typically:
///   %target = add i64 %image_base, <RVA_const>
/// or:
///   %target = add i64 <RVA_const>, %image_base
///
/// We extract the constant operand as the RVA.
static std::optional<uint32_t> extractRVAFromTarget(llvm::Value *target) {
  auto *bin_op = llvm::dyn_cast<llvm::BinaryOperator>(target);
  if (!bin_op || bin_op->getOpcode() != llvm::Instruction::Add)
    return std::nullopt;

  // Check both operands for a constant.
  for (unsigned i = 0; i < 2; ++i) {
    if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(bin_op->getOperand(i))) {
      uint64_t val = ci->getZExtValue();
      // RVAs are 32-bit values (< 4GB).
      if (val <= UINT32_MAX)
        return static_cast<uint32_t>(val);
    }
  }

  return std::nullopt;
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

}  // namespace

llvm::PreservedAnalyses VMDispatchResolutionPass::run(
    llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  auto &graph = MAM.getResult<VMHandlerGraphAnalysis>(M);
  if (graph.empty())
    return llvm::PreservedAnalyses::all();

  unsigned resolved_count = 0;
  unsigned skipped_count = 0;
  auto &ctx = M.getContext();

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    if (!F.hasFnAttribute("omill.vm_handler"))
      continue;

    auto va_opt = extractVAFromName(F.getName());
    if (!va_opt)
      continue;
    uint64_t handler_va = *va_opt;

    // Get known targets for this handler from the binary graph.
    auto targets = graph.getHandlerTargets(handler_va);

    // Collect dispatch calls in this function.
    llvm::SmallVector<llvm::CallInst *, 4> dispatch_jumps, dispatch_calls;
    collectDispatchCalls(F, dispatch_jumps, dispatch_calls);

    auto resolve_calls = [&](llvm::SmallVectorImpl<llvm::CallInst *> &calls) {
      for (auto *call : calls) {
        auto *target_val = call->getArgOperand(1);

        // Skip already-resolved (constant) targets.
        if (llvm::isa<llvm::ConstantInt>(target_val))
          continue;

        // Strategy 1: Extract RVA from the IR target computation and
        // look it up in the graph.
        if (auto rva = extractRVAFromTarget(target_val)) {
          if (auto resolved = graph.resolveRVA(*rva)) {
            auto *const_target =
                llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), *resolved);
            call->setArgOperand(1, const_target);
            ++resolved_count;
            continue;
          }
        }

        // Strategy 2: Single-target handler — the handler has exactly one
        // dispatch exit, so we know the target regardless of IR shape.
        if (targets.size() == 1 && calls.size() == 1) {
          auto *const_target =
              llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), targets[0]);
          call->setArgOperand(1, const_target);
          ++resolved_count;
          continue;
        }

        // Strategy 3: Two-target handler with two dispatch calls.
        // If the other call was already resolved (or can be), resolve this
        // one to the remaining target.
        if (targets.size() == 2 && calls.size() == 2) {
          // Find which target the other call resolved to.
          llvm::CallInst *other = (call == calls[0]) ? calls[1] : calls[0];
          auto *other_target = other->getArgOperand(1);
          if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(other_target)) {
            uint64_t other_va = ci->getZExtValue();
            uint64_t remaining =
                (other_va == targets[0]) ? targets[1] : targets[0];
            auto *const_target =
                llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), remaining);
            call->setArgOperand(1, const_target);
            ++resolved_count;
            continue;
          }
        }

        ++skipped_count;
      }
    };

    // Resolve in two passes: first pass may resolve some calls, enabling
    // Strategy 3 in the second pass.
    resolve_calls(dispatch_jumps);
    resolve_calls(dispatch_calls);
    resolve_calls(dispatch_jumps);
    resolve_calls(dispatch_calls);
  }

  llvm::errs() << "VMDispatchResolution: resolved " << resolved_count
               << " dispatch targets";
  if (skipped_count > 0)
    llvm::errs() << ", skipped " << skipped_count;
  llvm::errs() << "\n";

  if (resolved_count == 0)
    return llvm::PreservedAnalyses::all();
  return llvm::PreservedAnalyses::none();
}

}  // namespace omill
