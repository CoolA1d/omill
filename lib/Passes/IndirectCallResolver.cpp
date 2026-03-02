#include "omill/Passes/IndirectCallResolver.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/ADT/SmallPtrSet.h>

#include "omill/Analysis/BinaryMemoryMap.h"
#include "omill/Analysis/LiftedFunctionMap.h"
#include "omill/Utils/LiftedNames.h"

namespace omill {

namespace {

/// Maximum expression tree depth to prevent infinite recursion through PHIs.
static constexpr unsigned kMaxEvalDepth = 24;

/// Check if a pointer is a State struct GEP (getelementptr from arg0).
bool isStateGEP(llvm::Value *ptr) {
  auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(ptr);
  if (!gep)
    return false;
  return llvm::isa<llvm::Argument>(gep->getPointerOperand());
}

/// Walk backwards in the same basic block from `load` looking for a store to
/// the same pointer.  Returns the stored Value* or nullptr.  Skips stores
/// through inttoptr (they can't alias State GEPs) and stops at calls.
llvm::Value *forwardLoadInBlock(llvm::LoadInst *load) {
  auto *BB = load->getParent();
  auto *load_ptr = load->getPointerOperand();
  bool load_is_state = isStateGEP(load_ptr);
  for (auto it = load->getIterator(); it != BB->begin();) {
    --it;
    if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(&*it)) {
      if (SI->getPointerOperand() == load_ptr)
        return SI->getValueOperand();
      // If load is from State GEP, stores through inttoptr can't alias.
      if (load_is_state) {
        auto *sp = SI->getPointerOperand()->stripPointerCasts();
        if (llvm::isa<llvm::IntToPtrInst>(sp))
          continue;
      }
    }
    if (auto *CI2 = llvm::dyn_cast<llvm::CallInst>(&*it)) {
      auto *callee = CI2->getCalledFunction();
      if (callee && callee->isIntrinsic() &&
          (callee->doesNotAccessMemory() ||
           callee->onlyAccessesInaccessibleMemory()))
        continue;
      return nullptr;
    }
    if (llvm::isa<llvm::InvokeInst>(&*it))
      return nullptr;
  }
  return nullptr;
}

/// Normalize an integer address expression into (base, constant_offset).
/// Collapses chains of `add(add(X, A), B)` -> `(X, A+B)` and chases through
/// State GEP load->store forwarding.  Forwarding stops when it detects a
/// save/restore cycle (forwarded value resolves back to a load from the same
/// State GEP), which keeps both sides of a comparison at the same depth in
/// the RSP chain.
/// Returns (nullptr, C) for pure constants.
std::pair<llvm::Value *, int64_t> normalizeAddrExpr(llvm::Value *V,
                                                    unsigned max_fwd = 4) {
  int64_t offset = 0;
  unsigned fwd_remaining = max_fwd;
  while (true) {
    if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(V))
      return {nullptr, offset + CI->getSExtValue()};
    // add(X, C) or add(C, X)
    if (auto *BO = llvm::dyn_cast<llvm::BinaryOperator>(V)) {
      if (BO->getOpcode() == llvm::Instruction::Add) {
        if (auto *C = llvm::dyn_cast<llvm::ConstantInt>(BO->getOperand(1))) {
          offset += C->getSExtValue();
          V = BO->getOperand(0);
          continue;
        }
        if (auto *C = llvm::dyn_cast<llvm::ConstantInt>(BO->getOperand(0))) {
          offset += C->getSExtValue();
          V = BO->getOperand(1);
          continue;
        }
      }
      if (BO->getOpcode() == llvm::Instruction::Sub) {
        if (auto *C = llvm::dyn_cast<llvm::ConstantInt>(BO->getOperand(1))) {
          offset -= C->getSExtValue();
          V = BO->getOperand(0);
          continue;
        }
      }
    }
    // Chase through State GEP loads via same-block store forwarding.
    if (fwd_remaining > 0) {
      if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(V)) {
        if (isStateGEP(LI->getPointerOperand())) {
          if (auto *fwd = forwardLoadInBlock(LI)) {
            // Check if forwarding leads back to a load from the same State
            // GEP (save/restore cycle, e.g. RSP save then RSP restore).
            // If so, stop: this load is the canonical base.
            auto [fwd_base, fwd_off] = normalizeAddrExpr(fwd, 0);
            if (auto *fwd_load = llvm::dyn_cast_or_null<llvm::LoadInst>(fwd_base)) {
              if (fwd_load->getPointerOperand() == LI->getPointerOperand()) {
                // Cycle detected: forwarded value is another load from the
                // same register.  Use current load as canonical base.
                return {V, offset};
              }
            }
            --fwd_remaining;
            V = fwd;
            continue;
          }
        }
      }
    }
    return {V, offset};
  }
}

/// Check if two address-expression bases are equivalent.
/// Handles: SSA equality, and loads-from-same-State-GEP where one
/// store-forwards to the other (one level).
bool addressBasesEqual(llvm::Value *A, llvm::Value *B) {
  if (A == B)
    return true;
  // Both must be loads from the same pointer (State GEP).
  auto *LA = llvm::dyn_cast_or_null<llvm::LoadInst>(A);
  auto *LB = llvm::dyn_cast_or_null<llvm::LoadInst>(B);
  if (!LA || !LB)
    return false;
  if (LA->getPointerOperand() != LB->getPointerOperand())
    return false;
  // Try forwarding one level: does A forward to a value equal to B,
  // or B forward to a value equal to A?
  if (auto *fwdA = forwardLoadInBlock(LA)) {
    auto [baseA, offA] = normalizeAddrExpr(fwdA, 0);
    if (offA == 0 && baseA == B)
      return true;
  }
  if (auto *fwdB = forwardLoadInBlock(LB)) {
    auto [baseB, offB] = normalizeAddrExpr(fwdB, 0);
    if (offB == 0 && baseB == A)
      return true;
  }
  return false;
}

/// Recursively evaluate an SSA value to a concrete uint64_t using binary
/// memory reads for loads from constant addresses.
///
/// This is the core of the pass: unlike ConstantMemoryFolding (which folds
/// individual loads) or InstCombine (which folds arithmetic), this evaluator
/// walks the entire expression tree in one shot, resolving multi-hop chains
/// like load(load(0x140008000) + 0x10) without requiring multiple pass
/// iterations.
std::optional<uint64_t> evaluateToConstant(llvm::Value *V,
                                           const BinaryMemoryMap &map,
                                           unsigned depth = 0) {
  if (depth > kMaxEvalDepth)
    return std::nullopt;

  // Function argument: arg1 in lifted functions is the entry PC.
  // The function name encodes it as sub_<hex_va>.
  if (auto *Arg = llvm::dyn_cast<llvm::Argument>(V)) {
    if (Arg->getArgNo() == 1) {  // program_counter
      auto *F = Arg->getParent();
      auto name = F->getName();
      uint64_t va = extractEntryVA(name);
      if (va != 0)
        return va;
    }
    return std::nullopt;
  }

  // Already a constant integer.
  if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(V)) {
    if (CI->getBitWidth() <= 64)
      return CI->getZExtValue();
    return std::nullopt;
  }

  // ZExt / SExt: evaluate the inner value.
  if (auto *zext = llvm::dyn_cast<llvm::ZExtInst>(V))
    return evaluateToConstant(zext->getOperand(0), map, depth + 1);
  if (auto *sext = llvm::dyn_cast<llvm::SExtInst>(V)) {
    auto inner = evaluateToConstant(sext->getOperand(0), map, depth + 1);
    if (!inner)
      return std::nullopt;
    unsigned src_bits = sext->getOperand(0)->getType()->getIntegerBitWidth();
    // Sign extend from src_bits to 64 bits.
    uint64_t val = *inner;
    if (src_bits < 64 && (val & (1ULL << (src_bits - 1))))
      val |= ~((1ULL << src_bits) - 1);
    return val;
  }

  // Trunc: evaluate inner and mask.
  if (auto *trunc = llvm::dyn_cast<llvm::TruncInst>(V)) {
    auto inner = evaluateToConstant(trunc->getOperand(0), map, depth + 1);
    if (!inner)
      return std::nullopt;
    unsigned dst_bits = trunc->getType()->getIntegerBitWidth();
    if (dst_bits >= 64)
      return *inner;
    return *inner & ((1ULL << dst_bits) - 1);
  }

  // Binary operators: add, sub, xor, or, and, shl, lshr, ashr, mul.
  if (auto *bin = llvm::dyn_cast<llvm::BinaryOperator>(V)) {
    auto lhs = evaluateToConstant(bin->getOperand(0), map, depth + 1);
    auto rhs = evaluateToConstant(bin->getOperand(1), map, depth + 1);
    if (!lhs || !rhs)
      return std::nullopt;

    switch (bin->getOpcode()) {
    case llvm::Instruction::Add:  return *lhs + *rhs;
    case llvm::Instruction::Sub:  return *lhs - *rhs;
    case llvm::Instruction::Mul:  return *lhs * *rhs;
    case llvm::Instruction::Xor:  return *lhs ^ *rhs;
    case llvm::Instruction::Or:   return *lhs | *rhs;
    case llvm::Instruction::And:  return *lhs & *rhs;
    case llvm::Instruction::Shl:  return (*rhs < 64) ? (*lhs << *rhs) : 0ULL;
    case llvm::Instruction::LShr: return (*rhs < 64) ? (*lhs >> *rhs) : 0ULL;
    case llvm::Instruction::AShr: {
      if (*rhs >= 64)
        return (*lhs & (1ULL << 63)) ? ~0ULL : 0ULL;
      return static_cast<uint64_t>(
          static_cast<int64_t>(*lhs) >> static_cast<int64_t>(*rhs));
    }
    default:
      return std::nullopt;
    }
  }

  // Load: try binary memory read, then same-block store forwarding.
  if (auto *load = llvm::dyn_cast<llvm::LoadInst>(V)) {
    auto *ptr = load->getPointerOperand()->stripPointerCasts();
    // Detect inttoptr addressing.
    llvm::Value *int_val = nullptr;
    if (auto *itp = llvm::dyn_cast<llvm::IntToPtrInst>(ptr))
      int_val = itp->getOperand(0);
    else if (auto *ce = llvm::dyn_cast<llvm::ConstantExpr>(ptr)) {
      if (ce->getOpcode() == llvm::Instruction::IntToPtr)
        int_val = ce->getOperand(0);
    }

    // Path 1: inttoptr(X) where X evaluates to a constant → binary memory read.
    if (int_val) {
      auto addr = evaluateToConstant(int_val, map, depth + 1);
      if (addr) {
        unsigned load_size = load->getType()->getIntegerBitWidth() / 8;
        if (load_size <= 8) {
          uint8_t buf[8] = {};
          if (map.read(*addr, buf, load_size)) {
            uint64_t result = 0;
            for (unsigned i = 0; i < load_size; ++i)
              result |= static_cast<uint64_t>(buf[i]) << (i * 8);
            unsigned bits = load->getType()->getIntegerBitWidth();
            if (bits < 64)
              result &= (1ULL << bits) - 1;
            return result;
          }
        }
      }
    }

    // Path 2: Same-block store-to-load forwarding.
    // Handles two cases:
    //  a) Exact pointer match (State GEP stores, e.g. load %R12 / store %R12)
    //  b) Symbolic inttoptr address match (computed-address stores/loads
    //     that share the same (base,offset) after normalization)
    auto *BB = load->getParent();
    auto *load_ptr = load->getPointerOperand();
    bool load_is_state_gep = isStateGEP(load_ptr);
    auto load_addr_norm = int_val
        ? std::optional(normalizeAddrExpr(int_val))
        : std::nullopt;
    for (auto it = load->getIterator(); it != BB->begin();) {
      --it;
      if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(&*it)) {
        auto *store_ptr = SI->getPointerOperand();

        // Case (a): exact pointer match (same GEP/alloca/SSA value).
        if (store_ptr == load_ptr) {
          return evaluateToConstant(SI->getValueOperand(), map, depth + 1);
        }

        // Classify the store pointer.
        auto *sp_stripped = store_ptr->stripPointerCasts();
        bool store_is_inttoptr = llvm::isa<llvm::IntToPtrInst>(sp_stripped);
        bool store_is_state_gep = isStateGEP(store_ptr);

        if (int_val) {
          // Load is from inttoptr.  Check symbolic address match.
          if (store_is_inttoptr) {
            llvm::Value *store_int_val = nullptr;
            if (auto *itp2 = llvm::dyn_cast<llvm::IntToPtrInst>(sp_stripped))
              store_int_val = itp2->getOperand(0);

            if (store_int_val) {
              // Fast path: same SSA address.
              if (store_int_val == int_val) {
                return evaluateToConstant(SI->getValueOperand(), map, depth + 1);
              }

              // Normalize both address expressions and compare.
              auto [loadBase, loadOff] = *load_addr_norm;
              auto [storeBase, storeOff] = normalizeAddrExpr(store_int_val);
              if (loadOff == storeOff &&
                  addressBasesEqual(loadBase, storeBase)) {
                return evaluateToConstant(SI->getValueOperand(), map,
                                         depth + 1);
              }
            }
            // Non-matching inttoptr store — skip (different runtime addr).
            continue;
          }
          // Store to State GEP or other — can't alias inttoptr load.
          if (store_is_state_gep)
            continue;
        } else if (load_is_state_gep) {
          // Load is from State GEP.
          if (store_is_inttoptr)
            continue;  // inttoptr store can't alias State GEP
          if (store_is_state_gep)
            continue;  // Different State GEP — different register
        }
        // Unknown store — conservatively stop walking.
        break;
      }
      // Stop at call/invoke that may modify memory.
      if (auto *CI2 = llvm::dyn_cast<llvm::CallInst>(&*it)) {
        auto *callee = CI2->getCalledFunction();
        if (callee && callee->isIntrinsic() &&
            (callee->doesNotAccessMemory() ||
             callee->onlyAccessesInaccessibleMemory()))
          continue;
        break;
      }
      if (llvm::isa<llvm::InvokeInst>(&*it))
        break;
    }

    return std::nullopt;
  }

  // Select with one evaluable arm (or both).
  if (auto *sel = llvm::dyn_cast<llvm::SelectInst>(V)) {
    auto cond = evaluateToConstant(sel->getCondition(), map, depth + 1);
    if (cond) {
      // Condition is known — evaluate only the selected arm.
      auto *arm = (*cond != 0) ? sel->getTrueValue() : sel->getFalseValue();
      return evaluateToConstant(arm, map, depth + 1);
    }
    // If both arms evaluate to the same value, use that.
    auto true_val = evaluateToConstant(sel->getTrueValue(), map, depth + 1);
    auto false_val = evaluateToConstant(sel->getFalseValue(), map, depth + 1);
    if (true_val && false_val && *true_val == *false_val)
      return *true_val;
    return std::nullopt;
  }

  // PHI with all evaluable incoming values that agree.
  if (auto *phi = llvm::dyn_cast<llvm::PHINode>(V)) {
    if (phi->getNumIncomingValues() == 0)
      return std::nullopt;
    std::optional<uint64_t> agreed;
    for (unsigned i = 0, e = phi->getNumIncomingValues(); i < e; ++i) {
      auto val = evaluateToConstant(phi->getIncomingValue(i), map, depth + 1);
      if (!val)
        return std::nullopt;
      if (!agreed)
        agreed = val;
      else if (*agreed != *val)
        return std::nullopt;
    }
    return agreed;
  }

  return std::nullopt;
}

// ============================================================
// Monte Carlo Concrete Evaluation for VM Dispatch Targets
// ============================================================
// When the deterministic evaluateToConstant fails (because some operands
// are symbolic — e.g., unknown register values from the caller), we fall
// back to Monte Carlo evaluation.  VM dispatch targets are deterministic:
// symbolic operands are mixed via MBA but cancel out, always producing
// the same constant.  We verify this by running multiple trials with
// different random values for unknowns and checking consistency.

/// Map from {AllocaInst*, byte_offset} to stored values.
struct MCStoreEntry {
  llvm::Value *Val;
  unsigned BitWidth;
  llvm::StoreInst *SI;
};
using MCStoreMap = llvm::DenseMap<std::pair<llvm::AllocaInst *, int64_t>,
                                  llvm::SmallVector<MCStoreEntry, 2>>;

/// Decompose a GEP to {alloca, byte_offset}.
static std::optional<std::pair<llvm::AllocaInst *, int64_t>>
mcDecomposeGEP(llvm::Value *Ptr, const llvm::DataLayout &DL) {
  if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(Ptr))
    return std::make_pair(AI, int64_t(0));
  if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(Ptr)) {
    if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(GEP->getPointerOperand())) {
      llvm::APInt Off(64, 0);
      if (GEP->accumulateConstantOffset(DL, Off))
        return std::make_pair(AI, Off.getSExtValue());
    }
  }
  return std::nullopt;
}

/// Build a store map tracking all integer-type stores to alloca-based GEPs.
static MCStoreMap buildMCStoreMap(llvm::Function &F,
                                   const llvm::DataLayout &DL) {
  MCStoreMap Map;
  for (auto &BB : F)
    for (auto &I : BB) {
      auto *SI = llvm::dyn_cast<llvm::StoreInst>(&I);
      if (!SI)
        continue;
      auto *ValTy = SI->getValueOperand()->getType();
      if (!ValTy->isIntegerTy())
        continue;
      auto Decomp = mcDecomposeGEP(SI->getPointerOperand(), DL);
      if (!Decomp)
        continue;
      Map[*Decomp].push_back(
          {SI->getValueOperand(), ValTy->getIntegerBitWidth(), SI});
    }
  return Map;
}

/// Advance a simple xorshift64 PRNG.
static uint64_t mcXorshift64(uint64_t &S) {
  S ^= S << 13;
  S ^= S >> 7;
  S ^= S << 17;
  return S;
}

/// Concrete evaluation of an integer expression tree.
/// Unknown leaves (arguments, unresolved loads, PHIs, calls) get random
/// values from the RNG.  The SpecStoreMap enables following loads through
/// alloca-based stores.
static std::optional<uint64_t>
mcConcreteEval(llvm::Value *V,
               llvm::DenseMap<llvm::Value *, uint64_t> &Env,
               const MCStoreMap &SSM,
               const llvm::DataLayout &DL,
               const BinaryMemoryMap *BMM,
               llvm::SmallPtrSetImpl<llvm::Value *> &InProgress,
               uint64_t &RNG) {
  auto CIt = Env.find(V);
  if (CIt != Env.end())
    return CIt->second;
  if (!InProgress.insert(V).second)
    return std::nullopt;

  std::optional<uint64_t> Result;

  if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(V)) {
    Result = CI->getZExtValue();
  } else if (llvm::isa<llvm::UndefValue>(V) || llvm::isa<llvm::PoisonValue>(V)) {
    Result = uint64_t(0);
  } else if (auto *P2I = llvm::dyn_cast<llvm::PtrToIntInst>(V)) {
    llvm::Value *Ptr = P2I->getOperand(0);
    if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(Ptr)) {
      auto It = Env.find(AI);
      if (It != Env.end())
        Result = It->second;
    } else if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(Ptr)) {
      if (auto *AI =
              llvm::dyn_cast<llvm::AllocaInst>(GEP->getPointerOperand())) {
        auto It = Env.find(AI);
        if (It != Env.end()) {
          llvm::APInt Off(64, 0);
          if (GEP->accumulateConstantOffset(DL, Off))
            Result = It->second + Off.getZExtValue();
        }
      }
    }
  } else if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(V)) {
    if (LI->getType()->isIntegerTy()) {
      unsigned LoadBW = LI->getType()->getIntegerBitWidth();
      // Try alloca store map first.
      auto Decomp = mcDecomposeGEP(LI->getPointerOperand(), DL);
      if (Decomp) {
        auto SIt = SSM.find(*Decomp);
        if (SIt != SSM.end()) {
          llvm::Value *BestVal = nullptr;
          for (auto &E : SIt->second) {
            if (E.BitWidth != LoadBW)
              continue;
            if (E.SI->getParent() == LI->getParent()) {
              if (E.SI->comesBefore(LI))
                BestVal = E.Val;
            } else {
              BestVal = E.Val;
            }
          }
          if (BestVal)
            Result =
                mcConcreteEval(BestVal, Env, SSM, DL, BMM, InProgress, RNG);
        }
      }
      // Try binary memory read for inttoptr loads.
      if (!Result && BMM) {
        auto *Ptr = LI->getPointerOperand()->stripPointerCasts();
        llvm::Value *IntVal = nullptr;
        if (auto *ITP = llvm::dyn_cast<llvm::IntToPtrInst>(Ptr))
          IntVal = ITP->getOperand(0);
        if (IntVal) {
          auto Addr =
              mcConcreteEval(IntVal, Env, SSM, DL, BMM, InProgress, RNG);
          if (Addr) {
            unsigned LoadSize = LoadBW / 8;
            if (LoadSize <= 8) {
              uint8_t Buf[8] = {};
              if (BMM->read(*Addr, Buf, LoadSize)) {
                uint64_t Val = 0;
                for (unsigned i = 0; i < LoadSize; ++i)
                  Val |= static_cast<uint64_t>(Buf[i]) << (i * 8);
                if (LoadBW < 64)
                  Val &= (1ULL << LoadBW) - 1;
                Result = Val;
              }
            }
          }
        }
      }
    }
  } else if (auto *BO = llvm::dyn_cast<llvm::BinaryOperator>(V)) {
    auto L = mcConcreteEval(BO->getOperand(0), Env, SSM, DL, BMM, InProgress, RNG);
    auto R = mcConcreteEval(BO->getOperand(1), Env, SSM, DL, BMM, InProgress, RNG);
    if (L && R) {
      switch (BO->getOpcode()) {
      case llvm::Instruction::Add:  Result = *L + *R; break;
      case llvm::Instruction::Sub:  Result = *L - *R; break;
      case llvm::Instruction::Mul:  Result = *L * *R; break;
      case llvm::Instruction::Xor:  Result = *L ^ *R; break;
      case llvm::Instruction::And:  Result = *L & *R; break;
      case llvm::Instruction::Or:   Result = *L | *R; break;
      case llvm::Instruction::Shl:  Result = *L << (*R & 63); break;
      case llvm::Instruction::LShr: Result = *L >> (*R & 63); break;
      case llvm::Instruction::AShr:
        Result = static_cast<uint64_t>(
            static_cast<int64_t>(*L) >> (*R & 63));
        break;
      case llvm::Instruction::URem:
        Result = *R ? std::optional(*L % *R) : std::nullopt;
        break;
      case llvm::Instruction::UDiv:
        Result = *R ? std::optional(*L / *R) : std::nullopt;
        break;
      default: break;
      }
    }
  } else if (auto *Cast = llvm::dyn_cast<llvm::CastInst>(V)) {
    auto Op =
        mcConcreteEval(Cast->getOperand(0), Env, SSM, DL, BMM, InProgress, RNG);
    if (Op) {
      unsigned SrcBits = Cast->getSrcTy()->getScalarSizeInBits();
      unsigned DstBits = Cast->getDestTy()->getScalarSizeInBits();
      switch (Cast->getOpcode()) {
      case llvm::Instruction::ZExt:
        Result = (SrcBits < 64) ? (*Op & ((1ULL << SrcBits) - 1)) : *Op;
        break;
      case llvm::Instruction::SExt: {
        uint64_t v = *Op;
        if (SrcBits < 64) {
          v &= (1ULL << SrcBits) - 1;
          if (v & (1ULL << (SrcBits - 1)))
            v |= ~((1ULL << SrcBits) - 1);
        }
        Result = v;
        break;
      }
      case llvm::Instruction::Trunc:
        Result = (DstBits < 64) ? (*Op & ((1ULL << DstBits) - 1)) : *Op;
        break;
      case llvm::Instruction::BitCast:
      case llvm::Instruction::PtrToInt:
      case llvm::Instruction::IntToPtr:
        Result = *Op;
        break;
      default: break;
      }
    }
  } else if (auto *IC = llvm::dyn_cast<llvm::ICmpInst>(V)) {
    auto L = mcConcreteEval(IC->getOperand(0), Env, SSM, DL, BMM, InProgress, RNG);
    auto R = mcConcreteEval(IC->getOperand(1), Env, SSM, DL, BMM, InProgress, RNG);
    if (L && R) {
      bool Res = false;
      switch (IC->getPredicate()) {
      case llvm::ICmpInst::ICMP_EQ:  Res = *L == *R; break;
      case llvm::ICmpInst::ICMP_NE:  Res = *L != *R; break;
      case llvm::ICmpInst::ICMP_UGT: Res = *L > *R; break;
      case llvm::ICmpInst::ICMP_UGE: Res = *L >= *R; break;
      case llvm::ICmpInst::ICMP_ULT: Res = *L < *R; break;
      case llvm::ICmpInst::ICMP_ULE: Res = *L <= *R; break;
      case llvm::ICmpInst::ICMP_SGT:
        Res = static_cast<int64_t>(*L) > static_cast<int64_t>(*R); break;
      case llvm::ICmpInst::ICMP_SGE:
        Res = static_cast<int64_t>(*L) >= static_cast<int64_t>(*R); break;
      case llvm::ICmpInst::ICMP_SLT:
        Res = static_cast<int64_t>(*L) < static_cast<int64_t>(*R); break;
      case llvm::ICmpInst::ICMP_SLE:
        Res = static_cast<int64_t>(*L) <= static_cast<int64_t>(*R); break;
      default: goto mc_done;
      }
      Result = Res ? 1ULL : 0ULL;
    }
  } else if (auto *Sel = llvm::dyn_cast<llvm::SelectInst>(V)) {
    auto Cond =
        mcConcreteEval(Sel->getCondition(), Env, SSM, DL, BMM, InProgress, RNG);
    if (Cond)
      Result = (*Cond & 1)
          ? mcConcreteEval(Sel->getTrueValue(), Env, SSM, DL, BMM, InProgress, RNG)
          : mcConcreteEval(Sel->getFalseValue(), Env, SSM, DL, BMM, InProgress, RNG);
  } else if (auto *PHI = llvm::dyn_cast<llvm::PHINode>(V)) {
    // For PHI nodes, try evaluating all incoming values.
    // If they all agree, use that.  Otherwise treat as free variable.
    if (PHI->getNumIncomingValues() > 0) {
      std::optional<uint64_t> agreed;
      bool all_agree = true;
      for (unsigned i = 0, e = PHI->getNumIncomingValues(); i < e; ++i) {
        auto val = mcConcreteEval(PHI->getIncomingValue(i), Env, SSM, DL, BMM,
                                  InProgress, RNG);
        if (!val) {
          all_agree = false;
          break;
        }
        if (!agreed)
          agreed = val;
        else if (*agreed != *val) {
          all_agree = false;
          break;
        }
      }
      if (all_agree && agreed)
        Result = *agreed;
    }
  }

mc_done:
  InProgress.erase(V);

  // Unknown leaves get random concrete values.
  if (!Result) {
    if (llvm::isa<llvm::Argument>(V) || llvm::isa<llvm::PHINode>(V) ||
        llvm::isa<llvm::CallBase>(V) || llvm::isa<llvm::LoadInst>(V)) {
      Result = mcXorshift64(RNG);
    } else {
      return std::nullopt;
    }
  }

  Env[V] = *Result;
  return *Result;
}

/// Try to resolve a dispatch target using Monte Carlo evaluation.
/// Returns the constant value if all trials produce the same result
/// and the result is a valid address in the binary.
static std::optional<uint64_t>
tryMonteCarloResolve(llvm::Value *V, llvm::Function &F,
                     const BinaryMemoryMap *BMM) {
  constexpr unsigned NumTrials = 32;
  const llvm::DataLayout &DL = F.getDataLayout();

  MCStoreMap SSM = buildMCStoreMap(F, DL);

  // Collect all allocas for base assignment.
  llvm::SmallVector<llvm::AllocaInst *, 8> AllAllocas;
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *A = llvm::dyn_cast<llvm::AllocaInst>(&I))
        AllAllocas.push_back(A);

  auto populateAllocaBases = [&](llvm::DenseMap<llvm::Value *, uint64_t> &Env,
                                  uint64_t StartBase) {
    uint64_t Base = StartBase;
    for (auto *A : AllAllocas) {
      Env[A] = Base;
      Base += 0x0001'0000'0000ULL;
    }
  };

  // Run trials with first alloca base set.
  constexpr uint64_t AllocaBase1 = 0x7FFE'0000'0000ULL;
  llvm::DenseMap<uint64_t, unsigned> Freq;

  for (unsigned Trial = 0; Trial < NumTrials; ++Trial) {
    llvm::DenseMap<llvm::Value *, uint64_t> Env;
    populateAllocaBases(Env, AllocaBase1);
    uint64_t RNG = 0xDEADBEEF12345678ULL ^
                   (uint64_t(Trial) * 0x9E3779B97F4A7C15ULL);
    llvm::SmallPtrSet<llvm::Value *, 32> InProgress;
    auto Res = mcConcreteEval(V, Env, SSM, DL, BMM, InProgress, RNG);
    if (!Res)
      return std::nullopt;
    Freq[*Res]++;
  }

  uint64_t Candidate = 0;

  if (Freq.size() == 1) {
    Candidate = Freq.begin()->first;
  } else if (Freq.size() == 2 && BMM) {
    // Two-value split: integrity check branch.  Pick the majority value
    // if it's a valid binary address.
    auto It = Freq.begin();
    uint64_t ValA = It->first;
    unsigned CountA = It->second;
    ++It;
    uint64_t ValB = It->first;
    unsigned CountB = It->second;

    uint8_t ByteA = 0, ByteB = 0;
    bool ValidA = BMM->read(ValA, &ByteA, 1);
    bool ValidB = BMM->read(ValB, &ByteB, 1);

    if (ValidA && !ValidB)
      Candidate = ValA;
    else if (ValidB && !ValidA)
      Candidate = ValB;
    else if (ValidA && ValidB)
      Candidate = (CountA >= CountB) ? ValA : ValB;
    else
      return std::nullopt;
  } else {
    return std::nullopt;
  }

  // Cross-check: verify the result is NOT alloca-dependent.
  constexpr uint64_t AllocaBase2 = 0x7FFF'0000'0000ULL;
  {
    llvm::DenseMap<llvm::Value *, uint64_t> Env;
    populateAllocaBases(Env, AllocaBase2);
    uint64_t RNG = 0xDEADBEEF12345678ULL; // same seed as trial 0
    llvm::SmallPtrSet<llvm::Value *, 32> InProgress;
    auto Res = mcConcreteEval(V, Env, SSM, DL, BMM, InProgress, RNG);
    if (!Res)
      return std::nullopt;
    if (Freq.size() == 1 && *Res != Candidate)
      return std::nullopt; // alloca-dependent
    if (Freq.size() == 2 && Freq.find(*Res) == Freq.end())
      return std::nullopt; // third distinct value
  }

  // Validate: candidate should be a valid binary address (if BMM available).
  if (BMM) {
    uint8_t Byte = 0;
    if (!BMM->read(Candidate, &Byte, 1))
      return std::nullopt;
  }

  return Candidate;
}

/// Check if a dispatch target is already resolved (ptrtoint of a Function).
bool isAlreadyResolved(llvm::Value *target) {
  if (auto *ptoi = llvm::dyn_cast<llvm::PtrToIntOperator>(target))
    if (llvm::isa<llvm::Function>(ptoi->getPointerOperand()))
      return true;
  if (llvm::isa<llvm::ConstantInt>(target))
    return true;
  return false;
}

/// Resolve a dispatch_call target: replace with constant or ptrtoint(@import).
/// Returns true if the call was modified.
bool resolveDispatchCall(llvm::CallInst *call, uint64_t resolved_pc,
                         const BinaryMemoryMap *map,
                         const LiftedFunctionMap *lifted) {
  auto &M = *call->getFunction()->getParent();
  auto &Ctx = call->getContext();
  auto *i64_ty = llvm::Type::getInt64Ty(Ctx);

  // Priority 1: IAT import.
  if (map && map->hasImports()) {
    if (auto *imp = map->lookupImport(resolved_pc)) {
      auto *fn_type = llvm::FunctionType::get(llvm::Type::getVoidTy(Ctx), false);
      auto fn_callee = M.getOrInsertFunction(imp->function, fn_type);
      auto *fn = llvm::dyn_cast<llvm::Function>(fn_callee.getCallee());
      if (fn) {
        fn->setDLLStorageClass(llvm::GlobalValue::DLLImportStorageClass);
        llvm::IRBuilder<> Builder(call);
        auto *fn_addr = Builder.CreatePtrToInt(fn, i64_ty,
                                               imp->function + ".addr");
        call->setArgOperand(1, fn_addr);
        return true;
      }
    }
  }

  // Priority 2: Direct call to lifted function.
  auto *target_fn = lifted ? lifted->lookup(resolved_pc) : nullptr;
  if (target_fn) {
    llvm::IRBuilder<> Builder(call);
    auto *direct_call = Builder.CreateCall(
        target_fn,
        {call->getArgOperand(0),
         llvm::ConstantInt::get(i64_ty, resolved_pc),
         call->getArgOperand(2)});
    call->replaceAllUsesWith(direct_call);
    call->eraseFromParent();
    return true;
  }

  // Priority 3: Replace target with constant (for downstream passes to handle).
  call->setArgOperand(1, llvm::ConstantInt::get(i64_ty, resolved_pc));
  return true;
}

/// Resolve a dispatch_jump target.
/// Returns true if the call was modified.
bool resolveDispatchJump(llvm::CallInst *call, uint64_t resolved_pc,
                         const LiftedFunctionMap *lifted) {
  auto *F = call->getFunction();

  // Must be followed by ret.
  auto *ret = llvm::dyn_cast<llvm::ReturnInst>(call->getNextNode());
  if (!ret)
    return false;

  auto *BB = call->getParent();
  llvm::SmallVector<llvm::BasicBlock *, 4> old_succs(successors(BB));

  // Priority 1: Intra-function branch.
  if (auto *target_bb = findBlockForPC(*F, resolved_pc)) {
    llvm::IRBuilder<> Builder(call);
    auto *br = Builder.CreateBr(target_bb);

    call->replaceAllUsesWith(llvm::PoisonValue::get(call->getType()));
    ret->eraseFromParent();
    call->eraseFromParent();

    // Clean dead instructions between branch and end of block.
    while (&BB->back() != br) {
      auto &dead = BB->back();
      if (!dead.use_empty())
        dead.replaceAllUsesWith(llvm::PoisonValue::get(dead.getType()));
      dead.eraseFromParent();
    }

    llvm::SmallPtrSet<llvm::BasicBlock *, 4> new_succs(
        successors(BB).begin(), successors(BB).end());
    for (auto *old_succ : old_succs)
      if (!new_succs.count(old_succ))
        old_succ->removePredecessor(BB);

    return true;
  }

  // Priority 2: Inter-function musttail call.
  auto *target_fn = lifted ? lifted->lookup(resolved_pc) : nullptr;
  if (target_fn) {
    auto &Ctx = F->getContext();
    auto *i64_ty = llvm::Type::getInt64Ty(Ctx);

    llvm::IRBuilder<> Builder(call);
    auto *tail_call = Builder.CreateCall(
        target_fn,
        {call->getArgOperand(0),
         llvm::ConstantInt::get(i64_ty, resolved_pc),
         call->getArgOperand(2)});
    tail_call->setTailCallKind(llvm::CallInst::TCK_MustTail);
    auto *new_ret = Builder.CreateRet(tail_call);

    call->replaceAllUsesWith(llvm::PoisonValue::get(call->getType()));
    ret->eraseFromParent();
    call->eraseFromParent();

    while (&BB->back() != new_ret) {
      auto &dead = BB->back();
      if (!dead.use_empty())
        dead.replaceAllUsesWith(llvm::PoisonValue::get(dead.getType()));
      dead.eraseFromParent();
    }

    llvm::SmallPtrSet<llvm::BasicBlock *, 4> new_succs(
        successors(BB).begin(), successors(BB).end());
    for (auto *old_succ : old_succs)
      if (!new_succs.count(old_succ))
        old_succ->removePredecessor(BB);

    return true;
  }

  // Priority 3: Replace target with constant for downstream passes.
  auto *i64_ty = llvm::Type::getInt64Ty(call->getContext());
  call->setArgOperand(1, llvm::ConstantInt::get(i64_ty, resolved_pc));
  return true;
}

}  // namespace

llvm::PreservedAnalyses IndirectCallResolverPass::run(
    llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  if (F.isDeclaration())
    return llvm::PreservedAnalyses::all();

  auto &MAMProxy = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  auto *map = MAMProxy.getCachedResult<BinaryMemoryAnalysis>(*F.getParent());
  if (!map)
    return llvm::PreservedAnalyses::all();

  auto *lifted =
      MAMProxy.getCachedResult<LiftedFunctionAnalysis>(*F.getParent());

  // Collect candidates: dispatch_call and dispatch_jump with non-constant,
  // non-resolved targets.
  struct Candidate {
    llvm::CallInst *call;
    bool is_jump;
  };
  llvm::SmallVector<Candidate, 8> candidates;

  for (auto &BB : F) {
    for (auto &I : BB) {
      auto *call = llvm::dyn_cast<llvm::CallInst>(&I);
      if (!call)
        continue;
      auto *callee = call->getCalledFunction();
      if (!callee)
        continue;
      if (call->arg_size() < 3)
        continue;

      bool is_call = (callee->getName() == "__omill_dispatch_call");
      bool is_jump = (callee->getName() == "__omill_dispatch_jump");
      if (!is_call && !is_jump)
        continue;

      auto *target = call->getArgOperand(1);
      if (isAlreadyResolved(target))
        continue;

      candidates.push_back({call, is_jump});
    }
  }

  if (candidates.empty())
    return llvm::PreservedAnalyses::all();

  bool changed = false;

  for (auto &cand : candidates) {
    auto *target = cand.call->getArgOperand(1);
    auto resolved = evaluateToConstant(target, *map);
    // Monte Carlo fallback: if the deterministic evaluator fails (e.g., VM
    // handler MBA with symbolic operands that cancel out), try concrete
    // evaluation with random values for unknowns.
    if (!resolved)
      resolved = tryMonteCarloResolve(target, F, map);
    if (!resolved) {
      continue;
    }
    if (cand.is_jump) {
      changed |= resolveDispatchJump(cand.call, *resolved, lifted);
    } else {
      changed |= resolveDispatchCall(cand.call, *resolved, map, lifted);
    }
  }

  return changed ? llvm::PreservedAnalyses::none()
                 : llvm::PreservedAnalyses::all();
}

}  // namespace omill
