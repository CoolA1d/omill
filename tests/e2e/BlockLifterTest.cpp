/// \file BlockLifterTest.cpp
/// \brief End-to-end tests for the per-block lifter.
///
/// These tests create a BlockManager backed by raw x86-64 machine code
/// and verify that BlockLifter produces correct block-function IR.

#include <gtest/gtest.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>

#include <remill/Arch/Arch.h>
#include <remill/Arch/Name.h>
#include <remill/BC/Util.h>
#include <remill/OS/OS.h>

#include "omill/BC/BlockLifter.h"

#include <cstdint>
#include <map>

namespace {

/// Simple BlockManager backed by a contiguous byte buffer at a given VA.
class TestBlockManager : public omill::BlockManager {
 public:
  TestBlockManager() = default;

  void AddCode(uint64_t addr, const std::vector<uint8_t> &bytes) {
    for (size_t i = 0; i < bytes.size(); ++i) {
      code_[addr + i] = bytes[i];
    }
  }

  bool TryReadExecutableByte(uint64_t addr, uint8_t *byte) override {
    auto it = code_.find(addr);
    if (it == code_.end()) return false;
    *byte = it->second;
    return true;
  }

  void SetLiftedBlockDefinition(uint64_t addr,
                                llvm::Function *fn) override {
    blocks_[addr] = fn;
  }

  llvm::Function *GetLiftedBlockDeclaration(uint64_t addr) override {
    auto it = blocks_.find(addr);
    return (it != blocks_.end()) ? it->second : nullptr;
  }

  llvm::Function *GetLiftedBlockDefinition(uint64_t addr) override {
    auto it = blocks_.find(addr);
    if (it == blocks_.end()) return nullptr;
    return it->second->isDeclaration() ? nullptr : it->second;
  }

  const omill::BlockMap &GetBlocks() const { return blocks_; }

 private:
  std::map<uint64_t, uint8_t> code_;
  omill::BlockMap blocks_;
};

class BlockLifterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    arch_ = remill::Arch::Get(ctx_, remill::kOSWindows,
                              remill::kArchAMD64_AVX);
    ASSERT_NE(arch_, nullptr);

    module_ = remill::LoadArchSemantics(arch_.get());
    ASSERT_NE(module_, nullptr) << "Failed to load arch semantics";
  }

  llvm::LLVMContext ctx_;
  std::unique_ptr<const remill::Arch> arch_;
  std::unique_ptr<llvm::Module> module_;
};

// -----------------------------------------------------------------------
// Test: Single-instruction RET block
// -----------------------------------------------------------------------
TEST_F(BlockLifterTest, SingleRetBlock) {
  TestBlockManager mgr;
  // x86-64: ret = 0xC3
  mgr.AddCode(0x1000, {0xC3});

  omill::BlockLifter lifter(arch_.get(), mgr);

  llvm::SmallVector<uint64_t, 4> targets;
  auto *fn = lifter.LiftBlock(0x1000, targets);
  ASSERT_NE(fn, nullptr);
  EXPECT_FALSE(fn->isDeclaration());
  EXPECT_EQ(fn->getName(), "blk_1000");

  // A return block should have no successor targets.
  EXPECT_TRUE(targets.empty());

  // Verify the function is valid IR.
  std::string err;
  llvm::raw_string_ostream err_os(err);
  EXPECT_FALSE(llvm::verifyFunction(*fn, &err_os))
      << "Verifier failed: " << err;
}

// -----------------------------------------------------------------------
// Test: Two-instruction NOP + RET block
// -----------------------------------------------------------------------
TEST_F(BlockLifterTest, NopThenRet) {
  TestBlockManager mgr;
  // nop (0x90) + ret (0xC3)
  mgr.AddCode(0x2000, {0x90, 0xC3});

  omill::BlockLifter lifter(arch_.get(), mgr);

  llvm::SmallVector<uint64_t, 4> targets;
  auto *fn = lifter.LiftBlock(0x2000, targets);
  ASSERT_NE(fn, nullptr);
  EXPECT_TRUE(targets.empty());  // ret, no successors

  std::string err;
  llvm::raw_string_ostream err_os(err);
  EXPECT_FALSE(llvm::verifyFunction(*fn, &err_os)) << err;
}

// -----------------------------------------------------------------------
// Test: Direct unconditional JMP → discovers successor
// -----------------------------------------------------------------------
TEST_F(BlockLifterTest, DirectJumpDiscoversTarget) {
  TestBlockManager mgr;
  // jmp rel32 to 0x3010:  E9 <offset>
  // At 0x3000: E9 0B 00 00 00 → jumps to 0x3000 + 5 + 0x0B = 0x3010
  mgr.AddCode(0x3000, {0xE9, 0x0B, 0x00, 0x00, 0x00});
  // Target block: ret
  mgr.AddCode(0x3010, {0xC3});

  omill::BlockLifter lifter(arch_.get(), mgr);

  llvm::SmallVector<uint64_t, 4> targets;
  auto *fn = lifter.LiftBlock(0x3000, targets);
  ASSERT_NE(fn, nullptr);

  // Should discover 0x3010 as a successor.
  ASSERT_EQ(targets.size(), 1u);
  EXPECT_EQ(targets[0], 0x3010u);

  // The block-function should end with a musttail call to blk_3010.
  bool found_musttail = false;
  for (auto &BB : *fn) {
    if (auto *ret = llvm::dyn_cast<llvm::ReturnInst>(BB.getTerminator())) {
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(ret->getPrevNode())) {
        if (call->isMustTailCall()) {
          found_musttail = true;
          auto *callee = call->getCalledFunction();
          ASSERT_NE(callee, nullptr);
          EXPECT_TRUE(callee->getName().starts_with("blk_"));
        }
      }
    }
  }
  EXPECT_TRUE(found_musttail) << "Expected musttail call to successor block";

  std::string err;
  llvm::raw_string_ostream err_os(err);
  EXPECT_FALSE(llvm::verifyFunction(*fn, &err_os)) << err;
}

// -----------------------------------------------------------------------
// Test: Conditional branch → discovers both targets
// -----------------------------------------------------------------------
TEST_F(BlockLifterTest, ConditionalBranchDiscoversBothTargets) {
  TestBlockManager mgr;
  // cmp eax, eax (sets ZF=1) + je rel32
  // At 0x4000: 39 C0           → cmp eax, eax
  // At 0x4002: 0F 84 08 00 00 00 → je 0x4002+6+8 = 0x4010
  // Fall-through at 0x4008: ret
  mgr.AddCode(0x4000, {0x39, 0xC0, 0x0F, 0x84, 0x08, 0x00, 0x00, 0x00,
                        0xC3});
  mgr.AddCode(0x4010, {0xC3});

  omill::BlockLifter lifter(arch_.get(), mgr);

  llvm::SmallVector<uint64_t, 4> targets;
  auto *fn = lifter.LiftBlock(0x4000, targets);
  ASSERT_NE(fn, nullptr);

  // Should discover both branch targets.
  // Note: the cmp+je is two instructions lifted into the same block-function.
  // The taken target (0x4010) and not-taken target (0x4008) should both appear.
  EXPECT_GE(targets.size(), 2u);

  // Both targets should be present.
  bool has_taken = false, has_not_taken = false;
  for (auto t : targets) {
    if (t == 0x4010) has_taken = true;
    if (t == 0x4008) has_not_taken = true;
  }
  EXPECT_TRUE(has_taken) << "Missing taken target 0x4010";
  EXPECT_TRUE(has_not_taken) << "Missing not-taken target 0x4008";

  std::string err;
  llvm::raw_string_ostream err_os(err);
  EXPECT_FALSE(llvm::verifyFunction(*fn, &err_os)) << err;
}

// -----------------------------------------------------------------------
// Test: LiftReachable follows direct jumps
// -----------------------------------------------------------------------
TEST_F(BlockLifterTest, LiftReachableFollowsDirectJumps) {
  TestBlockManager mgr;
  // Block 0x5000: jmp 0x5010  (E9 0B 00 00 00)
  mgr.AddCode(0x5000, {0xE9, 0x0B, 0x00, 0x00, 0x00});
  // Block 0x5010: jmp 0x5020  (E9 0B 00 00 00)
  mgr.AddCode(0x5010, {0xE9, 0x0B, 0x00, 0x00, 0x00});
  // Block 0x5020: ret
  mgr.AddCode(0x5020, {0xC3});

  omill::BlockLifter lifter(arch_.get(), mgr);
  unsigned count = lifter.LiftReachable(0x5000);

  // Should have lifted 3 blocks.
  EXPECT_EQ(count, 3u);

  auto &blocks = mgr.GetBlocks();
  EXPECT_NE(blocks.count(0x5000), 0u);
  EXPECT_NE(blocks.count(0x5010), 0u);
  EXPECT_NE(blocks.count(0x5020), 0u);

  // All should be defined (non-declaration).
  for (auto &[pc, fn] : blocks) {
    EXPECT_FALSE(fn->isDeclaration())
        << "Block at 0x" << std::hex << pc << " should be defined";
  }
}

// -----------------------------------------------------------------------
// Test: Indirect jump produces __omill_dispatch_jump
// -----------------------------------------------------------------------
TEST_F(BlockLifterTest, IndirectJumpProducesDispatch) {
  TestBlockManager mgr;
  // jmp rax = FF E0
  mgr.AddCode(0x6000, {0xFF, 0xE0});

  omill::BlockLifter lifter(arch_.get(), mgr);

  llvm::SmallVector<uint64_t, 4> targets;
  auto *fn = lifter.LiftBlock(0x6000, targets);
  ASSERT_NE(fn, nullptr);

  // Should NOT have discovered any targets (no devirtualization info).
  EXPECT_TRUE(targets.empty());

  // Should contain a call to __omill_dispatch_jump.
  bool found_dispatch = false;
  for (auto &BB : *fn) {
    for (auto &I : BB) {
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(&I)) {
        if (auto *callee = call->getCalledFunction()) {
          if (callee->getName() == "__omill_dispatch_jump") {
            found_dispatch = true;
          }
        }
      }
    }
  }
  EXPECT_TRUE(found_dispatch)
      << "Expected call to __omill_dispatch_jump for indirect jmp";

  std::string err;
  llvm::raw_string_ostream err_os(err);
  EXPECT_FALSE(llvm::verifyFunction(*fn, &err_os)) << err;
}

// -----------------------------------------------------------------------
// Test: Direct CALL → dispatch_call + musttail to fall-through
// -----------------------------------------------------------------------
TEST_F(BlockLifterTest, DirectCallEmitsDispatchAndFallthrough) {
  TestBlockManager mgr;
  // call rel32 to 0x7100:  E8 <offset>
  // At 0x7000: E8 FB 00 00 00 → call 0x7000 + 5 + 0xFB = 0x7100
  // Fall-through at 0x7005: ret
  mgr.AddCode(0x7000, {0xE8, 0xFB, 0x00, 0x00, 0x00, 0xC3});
  mgr.AddCode(0x7100, {0xC3});

  omill::BlockLifter lifter(arch_.get(), mgr);

  llvm::SmallVector<uint64_t, 4> targets;
  auto *fn = lifter.LiftBlock(0x7000, targets);
  ASSERT_NE(fn, nullptr);

  // Should discover the fall-through (0x7005).
  bool has_fallthrough = false;
  for (auto t : targets) {
    if (t == 0x7005) has_fallthrough = true;
  }
  EXPECT_TRUE(has_fallthrough) << "Missing fall-through target 0x7005";

  // Should contain __omill_dispatch_call.
  bool found_dispatch_call = false;
  for (auto &BB : *fn) {
    for (auto &I : BB) {
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(&I)) {
        if (auto *callee = call->getCalledFunction()) {
          if (callee->getName() == "__omill_dispatch_call") {
            found_dispatch_call = true;
          }
        }
      }
    }
  }
  EXPECT_TRUE(found_dispatch_call) << "Expected __omill_dispatch_call";

  std::string err;
  llvm::raw_string_ostream err_os(err);
  EXPECT_FALSE(llvm::verifyFunction(*fn, &err_os)) << err;
}

// -----------------------------------------------------------------------
// Test: Duplicate LiftBlock call returns same function
// -----------------------------------------------------------------------
TEST_F(BlockLifterTest, DuplicateLiftBlockReturnsCached) {
  TestBlockManager mgr;
  mgr.AddCode(0x8000, {0xC3});

  omill::BlockLifter lifter(arch_.get(), mgr);

  llvm::SmallVector<uint64_t, 4> t1, t2;
  auto *fn1 = lifter.LiftBlock(0x8000, t1);
  auto *fn2 = lifter.LiftBlock(0x8000, t2);
  EXPECT_EQ(fn1, fn2) << "Second LiftBlock should return cached function";
}

}  // namespace
