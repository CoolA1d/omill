#include "omill/Arch/ArchABI.h"

namespace omill {

ArchABI ArchABI::getWin64ABI() {
  ArchABI abi;
  abi.arch = TargetArch::kX86_64;
  abi.os_name = "windows";
  abi.param_regs = {"RCX", "RDX", "R8", "R9"};
  abi.fp_param_regs = {"XMM0", "XMM1", "XMM2", "XMM3"};
  abi.callee_saved = {"RBX", "RBP", "RDI", "RSI",
                      "R12", "R13", "R14", "R15"};
  abi.stack_pointer = "RSP";
  abi.program_counter = "RIP";
  abi.return_reg = "RAX";
  abi.vec_return_reg = "XMM0";
  abi.shadow_space = 32;
  abi.red_zone = 0;
  abi.stack_alignment = 16;
  abi.volatile_scratch = {"RAX", "R10", "R11"};
  abi.max_vec_params = 4;
  abi.position_based_params = true;
  return abi;
}

ArchABI ArchABI::getAAPCS64DarwinABI() {
  ArchABI abi;
  abi.arch = TargetArch::kAArch64;
  abi.os_name = "darwin";
  // Remill AArch64 uses lowercase register names (matching C struct fields).
  abi.param_regs = {"x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"};
  abi.fp_param_regs = {"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"};
  abi.callee_saved = {"x19", "x20", "x21", "x22", "x23", "x24",
                      "x25", "x26", "x27", "x28", "x29", "x30"};
  abi.stack_pointer = "sp";
  abi.program_counter = "pc";
  abi.return_reg = "x0";
  abi.vec_return_reg = "v0";
  abi.shadow_space = 0;
  abi.red_zone = 128;
  abi.stack_alignment = 16;
  abi.volatile_scratch = {"x9", "x10", "x11", "x12", "x13",
                          "x14", "x15", "x16", "x17", "x18"};
  abi.max_vec_params = 8;
  abi.position_based_params = false;
  return abi;
}

ArchABI ArchABI::getAAPCS64LinuxABI() {
  ArchABI abi = getAAPCS64DarwinABI();
  abi.os_name = "linux";
  abi.red_zone = 0;
  // X18 is reserved as platform register on Linux (TLS).
  // Remove from volatile scratch, add to callee_saved conceptually.
  // In practice, lifted code shouldn't touch X18 at all.
  return abi;
}

ArchABI ArchABI::get(TargetArch arch, llvm::StringRef os) {
  if (arch == TargetArch::kAArch64) {
    if (os.contains("darwin") || os.contains("macos") || os.contains("ios"))
      return getAAPCS64DarwinABI();
    return getAAPCS64LinuxABI();
  }
  // Default: x86_64 / Win64
  return getWin64ABI();
}

}  // namespace omill
