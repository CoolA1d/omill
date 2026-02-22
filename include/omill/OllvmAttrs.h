#pragma once

/// @file OllvmAttrs.h
/// Clang-compatible function annotations for OLLVM obfuscation passes.
///
/// Usage:
///   OLLVM_FLATTEN void my_function() { ... }
///   OLLVM_FLATTEN OLLVM_SUBSTITUTE void hardened() { ... }

#if defined(__clang__) || defined(__GNUC__)
  #define OLLVM_ANNOTATE(x) __attribute__((annotate(x)))
#elif defined(_MSC_VER)
  // clang-cl supports __attribute__((annotate(...))); pure MSVC does not.
  // When compiling with cl.exe the macros expand to nothing — annotations are
  // only meaningful when producing LLVM IR via Clang.
  #define OLLVM_ANNOTATE(x)
#else
  #define OLLVM_ANNOTATE(x)
#endif

#define OLLVM_FLATTEN         OLLVM_ANNOTATE("ollvm_flatten")
#define OLLVM_SUBSTITUTE      OLLVM_ANNOTATE("ollvm_substitute")
#define OLLVM_STRING_ENCRYPT  OLLVM_ANNOTATE("ollvm_string_encrypt")
#define OLLVM_CONST_UNFOLD    OLLVM_ANNOTATE("ollvm_const_unfold")
#define OLLVM_VECTORIZE       OLLVM_ANNOTATE("ollvm_vectorize")
#define OLLVM_OPAQUE_PRED     OLLVM_ANNOTATE("ollvm_opaque_predicates")
#define OLLVM_BOGUS_CF        OLLVM_ANNOTATE("ollvm_bogus_cf")
#define OLLVM_ALL             OLLVM_ANNOTATE("ollvm_all")
#define OLLVM_NONE            OLLVM_ANNOTATE("ollvm_none")
