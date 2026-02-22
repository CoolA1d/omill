#include <cstdint>

#define EXPORT extern "C" __declspec(dllexport)

// ---------------------------------------------------------------------------
// Pure-computation functions for deobfuscation round-trip testing.
// Each function is small, self-contained, uses no global data (unless noted),
// and has a deterministic signature that can be verified via JIT.
// ---------------------------------------------------------------------------

/// Identity — baseline sanity check.
EXPORT int rt_identity(int x) {
  return x;
}

/// Collatz stopping time — loop with conditional branches (CFF target).
EXPORT int rt_collatz_steps(int n) {
  if (n <= 0) return -1;
  int steps = 0;
  while (n != 1) {
    if (n % 2 == 0)
      n = n / 2;
    else
      n = 3 * n + 1;
    steps++;
  }
  return steps;
}

/// Pack four bytes into a uint64_t — shifts + OR (MBA / vectorize target).
EXPORT uint64_t rt_bitfield_pack(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  return (static_cast<uint64_t>(a))
       | (static_cast<uint64_t>(b) << 8)
       | (static_cast<uint64_t>(c) << 16)
       | (static_cast<uint64_t>(d) << 24);
}

/// Extract byte N (0-3) from a uint64_t — shifts + AND + mask.
EXPORT uint8_t rt_bitfield_unpack(uint64_t val, int idx) {
  return static_cast<uint8_t>((val >> (idx * 8)) & 0xFF);
}

/// Iterative Fibonacci — simple loop (CFF / const-unfold target).
EXPORT int rt_fibonacci(int n) {
  if (n <= 0) return 0;
  if (n == 1) return 1;
  int a = 0, b = 1;
  for (int i = 2; i <= n; i++) {
    int t = a + b;
    a = b;
    b = t;
  }
  return b;
}

/// GCD via Euclidean algorithm — while loop with modulo.
EXPORT int rt_gcd(int a, int b) {
  if (a < 0) a = -a;
  if (b < 0) b = -b;
  while (b != 0) {
    int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

/// Multi-way classification — switch statement (CFF target).
EXPORT int rt_switch_classify(int x) {
  if (x < 0)        return 0;  // negative
  else if (x == 0)   return 1;  // zero
  else if (x < 10)   return 2;  // small
  else if (x < 100)  return 3;  // medium
  else if (x < 1000) return 4;  // large
  else               return 5;  // huge
}

/// Deeply nested branches — opaque predicate / bogus CF target.
EXPORT int rt_nested_branch(int x, int y) {
  int result = 0;
  if (x > 0) {
    if (y > 0) {
      result = x + y;
    } else if (y == 0) {
      result = x * 2;
    } else {
      result = x - y;
    }
  } else if (x == 0) {
    if (y > 0) {
      result = y * 3;
    } else {
      result = -1;
    }
  } else {
    if (y > 0) {
      result = y - x;
    } else {
      result = x * y;
    }
  }
  return result;
}

/// Population count — bitwise manipulation (vectorize / MBA target).
EXPORT int rt_popcount(uint32_t v) {
  int count = 0;
  while (v) {
    count += v & 1;
    v >>= 1;
  }
  return count;
}

/// In-place XOR cipher — loop + XOR (string-encrypt / const-unfold target).
/// Returns the sum of output bytes as a checksum.
EXPORT int rt_xor_cipher(uint8_t *buf, int len, uint8_t key) {
  int sum = 0;
  for (int i = 0; i < len; i++) {
    buf[i] ^= key;
    sum += buf[i];
  }
  return sum;
}

/// Sum of an integer array — pointer + loop (memory coalesce target).
EXPORT int rt_array_sum(const int *arr, int n) {
  int sum = 0;
  for (int i = 0; i < n; i++)
    sum += arr[i];
  return sum;
}

/// Trace of an NxN matrix (sum of diagonal) — strided access pattern.
EXPORT int rt_matrix_trace(const int *mat, int n) {
  int trace = 0;
  for (int i = 0; i < n; i++)
    trace += mat[i * n + i];
  return trace;
}
