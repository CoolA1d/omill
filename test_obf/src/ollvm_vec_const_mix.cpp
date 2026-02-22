#include <cstdint>

#define EXPORT extern "C" __declspec(dllexport)

static inline uint32_t rol32(uint32_t v, unsigned n) {
  n &= 31u;
  if (n == 0u)
    return v;
  return (v << n) | (v >> (32u - n));
}

// Mixed workload for OLLVM tests:
// 1) linear algebra-ish 3x3 * vec multiply,
// 2) bit-level transforms (rotate/xor/shift),
// 3) explicit data movement across blocks.
EXPORT uint32_t ollvm_vec_const_mix(const uint32_t *in) {
  // Linear algebra block: matrix-vector products.
  const uint32_t x0 = in[0];
  const uint32_t x1 = in[1];
  const uint32_t x2 = in[2];

  uint32_t d0 = 3u * x0 + 5u * x1 + 7u * x2;
  uint32_t d1 = 11u * x0 + 13u * x1 + 17u * x2;
  uint32_t d2 = 19u * x0 + 23u * x1 + 29u * x2;

  uint32_t lanes[4];
  lanes[0] = d0;
  lanes[1] = d1;
  lanes[2] = d2;
  lanes[3] = d0 ^ d1 ^ d2;

  // Data motion block: branch-driven lane shuffles.
  if (((d0 ^ d1) & 1u) != 0u) {
    uint32_t t = lanes[0];
    lanes[0] = lanes[2];
    lanes[2] = t;
    lanes[3] ^= lanes[1];
  } else {
    lanes[1] += lanes[3];
    lanes[2] ^= lanes[0];
  }

  // Bit-level block: rotate/xor/shift mixing with constants.
  static const uint32_t kMask[4] = {
      0xA3B1BAC6u, 0x56AA3350u, 0x677D9197u, 0xB27022DCu};
  for (unsigned i = 0; i < 4; ++i) {
    uint32_t v = lanes[i];
    v = rol32(v ^ kMask[i], i + 3u);
    v ^= (v >> (i + 1u));
    lanes[i] = v;
  }

  // Final data move + reduction block.
  uint32_t moved[4];
  moved[0] = lanes[1] ^ rol32(lanes[3], 5u);
  moved[1] = lanes[2] + (lanes[0] & 0x00FF00FFu);
  moved[2] = lanes[3] ^ (lanes[1] >> 3u);
  moved[3] = lanes[0] + lanes[2] + 0x9E3779B9u;

  uint32_t out = moved[0];
  out = rol32(out + moved[1], 7u);
  out ^= moved[2];
  out += (moved[3] ^ 0x85EBCA6Bu);
  return out;
}
