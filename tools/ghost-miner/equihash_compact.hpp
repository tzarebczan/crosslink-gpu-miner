#pragma once
/*
 * Equihash(200,9) helpers shared between main.cpp and stratum.cpp.
 *
 * Includes:
 *   - Blake2b-256 (host) for verifying header hash <= target before submit
 *   - Equihash compact-solution encoding (512 indices x 21 bits packed
 *     big-endian into 1344 bytes), matching ZcashStratum/CompressArray
 *   - Hex helpers
 */

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ghost {

// ---------------------------------------------------------------- hex helpers

inline std::string to_hex(const uint8_t* data, size_t len)
{
    static const char* digits = "0123456789abcdef";
    std::string out(len * 2, '0');
    for (size_t i = 0; i < len; i++) {
        out[2 * i]     = digits[(data[i] >> 4) & 0xF];
        out[2 * i + 1] = digits[data[i] & 0xF];
    }
    return out;
}

inline std::vector<uint8_t> from_hex(const std::string& s)
{
    if (s.size() % 2 != 0) throw std::runtime_error("from_hex: odd length");
    std::vector<uint8_t> out(s.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
        auto nib = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            throw std::runtime_error("from_hex: bad nibble");
        };
        out[i] = (uint8_t)((nib(s[2 * i]) << 4) | nib(s[2 * i + 1]));
    }
    return out;
}

// --------------------------------------------------- compact solution encoding
//
// Pack 512 indices, each (cBitLen+1) = 21 bits, big-endian into a 1344-byte
// buffer. Mirrors ZcashStratum.cpp's CompressArray + GetMinimalFromIndices.
inline std::vector<uint8_t> compress_indices(const std::vector<uint32_t>& indices,
                                             size_t cBitLen = 20)
{
    if (indices.empty()) return {};
    const size_t bit_len  = cBitLen + 1;        // 21
    const size_t byte_pad = sizeof(uint32_t) - (bit_len + 7) / 8;  // 4-3=1
    const size_t in_width = (bit_len + 7) / 8 + byte_pad;          // 4
    const size_t in_len   = indices.size() * sizeof(uint32_t);     // 2048
    const size_t out_len  = bit_len * in_len / (8 * in_width);     // 1344

    std::vector<uint8_t> in(in_len);
    for (size_t i = 0; i < indices.size(); i++) {
        uint32_t bei = ((indices[i] & 0xFF000000u) >> 24) |
                       ((indices[i] & 0x00FF0000u) >> 8)  |
                       ((indices[i] & 0x0000FF00u) << 8)  |
                       ((indices[i] & 0x000000FFu) << 24);
        std::memcpy(&in[i * sizeof(uint32_t)], &bei, sizeof(uint32_t));
    }

    std::vector<uint8_t> out(out_len);
    const uint32_t bit_len_mask = ((uint32_t)1 << bit_len) - 1;
    size_t   acc_bits  = 0;
    uint32_t acc_value = 0;
    size_t   j = 0;
    for (size_t i = 0; i < out_len; i++) {
        if (acc_bits < 8) {
            acc_value = acc_value << bit_len;
            for (size_t x = byte_pad; x < in_width; x++) {
                acc_value |= ((uint32_t)in[j + x] &
                              ((bit_len_mask >> (8 * (in_width - x - 1))) & 0xFF))
                             << (8 * (in_width - x - 1));
            }
            j += in_width;
            acc_bits += bit_len;
        }
        acc_bits -= 8;
        out[i] = (uint8_t)((acc_value >> acc_bits) & 0xFF);
    }
    return out;
}

// ---------------------------------------------------------- Zcash VarInt prefix
inline std::vector<uint8_t> varint_prefix(uint64_t n)
{
    std::vector<uint8_t> out;
    if (n < 253) {
        out.push_back((uint8_t)n);
    } else if (n <= 0xFFFFu) {
        out.push_back(0xfd);
        out.push_back((uint8_t)(n & 0xff));
        out.push_back((uint8_t)((n >> 8) & 0xff));
    } else if (n <= 0xFFFFFFFFul) {
        out.push_back(0xfe);
        for (int i = 0; i < 4; i++) out.push_back((uint8_t)((n >> (i * 8)) & 0xff));
    } else {
        out.push_back(0xff);
        for (int i = 0; i < 8; i++) out.push_back((uint8_t)((n >> (i * 8)) & 0xff));
    }
    return out;
}

// ----------------------------------------------------------- Blake2b (generic)
//
// Generic Blake2b for arbitrary digest_len in [1..64] bytes, single-block
// messages (we only ever feed at most 140 bytes + 1344-byte solution).
// Matches RFC 7693. Used to compute the block-header hash for local
// share validation against a target.

struct Blake2b {
    uint64_t h[8];
    uint64_t t[2];
    uint8_t  buf[128];
    size_t   buflen;
    size_t   outlen;
};

inline uint64_t bb_rotr64(uint64_t x, unsigned n) { return (x >> n) | (x << (64 - n)); }

inline void bb_compress(Blake2b& s, bool last, const uint8_t* block)
{
    static const uint64_t IV[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
    };
    static const uint8_t SIG[12][16] = {
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
        {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
        {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},
        {7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},
        {9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},
        {2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},
        {12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11},
        {13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},
        {6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5},
        {10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0},
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
        {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
    };
    uint64_t v[16];
    uint64_t m[16];
    for (int i = 0; i < 16; i++) {
        std::memcpy(&m[i], block + i * 8, 8);
    }
    for (int i = 0; i < 8; i++) v[i]     = s.h[i];
    for (int i = 0; i < 8; i++) v[i + 8] = IV[i];
    v[12] ^= s.t[0];
    v[13] ^= s.t[1];
    if (last) v[14] = ~v[14];
#define G(a,b,c,d,x,y) do { \
    a = a + b + (x); d = bb_rotr64(d ^ a, 32); \
    c = c + d;       b = bb_rotr64(b ^ c, 24); \
    a = a + b + (y); d = bb_rotr64(d ^ a, 16); \
    c = c + d;       b = bb_rotr64(b ^ c, 63); \
} while (0)
    for (int r = 0; r < 12; r++) {
        const uint8_t* p = SIG[r];
        G(v[0], v[4], v[ 8], v[12], m[p[ 0]], m[p[ 1]]);
        G(v[1], v[5], v[ 9], v[13], m[p[ 2]], m[p[ 3]]);
        G(v[2], v[6], v[10], v[14], m[p[ 4]], m[p[ 5]]);
        G(v[3], v[7], v[11], v[15], m[p[ 6]], m[p[ 7]]);
        G(v[0], v[5], v[10], v[15], m[p[ 8]], m[p[ 9]]);
        G(v[1], v[6], v[11], v[12], m[p[10]], m[p[11]]);
        G(v[2], v[7], v[ 8], v[13], m[p[12]], m[p[13]]);
        G(v[3], v[4], v[ 9], v[14], m[p[14]], m[p[15]]);
    }
#undef G
    for (int i = 0; i < 8; i++) s.h[i] ^= v[i] ^ v[i + 8];
}

inline void blake2b_init(Blake2b& s, size_t outlen)
{
    static const uint64_t IV[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
    };
    for (int i = 0; i < 8; i++) s.h[i] = IV[i];
    s.h[0] ^= 0x01010000ULL ^ (uint64_t)outlen;
    s.t[0] = s.t[1] = 0;
    s.buflen = 0;
    s.outlen = outlen;
}

inline void blake2b_update(Blake2b& s, const uint8_t* in, size_t in_len)
{
    while (in_len > 0) {
        if (s.buflen == 128) {
            s.t[0] += 128;
            if (s.t[0] < 128) s.t[1]++;
            bb_compress(s, false, s.buf);
            s.buflen = 0;
        }
        size_t left = 128 - s.buflen;
        size_t take = in_len < left ? in_len : left;
        std::memcpy(s.buf + s.buflen, in, take);
        s.buflen += take;
        in       += take;
        in_len   -= take;
    }
}

inline void blake2b_final(Blake2b& s, uint8_t* out)
{
    s.t[0] += s.buflen;
    if (s.t[0] < s.buflen) s.t[1]++;
    std::memset(s.buf + s.buflen, 0, 128 - s.buflen);
    bb_compress(s, true, s.buf);
    for (size_t i = 0; i < s.outlen; i++) {
        out[i] = (uint8_t)(s.h[i / 8] >> ((i % 8) * 8));
    }
}

// Convenience wrapper.
inline std::array<uint8_t, 32> blake2b_256(const std::vector<uint8_t>& in)
{
    Blake2b s;
    blake2b_init(s, 32);
    blake2b_update(s, in.data(), in.size());
    std::array<uint8_t, 32> out;
    blake2b_final(s, out.data());
    return out;
}

// ------------------------------------------------------- SHA-256 (FIPS 180-4)
//
// Self-contained SHA-256 + SHA-256d used for the stratum share hash.
//
// SNOMP-derived Zcash pools compute share diff as
//     headerHash = sha256d(header140 || varint(1344) || solution_compact)
// and compare that against the stratum-target advertised via mining.set_target.
// Note: this is DIFFERENT from the network PoW hash (blake2b-256 of the 140-B
// header alone) -- it's a historical SNOMP quirk. Miners that use blake2b for
// the share-diff check will see 100% "low difficulty" rejects because the two
// hash functions produce uncorrelated values over the same input bytes.

namespace sha256_detail {
inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static constexpr uint32_t K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};
inline void compress(uint32_t state[8], const uint8_t block[64])
{
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4]     << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] <<  8) |
               ((uint32_t)block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}
} // namespace sha256_detail

inline std::array<uint8_t, 32> sha256(const uint8_t* data, size_t len)
{
    uint32_t state[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    uint64_t bitlen = (uint64_t)len * 8;
    size_t i = 0;
    for (; i + 64 <= len; i += 64) sha256_detail::compress(state, data + i);

    uint8_t pad[128] = {0};
    size_t tail = len - i;
    std::memcpy(pad, data + i, tail);
    pad[tail] = 0x80;
    size_t padlen = (tail < 56) ? 64 : 128;
    for (int j = 0; j < 8; j++) pad[padlen - 1 - j] = (uint8_t)(bitlen >> (j * 8));
    sha256_detail::compress(state, pad);
    if (padlen == 128) sha256_detail::compress(state, pad + 64);

    std::array<uint8_t, 32> out;
    for (int j = 0; j < 8; j++) {
        out[j * 4]     = (uint8_t)(state[j] >> 24);
        out[j * 4 + 1] = (uint8_t)(state[j] >> 16);
        out[j * 4 + 2] = (uint8_t)(state[j] >>  8);
        out[j * 4 + 3] = (uint8_t)(state[j]);
    }
    return out;
}

inline std::array<uint8_t, 32> sha256d(const uint8_t* data, size_t len)
{
    auto h1 = sha256(data, len);
    return sha256(h1.data(), h1.size());
}

inline std::array<uint8_t, 32> sha256d(const std::vector<uint8_t>& v)
{
    return sha256d(v.data(), v.size());
}

// ----------------------------------------------------------- target compare
//
// Both `hash` and `target` are 32-byte LE (Zcash convention). Returns true
// iff hash < target (i.e. share meets pool diff).
inline bool hash_le_target(const uint8_t hash[32], const uint8_t target[32])
{
    for (int i = 31; i >= 0; i--) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return true;
}

} // namespace ghost
