/*
 * ghost-equihash-miner -- amd_hip backend implementation.
 *
 * Single translation unit compiled by hipcc. Produces an MSVC-ABI .obj
 * that the rest of nheqminer (MSVC-built) links against. The public
 * surface is in amd_hip.hpp; everything below runs on AMD GPUs via HIP.
 *
 * This file is a HIP port of amd_opencl/kernels/equihash.cl +
 * amd_opencl/amd_opencl.cpp (the orchestration piece). Scope:
 *   - NR_ROWS_LOG = 20 only.
 *   - Two-phase collision path.
 *   - gfx10xx/gfx11xx/gfx12xx targets (wave32/wave64 both work).
 *
 * Correctness verified standalone via tools/smoke_amd_hip/full_smoke.exe
 * (958/958 valid sols vs the OpenCL backend on gfx1201). This file is
 * essentially the same kernels + host driver, repackaged with the
 * amd_hip struct method implementations bolted on.
 */

#include "amd_hip.hpp"
#include "gpu_compat.h"  /* HIP headers on hipcc, CUDA aliases on plain nvcc */

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* --- Equihash(200,9) parameters (mirrors amd_opencl/kernels/param.h) ------ */
#define EQ_PARAM_N          200
#define EQ_PARAM_K          9
#define EQ_PREFIX           (EQ_PARAM_N / (EQ_PARAM_K + 1))  /* 20 */
#define EQ_NR_INPUTS        (1 << EQ_PREFIX)                 /* 2^20 */
#define EQ_APX_NR_ELMS_LOG  (EQ_PREFIX + 1)                  /* 21 */

#define NR_ROWS_LOG          20
#ifndef OVERHEAD
#define OVERHEAD             6
#endif
#define NR_ROWS              (1 << NR_ROWS_LOG)                          /* 1M */
#define NR_SLOTS             ((1 << (EQ_APX_NR_ELMS_LOG - NR_ROWS_LOG)) * OVERHEAD) /* 2*OVERHEAD */
#define SLOT_LEN             32
#define HT_SIZE              ((size_t)NR_ROWS * NR_SLOTS * SLOT_LEN)

#define ZCASH_BLOCK_HEADER_LEN  140
#define ZCASH_HASH_LEN          50
#define MAX_SOLS                10

/* Row-counter packing: 4 bits/row if NR_SLOTS < 16, else 8 bits. Mirrors
 * amd_opencl/kernels/param.h. */
#if NR_SLOTS < 16
#define BITS_PER_ROW   4
#define ROWS_PER_UINT  8
#define ROW_MASK       0x0F
#else
#define BITS_PER_ROW   8
#define ROWS_PER_UINT  4
#define ROW_MASK       0xFF
#endif

#ifndef AMD_HIP_DEFAULT_LOCAL_WORK_SIZE
#define AMD_HIP_DEFAULT_LOCAL_WORK_SIZE 64
#endif
#define COLL_DATA_SIZE_PER_TH  (NR_SLOTS * 5)

enum class HipCollisionMode : uint8_t {
    Lds = 0,
    Wave32 = 1,
    Wave32Coop = 2,
};

/* --- Device-visible struct ------------------------------------------------ */
struct sols_t {
    uint32_t nr;
    uint32_t likely_invalids;
    uint8_t  valid[MAX_SOLS];
    uint32_t values[MAX_SOLS][1 << EQ_PARAM_K];
};

__device__ __constant__ uint64_t d_blake_iv[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
};

/* --- Shared helpers ------------------------------------------------------- */
static __host__ __device__ __forceinline__ uint64_t rotr64(uint64_t x, int n)
{
    return (x >> n) | (x << (64 - n));
}

#define BLAKE_MIX(a, b, c, d, x, y) do { \
    a = a + b + (x); d = rotr64(d ^ a, 32); \
    c = c + d;       b = rotr64(b ^ c, 24); \
    a = a + b + (y); d = rotr64(d ^ a, 16); \
    c = c + d;       b = rotr64(b ^ c, 63); \
} while (0)

static __host__ __device__ __forceinline__ uint32_t xi_offset_for_round(int round)
{
    return 8 + (round / 2) * 4;
}

static __device__ __forceinline__ uint32_t encode_inputs(uint32_t row, uint32_t s0, uint32_t s1)
{
    return (row << 12) | ((s1 & 0x3f) << 6) | (s0 & 0x3f);
}
static __device__ __forceinline__ uint32_t decode_row(uint32_t ref)   { return ref >> 12; }
static __device__ __forceinline__ uint32_t decode_slot0(uint32_t ref) { return ref & 0x3f; }
static __device__ __forceinline__ uint32_t decode_slot1(uint32_t ref) { return (ref >> 6) & 0x3f; }
static __device__ __forceinline__ void decode_collision_pair(uint32_t pair_index, uint32_t cnt,
                                                             uint32_t& slot_a, uint32_t& slot_b)
{
    slot_a = 0;
    uint32_t row_pairs = cnt - 1;
    while (pair_index >= row_pairs) {
        pair_index -= row_pairs;
        slot_a++;
        row_pairs--;
    }
    slot_b = slot_a + 1 + pair_index;
}

static __device__ __forceinline__ uint64_t half_aligned_long(const uint64_t* p, uint32_t off)
{
    const uint8_t* bp = (const uint8_t*)p + off;
    uint64_t lo = *(const uint32_t*)(bp);
    uint64_t hi = *(const uint32_t*)(bp + 4);
    return lo | (hi << 32);
}
static __device__ __forceinline__ uint32_t well_aligned_int(const uint64_t* p, uint32_t off)
{
    return *(const uint32_t*)((const uint8_t*)p + off);
}

/* --- ht_store (NR_ROWS_LOG=20 specialization) ----------------------------- */
static __device__ uint32_t ht_store(int round, uint8_t* __restrict__ ht, uint32_t i,
    uint64_t xi0, uint64_t xi1, uint64_t xi2, uint64_t xi3, uint32_t* __restrict__ rowCounters)
{
    uint32_t row;
    if (!(round & 1)) {
        row = (xi0 & 0xffff) | ((xi0 & 0xf00000) >> 4);
    } else {
        row = ((xi0 & 0xf0000) >> 0) |
              ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
              ((xi0 & 0xf) << 4)   | ((xi0 & 0xf000)   >> 12);
    }
    xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
    xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
    xi2 = (xi2 >> 16) | (xi3 << (64 - 16));
    uint8_t* p = ht + (size_t)row * NR_SLOTS * SLOT_LEN;
    uint32_t rowIdx = row / ROWS_PER_UINT;
    uint32_t rowOffset = BITS_PER_ROW * (row % ROWS_PER_UINT);
    uint32_t xcnt = atomicAdd(&rowCounters[rowIdx], (uint32_t)(1u << rowOffset));
    xcnt = (xcnt >> rowOffset) & ROW_MASK;
    if (xcnt >= NR_SLOTS) {
        atomicSub(&rowCounters[rowIdx], (uint32_t)(1u << rowOffset));
        return 1;
    }
    uint32_t cnt = xcnt;
    p += cnt * SLOT_LEN + xi_offset_for_round(round);
    *(uint32_t*)(p - 4) = i;
    if (round == 0 || round == 1) {
        *(uint64_t*)(p + 0)  = xi0;
        *(uint64_t*)(p + 8)  = xi1;
        *(uint64_t*)(p + 16) = xi2;
    } else if (round == 2) {
        *(uint32_t*)(p + 0)  = (uint32_t)xi0;
        *(uint64_t*)(p + 4)  = (xi0 >> 32) | (xi1 << 32);
        *(uint64_t*)(p + 12) = (xi1 >> 32) | (xi2 << 32);
    } else if (round == 3) {
        *(uint32_t*)(p + 0)  = (uint32_t)xi0;
        *(uint64_t*)(p + 4)  = (xi0 >> 32) | (xi1 << 32);
        *(uint32_t*)(p + 12) = (uint32_t)(xi1 >> 32);
    } else if (round == 4) {
        *(uint64_t*)(p + 0) = xi0;
        *(uint64_t*)(p + 8) = xi1;
    } else if (round == 5) {
        *(uint64_t*)(p + 0) = xi0;
        *(uint32_t*)(p + 8) = (uint32_t)xi1;
    } else if (round == 6 || round == 7) {
        *(uint32_t*)(p + 0) = (uint32_t)xi0;
        *(uint32_t*)(p + 4) = (uint32_t)(xi0 >> 32);
    } else if (round == 8) {
        *(uint32_t*)(p + 0) = (uint32_t)xi0;
    }
    return 0;
}

static __device__ uint32_t xor_and_store(int round, uint8_t* __restrict__ ht_dst, uint32_t row,
    uint32_t slot_a, uint32_t slot_b,
    const uint64_t* __restrict__ a, const uint64_t* __restrict__ b,
    uint32_t* __restrict__ rowCounters)
{
    uint64_t xi0, xi1, xi2 = 0;
    if (round == 1 || round == 2) {
        xi0 = a[0] ^ b[0];
        xi1 = a[1] ^ b[1];
        xi2 = a[2] ^ b[2];
        if (round == 2) {
            xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
            xi1 = (xi1 >> 8) | (xi2 << (64 - 8));
            xi2 = (xi2 >> 8);
        }
    } else if (round == 3) {
        xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
        xi1 = half_aligned_long(a, 8) ^ half_aligned_long(b, 8);
        xi2 = (uint64_t)(well_aligned_int(a, 16) ^ well_aligned_int(b, 16));
    } else if (round == 4 || round == 5) {
        xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
        xi1 = half_aligned_long(a, 8) ^ half_aligned_long(b, 8);
        xi2 = 0;
        if (round == 4) {
            xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
            xi1 = (xi1 >> 8);
        }
    } else if (round == 6) {
        xi0 = a[0] ^ b[0];
        xi1 = (uint64_t)(*(const uint32_t*)(a + 1) ^ *(const uint32_t*)(b + 1));
        xi2 = 0;
        xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
        xi1 = (xi1 >> 8);
    } else { /* round == 7 or 8 */
        xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
        xi1 = 0;
        xi2 = 0;
        if (round == 8) xi0 = (xi0 >> 8);
    }
    if (!xi0 && !xi1) return 0;
    return ht_store(round, ht_dst, encode_inputs(row, slot_a, slot_b),
        xi0, xi1, xi2, 0, rowCounters);
}

/* --- Kernels -------------------------------------------------------------- */
__global__ void kernel_init_ht(uint32_t* __restrict__ rowCounters)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    rowCounters[gid] = 0;
}

static __device__ void kernel_round0_body(const uint64_t* __restrict__ blake_state,
                                          uint8_t* ht, uint32_t* rowCounters)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t inputs_per_thread = EQ_NR_INPUTS / (gridDim.x * blockDim.x);
    uint32_t input     = tid * inputs_per_thread;
    uint32_t input_end = (tid + 1) * inputs_per_thread;
    uint64_t v[16];
    while (input < input_end) {
        uint64_t word1 = ((uint64_t)input) << 32;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i]     = blake_state[i];
#pragma unroll
        for (int i = 0; i < 8; i++) v[i + 8] = d_blake_iv[i];
        v[12] ^= (uint64_t)(ZCASH_BLOCK_HEADER_LEN + 4);
        v[14] ^= (uint64_t)-1;

        /* 12 rounds; only m[word1_pos] is non-zero per round. */
        /* r1  */ BLAKE_MIX(v[0],v[4],v[ 8],v[12], 0,     word1);
                  BLAKE_MIX(v[1],v[5],v[ 9],v[13], 0,     0);
                  BLAKE_MIX(v[2],v[6],v[10],v[14], 0,     0);
                  BLAKE_MIX(v[3],v[7],v[11],v[15], 0,     0);
                  BLAKE_MIX(v[0],v[5],v[10],v[15], 0,     0);
                  BLAKE_MIX(v[1],v[6],v[11],v[12], 0,     0);
                  BLAKE_MIX(v[2],v[7],v[ 8],v[13], 0,     0);
                  BLAKE_MIX(v[3],v[4],v[ 9],v[14], 0,     0);
        /* r2  */ BLAKE_MIX(v[0],v[4],v[ 8],v[12], 0,     0);
                  BLAKE_MIX(v[1],v[5],v[ 9],v[13], 0,     0);
                  BLAKE_MIX(v[2],v[6],v[10],v[14], 0,     0);
                  BLAKE_MIX(v[3],v[7],v[11],v[15], 0,     0);
                  BLAKE_MIX(v[0],v[5],v[10],v[15], word1, 0);
                  BLAKE_MIX(v[1],v[6],v[11],v[12], 0,     0);
                  BLAKE_MIX(v[2],v[7],v[ 8],v[13], 0,     0);
                  BLAKE_MIX(v[3],v[4],v[ 9],v[14], 0,     0);
        /* r3  */ BLAKE_MIX(v[0],v[4],v[ 8],v[12], 0,     0);
                  BLAKE_MIX(v[1],v[5],v[ 9],v[13], 0,     0);
                  BLAKE_MIX(v[2],v[6],v[10],v[14], 0,     0);
                  BLAKE_MIX(v[3],v[7],v[11],v[15], 0,     0);
                  BLAKE_MIX(v[0],v[5],v[10],v[15], 0,     0);
                  BLAKE_MIX(v[1],v[6],v[11],v[12], 0,     0);
                  BLAKE_MIX(v[2],v[7],v[ 8],v[13], 0,     word1);
                  BLAKE_MIX(v[3],v[4],v[ 9],v[14], 0,     0);
        /* r4  */ BLAKE_MIX(v[0],v[4],v[ 8],v[12], 0,     0);
                  BLAKE_MIX(v[1],v[5],v[ 9],v[13], 0,     word1);
                  BLAKE_MIX(v[2],v[6],v[10],v[14], 0,     0);
                  BLAKE_MIX(v[3],v[7],v[11],v[15], 0,     0);
                  BLAKE_MIX(v[0],v[5],v[10],v[15], 0,     0);
                  BLAKE_MIX(v[1],v[6],v[11],v[12], 0,     0);
                  BLAKE_MIX(v[2],v[7],v[ 8],v[13], 0,     0);
                  BLAKE_MIX(v[3],v[4],v[ 9],v[14], 0,     0);
        /* r5  */ BLAKE_MIX(v[0],v[4],v[ 8],v[12], 0,     0);
                  BLAKE_MIX(v[1],v[5],v[ 9],v[13], 0,     0);
                  BLAKE_MIX(v[2],v[6],v[10],v[14], 0,     0);
                  BLAKE_MIX(v[3],v[7],v[11],v[15], 0,     0);
                  BLAKE_MIX(v[0],v[5],v[10],v[15], 0,     word1);
                  BLAKE_MIX(v[1],v[6],v[11],v[12], 0,     0);
                  BLAKE_MIX(v[2],v[7],v[ 8],v[13], 0,     0);
                  BLAKE_MIX(v[3],v[4],v[ 9],v[14], 0,     0);
        /* r6  */ BLAKE_MIX(v[0],v[4],v[ 8],v[12], 0,     0);
                  BLAKE_MIX(v[1],v[5],v[ 9],v[13], 0,     0);
                  BLAKE_MIX(v[2],v[6],v[10],v[14], 0,     0);
                  BLAKE_MIX(v[3],v[7],v[11],v[15], 0,     0);
                  BLAKE_MIX(v[0],v[5],v[10],v[15], 0,     0);
                  BLAKE_MIX(v[1],v[6],v[11],v[12], 0,     0);
                  BLAKE_MIX(v[2],v[7],v[ 8],v[13], 0,     0);
                  BLAKE_MIX(v[3],v[4],v[ 9],v[14], word1, 0);
        /* r7  */ BLAKE_MIX(v[0],v[4],v[ 8],v[12], 0,     0);
                  BLAKE_MIX(v[1],v[5],v[ 9],v[13], word1, 0);
                  BLAKE_MIX(v[2],v[6],v[10],v[14], 0,     0);
                  BLAKE_MIX(v[3],v[7],v[11],v[15], 0,     0);
                  BLAKE_MIX(v[0],v[5],v[10],v[15], 0,     0);
                  BLAKE_MIX(v[1],v[6],v[11],v[12], 0,     0);
                  BLAKE_MIX(v[2],v[7],v[ 8],v[13], 0,     0);
                  BLAKE_MIX(v[3],v[4],v[ 9],v[14], 0,     0);
        /* r8  */ BLAKE_MIX(v[0],v[4],v[ 8],v[12], 0,     0);
                  BLAKE_MIX(v[1],v[5],v[ 9],v[13], 0,     0);
                  BLAKE_MIX(v[2],v[6],v[10],v[14], 0,     word1);
                  BLAKE_MIX(v[3],v[7],v[11],v[15], 0,     0);
                  BLAKE_MIX(v[0],v[5],v[10],v[15], 0,     0);
                  BLAKE_MIX(v[1],v[6],v[11],v[12], 0,     0);
                  BLAKE_MIX(v[2],v[7],v[ 8],v[13], 0,     0);
                  BLAKE_MIX(v[3],v[4],v[ 9],v[14], 0,     0);
        /* r9  */ BLAKE_MIX(v[0],v[4],v[ 8],v[12], 0,     0);
                  BLAKE_MIX(v[1],v[5],v[ 9],v[13], 0,     0);
                  BLAKE_MIX(v[2],v[6],v[10],v[14], 0,     0);
                  BLAKE_MIX(v[3],v[7],v[11],v[15], 0,     0);
                  BLAKE_MIX(v[0],v[5],v[10],v[15], 0,     0);
                  BLAKE_MIX(v[1],v[6],v[11],v[12], 0,     0);
                  BLAKE_MIX(v[2],v[7],v[ 8],v[13], word1, 0);
                  BLAKE_MIX(v[3],v[4],v[ 9],v[14], 0,     0);
        /* r10 */ BLAKE_MIX(v[0],v[4],v[ 8],v[12], 0,     0);
                  BLAKE_MIX(v[1],v[5],v[ 9],v[13], 0,     0);
                  BLAKE_MIX(v[2],v[6],v[10],v[14], 0,     0);
                  BLAKE_MIX(v[3],v[7],v[11],v[15], word1, 0);
                  BLAKE_MIX(v[0],v[5],v[10],v[15], 0,     0);
                  BLAKE_MIX(v[1],v[6],v[11],v[12], 0,     0);
                  BLAKE_MIX(v[2],v[7],v[ 8],v[13], 0,     0);
                  BLAKE_MIX(v[3],v[4],v[ 9],v[14], 0,     0);
        /* r11 */ BLAKE_MIX(v[0],v[4],v[ 8],v[12], 0,     word1);
                  BLAKE_MIX(v[1],v[5],v[ 9],v[13], 0,     0);
                  BLAKE_MIX(v[2],v[6],v[10],v[14], 0,     0);
                  BLAKE_MIX(v[3],v[7],v[11],v[15], 0,     0);
                  BLAKE_MIX(v[0],v[5],v[10],v[15], 0,     0);
                  BLAKE_MIX(v[1],v[6],v[11],v[12], 0,     0);
                  BLAKE_MIX(v[2],v[7],v[ 8],v[13], 0,     0);
                  BLAKE_MIX(v[3],v[4],v[ 9],v[14], 0,     0);
        /* r12 */ BLAKE_MIX(v[0],v[4],v[ 8],v[12], 0,     0);
                  BLAKE_MIX(v[1],v[5],v[ 9],v[13], 0,     0);
                  BLAKE_MIX(v[2],v[6],v[10],v[14], 0,     0);
                  BLAKE_MIX(v[3],v[7],v[11],v[15], 0,     0);
                  BLAKE_MIX(v[0],v[5],v[10],v[15], word1, 0);
                  BLAKE_MIX(v[1],v[6],v[11],v[12], 0,     0);
                  BLAKE_MIX(v[2],v[7],v[ 8],v[13], 0,     0);
                  BLAKE_MIX(v[3],v[4],v[ 9],v[14], 0,     0);

        uint64_t h[7];
        h[0] = blake_state[0] ^ v[0] ^ v[ 8];
        h[1] = blake_state[1] ^ v[1] ^ v[ 9];
        h[2] = blake_state[2] ^ v[2] ^ v[10];
        h[3] = blake_state[3] ^ v[3] ^ v[11];
        h[4] = blake_state[4] ^ v[4] ^ v[12];
        h[5] = blake_state[5] ^ v[5] ^ v[13];
        h[6] = (blake_state[6] ^ v[6] ^ v[14]) & 0xffff;

        ht_store(0, ht, input * 2,
                 h[0], h[1], h[2], h[3], rowCounters);
        ht_store(0, ht, input * 2 + 1,
                 (h[3] >> 8) | (h[4] << (64 - 8)),
                 (h[4] >> 8) | (h[5] << (64 - 8)),
                 (h[5] >> 8) | (h[6] << (64 - 8)),
                 (h[6] >> 8), rowCounters);
        input++;
    }
}

template<int LWS>
static __device__ void equihash_round_lds(int round,
    uint8_t* ht_src, uint8_t* ht_dst,
    uint8_t* first_words_data, uint32_t* collisionsData, uint32_t* collisionsNum,
    uint32_t* rowCountersSrc, uint32_t* rowCountersDst)
{
    // Note: silentarmy (and this port) set `mask = 0`, making the first-byte
    // equality test a no-op -- every pair (i,j) in the row is emitted. We
    // previously tried removing the LDS first_words write/read as dead code,
    // but that REGRESSED by ~23% on RDNA4 (215 -> 165 Sol/s). The LDS dance
    // apparently helps hide latency of the collisionsData atomicAdd. Keep
    // the original structure.
    uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tlid = threadIdx.x;
    uint8_t* first_words = &first_words_data[(NR_SLOTS + 2) * tlid];
    uint8_t mask = 0;
    uint32_t thCollNum = 0;
    if (tlid == 0) *collisionsNum = 0;
    __syncthreads();

    uint32_t cnt;
    uint32_t xi_offset = xi_offset_for_round(round - 1);
    uint8_t* p = ht_src + (size_t)tid * NR_SLOTS * SLOT_LEN;
    uint32_t rowIdx = tid / ROWS_PER_UINT;
    uint32_t rowOffset = BITS_PER_ROW * (tid % ROWS_PER_UINT);
    cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
    cnt = min(cnt, (uint32_t)NR_SLOTS);
    if (cnt) {
        p += xi_offset;
        for (uint32_t i = 0; i < cnt; i++) {
            first_words[i] = (*(uint8_t*)(p + i * SLOT_LEN)) & mask;
        }
        for (uint32_t i = 0; i < cnt - 1 && thCollNum < COLL_DATA_SIZE_PER_TH; i++) {
            uint8_t data_i = first_words[i];
            uint32_t collision = (tid << 10) | (i << 5) | (i + 1);
            for (uint32_t j = i + 1; j < cnt; j++) {
                if (data_i == first_words[j]) {
                    thCollNum++;
                    uint32_t index = atomicAdd(collisionsNum, 1u);
                    collisionsData[index] = collision;
                }
                collision++;
            }
        }
    }
    __syncthreads();

    uint32_t totalCollisions = *collisionsNum;
    for (uint32_t index = tlid; index < totalCollisions; index += blockDim.x) {
        uint32_t collision = collisionsData[index];
        uint32_t collisionTid = collision >> 10;
        uint32_t i = (collision >> 5) & 0x1F;
        uint32_t j = collision & 0x1F;
        uint8_t* ptr = ht_src + (size_t)collisionTid * NR_SLOTS * SLOT_LEN + xi_offset;
        const uint64_t* a = (const uint64_t*)(ptr + i * SLOT_LEN);
        const uint64_t* b = (const uint64_t*)(ptr + j * SLOT_LEN);
        xor_and_store(round, ht_dst, collisionTid, i, j, a, b, rowCountersDst);
    }
}

template<int LWS>
static __device__ void equihash_round_wave32_row(int round, uint32_t row, uint32_t cnt,
    uint8_t* ht_src, uint8_t* ht_dst, uint32_t* rowCountersDst)
{
    uint32_t lane = threadIdx.x & 31;
    uint32_t xi_offset = xi_offset_for_round(round - 1);
    uint8_t* ptr = ht_src + (size_t)row * NR_SLOTS * SLOT_LEN + xi_offset;
    uint32_t totalCollisions = (cnt * (cnt - 1)) >> 1;
    for (uint32_t pair_index = lane; pair_index < totalCollisions; pair_index += 32) {
        uint32_t slot_a, slot_b;
        decode_collision_pair(pair_index, cnt, slot_a, slot_b);
        const uint64_t* a = (const uint64_t*)(ptr + slot_a * SLOT_LEN);
        const uint64_t* b = (const uint64_t*)(ptr + slot_b * SLOT_LEN);
        xor_and_store(round, ht_dst, row, slot_a, slot_b, a, b, rowCountersDst);
    }
}

template<int LWS>
static __device__ void equihash_round_wave32(int round,
    uint8_t* ht_src, uint8_t* ht_dst,
    uint32_t* rowCountersSrc, uint32_t* rowCountersDst)
{
    static_assert((LWS % 32) == 0, "wave32 prototype requires a multiple-of-32 block size");
    uint32_t lane = threadIdx.x & 31;
    uint32_t wave = threadIdx.x >> 5;
    uint32_t rows_per_block = LWS / 32;
    uint32_t row = blockIdx.x * rows_per_block + wave;
    uint32_t cnt = 0;
    if (lane == 0 && row < NR_ROWS) {
        uint32_t rowIdx = row / ROWS_PER_UINT;
        uint32_t rowOffset = BITS_PER_ROW * (row % ROWS_PER_UINT);
        cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
        cnt = min(cnt, (uint32_t)NR_SLOTS);
    }
    cnt = ghost_shfl(cnt, 0, 32);
    if (row >= NR_ROWS || cnt < 2) return;
    equihash_round_wave32_row<LWS>(round, row, cnt, ht_src, ht_dst, rowCountersDst);
}

/* --- Wave32 cooperative-load variant -------------------------------------
 *
 * The plain wave32 path has each of 32 lanes take a distinct (i,j) pair of
 * the row. For cnt=12 that reads the same slot up to cnt-1 = 11 times from
 * global memory across different lanes doing different pairs; for cnt=2
 * only lane 0 does anything so 31 lanes sit idle. Both are inefficient: the
 * first wastes bandwidth, the second wastes compute.
 *
 * Coop variant: lane L (L < cnt) loads slot L's Xi blob into private
 * registers up front. All 32 lanes then enumerate pairs cooperatively, but
 * fetch slot operands from the owning lane via __shfl (= ds_bpermute on
 * RDNA) instead of global_load. Each slot is fetched once from DRAM per
 * wave, then shuffled as needed. For high cnt rows this cuts the per-row
 * DRAM transactions by ~cnt× relative to the plain wave32 path.
 *
 * The xi_offset alignment varies per round (8, 12, 16, 20, 24 bytes).
 * Round 3 and 6 are 4-byte-aligned so we load via two uint32_t halves;
 * other rounds are 8-byte-aligned. The XOR logic is specialised per round
 * via `if constexpr` against the ROUND template parameter so the unused
 * xi1/xi2 loads and shuffles drop out of late rounds (7, 8 need only xi0).
 */
template<int ROUND>
static __device__ uint32_t xor_and_store_vals(uint8_t* __restrict__ ht_dst, uint32_t row,
    uint32_t slot_a, uint32_t slot_b,
    uint64_t a0, uint64_t a1, uint64_t a2,
    uint64_t b0, uint64_t b1, uint64_t b2,
    uint32_t* __restrict__ rowCounters)
{
    uint64_t xi0, xi1 = 0, xi2 = 0;
    if constexpr (ROUND == 1 || ROUND == 2) {
        xi0 = a0 ^ b0;
        xi1 = a1 ^ b1;
        xi2 = a2 ^ b2;
        if constexpr (ROUND == 2) {
            xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
            xi1 = (xi1 >> 8) | (xi2 << (64 - 8));
            xi2 = (xi2 >> 8);
        }
    } else if constexpr (ROUND == 3) {
        xi0 = a0 ^ b0;
        xi1 = a1 ^ b1;
        xi2 = (uint64_t)((uint32_t)a2 ^ (uint32_t)b2);
    } else if constexpr (ROUND == 4 || ROUND == 5) {
        xi0 = a0 ^ b0;
        xi1 = a1 ^ b1;
        if constexpr (ROUND == 4) {
            xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
            xi1 = (xi1 >> 8);
        }
    } else if constexpr (ROUND == 6) {
        xi0 = a0 ^ b0;
        xi1 = (uint64_t)((uint32_t)a1 ^ (uint32_t)b1);
        xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
        xi1 = (xi1 >> 8);
    } else { /* ROUND == 7 or 8 */
        xi0 = a0 ^ b0;
        if constexpr (ROUND == 8) xi0 = (xi0 >> 8);
    }
    if (!xi0 && !xi1) return 0;
    return ht_store(ROUND, ht_dst, encode_inputs(row, slot_a, slot_b),
        xi0, xi1, xi2, 0, rowCounters);
}

template<int ROUND, int LWS>
static __device__ void equihash_round_wave32_coop(
    uint8_t* ht_src, uint8_t* ht_dst,
    uint32_t* rowCountersSrc, uint32_t* rowCountersDst)
{
    static_assert((LWS % 32) == 0, "coop wave32 requires multiple-of-32 block size");
    uint32_t lane = threadIdx.x & 31;
    uint32_t wave = threadIdx.x >> 5;
    uint32_t rows_per_block = LWS / 32;
    uint32_t row = blockIdx.x * rows_per_block + wave;
    uint32_t cnt = 0;
    if (lane == 0 && row < NR_ROWS) {
        uint32_t rowIdx = row / ROWS_PER_UINT;
        uint32_t rowOffset = BITS_PER_ROW * (row % ROWS_PER_UINT);
        cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
        cnt = min(cnt, (uint32_t)NR_SLOTS);
    }
    cnt = ghost_shfl(cnt, 0, 32);
    if (row >= NR_ROWS || cnt < 2) return;

    constexpr uint32_t xi_offset = 8 + ((ROUND - 1) / 2) * 4;
    uint8_t* row_base = ht_src + (size_t)row * NR_SLOTS * SLOT_LEN;

    /* Cooperative load: lanes 0..cnt-1 each read ONE slot's Xi into regs. */
    uint64_t my_xi0 = 0, my_xi1 = 0, my_xi2 = 0;
    if (lane < cnt) {
        uint8_t* p = row_base + lane * SLOT_LEN + xi_offset;
        if constexpr (ROUND == 1 || ROUND == 2) {
            /* 24B xi at xi_offset=8, 8B-aligned */
            my_xi0 = *(const uint64_t*)(p + 0);
            my_xi1 = *(const uint64_t*)(p + 8);
            my_xi2 = *(const uint64_t*)(p + 16);
        } else if constexpr (ROUND == 3) {
            /* 20B xi at xi_offset=12, 4B-aligned -> pair of u32 loads */
            uint32_t lo = *(const uint32_t*)(p + 0);
            uint32_t hi = *(const uint32_t*)(p + 4);
            my_xi0 = (uint64_t)lo | ((uint64_t)hi << 32);
            lo = *(const uint32_t*)(p + 8);
            hi = *(const uint32_t*)(p + 12);
            my_xi1 = (uint64_t)lo | ((uint64_t)hi << 32);
            my_xi2 = *(const uint32_t*)(p + 16);
        } else if constexpr (ROUND == 4) {
            /* 16B xi at xi_offset=12, only 4B-aligned -> half-aligned loads */
            uint32_t lo = *(const uint32_t*)(p + 0);
            uint32_t hi = *(const uint32_t*)(p + 4);
            my_xi0 = (uint64_t)lo | ((uint64_t)hi << 32);
            lo = *(const uint32_t*)(p + 8);
            hi = *(const uint32_t*)(p + 12);
            my_xi1 = (uint64_t)lo | ((uint64_t)hi << 32);
        } else if constexpr (ROUND == 5) {
            /* 16B xi at xi_offset=16, 16B-aligned */
            my_xi0 = *(const uint64_t*)(p + 0);
            my_xi1 = *(const uint64_t*)(p + 8);
        } else if constexpr (ROUND == 6) {
            /* 12B xi at xi_offset=16, 16B-aligned */
            my_xi0 = *(const uint64_t*)(p + 0);
            my_xi1 = *(const uint32_t*)(p + 8);
        } else if constexpr (ROUND == 7) {
            /* 8B xi at xi_offset=20, 4B-aligned */
            uint32_t lo = *(const uint32_t*)(p + 0);
            uint32_t hi = *(const uint32_t*)(p + 4);
            my_xi0 = (uint64_t)lo | ((uint64_t)hi << 32);
        } else { /* ROUND == 8 */
            /* 8B xi at xi_offset=20, 4B-aligned */
            uint32_t lo = *(const uint32_t*)(p + 0);
            uint32_t hi = *(const uint32_t*)(p + 4);
            my_xi0 = (uint64_t)lo | ((uint64_t)hi << 32);
        }
    }

    /* Process pairs: fetch slot operands via cross-lane shuffles. The loop
     * counter `base` advances uniformly across all 32 lanes so every lane
     * participates in each __shfl call -- required for correct wave-level
     * ds_bpermute semantics (partial-wave shuffle is UB). Lanes whose
     * pair_index is out of range set slot_a=slot_b=0 and skip the store. */
    uint32_t totalCollisions = (cnt * (cnt - 1)) >> 1;
    for (uint32_t base = 0; base < totalCollisions; base += 32) {
        uint32_t pair_index = base + lane;
        bool active = pair_index < totalCollisions;
        uint32_t slot_a = 0, slot_b = 0;
        if (active) {
            decode_collision_pair(pair_index, cnt, slot_a, slot_b);
        }

        uint64_t a0 = ghost_shfl(my_xi0, (int)slot_a, 32);
        uint64_t b0 = ghost_shfl(my_xi0, (int)slot_b, 32);
        uint64_t a1 = 0, a2 = 0, b1 = 0, b2 = 0;
        if constexpr (ROUND == 1 || ROUND == 2) {
            a1 = ghost_shfl(my_xi1, (int)slot_a, 32);
            a2 = ghost_shfl(my_xi2, (int)slot_a, 32);
            b1 = ghost_shfl(my_xi1, (int)slot_b, 32);
            b2 = ghost_shfl(my_xi2, (int)slot_b, 32);
        } else if constexpr (ROUND == 3) {
            a1 = ghost_shfl(my_xi1, (int)slot_a, 32);
            b1 = ghost_shfl(my_xi1, (int)slot_b, 32);
            a2 = (uint64_t)ghost_shfl((uint32_t)my_xi2, (int)slot_a, 32);
            b2 = (uint64_t)ghost_shfl((uint32_t)my_xi2, (int)slot_b, 32);
        } else if constexpr (ROUND == 4 || ROUND == 5) {
            a1 = ghost_shfl(my_xi1, (int)slot_a, 32);
            b1 = ghost_shfl(my_xi1, (int)slot_b, 32);
        } else if constexpr (ROUND == 6) {
            a1 = (uint64_t)ghost_shfl((uint32_t)my_xi1, (int)slot_a, 32);
            b1 = (uint64_t)ghost_shfl((uint32_t)my_xi1, (int)slot_b, 32);
        }
        /* ROUND 7, 8: only xi0 is needed */

        if (active) {
            xor_and_store_vals<ROUND>(ht_dst, row, slot_a, slot_b,
                                       a0, a1, a2, b0, b1, b2, rowCountersDst);
        }
    }
}

/* __restrict__ on kernel arguments: ht_src vs ht_dst, and the two row-counter
 * buffers, are always distinct allocations (see StreamInstance::alloc). Telling
 * the compiler they do not alias unlocks better load/store scheduling and lets
 * read-only loads route through NVIDIA's RO cache without an explicit __ldg.
 * Safe on both HIP and CUDA. */

#define DEFINE_ROUND_LDS_KERNEL(ROUND, SFX, LWS) \
    __global__ __launch_bounds__(LWS) \
    void kernel_round##ROUND##_##SFX##_lds(const uint8_t* __restrict__ ht_src, \
                                           uint8_t* __restrict__ ht_dst, \
                                           const uint32_t* __restrict__ rowCountersSrc, \
                                           uint32_t* __restrict__ rowCountersDst) \
    { \
        __shared__ uint8_t  first_words_data[(NR_SLOTS + 2) * (LWS)]; \
        __shared__ uint32_t collisionsData[COLL_DATA_SIZE_PER_TH * (LWS)]; \
        __shared__ uint32_t collisionsNum; \
        equihash_round_lds<LWS>(ROUND, const_cast<uint8_t*>(ht_src), ht_dst, \
                                first_words_data, collisionsData, \
                                &collisionsNum, \
                                const_cast<uint32_t*>(rowCountersSrc), rowCountersDst); \
    }

#define DEFINE_ROUND_WAVE32_KERNEL(ROUND, SFX, LWS) \
    __global__ __launch_bounds__(LWS) \
    void kernel_round##ROUND##_##SFX##_wave32(const uint8_t* __restrict__ ht_src, \
                                              uint8_t* __restrict__ ht_dst, \
                                              const uint32_t* __restrict__ rowCountersSrc, \
                                              uint32_t* __restrict__ rowCountersDst) \
    { equihash_round_wave32<LWS>(ROUND, const_cast<uint8_t*>(ht_src), ht_dst, \
                                 const_cast<uint32_t*>(rowCountersSrc), rowCountersDst); }

#define DEFINE_ROUND8_LDS_KERNEL(SFX, LWS) \
    __global__ __launch_bounds__(LWS) \
    void kernel_round8_##SFX##_lds(const uint8_t* __restrict__ ht_src, \
                                   uint8_t* __restrict__ ht_dst, \
                                   const uint32_t* __restrict__ rowCountersSrc, \
                                   uint32_t* __restrict__ rowCountersDst, \
                                   sols_t* __restrict__ sols) \
    { \
        __shared__ uint8_t  first_words_data[(NR_SLOTS + 2) * (LWS)]; \
        __shared__ uint32_t collisionsData[COLL_DATA_SIZE_PER_TH * (LWS)]; \
        __shared__ uint32_t collisionsNum; \
        equihash_round_lds<LWS>(8, const_cast<uint8_t*>(ht_src), ht_dst, \
                                first_words_data, collisionsData, \
                                &collisionsNum, \
                                const_cast<uint32_t*>(rowCountersSrc), rowCountersDst); \
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; \
        if (tid == 0) { sols->nr = 0; sols->likely_invalids = 0; } \
    }

#define DEFINE_ROUND8_WAVE32_KERNEL(SFX, LWS) \
    __global__ __launch_bounds__(LWS) \
    void kernel_round8_##SFX##_wave32(const uint8_t* __restrict__ ht_src, \
                                      uint8_t* __restrict__ ht_dst, \
                                      const uint32_t* __restrict__ rowCountersSrc, \
                                      uint32_t* __restrict__ rowCountersDst, \
                                      sols_t* __restrict__ sols) \
    { \
        equihash_round_wave32<LWS>(8, const_cast<uint8_t*>(ht_src), ht_dst, \
                                   const_cast<uint32_t*>(rowCountersSrc), rowCountersDst); \
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; \
        if (tid == 0) { sols->nr = 0; sols->likely_invalids = 0; } \
    }

#define DEFINE_ROUND_WAVE32_COOP_KERNEL(ROUND, SFX, LWS) \
    __global__ __launch_bounds__(LWS) \
    void kernel_round##ROUND##_##SFX##_wave32_coop(const uint8_t* __restrict__ ht_src, \
                                                   uint8_t* __restrict__ ht_dst, \
                                                   const uint32_t* __restrict__ rowCountersSrc, \
                                                   uint32_t* __restrict__ rowCountersDst) \
    { equihash_round_wave32_coop<ROUND, LWS>(const_cast<uint8_t*>(ht_src), ht_dst, \
                                             const_cast<uint32_t*>(rowCountersSrc), rowCountersDst); }

#define DEFINE_ROUND8_WAVE32_COOP_KERNEL(SFX, LWS) \
    __global__ __launch_bounds__(LWS) \
    void kernel_round8_##SFX##_wave32_coop(const uint8_t* __restrict__ ht_src, \
                                           uint8_t* __restrict__ ht_dst, \
                                           const uint32_t* __restrict__ rowCountersSrc, \
                                           uint32_t* __restrict__ rowCountersDst, \
                                           sols_t* __restrict__ sols) \
    { \
        equihash_round_wave32_coop<8, LWS>(const_cast<uint8_t*>(ht_src), ht_dst, \
                                           const_cast<uint32_t*>(rowCountersSrc), rowCountersDst); \
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; \
        if (tid == 0) { sols->nr = 0; sols->likely_invalids = 0; } \
    }

static __device__ uint32_t expand_ref(const uint8_t* ht, uint32_t xi_offset,
                                      uint32_t row, uint32_t slot)
{
    return *(const uint32_t*)(ht + (size_t)row * NR_SLOTS * SLOT_LEN +
                              slot * SLOT_LEN + xi_offset - 4);
}

static __device__ uint32_t expand_refs(uint32_t* ins, uint32_t nr_inputs,
                                       uint8_t* const htabs[2], uint32_t round)
{
    const uint8_t* ht = htabs[round % 2];
    uint32_t xi_offset = xi_offset_for_round(round);
    uint32_t i = nr_inputs - 1;
    uint32_t j = nr_inputs * 2 - 1;
    int32_t dup_to_watch = -1;
    while (true) {
        ins[j]     = expand_ref(ht, xi_offset, decode_row(ins[i]), decode_slot1(ins[i]));
        ins[j - 1] = expand_ref(ht, xi_offset, decode_row(ins[i]), decode_slot0(ins[i]));
        if (!round) {
            if (dup_to_watch == -1) dup_to_watch = (int32_t)ins[j];
            else if ((int32_t)ins[j] == dup_to_watch || (int32_t)ins[j - 1] == dup_to_watch) return 0;
        }
        if (!i) break;
        i--;
        j -= 2;
    }
    return 1;
}

static __device__ void potential_sol(uint8_t* const htabs[2], sols_t* sols,
                                     uint32_t ref0, uint32_t ref1)
{
    uint32_t values_tmp[1 << EQ_PARAM_K];
    uint32_t nr_values = 0;
    values_tmp[nr_values++] = ref0;
    values_tmp[nr_values++] = ref1;
    uint32_t round = EQ_PARAM_K - 1;
    do {
        round--;
        if (!expand_refs(values_tmp, nr_values, htabs, round)) return;
        nr_values *= 2;
    } while (round > 0);
    uint32_t sol_i = atomicAdd(&sols->nr, 1u);
    if (sol_i >= MAX_SOLS) return;
#pragma unroll
    for (int i = 0; i < (1 << EQ_PARAM_K); i++) sols->values[sol_i][i] = values_tmp[i];
    sols->valid[sol_i] = 1;
}

static __device__ void kernel_sols_row_body(uint8_t* ht0, uint8_t* ht1, sols_t* sols,
                                            uint32_t row, uint32_t cnt)
{
    uint8_t* htabs[2] = { ht0, ht1 };
    uint32_t ht_i = (EQ_PARAM_K - 1) % 2;
    uint32_t xi_offset = xi_offset_for_round(EQ_PARAM_K - 1);
    uint32_t mask = 0xffffff;

    uint8_t* a = htabs[ht_i] + (size_t)row * NR_SLOTS * SLOT_LEN + xi_offset;
    for (uint32_t i = 0; i < cnt; i++) {
        uint32_t a_data = (*(uint32_t*)(a + i * SLOT_LEN)) & mask;
        uint32_t ref_i  = *(uint32_t*)(a + i * SLOT_LEN - 4);
        for (uint32_t j = i + 1; j < cnt; j++) {
            uint32_t b_data = (*(uint32_t*)(a + j * SLOT_LEN)) & mask;
            if (a_data == b_data) {
                uint32_t ref_j = *(uint32_t*)(a + j * SLOT_LEN - 4);
                potential_sol(htabs, sols, ref_i, ref_j);
                return;
            }
        }
    }
}

static __device__ void kernel_sols_body(uint8_t* ht0, uint8_t* ht1, sols_t* sols,
                                        uint32_t* rowCountersSrc, uint32_t* /*rowCountersDst*/)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t rowIdx = tid / ROWS_PER_UINT;
    uint32_t rowOffset = BITS_PER_ROW * (tid % ROWS_PER_UINT);
    uint32_t cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
    cnt = min(cnt, (uint32_t)NR_SLOTS);
    kernel_sols_row_body(ht0, ht1, sols, tid, cnt);
}

#define DEFINE_ROUND0_KERNEL(SFX, LWS) \
    __global__ __launch_bounds__(LWS) \
    void kernel_round0_##SFX(const uint64_t* __restrict__ blake_state, \
                             uint8_t* __restrict__ ht, \
                             uint32_t* __restrict__ rowCounters) \
    { kernel_round0_body(blake_state, ht, rowCounters); }

/* kernel_sols: only ht0 is the "active" final table (round 8 output).
 * ht1 is only touched by ref-chain expansion in potential_sol() where it
 * re-reads earlier rounds' slots. Both are read-only within this kernel.
 * The explicit (LWS, 1) launch-bounds form tells the compiler we are fine
 * with a single block per SM -- kernel_sols is the heaviest kernel by
 * scratch/VGPR pressure (177 VGPR / 2064 B private scratch on gfx1201)
 * so pushing for high occupancy here tends to hurt rather than help. */
#define DEFINE_KERNEL_SOLS(SFX, LWS) \
    __global__ __launch_bounds__(LWS, 1) \
    void kernel_sols_##SFX(const uint8_t* __restrict__ ht0, \
                           const uint8_t* __restrict__ ht1, \
                           sols_t* __restrict__ sols, \
                           const uint32_t* __restrict__ rowCountersSrc, \
                           uint32_t* __restrict__ rowCountersDst) \
    { kernel_sols_body(const_cast<uint8_t*>(ht0), const_cast<uint8_t*>(ht1), \
                       sols, const_cast<uint32_t*>(rowCountersSrc), rowCountersDst); }

DEFINE_ROUND0_KERNEL(64, 64)
DEFINE_ROUND0_KERNEL(128, 128)

DEFINE_ROUND_LDS_KERNEL(1, 64, 64)
DEFINE_ROUND_LDS_KERNEL(2, 64, 64)
DEFINE_ROUND_LDS_KERNEL(3, 64, 64)
DEFINE_ROUND_LDS_KERNEL(4, 64, 64)
DEFINE_ROUND_LDS_KERNEL(5, 64, 64)
DEFINE_ROUND_LDS_KERNEL(6, 64, 64)
DEFINE_ROUND_LDS_KERNEL(7, 64, 64)
DEFINE_ROUND8_LDS_KERNEL(64, 64)
DEFINE_ROUND_WAVE32_KERNEL(1, 64, 64)
DEFINE_ROUND_WAVE32_KERNEL(2, 64, 64)
DEFINE_ROUND_WAVE32_KERNEL(3, 64, 64)
DEFINE_ROUND_WAVE32_KERNEL(4, 64, 64)
DEFINE_ROUND_WAVE32_KERNEL(5, 64, 64)
DEFINE_ROUND_WAVE32_KERNEL(6, 64, 64)
DEFINE_ROUND_WAVE32_KERNEL(7, 64, 64)
DEFINE_ROUND8_WAVE32_KERNEL(64, 64)
DEFINE_ROUND_WAVE32_COOP_KERNEL(1, 64, 64)
DEFINE_ROUND_WAVE32_COOP_KERNEL(2, 64, 64)
DEFINE_ROUND_WAVE32_COOP_KERNEL(3, 64, 64)
DEFINE_ROUND_WAVE32_COOP_KERNEL(4, 64, 64)
DEFINE_ROUND_WAVE32_COOP_KERNEL(5, 64, 64)
DEFINE_ROUND_WAVE32_COOP_KERNEL(6, 64, 64)
DEFINE_ROUND_WAVE32_COOP_KERNEL(7, 64, 64)
DEFINE_ROUND8_WAVE32_COOP_KERNEL(64, 64)
DEFINE_KERNEL_SOLS(64, 64)

DEFINE_ROUND_LDS_KERNEL(1, 128, 128)
DEFINE_ROUND_LDS_KERNEL(2, 128, 128)
DEFINE_ROUND_LDS_KERNEL(3, 128, 128)
DEFINE_ROUND_LDS_KERNEL(4, 128, 128)
DEFINE_ROUND_LDS_KERNEL(5, 128, 128)
DEFINE_ROUND_LDS_KERNEL(6, 128, 128)
DEFINE_ROUND_LDS_KERNEL(7, 128, 128)
DEFINE_ROUND8_LDS_KERNEL(128, 128)
DEFINE_ROUND_WAVE32_KERNEL(1, 128, 128)
DEFINE_ROUND_WAVE32_KERNEL(2, 128, 128)
DEFINE_ROUND_WAVE32_KERNEL(3, 128, 128)
DEFINE_ROUND_WAVE32_KERNEL(4, 128, 128)
DEFINE_ROUND_WAVE32_KERNEL(5, 128, 128)
DEFINE_ROUND_WAVE32_KERNEL(6, 128, 128)
DEFINE_ROUND_WAVE32_KERNEL(7, 128, 128)
DEFINE_ROUND8_WAVE32_KERNEL(128, 128)
DEFINE_ROUND_WAVE32_COOP_KERNEL(1, 128, 128)
DEFINE_ROUND_WAVE32_COOP_KERNEL(2, 128, 128)
DEFINE_ROUND_WAVE32_COOP_KERNEL(3, 128, 128)
DEFINE_ROUND_WAVE32_COOP_KERNEL(4, 128, 128)
DEFINE_ROUND_WAVE32_COOP_KERNEL(5, 128, 128)
DEFINE_ROUND_WAVE32_COOP_KERNEL(6, 128, 128)
DEFINE_ROUND_WAVE32_COOP_KERNEL(7, 128, 128)
DEFINE_ROUND8_WAVE32_COOP_KERNEL(128, 128)
DEFINE_KERNEL_SOLS(128, 128)

#undef DEFINE_ROUND0_KERNEL
#undef DEFINE_KERNEL_SOLS
#undef DEFINE_ROUND_WAVE32_COOP_KERNEL
#undef DEFINE_ROUND8_WAVE32_COOP_KERNEL
#undef DEFINE_ROUND_LDS_KERNEL
#undef DEFINE_ROUND_WAVE32_KERNEL
#undef DEFINE_ROUND8_LDS_KERNEL
#undef DEFINE_ROUND8_WAVE32_KERNEL

/* ============================================================= host side == */

#define CHECK_HIP_RET(expr) do { \
    hipError_t _e = (expr); \
    if (_e != hipSuccess) { \
        std::fprintf(stderr, "[amd_hip] HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
    } \
} while (0)

/* Host-side Blake2b midstate (mirrors amd_opencl/zcash_blake.c). */
static const uint64_t H_BLAKE_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
};
static const uint8_t BLAKE_SIGMA[12][16] = {
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
struct HostBlake {
    uint64_t h[8];
    uint64_t bytes;
};
static void host_blake_init(HostBlake& st)
{
    st.h[0] = H_BLAKE_IV[0] ^ (0x01010000u | (uint64_t)ZCASH_HASH_LEN);
    for (int i = 1; i <= 5; i++) st.h[i] = H_BLAKE_IV[i];
    uint64_t zpw;
    std::memcpy(&zpw, "ZcashPoW", 8);
    st.h[6] = H_BLAKE_IV[6] ^ zpw;
    st.h[7] = H_BLAKE_IV[7] ^ (((uint64_t)EQ_PARAM_K << 32) | (uint64_t)EQ_PARAM_N);
    st.bytes = 0;
}
static void host_blake_update(HostBlake& st, const uint8_t* msg, uint32_t msg_len, uint32_t is_final)
{
    uint64_t m[16] = {0};
    std::memcpy(m, msg, msg_len <= 128 ? msg_len : 128);
    uint64_t v[16];
    for (int i = 0; i < 8; i++) v[i] = st.h[i];
    for (int i = 0; i < 8; i++) v[8 + i] = H_BLAKE_IV[i];
    st.bytes += msg_len;
    v[12] ^= st.bytes;
    if (is_final) v[14] ^= (uint64_t)-1;
    for (int r = 0; r < 12; r++) {
        const uint8_t* s = BLAKE_SIGMA[r];
        BLAKE_MIX(v[0], v[4], v[ 8], v[12], m[s[ 0]], m[s[ 1]]);
        BLAKE_MIX(v[1], v[5], v[ 9], v[13], m[s[ 2]], m[s[ 3]]);
        BLAKE_MIX(v[2], v[6], v[10], v[14], m[s[ 4]], m[s[ 5]]);
        BLAKE_MIX(v[3], v[7], v[11], v[15], m[s[ 6]], m[s[ 7]]);
        BLAKE_MIX(v[0], v[5], v[10], v[15], m[s[ 8]], m[s[ 9]]);
        BLAKE_MIX(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        BLAKE_MIX(v[2], v[7], v[ 8], v[13], m[s[12]], m[s[13]]);
        BLAKE_MIX(v[3], v[4], v[ 9], v[14], m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; i++) st.h[i] ^= v[i] ^ v[i + 8];
}

/* --- amd_hip_context: owns device buffers + stream, one per amd_hip -------
 *
 * Every run_once() launches 13 kernels (2x init_ht + round0 + 7x collision
 * rounds + round8 + kernel_sols). On Windows KMD that's ~300us/launch of
 * pure dispatch overhead; ~3.9ms per iteration is burned in the driver
 * regardless of GPU work. To fix that we build a hipGraph once (the kernel
 * DAG is identical across iterations -- same pointers, same params) and
 * replay it via hipGraphLaunch every iteration with just H2D midstate +
 * D2H sols as the per-call work.
 *
 * Graph capture is gated by AMD_HIP_USE_GRAPH=0|1 (default 1 = on). When
 * off, we fall back to the legacy per-kernel launch path for A/B bench
 * or to unblock debugging.
 */
/* Read AMD_HIP_USE_GRAPH env once; default ON. */
static bool graph_enabled() {
    static const bool v = [](){
        char buf[16] = {0};
#ifdef _WIN32
        size_t n = 0;
        if (getenv_s(&n, buf, sizeof buf, "AMD_HIP_USE_GRAPH") != 0) return true;
        const char* e = (n ? buf : nullptr);
#else
        const char* e = std::getenv("AMD_HIP_USE_GRAPH");
#endif
        if (!e || !*e) return true;       // default ON
        char c = e[0];
        if (c == '0' || c == 'n' || c == 'N' || c == 'f' || c == 'F') return false;
        if ((c == 'o' || c == 'O') && (e[1] == 'f' || e[1] == 'F'))   return false;
        return true;
    }();
    return v;
}

static bool read_env_string(const char* name, char* buf, size_t buf_size)
{
    if (!buf || buf_size == 0) return false;
    buf[0] = '\0';
#ifdef _WIN32
    size_t n = 0;
    if (getenv_s(&n, buf, buf_size, name) != 0 || n == 0) return false;
    return true;
#else
    const char* e = std::getenv(name);
    if (!e || !*e) return false;
    std::strncpy(buf, e, buf_size - 1);
    buf[buf_size - 1] = '\0';
    return true;
#endif
}

/* Per-arch tuning defaults. Env vars (AMD_HIP_LOCAL_WORK_SIZE,
 * AMD_HIP_STREAMS, AMD_HIP_COLLISION_MODE) override these at runtime.
 *
 * AMD:
 *   gfx12xx (RDNA4): LWS=128 / streams=4 / wave32 -- validated 273 Sol/s
 *                     on RX 9070 XT in bench/results.md
 *   other:           conservative defaults
 * NVIDIA (sm_XX):   warp size is 32, so the wave32 kernel family is a
 *                   natural fit (same lane-per-row structure, no LDS).
 *                   LWS=128 keeps 4 warps/block with plenty of resident
 *                   warps under NVIDIA's reg/LDS budget. streams=2 is a
 *                   conservative starting point; bump to 4 once an NVIDIA
 *                   run establishes baseline. All three values are
 *                   untested on real NV hardware -- see README.NVIDIA.md. */

static bool arch_is_rdna4(const std::string& arch) { return arch.rfind("gfx12", 0) == 0; }
static bool arch_is_nvidia(const std::string& arch) { return arch.rfind("sm_", 0) == 0; }

static uint32_t default_runtime_lws_for_arch(const std::string& arch)
{
    if (arch_is_rdna4(arch)) return 128u;
    if (arch_is_nvidia(arch)) return 128u;
    return (uint32_t)AMD_HIP_DEFAULT_LOCAL_WORK_SIZE;
}

static int default_streams_for_arch(const std::string& arch)
{
    if (arch_is_rdna4(arch)) return 4;
    if (arch_is_nvidia(arch)) return 2;  /* conservative; tune up from here */
    return 1;
}

static HipCollisionMode default_collision_mode_for_arch(const std::string& arch)
{
    if (arch_is_rdna4(arch)) return HipCollisionMode::Wave32;
    if (arch_is_nvidia(arch)) return HipCollisionMode::Wave32;  /* warpSize=32 native */
    return HipCollisionMode::Lds;
}

static uint32_t read_runtime_lws_env(const std::string& arch)
{
    char buf[16] = {0};
    if (!read_env_string("AMD_HIP_LOCAL_WORK_SIZE", buf, sizeof buf)) {
        return default_runtime_lws_for_arch(arch);
    }
    int v = std::atoi(buf);
    if (v == 64 || v == 128) return (uint32_t)v;
    return default_runtime_lws_for_arch(arch);
}

static HipCollisionMode read_collision_mode_env(const std::string& arch)
{
    char buf[32] = {0};
    if (!read_env_string("AMD_HIP_COLLISION_MODE", buf, sizeof buf)) {
        return default_collision_mode_for_arch(arch);
    }
    char c0 = buf[0];
    /* "coop" or "c" or "2" -> Wave32Coop (cooperative slot loading via ds_bpermute) */
    if (c0 == '2' || c0 == 'c' || c0 == 'C') return HipCollisionMode::Wave32Coop;
    if (c0 == '1' || c0 == 'w' || c0 == 'W') return HipCollisionMode::Wave32;
    return HipCollisionMode::Lds;
}

/* Per-round coop mask: when collision_mode == Wave32Coop, bit R (1..8) of this
 * mask enables coop for round R. Other rounds fall back to plain wave32. Useful
 * for bisecting correctness issues. Default: 0x1FE = rounds 1..8 all coop. */
static uint32_t read_coop_round_mask_env()
{
    char buf[16] = {0};
    if (!read_env_string("AMD_HIP_COOP_ROUND_MASK", buf, sizeof buf)) {
        return 0x1FE;  /* rounds 1..8 */
    }
    return (uint32_t)std::strtoul(buf, nullptr, 0);
}

static uint32_t coop_round_mask()
{
    static const uint32_t v = read_coop_round_mask_env();
    return v;
}

static const char* collision_mode_name(HipCollisionMode mode)
{
    switch (mode) {
        case HipCollisionMode::Wave32Coop: return "wave32_coop";
        case HipCollisionMode::Wave32:     return "wave32";
        case HipCollisionMode::Lds:        return "lds";
    }
    return "?";
}

/* StreamInstance: one full set of per-stream GPU resources (stream, HTs,
 * row-counters, midstate, sols, graph, pinned host buffers, events, and
 * pending-iter callback storage). amd_hip_context holds 1 or 2 of these
 * depending on AMD_HIP_STREAMS (default 1).
 *
 * Two instances allow interleaving iter launches across two streams, which
 * *may* let the HIP/KMD scheduler overlap kernel execution on the same GPU
 * and improve effective memory BW utilization. Whether this actually
 * happens on gfx1201 / Windows WDDM is measured empirically -- see
 * AMD_HIP_STREAMS=2 bench. Costs ~770 MB of extra VRAM per added stream. */
struct StreamInstance {
    int device_id = 0;
    uint32_t      runtime_lws = AMD_HIP_DEFAULT_LOCAL_WORK_SIZE;
    HipCollisionMode collision_mode = HipCollisionMode::Lds;
    uint8_t*       d_ht[2]     = {nullptr, nullptr};
    uint32_t*      d_rc[2]     = {nullptr, nullptr};
    uint64_t*      d_midstate  = nullptr;
    sols_t*        d_sols      = nullptr;
    hipStream_t    stream      = nullptr;
    hipGraphExec_t graph_exec  = nullptr;
    bool           graph_ready = false;

    sols_t*        h_sols_pinned[2] = {nullptr, nullptr};
    hipEvent_t     iter_done_evt[2] = {nullptr, nullptr};
    uint64_t       pipeline_iters   = 0;
    uint64_t       pipeline_drained = 0;

    std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)>
                   pending_solf[2];
    std::function<void(void)> pending_hashdonef[2];
    std::function<bool()>     pending_cancelf[2];

    void alloc(int devid, uint32_t lws, HipCollisionMode mode) {
        device_id = devid;
        runtime_lws = lws;
        collision_mode = mode;
        CHECK_HIP_RET(hipSetDevice(devid));
        /* hipStreamNonBlocking: don't implicitly synchronize with the NULL
         * stream. With n>1 streams in use, this lets the HIP/KMD scheduler
         * overlap kernels from different StreamInstances more aggressively. */
        CHECK_HIP_RET(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        CHECK_HIP_RET(hipMalloc(&d_ht[0],    HT_SIZE));
        CHECK_HIP_RET(hipMalloc(&d_ht[1],    HT_SIZE));
        CHECK_HIP_RET(hipMalloc(&d_rc[0],    NR_ROWS));
        CHECK_HIP_RET(hipMalloc(&d_rc[1],    NR_ROWS));
        CHECK_HIP_RET(hipMalloc(&d_midstate, 8 * sizeof(uint64_t)));
        CHECK_HIP_RET(hipMalloc(&d_sols,     sizeof(sols_t)));
        CHECK_HIP_RET(hipHostMalloc((void**)&h_sols_pinned[0], sizeof(sols_t), hipHostMallocDefault));
        CHECK_HIP_RET(hipHostMalloc((void**)&h_sols_pinned[1], sizeof(sols_t), hipHostMallocDefault));
        CHECK_HIP_RET(hipEventCreateWithFlags(&iter_done_evt[0], hipEventDisableTiming));
        CHECK_HIP_RET(hipEventCreateWithFlags(&iter_done_evt[1], hipEventDisableTiming));
    }
    void free_all() {
        if (graph_exec) { (void)hipGraphExecDestroy(graph_exec); graph_exec = nullptr; }
        graph_ready = false;
        if (stream) (void)hipStreamSynchronize(stream);
        if (iter_done_evt[0]) (void)hipEventDestroy(iter_done_evt[0]);
        if (iter_done_evt[1]) (void)hipEventDestroy(iter_done_evt[1]);
        if (h_sols_pinned[0]) (void)hipHostFree(h_sols_pinned[0]);
        if (h_sols_pinned[1]) (void)hipHostFree(h_sols_pinned[1]);
        if (d_ht[0])    (void)hipFree(d_ht[0]);
        if (d_ht[1])    (void)hipFree(d_ht[1]);
        if (d_rc[0])    (void)hipFree(d_rc[0]);
        if (d_rc[1])    (void)hipFree(d_rc[1]);
        if (d_midstate) (void)hipFree(d_midstate);
        if (d_sols)     (void)hipFree(d_sols);
        if (stream)     (void)hipStreamDestroy(stream);
        d_ht[0] = d_ht[1] = nullptr;
        d_rc[0] = d_rc[1] = nullptr;
        d_midstate = nullptr;
        d_sols = nullptr;
        stream = nullptr;
        iter_done_evt[0] = iter_done_evt[1] = nullptr;
        h_sols_pinned[0] = h_sols_pinned[1] = nullptr;
        pipeline_iters   = 0;
        pipeline_drained = 0;
        pending_solf[0] = pending_solf[1] = nullptr;
        pending_hashdonef[0] = pending_hashdonef[1] = nullptr;
        pending_cancelf[0] = pending_cancelf[1] = nullptr;
        runtime_lws = AMD_HIP_DEFAULT_LOCAL_WORK_SIZE;
        collision_mode = HipCollisionMode::Lds;
    }

    /* Forward declarations; methods below (after amd_hip_context so we
     * can refer to shared helpers). */
    void record_pipeline(hipStream_t s);
    bool build_graph();
    uint32_t run_once(const uint8_t header140[140], sols_t& host_sols);
    int launch_iter(const uint8_t header140[140]);
    const sols_t* wait_iter(int slot);
};

struct amd_hip_context {
    int device_id = 0;
    bool ready = false;
    std::string dev_name;
    std::string dev_arch;
    int cu_count = 0;
    uint32_t runtime_lws = AMD_HIP_DEFAULT_LOCAL_WORK_SIZE;
    HipCollisionMode collision_mode = HipCollisionMode::Lds;

    /* 1..MAX_STREAMS, selected at alloc time from AMD_HIP_STREAMS env
     * (default 1). Each stream owns a full set of device buffers, so VRAM
     * cost is linear in n_streams. 8 * 770 MB = ~6 GB VRAM, leaving plenty
     * of headroom on a 16 GB card. */
    static constexpr int MAX_STREAMS = 8;
    int n_streams = 1;
    int next_stream = 0;
    StreamInstance s[MAX_STREAMS];

    /* --- Backward-compat "primary instance" accessors. Legacy code refers
     * to ctx->stream, ctx->d_ht[0], etc. These are thin wrappers around
     * s[0]. For multi-stream mode, use s[i] directly. */
    hipStream_t&    stream()       { return s[0].stream; }
    uint8_t**       d_ht()         { return s[0].d_ht; }
    uint32_t**      d_rc()         { return s[0].d_rc; }
    uint64_t*&      d_midstate()   { return s[0].d_midstate; }
    sols_t*&        d_sols()       { return s[0].d_sols; }
    hipGraphExec_t& graph_exec()   { return s[0].graph_exec; }
    bool&           graph_ready()  { return s[0].graph_ready; }
    sols_t**        h_sols_pinned(){ return s[0].h_sols_pinned; }
    hipEvent_t*     iter_done_evt(){ return s[0].iter_done_evt; }

    static int read_streams_env(const std::string& arch) {
        char buf[8] = {0};
#ifdef _WIN32
        size_t n = 0;
        if (getenv_s(&n, buf, sizeof buf, "AMD_HIP_STREAMS") != 0 || n == 0) {
            return default_streams_for_arch(arch);
        }
#else
        const char* e = std::getenv("AMD_HIP_STREAMS");
        if (!e || !*e) return default_streams_for_arch(arch);
        std::strncpy(buf, e, sizeof buf - 1);
#endif
        int v = std::atoi(buf);
        if (v < 1) v = 1;
        if (v > MAX_STREAMS) v = MAX_STREAMS;
        return v;
    }

    void alloc_device_memory() {
        if (ready) return;
        n_streams = read_streams_env(dev_arch);
        std::fprintf(stderr, "[amd_hip] kernel config: lws=%u collision=%s\n",
            runtime_lws, collision_mode_name(collision_mode));
        for (int i = 0; i < n_streams; i++) s[i].alloc(device_id, runtime_lws, collision_mode);
        if (n_streams > 1) {
            std::fprintf(stderr, "[amd_hip] allocated %d streams (%zu MB each, %zu MB total)\n",
                n_streams,
                (size_t)(2 * HT_SIZE + 2 * NR_ROWS + sizeof(sols_t)) / (1024*1024),
                (size_t)n_streams * (2 * HT_SIZE + 2 * NR_ROWS + sizeof(sols_t)) / (1024*1024));
        }
        ready = true;
    }
    void free_device_memory() {
        if (!ready) return;
        for (int i = 0; i < n_streams; i++) s[i].free_all();
        ready = false;
        n_streams = 1;
        next_stream = 0;
    }
};

static dim3 row_grid_for(const StreamInstance& si)
{
    /* Wave32 and Wave32Coop both launch one wave per row -> LWS/32 rows
     * per block. LDS path launches one thread per row -> LWS rows per block. */
    bool per_wave = (si.collision_mode == HipCollisionMode::Wave32) ||
                    (si.collision_mode == HipCollisionMode::Wave32Coop);
    uint32_t rows_per_block = per_wave ? (si.runtime_lws / 32u) : si.runtime_lws;
    return dim3((NR_ROWS + rows_per_block - 1) / rows_per_block);
}

static void launch_round0_dispatch(const StreamInstance& si, hipStream_t sm)
{
    dim3 block(si.runtime_lws);
    dim3 grid(512);
    if (si.runtime_lws == 128) {
        hipLaunchKernelGGL(kernel_round0_128, grid, block, 0, sm,
                           si.d_midstate, si.d_ht[0], si.d_rc[0]);
    } else {
        hipLaunchKernelGGL(kernel_round0_64, grid, block, 0, sm,
                           si.d_midstate, si.d_ht[0], si.d_rc[0]);
    }
}

#define GH_DISPATCH_ROUND_CASE(ROUND, SFX, MODE) \
    case ROUND: \
        hipLaunchKernelGGL(kernel_round##ROUND##_##SFX##_##MODE, grid, block, 0, sm, \
                           si.d_ht[src], si.d_ht[dst], si.d_rc[src], si.d_rc[dst]); \
        break

static void launch_round_dispatch(const StreamInstance& si, hipStream_t sm, int round, int src, int dst)
{
    dim3 block(si.runtime_lws);
    dim3 grid = row_grid_for(si);
    bool use_coop = (si.collision_mode == HipCollisionMode::Wave32Coop) &&
                    ((coop_round_mask() >> round) & 1u);
    if (use_coop) {
        if (si.runtime_lws == 128) {
            switch (round) {
                GH_DISPATCH_ROUND_CASE(1, 128, wave32_coop);
                GH_DISPATCH_ROUND_CASE(2, 128, wave32_coop);
                GH_DISPATCH_ROUND_CASE(3, 128, wave32_coop);
                GH_DISPATCH_ROUND_CASE(4, 128, wave32_coop);
                GH_DISPATCH_ROUND_CASE(5, 128, wave32_coop);
                GH_DISPATCH_ROUND_CASE(6, 128, wave32_coop);
                GH_DISPATCH_ROUND_CASE(7, 128, wave32_coop);
            }
        } else {
            switch (round) {
                GH_DISPATCH_ROUND_CASE(1, 64, wave32_coop);
                GH_DISPATCH_ROUND_CASE(2, 64, wave32_coop);
                GH_DISPATCH_ROUND_CASE(3, 64, wave32_coop);
                GH_DISPATCH_ROUND_CASE(4, 64, wave32_coop);
                GH_DISPATCH_ROUND_CASE(5, 64, wave32_coop);
                GH_DISPATCH_ROUND_CASE(6, 64, wave32_coop);
                GH_DISPATCH_ROUND_CASE(7, 64, wave32_coop);
            }
        }
    } else if (si.collision_mode == HipCollisionMode::Wave32 ||
               si.collision_mode == HipCollisionMode::Wave32Coop) {
        if (si.runtime_lws == 128) {
            switch (round) {
                GH_DISPATCH_ROUND_CASE(1, 128, wave32);
                GH_DISPATCH_ROUND_CASE(2, 128, wave32);
                GH_DISPATCH_ROUND_CASE(3, 128, wave32);
                GH_DISPATCH_ROUND_CASE(4, 128, wave32);
                GH_DISPATCH_ROUND_CASE(5, 128, wave32);
                GH_DISPATCH_ROUND_CASE(6, 128, wave32);
                GH_DISPATCH_ROUND_CASE(7, 128, wave32);
            }
        } else {
            switch (round) {
                GH_DISPATCH_ROUND_CASE(1, 64, wave32);
                GH_DISPATCH_ROUND_CASE(2, 64, wave32);
                GH_DISPATCH_ROUND_CASE(3, 64, wave32);
                GH_DISPATCH_ROUND_CASE(4, 64, wave32);
                GH_DISPATCH_ROUND_CASE(5, 64, wave32);
                GH_DISPATCH_ROUND_CASE(6, 64, wave32);
                GH_DISPATCH_ROUND_CASE(7, 64, wave32);
            }
        }
    } else {
        if (si.runtime_lws == 128) {
            switch (round) {
                GH_DISPATCH_ROUND_CASE(1, 128, lds);
                GH_DISPATCH_ROUND_CASE(2, 128, lds);
                GH_DISPATCH_ROUND_CASE(3, 128, lds);
                GH_DISPATCH_ROUND_CASE(4, 128, lds);
                GH_DISPATCH_ROUND_CASE(5, 128, lds);
                GH_DISPATCH_ROUND_CASE(6, 128, lds);
                GH_DISPATCH_ROUND_CASE(7, 128, lds);
            }
        } else {
            switch (round) {
                GH_DISPATCH_ROUND_CASE(1, 64, lds);
                GH_DISPATCH_ROUND_CASE(2, 64, lds);
                GH_DISPATCH_ROUND_CASE(3, 64, lds);
                GH_DISPATCH_ROUND_CASE(4, 64, lds);
                GH_DISPATCH_ROUND_CASE(5, 64, lds);
                GH_DISPATCH_ROUND_CASE(6, 64, lds);
                GH_DISPATCH_ROUND_CASE(7, 64, lds);
            }
        }
    }
}

static void launch_round8_dispatch(const StreamInstance& si, hipStream_t sm)
{
    dim3 block(si.runtime_lws);
    dim3 grid = row_grid_for(si);
    bool use_coop = (si.collision_mode == HipCollisionMode::Wave32Coop) &&
                    ((coop_round_mask() >> 8) & 1u);
    if (use_coop) {
        if (si.runtime_lws == 128) {
            hipLaunchKernelGGL(kernel_round8_128_wave32_coop, grid, block, 0, sm,
                               si.d_ht[1], si.d_ht[0], si.d_rc[1], si.d_rc[0], si.d_sols);
        } else {
            hipLaunchKernelGGL(kernel_round8_64_wave32_coop, grid, block, 0, sm,
                               si.d_ht[1], si.d_ht[0], si.d_rc[1], si.d_rc[0], si.d_sols);
        }
    } else if (si.collision_mode == HipCollisionMode::Wave32 ||
               si.collision_mode == HipCollisionMode::Wave32Coop) {
        if (si.runtime_lws == 128) {
            hipLaunchKernelGGL(kernel_round8_128_wave32, grid, block, 0, sm,
                               si.d_ht[1], si.d_ht[0], si.d_rc[1], si.d_rc[0], si.d_sols);
        } else {
            hipLaunchKernelGGL(kernel_round8_64_wave32, grid, block, 0, sm,
                               si.d_ht[1], si.d_ht[0], si.d_rc[1], si.d_rc[0], si.d_sols);
        }
    } else {
        if (si.runtime_lws == 128) {
            hipLaunchKernelGGL(kernel_round8_128_lds, grid, block, 0, sm,
                               si.d_ht[1], si.d_ht[0], si.d_rc[1], si.d_rc[0], si.d_sols);
        } else {
            hipLaunchKernelGGL(kernel_round8_64_lds, grid, block, 0, sm,
                               si.d_ht[1], si.d_ht[0], si.d_rc[1], si.d_rc[0], si.d_sols);
        }
    }
}

static void launch_kernel_sols_dispatch(const StreamInstance& si, hipStream_t sm)
{
    dim3 block(si.runtime_lws);
    dim3 grid((NR_ROWS + si.runtime_lws - 1) / si.runtime_lws);
    if (si.runtime_lws == 128) {
        hipLaunchKernelGGL(kernel_sols_128, grid, block, 0, sm,
                           si.d_ht[0], si.d_ht[1], si.d_sols, si.d_rc[0], si.d_rc[1]);
    } else {
        hipLaunchKernelGGL(kernel_sols_64, grid, block, 0, sm,
                           si.d_ht[0], si.d_ht[1], si.d_sols, si.d_rc[0], si.d_rc[1]);
    }
}

#undef GH_DISPATCH_ROUND_CASE

/* --- StreamInstance method definitions (out-of-class so they can use
 *     file-scope helpers and kernel declarations) ------------------------- */

void StreamInstance::record_pipeline(hipStream_t sm) {
    const uint32_t nr_uints = NR_ROWS / ROWS_PER_UINT;
    const uint32_t init_lws = 256;
    dim3 init_grid((nr_uints + init_lws - 1) / init_lws);
    auto launch_init_ht = [&](int t) {
        hipLaunchKernelGGL(kernel_init_ht, init_grid, dim3(init_lws), 0, sm, d_rc[t]);
    };

    launch_init_ht(0);
    launch_round0_dispatch(*this, sm);
    for (int round = 1; round <= 7; round++) {
        int src = (round - 1) & 1;
        int dst = round & 1;
        launch_init_ht(dst);
        launch_round_dispatch(*this, sm, round, src, dst);
    }
    launch_init_ht(0);
    launch_round8_dispatch(*this, sm);
    launch_kernel_sols_dispatch(*this, sm);
}

bool StreamInstance::build_graph() {
    if (graph_ready) return true;
    hipStream_t cap = nullptr;
    hipError_t e = hipStreamCreate(&cap);
    if (e != hipSuccess) {
        std::fprintf(stderr, "[amd_hip] graph: hipStreamCreate failed: %s\n",
            hipGetErrorString(e));
        return false;
    }
    e = hipStreamBeginCapture(cap, hipStreamCaptureModeGlobal);
    if (e != hipSuccess) {
        std::fprintf(stderr, "[amd_hip] graph: hipStreamBeginCapture failed: %s\n",
            hipGetErrorString(e));
        (void)hipStreamDestroy(cap);
        return false;
    }
    record_pipeline(cap);
    hipGraph_t graph = nullptr;
    e = hipStreamEndCapture(cap, &graph);
    (void)hipStreamDestroy(cap);
    if (e != hipSuccess) {
        std::fprintf(stderr, "[amd_hip] graph: hipStreamEndCapture failed: %s\n",
            hipGetErrorString(e));
        return false;
    }
    e = ghost_graph_instantiate(&graph_exec, graph);
    (void)hipGraphDestroy(graph);
    if (e != hipSuccess) {
        std::fprintf(stderr, "[amd_hip] graph: ghost_graph_instantiate failed: %s\n",
            hipGetErrorString(e));
        graph_exec = nullptr;
        return false;
    }
    graph_ready = true;
    return true;
}

uint32_t StreamInstance::run_once(const uint8_t header140[140], sols_t& host_sols) {
    HostBlake mid;
    host_blake_init(mid);
    host_blake_update(mid, header140, 128, 0);
    CHECK_HIP_RET(hipMemcpyAsync(d_midstate, mid.h, 8 * sizeof(uint64_t),
                                 hipMemcpyHostToDevice, stream));

    if (graph_enabled() && build_graph()) {
        CHECK_HIP_RET(hipGraphLaunch(graph_exec, stream));
    } else {
        record_pipeline(stream);
    }

    CHECK_HIP_RET(hipMemcpyAsync(h_sols_pinned[0], d_sols, sizeof(sols_t),
                                 hipMemcpyDeviceToHost, stream));
    CHECK_HIP_RET(hipStreamSynchronize(stream));
    std::memcpy(&host_sols, h_sols_pinned[0], sizeof(sols_t));
    return host_sols.nr < MAX_SOLS ? host_sols.nr : MAX_SOLS;
}

int StreamInstance::launch_iter(const uint8_t header140[140]) {
    HostBlake mid;
    host_blake_init(mid);
    host_blake_update(mid, header140, 128, 0);
    CHECK_HIP_RET(hipMemcpyAsync(d_midstate, mid.h, 8 * sizeof(uint64_t),
                                 hipMemcpyHostToDevice, stream));

    if (graph_enabled() && build_graph()) {
        CHECK_HIP_RET(hipGraphLaunch(graph_exec, stream));
    } else {
        record_pipeline(stream);
    }

    int slot = (int)(pipeline_iters % 2);
    CHECK_HIP_RET(hipMemcpyAsync(h_sols_pinned[slot], d_sols, sizeof(sols_t),
                                 hipMemcpyDeviceToHost, stream));
    CHECK_HIP_RET(hipEventRecord(iter_done_evt[slot], stream));
    pipeline_iters++;
    return slot;
}

const sols_t* StreamInstance::wait_iter(int slot) {
    CHECK_HIP_RET(hipEventSynchronize(iter_done_evt[slot]));
    return h_sols_pinned[slot];
}

/* Per-kernel profile; only ever run on the primary stream (s[0]). Returns
 * amd_hip::ProfileResult (public type). */
static amd_hip::ProfileResult run_once_profiled(StreamInstance& si, const uint8_t header140[140])
{
    amd_hip::ProfileResult r{};
    HostBlake mid;
    host_blake_init(mid);
    host_blake_update(mid, header140, 128, 0);

    hipEvent_t e_start, e_end;
    CHECK_HIP_RET(hipEventCreate(&e_start));
    CHECK_HIP_RET(hipEventCreate(&e_end));

    auto time_span = [&](auto&& f) {
        CHECK_HIP_RET(hipEventRecord(e_start, si.stream));
        f();
        CHECK_HIP_RET(hipEventRecord(e_end, si.stream));
        CHECK_HIP_RET(hipEventSynchronize(e_end));
        float ms = 0.0f;
        CHECK_HIP_RET(hipEventElapsedTime(&ms, e_start, e_end));
        return (double)ms;
    };

    r.ms_h2d_midstate = time_span([&]() {
        CHECK_HIP_RET(hipMemcpyAsync(si.d_midstate, mid.h, 8 * sizeof(uint64_t),
                                     hipMemcpyHostToDevice, si.stream));
    });

    const uint32_t nr_uints = NR_ROWS / ROWS_PER_UINT;
    const uint32_t init_lws = 256;
    dim3 init_grid((nr_uints + init_lws - 1) / init_lws);
    auto launch_init_ht = [&](int t) {
        hipLaunchKernelGGL(kernel_init_ht, init_grid, dim3(init_lws), 0, si.stream, si.d_rc[t]);
    };

    int ht_i = 0;
    r.ms_init_ht[ht_i++] = time_span([&]() { launch_init_ht(0); });

    r.ms_round[0] = time_span([&]() {
        launch_round0_dispatch(si, si.stream);
    });
    for (int round = 1; round <= 7; round++) {
        int src = (round - 1) & 1;
        int dst = round & 1;
        r.ms_init_ht[ht_i++] = time_span([&]() { launch_init_ht(dst); });
        r.ms_round[round] = time_span([&]() {
            launch_round_dispatch(si, si.stream, round, src, dst);
        });
    }

    r.ms_init_ht[ht_i++] = time_span([&]() { launch_init_ht(0); });
    r.ms_round[8] = time_span([&]() {
        launch_round8_dispatch(si, si.stream);
    });
    r.ht_count = ht_i;

    r.ms_kernel_sols = time_span([&]() {
        launch_kernel_sols_dispatch(si, si.stream);
    });

    r.ms_d2h = time_span([&]() {
        CHECK_HIP_RET(hipMemcpyAsync(si.h_sols_pinned[0], si.d_sols, sizeof(sols_t),
                                     hipMemcpyDeviceToHost, si.stream));
    });

    r.ms_total = r.ms_h2d_midstate + r.ms_kernel_sols + r.ms_d2h;
    for (int i = 0; i < r.ht_count; i++) r.ms_total += r.ms_init_ht[i];
    for (int i = 0; i < 9; i++)          r.ms_total += r.ms_round[i];

    (void)hipEventDestroy(e_start);
    (void)hipEventDestroy(e_end);
    return r;
}

/* --- amd_hip struct method implementations -------------------------------- */
static void fill_device_info(int dev_id, std::string& gpu_name, int& cu_count, std::string& version)
{
    hipDeviceProp_t prop{};
    hipError_t e = hipGetDeviceProperties(&prop, dev_id);
    if (e != hipSuccess) {
        gpu_name = "(unknown)";
        cu_count = 0;
        version = "";
        return;
    }
    gpu_name = prop.name;
    cu_count = prop.multiProcessorCount;
    /* AMD: "gfx1201"; NVIDIA: "sm_89". See gpu_compat.h. */
    version = GHOST_GPU_ARCH_STRING(prop);
}

amd_hip::amd_hip(int platf_id, int dev_id)
    : device_id(dev_id), platform_id(platf_id), context(nullptr), m_cu_count(0)
{
    int n_devices = 0;
    hipError_t e = hipGetDeviceCount(&n_devices);
    if (e != hipSuccess || dev_id < 0 || dev_id >= n_devices) {
        std::fprintf(stderr, "[amd_hip] device %d not available (hipGetDeviceCount=%s, n=%d)\n",
            dev_id, hipGetErrorString(e), n_devices);
        return;
    }
    CHECK_HIP_RET(hipSetDevice(dev_id));
    fill_device_info(dev_id, m_gpu_name, m_cu_count, m_version);
    context = new amd_hip_context();
    context->device_id = dev_id;
    context->dev_name = m_gpu_name;
    context->dev_arch = m_version;
    context->cu_count = m_cu_count;
    context->runtime_lws = read_runtime_lws_env(m_version);
    context->collision_mode = read_collision_mode_env(m_version);
}

amd_hip::~amd_hip()
{
    if (context) {
        context->free_device_memory();
        delete context;
        context = nullptr;
    }
}

std::string amd_hip::getdevinfo()
{
    char buf[256];
    std::snprintf(buf, sizeof buf, "%s [%s] (#%d) HIP %d CU",
        m_gpu_name.c_str(), m_version.c_str(), device_id, m_cu_count);
    return std::string(buf);
}

int amd_hip::getcount()
{
    int n = 0;
    hipError_t e = hipGetDeviceCount(&n);
    return (e == hipSuccess) ? n : 0;
}

void amd_hip::getinfo(int /*platf_id*/, int d_id, std::string& gpu_name, int& sm_count, std::string& version)
{
    fill_device_info(d_id, gpu_name, sm_count, version);
}

void amd_hip::print_amd_hip_devices()
{
    int n = getcount();
    std::printf("[amd_hip] HIP devices: %d\n", n);
    for (int i = 0; i < n; i++) {
        std::string name, arch;
        int cu = 0;
        fill_device_info(i, name, cu, arch);
        std::printf("  #%d %s [%s] %d CU\n", i, name.c_str(), arch.c_str(), cu);
    }
}

void amd_hip::start(amd_hip& ctx)
{
    if (!ctx.context) return;
    CHECK_HIP_RET(hipSetDevice(ctx.device_id));
    ctx.context->alloc_device_memory();
}

void amd_hip::stop(amd_hip& ctx)
{
    if (!ctx.context) return;
    ctx.context->free_device_memory();
}

/* --- solution post-processing ---------------------------------------------
 * Equihash's EhIsValidSolution rejects solutions with:
 *   (a) duplicate initial-round indices (ANY two indices equal), or
 *   (b) an incorrectly ordered index tree: at each binary-reduction level
 *       the left subtree's first index must be < the right subtree's first
 *       index. Silentarmy-derived GPU kernels don't enforce (b); the host
 *       must recursively sort pairs at every level. Without this fix the
 *       pool's Zcash verifier returns:
 *           "Invalid solution: Index tree incorrectly ordered"
 * Port of amd_opencl/amd_opencl.cpp::{sort_pair, verify_sol_indices}.
 */
static void sort_pair_host(uint32_t* a, uint32_t len)
{
    uint32_t* b = a + len;
    uint32_t tmp, need_sorting = 0;
    for (uint32_t i = 0; i < len; i++) {
        if (need_sorting || a[i] > b[i]) {
            need_sorting = 1;
            tmp = a[i]; a[i] = b[i]; b[i] = tmp;
        } else if (a[i] < b[i]) {
            return;
        }
    }
}

static bool verify_and_sort_indices(uint32_t* inputs)
{
    /* Dedup via 2^(PREFIX+1)-bit bitmap. For Equihash(200,9) that's
     * 2^21 / 8 = 256 KiB. Reused via thread_local so we pay the zero-fill
     * but never the heap alloc after the first call on each thread. */
    constexpr uint32_t seen_len = (1u << (EQ_PREFIX + 1)) / 8;
    thread_local std::vector<uint8_t> seen(seen_len, 0);
    std::memset(seen.data(), 0, seen_len);
    for (uint32_t i = 0; i < (1u << EQ_PARAM_K); i++) {
        uint32_t v = inputs[i];
        if ((v >> 3) >= seen_len) return false;
        uint8_t  m    = (uint8_t)(1u << (v & 7));
        uint8_t& slot = seen[v >> 3];
        if (slot & m) return false;         /* duplicate index */
        slot |= m;
    }
    /* Recursive pair sort at every binary-reduction level. */
    for (uint32_t level = 0; level < EQ_PARAM_K; level++)
        for (uint32_t i = 0; i < (1u << EQ_PARAM_K); i += (2u << level))
            sort_pair_host(&inputs[i], 1u << level);
    return true;
}

/* Host-side counters for observability. Incremented in solve(); the
 * difference tells us how many GPU candidates the pool would reject. */
static std::atomic<uint64_t> g_raw_candidates{0};
static std::atomic<uint64_t> g_dropped_by_filter{0};

int amd_hip::get_n_streams(amd_hip& ctx) {
    if (!ctx.context || !ctx.context->ready) return 0;
    return ctx.context->n_streams;
}

uint64_t amd_hip::get_raw_candidates()     { return g_raw_candidates.load(); }
uint64_t amd_hip::get_dropped_by_filter()  { return g_dropped_by_filter.load(); }

void amd_hip::solve(const char* tequihash_header, unsigned int tequihash_header_len,
    const char* nonce, unsigned int nonce_len,
    std::function<bool()> cancelf,
    std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
    std::function<void(void)> hashdonef,
    amd_hip& ctx)
{
    if (!ctx.context || !ctx.context->ready) return;
    if (cancelf && cancelf()) return;

    /* Assemble the 140-byte (header || nonce) buffer that Blake consumes. */
    uint8_t header140[140] = {0};
    unsigned int hlen = std::min<unsigned int>(tequihash_header_len, 140);
    std::memcpy(header140, tequihash_header, hlen);
    unsigned int nbytes = std::min<unsigned int>(nonce_len, 140 - hlen);
    std::memcpy(header140 + hlen, nonce, nbytes);

    CHECK_HIP_RET(hipSetDevice(ctx.device_id));

    /* Legacy sync path always uses the primary stream. */
    sols_t host_sols;
    uint32_t n = ctx.context->s[0].run_once(header140, host_sols);

    if (hashdonef) hashdonef();

    for (uint32_t s = 0; s < n; s++) {
        if (!host_sols.valid[s]) continue;
        if (cancelf && cancelf()) return;

        g_raw_candidates.fetch_add(1, std::memory_order_relaxed);
        /* Dedup + recursive pair-sort in place. Drops solutions with
         * duplicate indices; sorts the rest so EhIsValidSolution accepts
         * them at the pool. */
        if (!verify_and_sort_indices(host_sols.values[s])) {
            g_dropped_by_filter.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        std::vector<uint32_t> indices(host_sols.values[s],
                                      host_sols.values[s] + (1 << EQ_PARAM_K));
        /* cbitlen = N/(K+1) = 20 for Equihash(200,9). Matches the OpenCL
         * backend's contract so ZcashStratum's GetMinimalFromIndices works. */
        if (solutionf) solutionf(indices, EQ_PREFIX, nullptr);
    }
}

/* --- Pipelined variant ---------------------------------------------------- *
 *
 * Overlap GPU iter N with CPU post-processing of iter N-1 (and H2D of
 * iter N-1's next launch). Depth: 2 in-flight iters max. Slot rotation
 * via pipeline_iters / pipeline_drained counters.
 *
 * Contract for callers: the solutionf and hashdonef MUST capture any
 * iter-specific state (nonce, job, target) BY VALUE; by the time we
 * invoke them for iter N-1, the outer scope will have already advanced
 * to iter N.
 */
static void drain_one_slot(StreamInstance& si) {
    int slot = (int)(si.pipeline_drained % 2);
    const sols_t* sols = si.wait_iter(slot);
    uint32_t n = sols->nr < MAX_SOLS ? sols->nr : MAX_SOLS;

    if (si.pending_hashdonef[slot]) si.pending_hashdonef[slot]();

    for (uint32_t s = 0; s < n; s++) {
        if (!sols->valid[s]) continue;
        if (si.pending_cancelf[slot] && si.pending_cancelf[slot]()) break;

        /* Copy sol indices out of the pinned buffer -- verify_and_sort
         * writes in place, so we can't touch the source buffer after the
         * event fires because the NEXT launch that rotates into this slot
         * may overwrite it. We work on a local copy. */
        uint32_t local[1 << EQ_PARAM_K];
        std::memcpy(local, sols->values[s], sizeof(local));

        g_raw_candidates.fetch_add(1, std::memory_order_relaxed);
        if (!verify_and_sort_indices(local)) {
            g_dropped_by_filter.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        std::vector<uint32_t> indices(local, local + (1 << EQ_PARAM_K));
        if (si.pending_solf[slot]) si.pending_solf[slot](indices, EQ_PREFIX, nullptr);
    }

    si.pending_solf[slot]      = nullptr;
    si.pending_hashdonef[slot] = nullptr;
    si.pending_cancelf[slot]   = nullptr;
    si.pipeline_drained++;
}

void amd_hip::solve_pipelined(const char* tequihash_header, unsigned int tequihash_header_len,
    const char* nonce, unsigned int nonce_len,
    std::function<bool()> cancelf,
    std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
    std::function<void(void)> hashdonef,
    amd_hip& ctx)
{
    if (!ctx.context || !ctx.context->ready) return;
    auto& c = *ctx.context;
    if (cancelf && cancelf()) return;

    /* Round-robin across streams. For n_streams=1 this is always stream 0
     * and behaves exactly like the previous single-stream pipelined path.
     * For n_streams=2 we interleave: iter N on stream (N%2), which gives
     * both streams a chance to run kernels concurrently IF the HIP/KMD
     * scheduler is willing to overlap them on the same GPU. */
    int stream_idx = c.next_stream;
    c.next_stream = (c.next_stream + 1) % c.n_streams;
    StreamInstance& si = c.s[stream_idx];

    /* If THIS stream has 2 in flight, drain its oldest before launching
     * another onto it (our pinned sols slot is 2-way rotation). The newer
     * in-flight iter on this same stream keeps the GPU busy during the
     * event wait; on the other stream, GPU work also keeps running. */
    if (si.pipeline_iters - si.pipeline_drained >= 2) {
        drain_one_slot(si);
    }

    uint8_t header140[140] = {0};
    unsigned int hlen = std::min<unsigned int>(tequihash_header_len, 140);
    std::memcpy(header140, tequihash_header, hlen);
    unsigned int nbytes = std::min<unsigned int>(nonce_len, 140 - hlen);
    std::memcpy(header140 + hlen, nonce, nbytes);

    CHECK_HIP_RET(hipSetDevice(ctx.device_id));

    int slot = si.launch_iter(header140);
    si.pending_solf[slot]      = std::move(solutionf);
    si.pending_hashdonef[slot] = std::move(hashdonef);
    si.pending_cancelf[slot]   = std::move(cancelf);
}

void amd_hip::flush_pipeline(amd_hip& ctx)
{
    if (!ctx.context || !ctx.context->ready) return;
    auto& c = *ctx.context;
    /* Drain each stream independently. Order between streams doesn't
     * matter since iters are identified by (stream, slot) and the pool
     * consumes share submissions without ordering requirements. */
    for (int i = 0; i < c.n_streams; i++) {
        while (c.s[i].pipeline_iters > c.s[i].pipeline_drained) {
            drain_one_slot(c.s[i]);
        }
    }
}

amd_hip::ProfileResult amd_hip::profile(int n_iters, amd_hip& ctx)
{
    ProfileResult out{};
    if (!ctx.context || !ctx.context->ready || n_iters <= 0) return out;
    auto& c = *ctx.context;
    CHECK_HIP_RET(hipSetDevice(ctx.device_id));

    /* Synthetic header + nonce, reused across iters. */
    uint8_t header140[140] = {0};
    for (int i = 0; i < 140; i++) header140[i] = (uint8_t)(i * 37 + 13);

    /* Warm up GPU to steady-state clocks before timing. */
    for (int i = 0; i < 32; i++) (void)run_once_profiled(c.s[0], header140);

    ProfileResult acc{};
    acc.ht_count = 9;
    for (int i = 0; i < n_iters; i++) {
        auto r = run_once_profiled(c.s[0], header140);
        acc.ms_total         += r.ms_total;
        acc.ms_h2d_midstate  += r.ms_h2d_midstate;
        acc.ms_kernel_sols   += r.ms_kernel_sols;
        acc.ms_d2h           += r.ms_d2h;
        for (int k = 0; k < r.ht_count; k++) {
            acc.ms_init_ht[k]    += r.ms_init_ht[k];
            acc.ms_init_ht_total += r.ms_init_ht[k];
        }
        for (int k = 0; k < 9; k++) {
            acc.ms_round[k]    += r.ms_round[k];
            acc.ms_round_total += r.ms_round[k];
        }
    }
    double inv = 1.0 / (double)n_iters;
    acc.ms_total         *= inv;
    acc.ms_h2d_midstate  *= inv;
    acc.ms_kernel_sols   *= inv;
    acc.ms_d2h           *= inv;
    acc.ms_init_ht_total *= inv;
    acc.ms_round_total   *= inv;
    for (int k = 0; k < 9; k++) {
        acc.ms_init_ht[k] *= inv;
        acc.ms_round[k]   *= inv;
    }
    return acc;
}
