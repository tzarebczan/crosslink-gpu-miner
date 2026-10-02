/*
 * gpu_compat.h -- HIP-to-CUDA alias layer for the native-CUDA build path.
 *
 * On hipcc: forwards to <hip/hip_runtime.h>; on plain nvcc: provides
 * CUDA-runtime aliases for the HIP symbols used in hip_impl.hip.
 * Extending the HIP API surface in hip_impl.hip may require adding aliases
 * below; missing symbols produce a clean compile error.
 */
#pragma once

#if defined(__CUDACC__) && !defined(__HIPCC__)

#include <cuda_runtime.h>
#include <cstdio>

using hipError_t        = cudaError_t;
using hipStream_t       = cudaStream_t;
using hipEvent_t        = cudaEvent_t;
using hipGraph_t        = cudaGraph_t;
using hipGraphExec_t    = cudaGraphExec_t;
using hipDeviceProp_t   = cudaDeviceProp;

#define hipSuccess                     cudaSuccess
#define hipGetErrorString              cudaGetErrorString
#define hipSetDevice                   cudaSetDevice
#define hipGetDeviceCount              cudaGetDeviceCount
#define hipGetDeviceProperties         cudaGetDeviceProperties
#define hipMalloc                      cudaMalloc
#define hipFree                        cudaFree
#define hipMemcpyAsync                 cudaMemcpyAsync
#define hipMemcpyHostToDevice          cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost          cudaMemcpyDeviceToHost
#define hipHostMalloc                  cudaHostAlloc
#define hipHostMallocDefault           cudaHostAllocDefault
#define hipHostFree                    cudaFreeHost
#define hipStreamCreate                cudaStreamCreate
#define hipStreamCreateWithFlags       cudaStreamCreateWithFlags
#define hipStreamDestroy               cudaStreamDestroy
#define hipStreamSynchronize           cudaStreamSynchronize
#define hipStreamNonBlocking           cudaStreamNonBlocking
#define hipStreamBeginCapture          cudaStreamBeginCapture
#define hipStreamEndCapture            cudaStreamEndCapture
#define hipStreamCaptureModeGlobal     cudaStreamCaptureModeGlobal
#define hipEventCreate                 cudaEventCreate
#define hipEventCreateWithFlags        cudaEventCreateWithFlags
#define hipEventDestroy                cudaEventDestroy
#define hipEventRecord                 cudaEventRecord
#define hipEventSynchronize            cudaEventSynchronize
#define hipEventElapsedTime            cudaEventElapsedTime
#define hipEventDisableTiming          cudaEventDisableTiming
#define hipGraphLaunch                 cudaGraphLaunch
#define hipGraphDestroy                cudaGraphDestroy
#define hipGraphExecDestroy            cudaGraphExecDestroy

#define hipLaunchKernelGGL(kernel, grid, block, shmem, stream, ...) \
    kernel<<<(grid), (block), (shmem), (stream)>>>(__VA_ARGS__)

/* AMD exposes a gcnArchName ("gfx1201"); NVIDIA's cudaDeviceProp does not.
 * Synthesize "sm_XX" from compute cap. Caller must consume the returned
 * pointer before the next call on this thread (std::string assignment is
 * the expected use; do not chain two calls in one expression). */
static inline const char* gpu_arch_string(const cudaDeviceProp& p) {
    static thread_local char buf[16];
    std::snprintf(buf, sizeof buf, "sm_%d%d", p.major, p.minor);
    return buf;
}
#define GHOST_GPU_ARCH_STRING(p) gpu_arch_string(p)

/* cudaGraphInstantiate signature moved from 5-arg (error-reporting) to
 * 3-arg (flags) in CUDA 12.0. The HIP form is still 5-arg. Wrap both so
 * hip_impl.hip calls a single stable helper. */
static inline cudaError_t ghost_graph_instantiate(cudaGraphExec_t* exec, cudaGraph_t graph) {
#if CUDART_VERSION >= 12000
    return cudaGraphInstantiate(exec, graph, /*flags=*/0ULL);
#else
    return cudaGraphInstantiate(exec, graph, /*errorNode=*/nullptr,
                                /*logBuffer=*/nullptr, /*bufferSize=*/0);
#endif
}

/* __shfl is deprecated on CUDA 9+ (sm_70 Volta+ wants explicit active-lane
 * mask). We always drive shuffles with all 32 lanes active (see the coop
 * kernel re-structure in hip_impl.hip), so the safe mask is 0xFFFFFFFFu.
 * Hoisting into a helper keeps hip_impl.hip readable for both targets.
 *
 * NB: templated on T so uint32_t, uint64_t, int, float all work. nvcc's
 * __shfl_sync already has overloads for these types. */
template<typename T>
__device__ __forceinline__ T ghost_shfl(T val, int srcLane, int width) {
    return __shfl_sync(0xFFFFFFFFu, val, srcLane, width);
}

/* Read-only cache hint. On NVIDIA this routes the load through the 48KB
 * read-only data cache (separate from L1) which helps when the same
 * cache lines are read many times across the kernel -- e.g. a row's
 * hash-table slots re-read by every pair in the plain wave32 path.
 * Use only when the pointed-to memory is guaranteed not to be written
 * by any thread in this kernel launch.
 *
 * On HIP we keep the same helper surface to keep call sites portable;
 * AMD has __ldg() too and it generally maps to a cacheable load with a
 * similar "don't pollute the coherent path" hint. Both are semantically
 * a pure load -- just a codegen nudge. */
template<typename T>
__device__ __forceinline__ T ghost_ldg(const T* __restrict__ ptr) {
    return __ldg(ptr);
}

#else  /* hipcc path */

#include <hip/hip_runtime.h>
#define GHOST_GPU_ARCH_STRING(p) ((p).gcnArchName)

static inline hipError_t ghost_graph_instantiate(hipGraphExec_t* exec, hipGraph_t graph) {
    return hipGraphInstantiate(exec, graph, /*errorNode=*/nullptr,
                               /*logBuffer=*/nullptr, /*bufferSize=*/0);
}

template<typename T>
__device__ __forceinline__ T ghost_shfl(T val, int srcLane, int width) {
    return __shfl(val, srcLane, width);
}

template<typename T>
__device__ __forceinline__ T ghost_ldg(const T* __restrict__ ptr) {
    return __ldg(ptr);
}

#endif
