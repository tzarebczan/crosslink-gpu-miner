#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

/*
 * AMD HIP backend: drives the Equihash(200,9) solver via HIP kernels
 * compiled by hipcc (see hip_impl.hip). The struct's public API matches
 * amd_opencl so Solver<amd_hip> works unchanged.
 *
 * This header MUST stay MSVC-clean: no hip/hip_runtime.h, no HIP types.
 * The opaque amd_hip_context* is allocated & freed inside the .hip TU.
 */

struct amd_hip_context;

struct amd_hip {
	int device_id;
	int platform_id;
	amd_hip_context* context;

	std::string m_gpu_name;
	int m_cu_count;
	std::string m_version;

	amd_hip(int platf_id, int dev_id);
	~amd_hip();

	std::string getdevinfo();

	static int getcount();
	static void getinfo(int platf_id, int d_id, std::string& gpu_name, int& sm_count, std::string& version);

	static void start(amd_hip& device_context);
	static void stop(amd_hip& device_context);

	static void solve(const char* tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef,
		amd_hip& device_context);

	/* Pipelined entry: queues this iter's GPU work, then (if a previous
	 * iter is pending) blocks on that earlier iter, runs host-side sol
	 * validation, and dispatches solutionf/hashdonef for its sols. The
	 * net effect is that CPU post-processing of iter N-1 overlaps with
	 * GPU execution of iter N. Caller must finally call flush_pipeline
	 * to drain the last in-flight iter.
	 *
	 * Thread-safety: must be called from the same thread throughout a
	 * session (it mutates per-context slot bookkeeping). */
	static void solve_pipelined(const char* tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef,
		amd_hip& device_context);

	/* Drain any iters still in flight from solve_pipelined, running the
	 * solutionf/hashdonef that were passed in with each original
	 * solve_pipelined call. Safe to call when nothing is pending. */
	static void flush_pipeline(amd_hip& device_context);

	std::string getname() { return "AMD-HIP"; }

	static void print_amd_hip_devices();

	/* Diagnostic counters, updated by solve(). Helpful to distinguish raw
	 * GPU candidate throughput from pool-valid-sol throughput (the host
	 * dedup+sort filter typically drops ~15-25% of Equihash(200,9) raw
	 * candidates). Thread-safe; reads may race with writes but are fine
	 * for display purposes. */
	static uint64_t get_raw_candidates();
	static uint64_t get_dropped_by_filter();
	static int      get_n_streams(amd_hip& ctx);

	/* Per-kernel wall-clock profile of a single iter. Returns averaged
	 * timings (ms) over n_iters serialized runs (graph disabled so we can
	 * measure each kernel separately via hipEvent_t). For tuning only --
	 * not used on the hot path. */
	struct ProfileResult {
		double ms_total;
		double ms_h2d_midstate;
		double ms_init_ht_total;
		double ms_init_ht[9];
		int    ht_count;
		double ms_round[9];
		double ms_round_total;
		double ms_kernel_sols;
		double ms_d2h;
	};
	static ProfileResult profile(int n_iters, amd_hip& device_context);
};
