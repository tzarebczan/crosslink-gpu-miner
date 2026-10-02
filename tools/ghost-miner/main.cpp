/*
 * ghost-miner -- minimal Equihash(200,9) Zcash stratum miner.
 *
 * Pipes the proven amd_hip backend into a hand-rolled stratum client so
 * we can mine against a real pool / Zebra+s-nomp endpoint without dragging
 * in Boost / nheqminer's full plumbing.
 *
 * Usage:
 *   ghost-miner.exe -l host:port -u t1addr.worker [-p password] [-d 0]
 *
 * What it does:
 *   1. Connect, mining.subscribe (-> extranonce1) + mining.authorize
 *   2. On mining.notify: build 108-byte header prefix, store target
 *   3. Mining loop: pick a 32-byte nonce starting with extranonce1 + a
 *      monotonic counter, call amd_hip::solve(), for each candidate sol:
 *      - encode 512 indices into 1344-byte compact form
 *      - compute Blake2b-256(header_140 || varint || compact_sol)
 *      - if hash <= target, mining.submit
 *   4. On any new job (even non-clean), restart the loop with the new header.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <ctime>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../../amd_hip/amd_hip.hpp"
#include "equihash_compact.hpp"
#include "stratum.hpp"

namespace {

struct CliArgs {
    std::string host;
    uint16_t    port = 0;
    std::string user;
    std::string password = "x";
    int         device   = 0;
    bool        debug    = false;
    int         bench_iters = 0;   // if > 0, run standalone bench and exit.
    int         profile_iters = 0; // if > 0, run kernel-level profile and exit.
    double      min_diff = 0.0;    // if > 0, local floor intersected with set_target.
};

bool parse_args(int argc, char** argv, CliArgs& a)
{
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto need = [&](int& i_) -> const char* {
            if (i_ + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", argv[i_]);
                std::exit(2);
            }
            return argv[++i_];
        };
        if (s == "-l" || s == "--location") {
            std::string hp = need(i);
            auto colon = hp.find(':');
            if (colon == std::string::npos) {
                std::fprintf(stderr, "expected host:port for -l\n");
                return false;
            }
            a.host = hp.substr(0, colon);
            a.port = (uint16_t)std::atoi(hp.substr(colon + 1).c_str());
        } else if (s == "-u" || s == "--user") {
            a.user = need(i);
        } else if (s == "-p" || s == "--password") {
            a.password = need(i);
        } else if (s == "-d" || s == "--device") {
            a.device = std::atoi(need(i));
        } else if (s == "--debug") {
            a.debug = true;
        } else if (s == "--bench") {
            a.bench_iters = std::atoi(need(i));
        } else if (s == "--profile") {
            a.profile_iters = std::atoi(need(i));
        } else if (s == "--min-diff") {
            a.min_diff = std::strtod(need(i), nullptr);
        } else if (s == "-h" || s == "--help") {
            std::printf(
                "ghost-miner -- HIP Equihash(200,9) miner\n"
                "Usage: ghost-miner -l HOST:PORT -u WALLET[.WORKER] [-p PASS] [-d DEVICE]\n"
                "       ghost-miner --bench N [-d DEVICE]\n"
                "  -l HOST:PORT     stratum endpoint\n"
                "  -u WALLET[.W]    payout address (.worker optional)\n"
                "  -p PASS          pool password (default: x)\n"
                "  -d N             HIP device index (default: 0)\n"
                "  --bench N        run N solve iterations with a synthetic\n"
                "                   header and print Sol/s (no pool needed)\n"
                "  --min-diff D     local-side difficulty floor (e.g. 5.0). Shares\n"
                "                   below this diff are filtered out before submit,\n"
                "                   even if the pool's set_target is laxer.\n");
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return false;
        }
    }
    if (a.bench_iters > 0 || a.profile_iters > 0) return true;
    return !a.host.empty() && a.port != 0 && !a.user.empty();
}

// --- shared mining state, written by stratum thread, read by mining loop ---
struct JobState {
    std::mutex                mu;
    bool                      have_job = false;
    uint64_t                  job_seq  = 0;
    ghost::StratumJob         job;
    std::vector<uint8_t>      target_le;     // 32 bytes, little-endian compare
    std::vector<uint8_t>      nonce1_be;     // server-controlled bytes (raw, as received)
};

JobState g_job;
std::atomic<uint64_t> g_consumed_seq{0};
std::atomic<bool>     g_stop{false};

uint64_t now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        clock::now().time_since_epoch()).count();
}

/* Build a 32-byte LE target from a share difficulty using the Zcash
 * powLimit convention (diff1 = 2^243). Matches what SNOMP-style pools
 * use when constructing `mining.set_target` payloads.
 *
 *     target_bignum = 2^243 / diff
 *
 * Returned as a 32-byte little-endian array where byte[31] is the most
 * significant byte -- the same layout `hash_le_target` expects. Used by
 * the `--min-diff` floor to filter locally tighter than the pool asks. */
std::vector<uint8_t> target_from_diff(double diff)
{
    std::vector<uint8_t> out(32, 0);
    if (!(diff > 0.0)) return out;
    // log2(target) = 243 - log2(diff)
    double log2_tgt = 243.0 - std::log2(diff);
    if (log2_tgt < 0.0) {
        std::fill(out.begin(), out.end(), 0);
        out[0] = 1;
        return out;
    }
    if (log2_tgt >= 256.0) {
        std::fill(out.begin(), out.end(), 0xff);
        return out;
    }
    int bit = (int)log2_tgt;
    double frac = log2_tgt - (double)bit;
    uint64_t mant = (uint64_t)(std::pow(2.0, frac) * (double)(1ULL << 53));
    for (int b = bit; b >= 0 && b > bit - 54; b--) {
        int shift = bit - b;
        uint64_t bit_val = (mant >> (53 - shift)) & 1;
        if (bit_val) {
            int byte = b >> 3;
            int off  = b & 7;
            if (byte >= 0 && byte < 32) out[byte] |= (uint8_t)(1u << off);
        }
    }
    return out;
}

/* Return true iff `a <= b`, where both are 32-byte LE bignums (byte 31
 * = most significant). Used to intersect two targets and pick the
 * stricter (smaller-value) one. */
bool target_le_le(const uint8_t a[32], const uint8_t b[32])
{
    for (int i = 31; i >= 0; i--) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return true;
}

/* Wall-clock timestamp as "HH:MM:SS" for human-facing log lines. Local
 * time (what the user sees on their machine's clock). Thread-safe via
 * localtime_s on Windows / localtime_r on POSIX. */
const char* ts() {
    thread_local char buf[16];
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::snprintf(buf, sizeof buf, "%02d:%02d:%02d",
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

} // namespace

int main(int argc, char** argv)
{
    CliArgs a;
    if (!parse_args(argc, argv, a)) {
        std::fprintf(stderr, "Usage: ghost-miner -l HOST:PORT -u WALLET[.WORKER] [-p PASS] [-d DEVICE]\n");
        return 2;
    }

    std::printf("[ghost-miner] starting\n");
    if (a.bench_iters > 0) {
        std::printf("  mode   : bench (%d iters)\n", a.bench_iters);
    } else {
        std::printf("  pool   : %s:%u\n", a.host.c_str(), (unsigned)a.port);
        std::printf("  user   : %s\n",    a.user.c_str());
    }
    std::printf("  device : HIP #%d\n", a.device);

    if (amd_hip::getcount() == 0) {
        std::fprintf(stderr, "[ghost-miner] no HIP devices found\n");
        return 1;
    }
    amd_hip::print_amd_hip_devices();

    // Clean shutdown on Ctrl-C: stop the mining + reconnect loops.
    std::signal(SIGINT,  [](int){ g_stop = true; });
    std::signal(SIGTERM, [](int){ g_stop = true; });

    amd_hip ctx(0, a.device);
    std::printf("[ghost-miner] device: %s\n", ctx.getdevinfo().c_str());
    amd_hip::start(ctx);

    // ---- Profile mode: dump per-kernel wall-clock averages and exit. ------
    if (a.profile_iters > 0) {
        std::printf("[profile] running %d iters, warming up first...\n", a.profile_iters);
        auto p = amd_hip::profile(a.profile_iters, ctx);
        static const char* round_names[9] = {
            "round0", "round1", "round2", "round3", "round4",
            "round5", "round6", "round7", "round8"
        };
        std::printf("\n=== amd_hip profile (ms/iter, averaged over %d iters) ===\n", a.profile_iters);
        std::printf("  %-20s %8.4f ms\n", "h2d_midstate", p.ms_h2d_midstate);
        std::printf("  %-20s %8.4f ms  (%d calls)\n", "init_ht_total",
                    p.ms_init_ht_total, p.ht_count);
        for (int i = 0; i < p.ht_count; i++) {
            std::printf("    %-18s %8.4f ms\n", "init_ht[i]", p.ms_init_ht[i]);
        }
        std::printf("  %-20s %8.4f ms\n", "round_total", p.ms_round_total);
        for (int i = 0; i < 9; i++) {
            std::printf("    %-18s %8.4f ms\n", round_names[i], p.ms_round[i]);
        }
        std::printf("  %-20s %8.4f ms\n", "kernel_sols",   p.ms_kernel_sols);
        std::printf("  %-20s %8.4f ms\n", "d2h_sols",      p.ms_d2h);
        std::printf("  %-20s %8.4f ms\n", "==> total",     p.ms_total);
        std::printf("\n  (vs graph-path hot loop ~9.4 ms, the overhead here is\n"
                    "   per-kernel launch + event sync; treat the relative\n"
                    "   ordering between kernels as the signal, not absolutes.)\n");
        amd_hip::stop(ctx);
        return 0;
    }

    // ---- Bench mode: synthetic header, N iters, print timings, exit. ------
    if (a.bench_iters > 0) {
        uint8_t hdr[108] = {0};
        // A deterministic but non-trivial header so we don't accidentally
        // measure a degenerate case. The exact bytes don't matter for perf.
        for (int i = 0; i < 108; i++) hdr[i] = (uint8_t)(i * 37 + 13);
        std::vector<uint8_t> nonce(32, 0);

        std::atomic<uint64_t> iters{0}, sols{0};
        std::function<bool()> cancelf = [&]() { return g_stop.load(); };
        std::function<void(void)> hashdonef = [&]() { iters++; };
        std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solf =
            [&](const std::vector<uint32_t>& idx, size_t, const unsigned char*) {
                if (idx.size() == 512) sols++;
            };

        // AMD_HIP_PIPELINE=0 opts out for A/B comparison; default ON.
        bool pipelined = true;
        {
            char buf[8] = {0};
#ifdef _WIN32
            size_t n = 0;
            if (getenv_s(&n, buf, sizeof buf, "AMD_HIP_PIPELINE") == 0 && n > 0)
                pipelined = !(buf[0] == '0');
#else
            const char* e = std::getenv("AMD_HIP_PIPELINE");
            if (e && *e) pipelined = !(e[0] == '0');
#endif
        }
        std::printf("[bench] pipeline = %s, streams = %d\n",
                    pipelined ? "ON" : "OFF", amd_hip::get_n_streams(ctx));

        auto do_solve = [&](const uint8_t* nb) {
            if (pipelined) {
                amd_hip::solve_pipelined((const char*)hdr, 108,
                    (const char*)nb, 32, cancelf, solf, hashdonef, ctx);
            } else {
                amd_hip::solve((const char*)hdr, 108,
                    (const char*)nb, 32, cancelf, solf, hashdonef, ctx);
            }
        };

        // Proper thermal warm-up: RDNA4 takes several seconds to reach
        // steady-state clocks under sustained load. We run until either
        // 5 seconds pass or 20% of the total bench iters elapse, whichever
        // comes first. Without this, short benches report cold-GPU
        // numbers that are 15-20% above warm steady state.
        std::printf("[bench] warming up...\n");
        {
            auto wt0 = std::chrono::steady_clock::now();
            uint64_t wctr = 0;
            int warmup_cap = a.bench_iters / 5;
            if (warmup_cap < 50) warmup_cap = 50;
            for (int i = 0; i < warmup_cap && !g_stop; i++) {
                wctr++;
                std::memcpy(nonce.data(), &wctr, sizeof wctr);
                do_solve(nonce.data());
                double wsecs = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - wt0).count();
                if (wsecs >= 5.0) break;
            }
            if (pipelined) amd_hip::flush_pipeline(ctx);
        }
        iters = 0; sols = 0;
        uint64_t raw_c_before  = amd_hip::get_raw_candidates();
        uint64_t drop_before   = amd_hip::get_dropped_by_filter();

        auto t0 = std::chrono::steady_clock::now();
        uint64_t nctr = 1000000;
        for (int i = 0; i < a.bench_iters && !g_stop; i++) {
            nctr++;
            std::memcpy(nonce.data(), &nctr, sizeof nctr);
            do_solve(nonce.data());
        }
        if (pipelined) amd_hip::flush_pipeline(ctx);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        uint64_t raw_c = amd_hip::get_raw_candidates() - raw_c_before;
        uint64_t drop  = amd_hip::get_dropped_by_filter() - drop_before;
        double   drop_pct = raw_c ? 100.0 * (double)drop / (double)raw_c : 0.0;

        std::printf("\n=== bench results ===\n");
        std::printf("  iters       : %llu\n", (unsigned long long)iters.load());
        std::printf("  elapsed     : %.3f s\n", secs);
        std::printf("  ms / iter   : %.3f\n", (secs * 1000.0) / (double)iters.load());
        std::printf("  valid sols  : %llu\n", (unsigned long long)sols.load());
        std::printf("  raw cand.   : %llu  (filter drop %.1f%%)\n",
            (unsigned long long)raw_c, drop_pct);
        std::printf("  Sol/s (val) : %.2f\n", (double)sols.load() / secs);
        std::printf("  Sol/s (raw) : %.2f\n", (double)raw_c / secs);
        amd_hip::stop(ctx);
        return 0;
    }
    // -----------------------------------------------------------------------

    ghost::StratumClient client;
    client.on_job([](const ghost::StratumJob& j,
                     const std::vector<uint8_t>& target,
                     const std::vector<uint8_t>& nonce1) {
        std::lock_guard<std::mutex> lk(g_job.mu);
        g_job.job        = j;
        g_job.target_le  = target;
        g_job.nonce1_be  = nonce1;
        g_job.have_job   = true;
        g_job.job_seq++;
    });
    std::atomic<uint64_t> shares_total{0}, shares_ok{0}, block_submits{0};
    client.on_accept([&](bool ok, const std::string& reason) {
        if (ok) { shares_ok++; std::printf("[%s][ghost-miner] share ACCEPTED (%llu/%llu)\n",
            ts(), (unsigned long long)shares_ok.load(),
            (unsigned long long)shares_total.load()); }
        else    { std::printf("[%s][ghost-miner] share REJECTED (%s)\n", ts(), reason.c_str()); }
    });

    // Outer reconnect loop with exponential backoff (1s, 2s, 4s, ... capped
    // at 60s). We reset the mining stats on every successful (re)connect so
    // the displayed Sol/s is for the current session, not a weighted blend
    // of a stalled one.
    uint64_t backoff_ms = 1000;
    const uint64_t backoff_cap_ms = 60000;
    uint64_t attempt = 0;

    while (!g_stop) {
        attempt++;
        if (attempt > 1) {
            std::printf("[ghost-miner] reconnect attempt #%llu in %llu ms\n",
                (unsigned long long)attempt, (unsigned long long)backoff_ms);
            for (uint64_t slept = 0; slept < backoff_ms && !g_stop; slept += 100)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (g_stop) break;
        }

        if (!client.connect(a.host, a.port, a.user, a.password, "ghost-miner/0.1")) {
            backoff_ms = std::min(backoff_cap_ms, backoff_ms * 2);
            continue;
        }
        // Forget any stale job + state from a previous session.
        {
            std::lock_guard<std::mutex> lk(g_job.mu);
            g_job.have_job = false;
            g_job.target_le.clear();
            g_job.nonce1_be.clear();
            g_job.job_seq++;   // invalidates any in-flight cancelf
        }

        // Wait up to 30s for first job + target. If the pool never sends one,
        // treat as a failed connection and back off.
        std::printf("[ghost-miner] waiting for first job...\n");
        bool got_work = false;
        for (int i = 0; i < 300 && !g_stop && client.connected(); i++) {
            {
                std::lock_guard<std::mutex> lk(g_job.mu);
                if (g_job.have_job && !g_job.target_le.empty()) { got_work = true; break; }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (g_stop) break;
        if (!got_work || !client.connected()) {
            std::fprintf(stderr, "[ghost-miner] no job received or socket dropped; reconnecting\n");
            client.disconnect();
            backoff_ms = std::min(backoff_cap_ms, backoff_ms * 2);
            continue;
        }

        // Successful connect -> reset backoff.
        backoff_ms = 1000;

        // Mining loop. Snapshot current job, pick nonce, solve, submit, repeat.
        uint64_t mine_start_ms = now_ms();
        uint64_t hash_iters    = 0;
        uint64_t sols_total    = 0;
        uint64_t nonce_ctr     = 0;

        // Pipelining toggle: default ON (overlap CPU post-processing with next
        // iter's GPU work). Set AMD_HIP_PIPELINE=0 to disable for debugging
        // or A/B. Pipelining requires per-iter lambdas to capture state BY
        // VALUE because callbacks fire 1-2 iters later (after the outer
        // loop's locals have already advanced).
        static const bool use_pipelined = []() {
            char buf[8] = {0};
#ifdef _WIN32
            size_t n = 0;
            if (getenv_s(&n, buf, sizeof buf, "AMD_HIP_PIPELINE") != 0) return true;
            return !(n > 0 && buf[0] == '0');
#else
            const char* e = std::getenv("AMD_HIP_PIPELINE");
            return !(e && *e && e[0] == '0');
#endif
        }();
        // GHOST_GPU_DUTY=1..100. 100 mines continuously. Lower values mine
        // for that fraction of each 500 ms window, then drain the GPU and
        // sleep so the card is not busy the whole time.
        struct GpuDuty {
            int percent = 100;
            int window_ms = 500;
            std::chrono::steady_clock::time_point window_start =
                std::chrono::steady_clock::now();
            int work_ms() const { return window_ms * percent / 100; }
        };
        static GpuDuty gpu_duty = []() {
            GpuDuty d;
            const char* e = std::getenv("GHOST_GPU_DUTY");
            if (e && *e) {
                int v = std::atoi(e);
                if (v < 1) v = 1;
                if (v > 100) v = 100;
                d.percent = v;
            }
            return d;
        }();
        if (hash_iters == 0) {
            std::printf("[ghost-miner] pipeline = %s, streams = %d\n",
                use_pipelined ? "ON" : "OFF", amd_hip::get_n_streams(ctx));
            if (gpu_duty.percent < 100) {
                std::printf("[ghost-miner] GPU duty = %d%% (%d ms on / %d ms off)\n",
                    gpu_duty.percent, gpu_duty.work_ms(),
                    gpu_duty.window_ms - gpu_duty.work_ms());
            }
            if (a.min_diff > 0.0) {
                std::printf("[ghost-miner] local min-diff floor = %.6f "
                    "(filters shares below this before submit)\n", a.min_diff);
            }
        }

        // Precompute the min-diff floor target once per reconnect. Constant
        // for the life of the connection; intersected with each job's
        // set_target below to get the effective filter.
        std::vector<uint8_t> floor_target;
        if (a.min_diff > 0.0) floor_target = target_from_diff(a.min_diff);

        while (!g_stop && client.connected()) {
        ghost::StratumJob job;
        std::vector<uint8_t> target_le;
        std::vector<uint8_t> nonce1;
        uint64_t local_seq;
        {
            std::lock_guard<std::mutex> lk(g_job.mu);
            job       = g_job.job;
            target_le = g_job.target_le;
            nonce1    = g_job.nonce1_be;
            local_seq = g_job.job_seq;
        }
        g_consumed_seq = local_seq;

        // Intersect pool's set_target with our local --min-diff floor (if
        // any): pick the stricter (smaller-value) target. Guarantees we
        // never submit below the user's chosen floor, regardless of what
        // the pool advertises.
        if (!floor_target.empty() && target_le.size() == 32) {
            if (!target_le_le(target_le.data(), floor_target.data())) {
                target_le = floor_target;
            }
        }

        std::vector<uint8_t> hdr108 = job.header_prefix();   // 108 B (no nonce / sol)
        if (hdr108.size() != 108) {
            std::fprintf(stderr, "[ghost-miner] header_prefix size = %zu (want 108)\n", hdr108.size());
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        // Build 32-byte nonce: [nonce1][counter LE in remaining bytes]
        std::vector<uint8_t> nonce(32, 0);
        size_t n1len = nonce1.size();
        if (n1len > 32) n1len = 32;
        std::memcpy(nonce.data(), nonce1.data(), n1len);
        for (size_t i = 0; i < 8 && n1len + i < 32; i++) {
            nonce[n1len + i] = (uint8_t)((nonce_ctr >> (8 * i)) & 0xff);
        }
        nonce_ctr++;

        // Stale check: if a new clean job arrived since we snapshot, break
        // out so the outer loop re-reads. With pipelining, we still launch
        // this iter (already-queued iters in flight will retire naturally
        // via flush_pipeline when we break). In sync mode, cancelf below
        // lets the solver abort early inside solve().
        {
            std::lock_guard<std::mutex> lk(g_job.mu);
            if (g_job.job_seq != local_seq && job.clean) {
                if (use_pipelined) amd_hip::flush_pipeline(ctx);
                continue;   // re-snapshot the new job; break would drop the pool
            }
        }

        // --- Per-iter lambdas. Capture iter-specific state BY VALUE so the
        // pipelined path can invoke them 1-2 iters later without dangling
        // refs into a by-then-advanced outer loop. Persistent state
        // (counters, client) is captured by ref since it outlives iters.

        std::function<bool()> cancelf = []() { return g_stop.load(); };
        std::function<void(void)> hashdonef = [&hash_iters]() { hash_iters++; };

        auto this_nonce   = nonce;      // by-value snapshots
        auto this_hdr108  = hdr108;
        auto this_job     = job;
        auto this_target  = target_le;
        auto this_n1len   = n1len;

        std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solf =
            [this_nonce, this_hdr108, this_job, this_target, this_n1len,
             &sols_total, &shares_total, &shares_ok, &block_submits, &client]
            (const std::vector<uint32_t>& indices, size_t cbitlen, const unsigned char* /*compact*/)
        {
            if (indices.size() != 512) return;
            sols_total++;
            // Pack 512 x 21-bit indices into 1344-byte compact solution.
            std::vector<uint8_t> compact = ghost::compress_indices(indices, cbitlen ? cbitlen : 20);
            if (compact.size() != 1344) {
                std::fprintf(stderr, "[ghost-miner] compact size = %zu (want 1344)\n", compact.size());
                return;
            }
            // Local share check: SHA-256d(header_140 || varint(1344) || compact).
            //
            // header_140 = hdr108 || nonce(32). This MUST match what the pool
            // hashes in jobManager.js:
            //     headerSolnBuffer = header || Buffer.from(soln, 'hex')
            //     headerHash       = util.sha256d(headerSolnBuffer)
            //     bignum.fromBuffer(headerHash, { endian: 'little' })
            //
            // Note that SNOMP-derived Zcash pools use SHA-256d here even though
            // the network PoW hash is Blake2b-256. Using Blake2b for the share
            // hash produces 100% "low difficulty" rejects because the two hash
            // functions produce uncorrelated values for the same pre-image.
            std::vector<uint8_t> hdr140;
            hdr140.reserve(140);
            hdr140.insert(hdr140.end(), this_hdr108.begin(), this_hdr108.end());
            hdr140.insert(hdr140.end(), this_nonce.begin(),  this_nonce.end());

            std::vector<uint8_t> preimage;
            preimage.reserve(140 + 3 + 1344);
            preimage.insert(preimage.end(), hdr140.begin(), hdr140.end());
            std::vector<uint8_t> vi = ghost::varint_prefix(1344);
            preimage.insert(preimage.end(), vi.begin(), vi.end());
            preimage.insert(preimage.end(), compact.begin(), compact.end());

            auto digest = ghost::sha256d(preimage);
            static std::atomic<uint64_t> debug_dumped{0};
            static bool debug_on = [](){
                const char* e = std::getenv("GHOST_DEBUG_HASH");
                return e && *e;
            }();
            if (debug_on && debug_dumped.fetch_add(1) < 5) {
                // Print hash (big-endian display = reverse of LE bytes) and
                // current target so we can eyeball if Blake2b+target are sane.
                std::string hbe, tbe;
                for (int i = 31; i >= 0; i--) {
                    char tmp[3]; std::snprintf(tmp, sizeof tmp, "%02x", digest.data()[i]);
                    hbe += tmp;
                }
                for (int i = 31; i >= 0; i--) {
                    char tmp[3]; std::snprintf(tmp, sizeof tmp, "%02x", this_target.data()[i]);
                    tbe += tmp;
                }
                std::printf("[debug] hash=%s\n[debug] tgt =%s\n", hbe.c_str(), tbe.c_str());
            }
            // GHOST_DEBUG_WIRE: dump the full 140-byte header + the submit
            // solution hex for the first share. Useful for diagnosing
            // [20,"invalid solution"] against Zebra/Crosslink SNOMP forks:
            // compare the 140-byte header bytes against what the pool
            // reconstructs from its template, and compare the compact
            // solution bytes against the pool's re-encode.
            static std::atomic<int> wire_dumped{0};
            static const bool wire_on = [](){
                char buf[8] = {0};
#ifdef _WIN32
                size_t n = 0;
                if (getenv_s(&n, buf, sizeof buf, "GHOST_DEBUG_WIRE") != 0) return false;
                return n > 0 && buf[0] && buf[0] != '0';
#else
                const char* e = std::getenv("GHOST_DEBUG_WIRE");
                return e && *e && e[0] != '0';
#endif
            }();
            if (wire_on && wire_dumped.fetch_add(1) < 1) {
                std::printf("[wire] version  = %s\n", this_job.version_hex.c_str());
                std::printf("[wire] prevhash = %s\n", this_job.prevhash_hex.c_str());
                std::printf("[wire] merkle   = %s\n", this_job.merkle_hex.c_str());
                std::printf("[wire] reserved = %s\n", this_job.reserved_hex.c_str());
                std::printf("[wire] time     = %s\n", this_job.time_hex.c_str());
                std::printf("[wire] bits     = %s\n", this_job.bits_hex.c_str());
                std::printf("[wire] nonce    = %s\n",
                    ghost::to_hex(this_nonce.data(), 32).c_str());
                std::printf("[wire] header140= %s\n",
                    ghost::to_hex(hdr140.data(), 140).c_str());
                std::printf("[wire] sol_bytes= %s...\n",
                    ghost::to_hex(compact.data(), 32).c_str());
            }
            static const bool force_submit = [](){
                const char* e = std::getenv("GHOST_FORCE_SUBMIT");
                return e && *e;
            }();
            static std::atomic<int> forced_count{0};
            bool passes_target = ghost::hash_le_target(digest.data(), this_target.data());
            // Share hash is SHA-256d(header||sol). The block hash is that
            // same preimage plus the fat pointer, so a real block often
            // misses the share target. Submit those anyway; the pool checks
            // the fat-pointer hash before it applies share difficulty.
            bool block_hit = false;
            if (this_job.block_target_le.size() == 32 && !this_job.fat_pointer.empty()) {
                std::vector<uint8_t> cand = preimage;
                cand.insert(cand.end(), this_job.fat_pointer.begin(), this_job.fat_pointer.end());
                auto cd = ghost::sha256d(cand);
                block_hit = ghost::hash_le_target(cd.data(), this_job.block_target_le.data());
            }
            if (!passes_target && !block_hit) {
                if (!force_submit || forced_count.fetch_add(1) >= 1) {
                    // sol valid, but doesn't meet pool diff -- skip.
                    return;
                }
                std::printf("[ghost-miner] FORCING submit for protocol test (expect reject)\n");
            }
            if (block_hit) {
                block_submits++;
                std::printf("[%s][ghost-miner] block candidate (share %s)\n",
                    ts(), passes_target ? "met" : "below target");
            }

            // Submit. nonce2 = bytes[n1len..32) of nonce.
            ghost::StratumSubmission sub;
            sub.job_id       = this_job.job_id;
            sub.time_hex     = this_job.time_hex;
            sub.nonce2_hex   = ghost::to_hex(this_nonce.data() + this_n1len, 32 - this_n1len);
            // VarInt(1344) = fd 40 05; total length = 3+1344 = 1347 bytes.
            std::vector<uint8_t> sol_with_prefix;
            sol_with_prefix.reserve(3 + 1344);
            sol_with_prefix.insert(sol_with_prefix.end(), vi.begin(), vi.end());
            sol_with_prefix.insert(sol_with_prefix.end(), compact.begin(), compact.end());
            sub.solution_hex = ghost::to_hex(sol_with_prefix.data(), sol_with_prefix.size());
            shares_total++;
            std::printf("[%s][ghost-miner] submit job=%s nonce2=%s hash=%s\n",
                ts(), sub.job_id.c_str(), sub.nonce2_hex.c_str(),
                ghost::to_hex(digest.data(), 8).c_str());
            client.submit(sub);
        };

        if (gpu_duty.percent < 100) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - gpu_duty.window_start).count();
            if (elapsed >= gpu_duty.window_ms) {
                gpu_duty.window_start = now;
            } else if (elapsed >= gpu_duty.work_ms()) {
                if (use_pipelined) amd_hip::flush_pipeline(ctx);
                auto deadline = now + std::chrono::milliseconds(gpu_duty.window_ms - elapsed);
                while (!g_stop && std::chrono::steady_clock::now() < deadline) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                gpu_duty.window_start = std::chrono::steady_clock::now();
                if (g_stop) break;
            }
        }

        if (use_pipelined) {
            amd_hip::solve_pipelined(reinterpret_cast<const char*>(hdr108.data()),
                                     (unsigned)hdr108.size(),
                                     reinterpret_cast<const char*>(nonce.data()),
                                     (unsigned)nonce.size(),
                                     cancelf, solf, hashdonef, ctx);
        } else {
            amd_hip::solve(reinterpret_cast<const char*>(hdr108.data()),
                           (unsigned)hdr108.size(),
                           reinterpret_cast<const char*>(nonce.data()),
                           (unsigned)nonce.size(),
                           cancelf, solf, hashdonef, ctx);
        }

        if ((nonce_ctr & 0x1F) == 0) {
            uint64_t dt = now_ms() - mine_start_ms;
            if (dt > 0) {
                // sols_total counts pool-valid candidates (post dedup+sort filter).
                // raw counts every GPU-emitted candidate before we drop the ~15-25%
                // with duplicate indices. Both are useful: raw = GPU throughput,
                // validated = what the pool would actually accept.
                double sps  = (double)sols_total * 1000.0 / (double)dt;
                uint64_t raw     = amd_hip::get_raw_candidates();
                uint64_t dropped = amd_hip::get_dropped_by_filter();
                double rsps = (double)raw * 1000.0 / (double)dt;
                double drop_pct = raw ? 100.0 * (double)dropped / (double)raw : 0.0;
                std::printf("[%s][ghost-miner] %.2f Sol/s (raw %.2f, filter drop %.1f%%) "
                    "iters=%llu sols=%llu shares=%llu accepted=%llu blocks=%llu\n",
                    ts(), sps, rsps, drop_pct,
                    (unsigned long long)hash_iters, (unsigned long long)sols_total,
                    (unsigned long long)shares_total.load(),
                    (unsigned long long)shares_ok.load(),
                    (unsigned long long)block_submits.load());
            }
        }
        }  // inner mining loop

        // Reached here via disconnect, g_stop, or stale-job break. Drain any
        // in-flight GPU iters before tearing down so their callbacks (which
        // hold refs into the client) fire before disconnect.
        if (use_pipelined) amd_hip::flush_pipeline(ctx);

        if (!g_stop) {
            std::fprintf(stderr, "[ghost-miner] pool connection dropped; will reconnect\n");
        }
        client.disconnect();
    }  // outer reconnect loop

    client.stop();
    amd_hip::stop(ctx);
    return 0;
}
