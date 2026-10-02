#pragma once
/*
 * Minimal Zcash NiceHash stratum client (newline-delimited JSON-RPC).
 *
 * Implements the producer/consumer side of:
 *   mining.subscribe / mining.authorize           (client -> server)
 *   mining.set_target / mining.notify              (server -> client)
 *   mining.submit                                  (client -> server)
 *
 * Tested against s-nomp / Zebra / equihash-style stratum endpoints.
 *
 * Networking: blocking sockets (Winsock on Windows). One reader thread that
 * pumps incoming lines into a thread-safe queue, callbacks from there. Submit
 * uses the same socket guarded by a mutex.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
#define GHOST_INVALID_SOCKET INVALID_SOCKET
#define ghost_closesocket    closesocket
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
#define GHOST_INVALID_SOCKET (-1)
#define ghost_closesocket    close
#endif

namespace ghost {

// --------------------------------------------------------- tiny JSON extractor
// We only need to pull a handful of well-known fields out of stratum JSON.
// A tiny scanner is enough; no need to drag in a JSON library.

inline std::string json_unescape(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char c = s[i + 1];
            switch (c) {
                case 'n': out.push_back('\n'); break;
                case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break;
                case '"': out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/');  break;
                default:  out.push_back(c);    break;
            }
            i++;
        } else out.push_back(s[i]);
    }
    return out;
}

// Returns the position after the matching ", ], or }; or std::string::npos on error.
inline size_t skip_json_value(const std::string& s, size_t pos)
{
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) pos++;
    if (pos >= s.size()) return std::string::npos;
    char c = s[pos];
    if (c == '"') {
        pos++;
        while (pos < s.size()) {
            if (s[pos] == '\\') { pos += 2; continue; }
            if (s[pos] == '"') return pos + 1;
            pos++;
        }
        return std::string::npos;
    }
    if (c == '{' || c == '[') {
        char close = (c == '{') ? '}' : ']';
        int depth = 1;
        pos++;
        while (pos < s.size() && depth > 0) {
            if (s[pos] == '"') {
                pos = skip_json_value(s, pos);
                if (pos == std::string::npos) return pos;
            } else if (s[pos] == '{' || s[pos] == '[') { depth++; pos++; }
            else if (s[pos] == '}' || s[pos] == ']')   { depth--; pos++; }
            else pos++;
        }
        return pos;
    }
    while (pos < s.size() && s[pos] != ',' && s[pos] != '}' && s[pos] != ']') pos++;
    return pos;
}

// Find the value associated with `"key":` at the JSON object root.
// Returns (start, end) of the raw value including its surrounding quotes/brackets.
inline bool find_json_field(const std::string& s, const std::string& key,
                            size_t& out_start, size_t& out_end)
{
    std::string needle = "\"" + key + "\"";
    size_t pos = 0;
    while ((pos = s.find(needle, pos)) != std::string::npos) {
        size_t after = pos + needle.size();
        while (after < s.size() && (s[after] == ' ' || s[after] == '\t')) after++;
        if (after < s.size() && s[after] == ':') {
            after++;
            while (after < s.size() && (s[after] == ' ' || s[after] == '\t')) after++;
            size_t end = skip_json_value(s, after);
            if (end == std::string::npos) return false;
            out_start = after;
            out_end   = end;
            return true;
        }
        pos = after;
    }
    return false;
}

inline bool get_json_string(const std::string& s, const std::string& key, std::string& val)
{
    size_t a, b;
    if (!find_json_field(s, key, a, b)) return false;
    if (s[a] != '"' || b - a < 2) return false;
    val = json_unescape(s.substr(a + 1, b - a - 2));
    return true;
}

// Pull params array contents (raw, including surrounding []) for a notify-style msg.
inline bool get_json_raw(const std::string& s, const std::string& key, std::string& val)
{
    size_t a, b;
    if (!find_json_field(s, key, a, b)) return false;
    val = s.substr(a, b - a);
    return true;
}

// Split a [...] JSON array at top level into raw element strings.
inline std::vector<std::string> split_json_array(const std::string& s)
{
    std::vector<std::string> out;
    if (s.empty() || s.front() != '[') return out;
    size_t pos = 1;
    while (pos < s.size()) {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == ',' || s[pos] == '\t')) pos++;
        if (pos >= s.size() || s[pos] == ']') break;
        size_t end = skip_json_value(s, pos);
        if (end == std::string::npos) break;
        out.push_back(s.substr(pos, end - pos));
        pos = end;
    }
    return out;
}

inline std::string strip_quotes(const std::string& s)
{
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return json_unescape(s.substr(1, s.size() - 2));
    return s;
}

// ---------------------------------------------------------- stratum data types

struct StratumJob {
    std::string job_id;
    std::string version_hex;     // 4 bytes LE -> 8 hex chars
    std::string prevhash_hex;    // 32 bytes -> 64 hex chars
    std::string merkle_hex;      // 32 bytes -> 64 hex chars
    std::string reserved_hex;    // 32 bytes -> 64 hex chars
    std::string time_hex;        // 4 bytes -> 8 hex chars
    std::string bits_hex;        // 4 bytes -> 8 hex chars
    bool        clean = true;
    // Crosslink s-nomp appends these after the usual notify fields.
    // fat_pointer is the raw fat_pointer_to_bft_block. block_target_le is
    // the block target as a 32-byte little-endian integer, same compare
    // the pool uses for candidateBigNumLE <= job.target. Both empty when
    // the pool does not send them.
    std::vector<uint8_t> fat_pointer;
    std::vector<uint8_t> block_target_le;

    // Concatenated 108-byte header prefix (everything except 32-byte nonce
    // and the variable-length solution).
    std::vector<uint8_t> header_prefix() const;
};

struct StratumSubmission {
    std::string job_id;
    std::string time_hex;
    std::string nonce2_hex;      // hex of bytes [nonce1Size .. 32) of the 32-byte nonce
    std::string solution_hex;    // varint-prefixed (3 + 1344) bytes hex
};

// ---------------------------------------------------------- stratum client

class StratumClient {
public:
    using JobCallback = std::function<void(const StratumJob&,
                                            const std::vector<uint8_t>& target,
                                            const std::vector<uint8_t>& nonce1)>;
    using AcceptCallback = std::function<void(bool accepted, const std::string& reason)>;

    StratumClient() = default;
    ~StratumClient() { stop(); }

    // host:port, "wallet[.worker]", "x" (or any pool password).
    // Safe to call repeatedly: prior connection/reader-thread are torn down
    // first so reconnect loops work.
    bool connect(const std::string& host, uint16_t port,
                 const std::string& user, const std::string& password,
                 const std::string& user_agent);

    // Gracefully close the socket but keep the client usable (connect() again).
    // Safe to call from any thread; unblocks reader_loop's recv().
    void disconnect();

    // Fully shut down (disconnect + join reader + Winsock cleanup).
    void stop();

    void on_job(JobCallback cb)    { job_cb_    = std::move(cb); }
    void on_accept(AcceptCallback cb) { accept_cb_ = std::move(cb); }

    bool submit(const StratumSubmission& sub);

    // True iff the TCP socket is still alive and authorized.
    bool connected() const { return alive_.load(); }

private:
    bool send_line(const std::string& line);
    void reader_loop();
    void handle_message(const std::string& line);
    void mark_dead();           // Atomic: close socket + set alive_=false.

    socket_t      socket_   = GHOST_INVALID_SOCKET;
    std::mutex    send_mu_;
    std::thread   reader_;
    std::atomic_bool stop_{false};
    std::atomic_bool alive_{false};

    std::string user_;
    std::string user_agent_;

    // Server-issued. Protected by latest_mu_.
    std::mutex            latest_mu_;
    std::vector<uint8_t>  cur_target_;
    std::vector<uint8_t>  cur_nonce1_;     // 0..32 bytes (LE within Zcash 32B nonce)

    JobCallback    job_cb_;
    AcceptCallback accept_cb_;

    std::atomic<int> next_id_{10};
    std::mutex pending_mu_;
    std::vector<int> pending_submit_ids_;

#ifdef _WIN32
    std::atomic<int> wsa_inited_{0};
#endif
};

// ============================================================ implementations

inline std::vector<uint8_t> StratumJob::header_prefix() const
{
    std::vector<uint8_t> hdr;
    hdr.reserve(140);
    auto append_hex = [&hdr](const std::string& h) {
        for (size_t i = 0; i < h.size(); i += 2) {
            int hi = h[i], lo = h[i + 1];
            auto nib = [](int c) {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return 0;
            };
            hdr.push_back((uint8_t)((nib(hi) << 4) | nib(lo)));
        }
    };
    append_hex(version_hex);
    append_hex(prevhash_hex);
    append_hex(merkle_hex);
    append_hex(reserved_hex);
    append_hex(time_hex);
    append_hex(bits_hex);
    return hdr;
}

inline void StratumClient::mark_dead()
{
    alive_ = false;
    socket_t s = socket_;
    socket_ = GHOST_INVALID_SOCKET;
    if (s != GHOST_INVALID_SOCKET) ghost_closesocket(s);
}

inline void StratumClient::disconnect()
{
    mark_dead();
    if (reader_.joinable() && std::this_thread::get_id() != reader_.get_id()) {
        reader_.join();
    }
    // Wipe transient pool state so a reconnect doesn't submit a stale session's
    // extranonce / target to a newly-connected pool.
    {
        std::lock_guard<std::mutex> lk(latest_mu_);
        cur_target_.clear();
        cur_nonce1_.clear();
    }
    {
        std::lock_guard<std::mutex> lk(pending_mu_);
        pending_submit_ids_.clear();
    }
}

inline bool StratumClient::connect(const std::string& host, uint16_t port,
                                   const std::string& user, const std::string& password,
                                   const std::string& user_agent)
{
    // Allow reconnect: tear down any prior reader / socket.
    disconnect();
    stop_ = false;

#ifdef _WIN32
    if (wsa_inited_.exchange(1) == 0) {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
    }
#endif
    user_       = user;
    user_agent_ = user_agent;

    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    char port_str[16];
    std::snprintf(port_str, sizeof port_str, "%u", (unsigned)port);
    if (getaddrinfo(host.c_str(), port_str, &hints, &res) != 0) {
        std::fprintf(stderr, "[stratum] DNS lookup failed for %s\n", host.c_str());
        return false;
    }

    socket_t s = GHOST_INVALID_SOCKET;
    for (addrinfo* p = res; p; p = p->ai_next) {
        s = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (s == GHOST_INVALID_SOCKET) continue;
        if (::connect(s, p->ai_addr, (int)p->ai_addrlen) == 0) break;
        ghost_closesocket(s);
        s = GHOST_INVALID_SOCKET;
    }
    freeaddrinfo(res);
    if (s == GHOST_INVALID_SOCKET) {
        std::fprintf(stderr, "[stratum] connect failed to %s:%u\n", host.c_str(), (unsigned)port);
        return false;
    }
    socket_ = s;
    alive_  = true;
    std::printf("[stratum] connected to %s:%u\n", host.c_str(), (unsigned)port);

    reader_ = std::thread([this]() { reader_loop(); });

    char buf[512];
    std::snprintf(buf, sizeof buf,
        "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"%s\",null,\"%s\",\"%u\"]}\n",
        user_agent.c_str(), host.c_str(), (unsigned)port);
    if (!send_line(buf)) { mark_dead(); return false; }

    std::snprintf(buf, sizeof buf,
        "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"%s\"]}\n",
        user.c_str(), password.c_str());
    if (!send_line(buf)) { mark_dead(); return false; }

    return true;
}

inline void StratumClient::stop()
{
    stop_ = true;
    mark_dead();
    if (reader_.joinable() && std::this_thread::get_id() != reader_.get_id()) {
        reader_.join();
    }
#ifdef _WIN32
    if (wsa_inited_.exchange(0) == 1) {
        WSACleanup();
    }
#endif
}

inline bool StratumClient::send_line(const std::string& line)
{
    std::lock_guard<std::mutex> lk(send_mu_);
    if (!alive_.load() || socket_ == GHOST_INVALID_SOCKET) return false;
    const char* p = line.c_str();
    size_t      remain = line.size();
    while (remain) {
        int n = ::send(socket_, p, (int)remain, 0);
        if (n <= 0) {
            std::fprintf(stderr, "[stratum] send failed\n");
            mark_dead();
            return false;
        }
        p      += n;
        remain -= n;
    }
    return true;
}

inline void StratumClient::reader_loop()
{
    std::string acc;
    char buf[4096];
    while (!stop_ && alive_.load()) {
        int n = ::recv(socket_, buf, sizeof buf, 0);
        if (n <= 0) {
            if (!stop_ && alive_.load())
                std::fprintf(stderr, "[stratum] recv ended (n=%d) -- pool disconnect\n", n);
            break;
        }
        acc.append(buf, n);
        size_t pos;
        while ((pos = acc.find('\n')) != std::string::npos) {
            std::string line = acc.substr(0, pos);
            acc.erase(0, pos + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) handle_message(line);
        }
    }
    // Exiting reader: mark dead so mining loop notices.
    mark_dead();
}

inline void StratumClient::handle_message(const std::string& line)
{
    std::string method;
    bool is_method = get_json_string(line, "method", method);

    if (is_method && method == "mining.set_target") {
        std::string params_raw;
        if (!get_json_raw(line, "params", params_raw)) return;
        auto arr = split_json_array(params_raw);
        if (arr.empty()) return;
        std::string thex = strip_quotes(arr[0]);
        std::vector<uint8_t> tgt;
        try { tgt = from_hex(thex); } catch (...) { return; }
        // Stratum target is BE in hex string; convert to LE for our compares.
        std::vector<uint8_t> tgt_le(tgt.rbegin(), tgt.rend());
        std::lock_guard<std::mutex> lk(latest_mu_);
        cur_target_ = std::move(tgt_le);
        std::printf("[stratum] new target = %s\n", thex.c_str());
        return;
    }

    if (is_method && method == "mining.set_difficulty") {
        // Fallback for SNOMP-family pools that only send set_difficulty
        // (no set_target). We derive target = powLimit / diff, where
        // powLimit = 0x0007ffff... is the Zcash mainnet genesis bound
        // (matches nheqminer's ZcashStratum target derivation).
        //
        // Our cur_target_ is authoritative if the pool already sent
        // set_target; we only fill in from set_difficulty if empty.
        std::string params_raw;
        if (!get_json_raw(line, "params", params_raw)) return;
        auto arr = split_json_array(params_raw);
        if (arr.empty()) return;
        double diff = std::strtod(strip_quotes(arr[0]).c_str(), nullptr);
        if (!(diff > 0.0)) return;

        {
            std::lock_guard<std::mutex> lk(latest_mu_);
            if (!cur_target_.empty()) {
                // set_target is authoritative; log and move on.
                std::printf("[stratum] set_difficulty %.6f (set_target active; ignored)\n", diff);
                return;
            }
        }

        // Compute target = powLimit / diff via log-space:
        //   log2(target) = log2(powLimit) - log2(diff)
        // Then materialize as a 256-bit BE byte array by placing a leading
        // mantissa at the appropriate bit position. This isn't exact, but
        // pool vardiff steps are coarse and any error << share granularity
        // (way below 1 ulp of the pool's own target rounding).
        const double log2_pow = 243.0 + std::log2((double)0x7ffffULL) - 19.0; // ~242.99
        double log2_tgt = log2_pow - std::log2(diff);
        if (log2_tgt < 0.0) log2_tgt = 0.0;
        if (log2_tgt > 255.0) log2_tgt = 255.0;
        int bit = (int)log2_tgt;
        // Mantissa in [1.0, 2.0) -> 53-bit integer mantissa.
        double frac = log2_tgt - (double)bit;
        uint64_t mant = (uint64_t)(std::pow(2.0, frac) * (double)(1ULL << 53));

        uint8_t tgt_be[32] = {0};
        // Place mant (54 bits: implicit leading 1 + 53 frac) at bit `bit`
        // down to bit `bit - 53`, big-endian in tgt_be.
        for (int b = bit; b >= 0 && b > bit - 54; b--) {
            int shift = bit - b;
            uint64_t bit_val = (mant >> (53 - shift)) & 1;
            if (bit_val) {
                int byte = 31 - (b >> 3);
                int off  = b & 7;
                if (byte >= 0 && byte < 32) tgt_be[byte] |= (uint8_t)(1u << off);
            }
        }

        std::vector<uint8_t> tgt_le(tgt_be, tgt_be + 32);
        std::reverse(tgt_le.begin(), tgt_le.end());
        std::lock_guard<std::mutex> lk(latest_mu_);
        cur_target_ = std::move(tgt_le);
        std::printf("[stratum] set_difficulty %.6f -> derived target (log-space)\n", diff);
        return;
    }

    // Some Zcash pools (including patched-SNOMP variants used for Zebra +
    // Crosslink) issue a mid-session extranonce1 change via
    // mining.set_extranonce. Format: params = [extranonce1_hex] or
    // [extranonce1_hex, extranonce2_size]. Update our cached nonce1 so the
    // next job uses it; the mining loop picks it up when it re-snapshots.
    if (is_method && method == "mining.set_extranonce") {
        std::string params_raw;
        if (!get_json_raw(line, "params", params_raw)) return;
        auto arr = split_json_array(params_raw);
        if (arr.empty()) return;
        std::string n1hex = strip_quotes(arr[0]);
        std::vector<uint8_t> n1;
        try { n1 = from_hex(n1hex); } catch (...) { return; }
        std::lock_guard<std::mutex> lk(latest_mu_);
        cur_nonce1_ = std::move(n1);
        std::printf("[stratum] set_extranonce: new nonce1 = %s (%zu B)\n",
                    n1hex.c_str(), cur_nonce1_.size());
        return;
    }

    // Pool-initiated redirect. We don't honor the new host/port (would
    // require plumbing back to main's reconnect loop); instead we just drop
    // the socket so the outer loop reconnects to the configured pool, which
    // is safer behavior for a miner anyway (avoids host-swap MITM).
    if (is_method && method == "client.reconnect") {
        std::fprintf(stderr, "[stratum] pool requested client.reconnect; dropping socket\n");
        mark_dead();
        return;
    }

    if (is_method && method == "mining.notify") {
        std::string params_raw;
        if (!get_json_raw(line, "params", params_raw)) return;
        auto arr = split_json_array(params_raw);
        if (arr.size() < 8) {
            std::fprintf(stderr, "[stratum] mining.notify with %zu params (want >=8)\n", arr.size());
            return;
        }
        StratumJob job;
        job.job_id       = strip_quotes(arr[0]);
        job.version_hex  = strip_quotes(arr[1]);
        job.prevhash_hex = strip_quotes(arr[2]);
        job.merkle_hex   = strip_quotes(arr[3]);
        job.reserved_hex = strip_quotes(arr[4]);
        job.time_hex     = strip_quotes(arr[5]);
        job.bits_hex     = strip_quotes(arr[6]);
        job.clean        = arr[7].find("true") != std::string::npos;
        // [8] algo, [9] personalization, [10] fat pointer hex, [11] block target hex.
        if (arr.size() >= 12) {
            std::string fphex = strip_quotes(arr[10]);
            std::string thex  = strip_quotes(arr[11]);
            if (fphex.size() >= 2 && fphex != "null" && (fphex.size() % 2) == 0) {
                try { job.fat_pointer = from_hex(fphex); }
                catch (...) { job.fat_pointer.clear(); }
            }
            if (thex.size() == 64) {
                try {
                    auto be = from_hex(thex);
                    if (be.size() == 32)
                        job.block_target_le.assign(be.rbegin(), be.rend());
                } catch (...) { job.block_target_le.clear(); }
            }
        }

        std::vector<uint8_t> tgt, n1;
        {
            std::lock_guard<std::mutex> lk(latest_mu_);
            tgt = cur_target_;
            n1  = cur_nonce1_;
        }
        std::printf("[stratum] new job %s (clean=%d, nonce1=%zu B, target=%zu B, fat=%zu B)\n",
                    job.job_id.c_str(), (int)job.clean, n1.size(), tgt.size(),
                    job.fat_pointer.size());
        if (job_cb_) job_cb_(job, tgt, n1);
        return;
    }

    // Result from a pending request (id-based).
    std::string id_str;
    if (!get_json_raw(line, "id", id_str)) return;
    int id = std::atoi(id_str.c_str());

    if (id == 1) {
        // NiceHash-Zcash mining.subscribe response:
        //   result = [session_id_or_null, extranonce1_hex]
        // Some pool flavors wrap the first slot, e.g.
        //   result = [[["mining.notify","x"]], "extranonce_hex"]
        // or append an extranonce2_size integer:
        //   result = [session_id, extranonce_hex, extranonce2_size]
        // Strategy: take result[1] if present and it looks like hex; else
        // fall back to scanning for the last hex-looking scalar.
        std::string result_raw;
        if (!get_json_raw(line, "result", result_raw)) return;
        auto arr = split_json_array(result_raw);
        auto is_hex_scalar = [](const std::string& v){
            if (v.empty()) return false;
            for (char c : v) {
                if (!((c >= '0' && c <= '9') ||
                      (c >= 'a' && c <= 'f') ||
                      (c >= 'A' && c <= 'F'))) return false;
            }
            return (v.size() % 2) == 0;
        };
        std::string n1hex;
        if (arr.size() >= 2) {
            std::string v = strip_quotes(arr[1]);
            if (is_hex_scalar(v)) n1hex = v;
        }
        if (n1hex.empty()) {
            for (auto it = arr.rbegin(); it != arr.rend(); ++it) {
                std::string v = strip_quotes(*it);
                if (is_hex_scalar(v)) { n1hex = v; break; }
            }
        }
        if (!n1hex.empty()) {
            std::vector<uint8_t> n1;
            try { n1 = from_hex(n1hex); } catch (...) { n1.clear(); }
            std::lock_guard<std::mutex> lk(latest_mu_);
            cur_nonce1_ = std::move(n1);
            std::printf("[stratum] subscribed; extranonce1 = %s (%zu B)\n",
                        n1hex.c_str(), cur_nonce1_.size());
        } else {
            std::printf("[stratum] subscribed (no extranonce)\n");
        }
        return;
    }

    if (id == 2) {
        if (line.find("\"result\":true") != std::string::npos) {
            std::printf("[stratum] authorized\n");
            // Opt-in to mid-session extranonce changes. Harmless if the pool
            // ignores or rejects the method -- we just lose that feature.
            send_line("{\"id\":3,\"method\":\"mining.extranonce.subscribe\",\"params\":[]}\n");
        } else {
            std::fprintf(stderr, "[stratum] authorize failed: %s\n", line.c_str());
        }
        return;
    }

    if (id == 3) {
        // Response to mining.extranonce.subscribe; log and move on.
        return;
    }

    // Submit response
    {
        std::lock_guard<std::mutex> lk(pending_mu_);
        for (auto it = pending_submit_ids_.begin(); it != pending_submit_ids_.end(); ++it) {
            if (*it == id) {
                bool ok = line.find("\"result\":true") != std::string::npos;
                std::string reason;
                // Most pools: top-level "reject-reason":"..." field.
                if (!get_json_string(line, "reject-reason", reason)) {
                    // nheqminer-compatible: {"error":[code,"message",...]}.
                    // Extract error[1] (the human-readable reason), which
                    // is what distinguishes [20,"invalid solution"] (crypto
                    // verify failed -> header/solution bytes wrong) from
                    // [23,"Low difficulty share"] (crypto ok, below target).
                    std::string err;
                    if (get_json_raw(line, "error", err) && err != "null") {
                        if (!err.empty() && err.front() == '[') {
                            auto parts = split_json_array(err);
                            if (parts.size() >= 2) {
                                std::string r1 = strip_quotes(parts[1]);
                                if (!r1.empty()) reason = r1;
                            }
                        }
                        if (reason.empty()) reason = err;
                    }
                }
                if (!ok && reason.empty()) reason = line;
                pending_submit_ids_.erase(it);
                if (accept_cb_) accept_cb_(ok, reason);
                else std::printf("[stratum] share %s\n", ok ? "accepted" : "rejected");
                return;
            }
        }
    }
}

inline bool StratumClient::submit(const StratumSubmission& sub)
{
    if (!alive_.load()) return false;
    int id = next_id_++;
    {
        std::lock_guard<std::mutex> lk(pending_mu_);
        pending_submit_ids_.push_back(id);
    }
    char buf[8192];
    std::snprintf(buf, sizeof buf,
        "{\"id\":%d,\"method\":\"mining.submit\",\"params\":[\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"]}\n",
        id, user_.c_str(), sub.job_id.c_str(), sub.time_hex.c_str(),
        sub.nonce2_hex.c_str(), sub.solution_hex.c_str());
    return send_line(buf);
}

} // namespace ghost
