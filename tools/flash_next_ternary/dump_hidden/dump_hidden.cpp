// Dump per-layer residual-stream tensors ("l_out-N") plus the final logits for a
// fixed prompt, so two models can be compared layer by layer offline.
//
// Built against the shipped fork binaries (bin/cuda-e8); does not modify the runtime.
// Output: <outdir>/<tensor>.f32 (row-major [n_tokens, n_embd]) and meta.json.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <map>
#include <regex>
#include <string>
#include <vector>

struct dump_state {
    std::regex filter;
    std::string outdir;
    std::map<std::string, std::vector<int64_t>> shapes;
    std::vector<uint8_t> buf;
};

// --- decode-timing mode: measure what a host-side expert cache would cost per token ---------
// mode 1: the scheduler stops at every ffn_moe_topk node (graph split + sync), reads nothing
// mode 2: + reads the top-k ids to the host
// mode 3: + runs a global (layer, expert) LRU of `slots` entries on them (no weight transfer)
struct timing_state {
    int mode = 0;
    int slots = 8000;
    int k = 10;
    std::regex topk_re{"^ffn_moe_topk-([0-9]+)$"};
    std::vector<int32_t> ids;
    // LRU: doubly linked list over an unordered_map<gid, node>
    struct node { int64_t gid; node * prev; node * next; };
    std::unordered_map<int64_t, node *> where;
    node * head = nullptr; node * tail = nullptr; size_t size = 0;   // head = most recent
    int64_t hits = 0, misses = 0, tokens = 0;
    void touch(node * n) {
        if (n == head) return;
        if (n->prev) n->prev->next = n->next;
        if (n->next) n->next->prev = n->prev; else tail = n->prev;
        n->prev = nullptr; n->next = head; if (head) head->prev = n; head = n; if (!tail) tail = n;
    }
    void insert(int64_t gid) {
        node * n = new node{gid, nullptr, head};
        if (head) head->prev = n; head = n; if (!tail) tail = n; where[gid] = n; size++;
        if ((int) size > slots) {                              // evict the least recent
            node * v = tail; tail = v->prev; if (tail) tail->next = nullptr; else head = nullptr;
            where.erase(v->gid); delete v; size--;
        }
    }
};

static bool cb_timing(struct ggml_tensor * t, bool ask, void * ud) {
    auto * ts = (timing_state *) ud;
    std::cmatch m;
    if (!std::regex_match(t->name, m, ts->topk_re)) return !ask;   // not interested (ask) / continue (exec)
    if (ask) return true;
    if (ts->mode < 2) return true;
    const int layer = atoi(m[1].str().c_str());
    const int64_t k = t->ne[0], ntok = t->ne[1];
    ts->ids.resize(k * ntok);
    for (int64_t i1 = 0; i1 < ntok; ++i1) {
        ggml_backend_tensor_get(t, ts->ids.data() + i1 * k, i1 * t->nb[1], k * sizeof(int32_t));
    }
    if (ts->mode < 3) return true;
    for (int64_t i1 = 0; i1 < ntok; ++i1) {
        // protect every selected expert that is resident before evicting for the misses
        std::vector<int64_t> miss;
        for (int64_t j = 0; j < k; ++j) {
            const int64_t gid = (int64_t) layer * 512 + ts->ids[i1 * k + j];
            auto it = ts->where.find(gid);
            if (it != ts->where.end()) { ts->touch(it->second); ts->hits++; } else { miss.push_back(gid); ts->misses++; }
        }
        for (int64_t gid : miss) ts->insert(gid);
    }
    return true;
}

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (dump_state *) ud;
    const std::string name = t->name;
    if (ask) {
        return std::regex_match(name, st->filter);
    }
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) {
        LOG_WRN("skipping %s: type %s\n", name.c_str(), ggml_type_name(t->type));
        return true;
    }
    const bool is_i32 = t->type == GGML_TYPE_I32;
    const size_t row = t->ne[0] * ggml_type_size(t->type);
    if (ggml_is_contiguous(t)) {
        st->buf.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, st->buf.data(), 0, st->buf.size());
    } else if (t->nb[0] == ggml_type_size(t->type) && t->ne[2] * t->ne[3] == 1) {
        // 2-D view with strided rows (e.g. the top-k view of an argsort): copy row by row
        st->buf.resize(row * t->ne[1]);
        for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
            ggml_backend_tensor_get(t, st->buf.data() + i1 * row, i1 * t->nb[1], row);
        }
    } else {
        LOG_WRN("skipping %s: not contiguous\n", name.c_str());
        return true;
    }
    std::ofstream f(st->outdir + "/" + name + (is_i32 ? ".i32" : ".f32"), std::ios::binary | std::ios::app);
    f.write((const char *) st->buf.data(), st->buf.size());
    auto & sh = st->shapes[name];
    if (sh.empty()) {
        sh = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    } else {
        sh[1] += t->ne[1]; // tokens accumulate across ubatches
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    dump_state st;
    st.filter = std::regex("^(l_out-[0-9]+|result_norm|result_output)$");
    st.outdir = "hidden_dump";
    int n_chunks = 0, chunk_len = 512, chunk_offset = 0;
    bool save_logits = true;
    bool all_rows = false;            // compute every token in the last layer (logits requested, not saved)
    int timing_tokens = 0;            // > 0: decode-timing mode (see timing_state)
    int timing_skip = 8;              // decode tokens excluded from the statistics (cache warm-up)
    bool dump_in_timing = false;      // also run the dump callback during decode-timing (single-token graphs)
    timing_state ts;

    // strip our own options before handing the rest to common_params_parse
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (!strcmp(argv[i], "--dump-dir") && i + 1 < argc) { st.outdir = argv[++i]; continue; }
        if (!strcmp(argv[i], "--dump-filter") && i + 1 < argc) { st.filter = std::regex(argv[++i]); continue; }
        if (!strcmp(argv[i], "--n-chunks") && i + 1 < argc) { n_chunks = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--chunk-len") && i + 1 < argc) { chunk_len = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--chunk-offset") && i + 1 < argc) { chunk_offset = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--no-logits")) { save_logits = false; continue; }
        if (!strcmp(argv[i], "--all-rows")) { all_rows = true; continue; }
        if (!strcmp(argv[i], "--decode-timing") && i + 1 < argc) { timing_tokens = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--timing-mode") && i + 1 < argc) { ts.mode = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--timing-slots") && i + 1 < argc) { ts.slots = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--timing-skip") && i + 1 < argc) { timing_skip = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--dump-in-timing")) { dump_in_timing = true; continue; }
        args.push_back(argv[i]);
    }
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    if (timing_tokens > 0) {
        if (ts.mode > 0) { params.cb_eval = cb_timing; params.cb_eval_user_data = &ts; }
        if (dump_in_timing) { params.cb_eval = cb; params.cb_eval_user_data = &st; }
    } else {
        params.cb_eval = cb;
        params.cb_eval_user_data = &st;
    }
    params.warmup = false;

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { LOG_ERR("init failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> all = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    // chunk mode: split the corpus into n_chunks independent sequences of chunk_len tokens
    // (memory cleared between them); single-prompt mode is n_chunks = 1, chunk_len = n_ctx
    std::vector<std::vector<llama_token>> seqs;
    if (n_chunks > 0) {
        for (int c = chunk_offset; c < chunk_offset + n_chunks && (size_t) (c + 1) * chunk_len <= all.size(); ++c) {
            seqs.emplace_back(all.begin() + (size_t) c * chunk_len, all.begin() + (size_t) (c + 1) * chunk_len);
        }
    } else {
        if ((int) all.size() > params.n_ctx) all.resize(params.n_ctx);
        seqs.push_back(all);
    }
    LOG_INF("n_sequences = %zu, tokens/seq = %zu\n", seqs.size(), seqs.empty() ? 0 : seqs[0].size());

    std::string mk = "mkdir -p '" + st.outdir + "'";
    if (system(mk.c_str()) != 0) { LOG_ERR("mkdir failed\n"); return 1; }
    // fresh files
    for (auto & kv : st.shapes) { (void) kv; }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (timing_tokens > 0 && dump_in_timing) {
        std::string mk2 = "mkdir -p '" + st.outdir + "'";
        if (system(mk2.c_str()) != 0) { LOG_ERR("mkdir failed\n"); return 1; }
    }
    if (timing_tokens > 0) {
        // prompt = first sequence in one batch, then greedy decode one token at a time
        const auto & sq = seqs.at(0);
        llama_memory_clear(llama_get_memory(ctx), true);
        // prompt in n_batch-sized pieces (the last token of the prompt requests logits)
        const auto tp0 = std::chrono::steady_clock::now();
        const size_t nb = (size_t) std::max(1, params.n_batch);
        for (size_t start = 0; start < sq.size(); start += nb) {
            const size_t n = std::min(nb, sq.size() - start);
            llama_batch batch = llama_batch_init((int) n, 0, 1);
            for (size_t i = 0; i < n; ++i) common_batch_add(batch, sq[start + i], (llama_pos) (start + i), {0}, start + i + 1 == sq.size());
            if (llama_decode(ctx, batch)) { LOG_ERR("prompt decode failed at %zu\n", start); return 1; }
            llama_batch_free(batch);
            if ((start / nb) % 16 == 15) LOG_INF("  prefill %zu / %zu tokens\n", start + n, sq.size());
        }
        const double prompt_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
        LOG_INF("PREFILL %zu tokens in %.1f s (%.1f t/s)\n", sq.size(), prompt_s, sq.size() / prompt_s);
        llama_pos pos = (llama_pos) sq.size();
        const int warm = timing_skip;
        double first_ms = 0; int n_first = 0;
        double total_ms = 0; std::vector<double> lat;
        llama_token tok = 0;
        for (int i = 0; i < timing_tokens; ++i) {
            const float * l = llama_get_logits_ith(ctx, -1);
            tok = (llama_token) (std::max_element(l, l + n_vocab) - l);
            llama_batch b1 = llama_batch_init(1, 0, 1);
            common_batch_add(b1, tok, pos++, {0}, true);
            const auto t0 = std::chrono::steady_clock::now();
            if (llama_decode(ctx, b1)) { LOG_ERR("decode failed\n"); return 1; }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            llama_batch_free(b1);
            if (i >= warm) { total_ms += ms; lat.push_back(ms); ts.tokens++; } else { first_ms += ms; n_first++; }
        }
        std::sort(lat.begin(), lat.end());
        const double p50 = lat[lat.size() / 2], p95 = lat[(size_t) (lat.size() * 0.95)];
        LOG_INF("TIMING mode=%d slots=%d tokens=%d  mean %.2f ms/tok (%.1f t/s)  p50 %.2f  p95 %.2f  | first %d tokens: %.2f ms/tok",
                ts.mode, ts.slots, (int) lat.size(), total_ms / lat.size(), 1000.0 * lat.size() / total_ms, p50, p95,
                n_first, n_first ? first_ms / n_first : 0.0);
        if (ts.mode >= 3) {
            LOG_INF("  LRU: hit %.4f  miss/token %.2f", (double) ts.hits / (ts.hits + ts.misses),
                    (double) ts.misses / ts.tokens);
        }
        LOG_INF("\n");
        llama_backend_free();
        return 0;
    }
    std::ofstream lf(st.outdir + "/logits.f32", std::ios::binary);
    std::vector<llama_token> toks;                  // concatenated token stream for meta
    for (size_t si = 0; si < seqs.size(); ++si) {
        const auto & sq = seqs[si];
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_init((int) sq.size(), 0, 1);
        for (size_t i = 0; i < sq.size(); ++i) {
            common_batch_add(batch, sq[i], (llama_pos) i, {0}, save_logits || all_rows);
        }
        if (llama_decode(ctx, batch)) { LOG_ERR("decode failed on seq %zu\n", si); return 1; }
        if (save_logits) {
            for (size_t i = 0; i < sq.size(); ++i) {
                const float * l = llama_get_logits_ith(ctx, (int) i);
                lf.write((const char *) l, sizeof(float) * n_vocab);
            }
        }
        toks.insert(toks.end(), sq.begin(), sq.end());
        llama_batch_free(batch);
        if (si % 10 == 9) LOG_INF("  %zu / %zu sequences\n", si + 1, seqs.size());
    }
    if (save_logits) st.shapes["logits"] = {n_vocab, (int64_t) toks.size(), 1, 1};
    std::ofstream meta(st.outdir + "/meta.json");
    meta << "{\n  \"n_tokens\": " << toks.size() << ",\n  \"n_sequences\": " << seqs.size() << ",\n  \"seq_len\": " << (seqs.empty() ? 0 : seqs[0].size()) << ",\n  \"tokens\": [";
    for (size_t i = 0; i < toks.size(); ++i) meta << (i ? "," : "") << toks[i];
    meta << "],\n  \"tensors\": {";
    bool first = true;
    for (auto & kv : st.shapes) {
        meta << (first ? "" : ",") << "\n    \"" << kv.first << "\": [" << kv.second[0] << "," << kv.second[1] << "]";
        first = false;
    }
    meta << "\n  }\n}\n";
    LOG_INF("dumped %zu tensors to %s\n", st.shapes.size(), st.outdir.c_str());
    llama_backend_free();
    return 0;
}
