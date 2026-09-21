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

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (dump_state *) ud;
    const std::string name = t->name;
    if (ask) {
        return std::regex_match(name, st->filter);
    }
    if (t->type != GGML_TYPE_F32) {
        LOG_WRN("skipping %s: type %s\n", name.c_str(), ggml_type_name(t->type));
        return true;
    }
    if (!ggml_is_contiguous(t)) {
        LOG_WRN("skipping %s: not contiguous\n", name.c_str());
        return true;
    }
    const size_t nbytes = ggml_nbytes(t);
    st->buf.resize(nbytes);
    ggml_backend_tensor_get(t, st->buf.data(), 0, nbytes);
    std::ofstream f(st->outdir + "/" + name + ".f32", std::ios::binary | std::ios::app);
    f.write((const char *) st->buf.data(), nbytes);
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

    // strip our own options before handing the rest to common_params_parse
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (!strcmp(argv[i], "--dump-dir") && i + 1 < argc) { st.outdir = argv[++i]; continue; }
        if (!strcmp(argv[i], "--dump-filter") && i + 1 < argc) { st.filter = std::regex(argv[++i]); continue; }
        if (!strcmp(argv[i], "--n-chunks") && i + 1 < argc) { n_chunks = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--chunk-len") && i + 1 < argc) { chunk_len = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--chunk-offset") && i + 1 < argc) { chunk_offset = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--no-logits")) { save_logits = false; continue; }
        args.push_back(argv[i]);
    }
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = cb;
    params.cb_eval_user_data = &st;
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
    std::ofstream lf(st.outdir + "/logits.f32", std::ios::binary);
    std::vector<llama_token> toks;                  // concatenated token stream for meta
    for (size_t si = 0; si < seqs.size(); ++si) {
        const auto & sq = seqs[si];
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_init((int) sq.size(), 0, 1);
        for (size_t i = 0; i < sq.size(); ++i) {
            common_batch_add(batch, sq[i], (llama_pos) i, {0}, save_logits);
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
