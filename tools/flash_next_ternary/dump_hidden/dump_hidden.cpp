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

    // strip our own options before handing the rest to common_params_parse
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (!strcmp(argv[i], "--dump-dir") && i + 1 < argc) { st.outdir = argv[++i]; continue; }
        if (!strcmp(argv[i], "--dump-filter") && i + 1 < argc) { st.filter = std::regex(argv[++i]); continue; }
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
    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if ((int) toks.size() > params.n_ctx) toks.resize(params.n_ctx);
    LOG_INF("n_tokens = %zu\n", toks.size());

    std::string mk = "mkdir -p '" + st.outdir + "'";
    if (system(mk.c_str()) != 0) { LOG_ERR("mkdir failed\n"); return 1; }
    // fresh files
    for (auto & kv : st.shapes) { (void) kv; }

    // request logits for every token so result_output covers the whole prompt
    llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(batch, toks[i], (llama_pos) i, {0}, true);
    }
    if (llama_decode(ctx, batch)) { LOG_ERR("decode failed\n"); return 1; }

    // also write the logits from the API (same as result_output but guaranteed full)
    {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        std::ofstream f(st.outdir + "/logits.f32", std::ios::binary);
        for (size_t i = 0; i < toks.size(); ++i) {
            const float * l = llama_get_logits_ith(ctx, (int) i);
            f.write((const char *) l, sizeof(float) * n_vocab);
        }
        st.shapes["logits"] = {n_vocab, (int64_t) toks.size(), 1, 1};
    }
    std::ofstream meta(st.outdir + "/meta.json");
    meta << "{\n  \"n_tokens\": " << toks.size() << ",\n  \"tokens\": [";
    for (size_t i = 0; i < toks.size(); ++i) meta << (i ? "," : "") << toks[i];
    meta << "],\n  \"tensors\": {";
    bool first = true;
    for (auto & kv : st.shapes) {
        meta << (first ? "" : ",") << "\n    \"" << kv.first << "\": [" << kv.second[0] << "," << kv.second[1] << "]";
        first = false;
    }
    meta << "\n  }\n}\n";
    LOG_INF("dumped %zu tensors to %s\n", st.shapes.size(), st.outdir.c_str());
    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
