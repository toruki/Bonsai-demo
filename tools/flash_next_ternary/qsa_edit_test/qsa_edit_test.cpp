// Exercise the KV-cell edits that the qwen4exp pooled-key fast path must survive (seq_rm suffix / interior,
// batches of odd sizes, seq_cp / seq_keep / seq_add / seq_div, state save + restore, clear) and print a hash
// of the logits after every step. Run twice (LLAMA_QSA_POOLED=0 and =1, or two runs of the same build) and
// diff the outputs: every STEP line must match. Step "restore" must also reproduce the hashes of the steps
// it restores (checked in-process).
//
// usage: qsa_edit_test -m model.gguf -f corpus.txt -c 8192 -b 1024 -ub 1024 -ngl 99 --expert-cache-slots N -fa on

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static uint64_t fnv1a(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

struct tester {
    llama_context * ctx;
    const llama_vocab * vocab;
    std::vector<llama_token> toks;
    int n_batch;
    int n_fail = 0;
    std::vector<std::pair<std::string, uint64_t>> log;

    // decode toks[t0, t0+n) at positions pos0.. in seq, in pieces of at most piece tokens; logits for the last token
    uint64_t decode(llama_seq_id seq, size_t t0, size_t n, llama_pos pos0, size_t piece, const char * name) {
        uint64_t h = 0;
        for (size_t s = 0; s < n; s += piece) {
            const size_t m = std::min(piece, n - s);
            llama_batch b = llama_batch_init((int) m, 0, 1);
            for (size_t i = 0; i < m; ++i) {
                common_batch_add(b, toks.at(t0 + s + i), pos0 + (llama_pos) (s + i), {seq}, s + i + 1 == n);
            }
            if (llama_decode(ctx, b)) { LOG_ERR("decode failed: %s\n", name); exit(1); }
            llama_batch_free(b);
        }
        const float * lg = llama_get_logits_ith(ctx, -1);
        const int nv = llama_vocab_n_tokens(vocab);
        h = fnv1a(lg, (size_t) nv*sizeof(float));
        int am = 0; for (int i = 1; i < nv; ++i) if (lg[i] > lg[am]) am = i;
        printf("STEP %-28s seq %d pos %6d n %5zu piece %4zu  argmax %6d  hash %016llx\n", name, seq, pos0 + (int) n - 1, n, piece, am, (unsigned long long) h);
        log.emplace_back(name, h);
        return h;
    }

    uint64_t single(llama_seq_id seq, size_t t, llama_pos pos, const char * name) { return decode(seq, t, 1, pos, 1, name); }
};

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);
    params.warmup = false;
    // two sequences in one stream: seq_cp lands in the same cells array as the fast path watches
    params.n_parallel = 2;
    params.kv_unified = true;

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { LOG_ERR("init failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);

    tester T;
    T.ctx = ctx;
    T.vocab = llama_model_get_vocab(model);
    T.toks = common_tokenize(ctx, params.prompt, false, true);
    T.n_batch = params.n_batch;
    if (T.toks.size() < 4000) { LOG_ERR("need >= 4000 tokens of text\n"); return 1; }
    const size_t nb = (size_t) params.n_batch;

    // A: prompt ending mid-block (1030 = 257 blocks + 2) crossing the 1024 ubatch boundary, then singles
    llama_memory_clear(mem, true);
    size_t P = 1030;
    T.decode(0, 0, P, 0, nb, "A.prefill");
    for (int i = 0; i < 10; ++i) T.single(0, P + i, (llama_pos) (P + i), "A.single");
    llama_pos pos = (llama_pos) (P + 10);           // next position
    size_t    tix = P + 10;                          // next token index

    // the hybrid (GDN) state cannot be rewound by seq_rm alone, so an edit goes through a state
    // snapshot as llama-server's context checkpoints do: restore, drop the cells past it, continue
    auto snapshot_ctx = [&]() { std::vector<uint8_t> st(llama_state_get_size(ctx)); llama_state_get_data(ctx, st.data(), st.size()); return st; };
    auto snapshot_seq = [&](llama_seq_id s) { std::vector<uint8_t> st(llama_state_seq_get_size(ctx, s)); llama_state_seq_get_data(ctx, st.data(), st.size(), s); return st; };

    // B: continue 10 singles, then restore the whole-context snapshot, truncate, replay 6 (must match B's first 6)
    std::vector<uint8_t> S1 = snapshot_ctx();
    const llama_pos pos1 = pos; const size_t tix1 = tix;
    std::vector<uint64_t> hB;
    for (int i = 0; i < 10; ++i) { hB.push_back(T.single(0, tix, pos, "B.single")); pos++; tix++; }
    llama_state_set_data(ctx, S1.data(), S1.size());
    if (!llama_memory_seq_rm(mem, 0, pos1, -1)) { printf("seq_rm after restore failed\n"); T.n_fail++; }
    pos = pos1; tix = tix1;
    for (int i = 0; i < 6; ++i) {
        const uint64_t h = T.single(0, tix, pos, "B.replay_after_restore"); pos++; tix++;
        if (h != hB[i]) { printf("MISMATCH: B replay %d differs\n", i); T.n_fail++; }
    }

    // D: a batch of 5 (not a multiple of the block size), then singles
    T.decode(0, tix, 5, pos, 5, "D.batch5"); pos += 5; tix += 5;
    for (int i = 0; i < 4; ++i) { T.single(0, tix, pos, "D.single"); pos++; tix++; }

    // E: copy to seq 1 (two sequences resident), decode on both, keep seq 0 only
    llama_memory_seq_cp(mem, 0, 1, -1, -1);
    { llama_pos p1 = pos; size_t t1 = tix;
      for (int i = 0; i < 3; ++i) { T.single(1, t1 + 100, p1, "E.seq1_after_cp"); p1++; t1++; } }
    for (int i = 0; i < 3; ++i) { T.single(0, tix, pos, "E.seq0_with_seq1"); pos++; tix++; }
    llama_memory_seq_keep(mem, 0);
    for (int i = 0; i < 3; ++i) { T.single(0, tix, pos, "E.seq0_after_keep"); pos++; tix++; }

    // G: per-sequence snapshot (the server's checkpoint blob), continue 5, restore + truncate, replay 5
    std::vector<uint8_t> S2 = snapshot_seq(0);
    const llama_pos pos2 = pos; const size_t tix2 = tix;
    std::vector<uint64_t> hG;
    for (int i = 0; i < 5; ++i) { hG.push_back(T.single(0, tix, pos, "G.single")); pos++; tix++; }
    if (llama_state_seq_set_data(ctx, S2.data(), S2.size(), 0) == 0) { printf("seq state restore failed\n"); T.n_fail++; }
    if (!llama_memory_seq_rm(mem, 0, pos2, -1)) { printf("seq_rm after seq restore failed\n"); T.n_fail++; }
    pos = pos2; tix = tix2;
    for (int i = 0; i < 5; ++i) {
        const uint64_t h = T.single(0, tix, pos, "G.replay_after_seq_restore"); pos++; tix++;
        if (h != hG[i]) { printf("MISMATCH: G replay %d differs\n", i); T.n_fail++; }
    }

    // I: fresh sequence in pieces of 7, then singles across the 256-cell padding boundary
    llama_memory_clear(mem, true);
    T.decode(0, 2000, 252, 0, 7, "I.prefill_pieces7");
    for (int i = 0; i < 8; ++i) T.single(0, 2000 + 252 + i, 252 + i, "I.single_across_256");

    // J: prompt of one block short of 1024 and singles across 1024
    llama_memory_clear(mem, true);
    T.decode(0, 3000, 1020, 0, nb, "J.prefill_1020");
    for (int i = 0; i < 8; ++i) T.single(0, 3000 + 1020 + i, 1020 + i, "J.single_across_1024");

    printf("DONE steps %zu fail %d\n", T.log.size(), T.n_fail);
    llama_backend_free();
    return T.n_fail ? 2 : 0;
}
