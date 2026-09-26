# qsa_edit_test

KV-cell edit scenarios for the qwen4exp pooled-indexer-key fast path (fork `qwen4exp-port`): prompt ending
mid-block, single-token decode, whole-context and per-sequence state snapshot → continue → restore → truncate
→ replay (the llama-server checkpoint pattern; the hybrid GDN state cannot be rewound by `seq_rm` alone),
odd batch sizes, `seq_cp` / `seq_keep` with two sequences in one stream, prompts in pieces of 7, and singles
across the 256-cell and 1024-token boundaries. Every step prints a hash of the logits; replays are checked
in-process, and two runs (`LLAMA_QSA_POOLED=0` / `=1`, or the same build twice) must print identical STEP lines.

Build (against the fork's `build-q4x`):

    L=llama.cpp; g++ -std=c++17 -O2 -o tools/flash_next_ternary/qsa_edit_test/qsa_edit_test \
      tools/flash_next_ternary/qsa_edit_test/qsa_edit_test.cpp -I$L/include -I$L/ggml/include -I$L/common \
      -L$L/build-q4x/bin -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,$PWD/$L/build-q4x/bin

Run: see `q4x_eval/edit_test.sh` (both variants, diff of the STEP lines).
