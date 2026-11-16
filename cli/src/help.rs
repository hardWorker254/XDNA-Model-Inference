//! The help text.
//!
//! # Why this is hand-written and not generated
//!
//! The C++ help is not a clap template and not a doc comment: it is a
//! free-form block whose verb sections are GUARDED by which binary is
//! running, and whose prose explains what each flag MEANS and what it
//! refuses. Generating it would lose the guards, and the guards are the
//! point -- help that lists a verb and then sends the reader to another
//! program for it is the failure the binary split exists to end.
//!
//! The text below is the C++ text, transcribed. It is checked against the
//! oracle's output by the parity harness, so a divergence is a failing gate
//! and not a thing a reader has to notice.

use npum_families::registry::{owns, verbs_of, ROSTER};

/// The one worked example in `--help`, per family.
///
/// It has to be a command the reader of THIS binary can type, or the example
/// teaches the wrong verb.
pub fn usage_example(binary: &str) -> &'static str {
    match binary {
        "npuaudio" => "npuaudio transcribe models/whisper-base.npue audio.wav",
        "npuimage" => "npuimage classify models/vit-base-patch16-224.npue image.png",
        _ => "npuembeddings embed models/all-MiniLM-L6-v2.npue in.txt",
    }
}

/// The whole `--help` text.
pub fn print_usage(binary: &str) -> String {
    let mut out = String::new();
    out.push_str("NpuEmbeddings -- BERT-family embeddings on the AMD NPU (XDNA1/npu1,\n");
    out.push_str("XDNA2/npu2; --dev selects the generation)\n");
    out.push('\n');

    // WHICH OF THE THREE THIS IS, before any verb: a reader who typed the
    // wrong binary learns it here rather than from a refusal later. This is
    // the SAME roster string the refusals print, so the help text and the
    // refusal cannot drift about which program to run next.
    out.push_str(&format!("  This is {binary}.\n\n"));
    out.push_str(ROSTER);
    out.push_str("\n\n");

    // EVERY VERB BLOCK IS GUARDED by owns(), so nothing here documents a
    // command this binary would refuse.
    if owns(binary, "list") {
        out.push_str(&format!(
            "  {binary} list\n\
             \x20       every model this build can run, and which are installed\n\n"
        ));
    }

    out.push_str(&format!(
        "  {binary} serve <model> [--port N] [--bind ADDR]\n\
         \x20       the HTTP endpoint for WHATEVER the container is. One verb,\n\
         \x20       four endpoints, and the model's arch picks which:\n\
         \x20         /v1/embeddings           a BERT-family text model\n\
         \x20         /v1/audio/transcriptions Whisper\n\
         \x20         /v1/classify             a ViT classifier\n\
         \x20         /v1/pose                 YOLOv8-pose\n\
         \x20       Downloads and verifies the model first if it is not\n\
         \x20       installed yet. The mode is chosen by the container's arch, never\n\
         \x20       by a flag, and each endpoint answers only for its own path -- a\n\
         \x20       wrong path is a 404 naming the right one, never another\n\
         \x20       endpoint's answer.\n\
         \x20       Only this binary's own family is served here ({binary}): another\n\
         \x20       binary's container is refused by name before the listener opens,\n\
         \x20       rather than answered by an endpoint this process is not named for.\n\n"
    ));

    if owns(binary, "embed") {
        out.push_str(&format!(
            "  {binary} embed <model> <in.txt> [out.f32]\n\
             \x20       embed a text file, one text per line\n\n"
        ));
    }

    if owns(binary, "transcribe") {
        out.push_str(&format!(
            "  {binary} transcribe <model> <audio.wav> [--language en]\n\
             \x20       transcribe 16 kHz audio with a Whisper model. The transcript\n\
             \x20       goes to stdout on its own and the status block to stderr, so\n\
             \x20       it pipes. --json prints the OpenAI-shaped object instead,\n\
             \x20       with one segment per 30 s window. --convert ingests through\n\
             \x20       ffmpeg (mp3, m4a, webm, or a WAV the reader refuses).\n\n"
        ));
    }

    if owns(binary, "classify") {
        out.push_str(&format!(
            "  {binary} classify <model> <image.png> [more.png ...]\n\
             \x20       classify images with a ViT model (vit-base-patch16-224), one\n\
             \x20       per argument. The label goes to stdout on its own and the\n\
             \x20       status block to stderr, so it pipes. PNG and JPEG only --\n\
             \x20       anything else is refused by name rather than guessed at, and\n\
             \x20       so is a non-square resize or a shortest-edge crop: the\n\
             \x20       container says which geometry it was trained with and this\n\
             \x20       reads it. --top-k prints the runners-up to stderr, --json\n\
             \x20       the objects to stdout. `embed` is not how this runs.\n\
             \x20       Over HTTP: `{binary} serve <model>`, which answers POST\n\
             \x20       /v1/classify with the same object `classify --json` prints (both\n\
             \x20       call npue::vit::prediction_json), one `image` part and an\n\
             \x20       optional `top_k`. GET /v1/labels returns the whole vocabulary, so\n\
             \x20       a client can turn an id into a name without the container.\n\n"
        ));
    }

    if owns(binary, "pose") {
        out.push_str(&format!(
            "  {binary} pose <model> <image.png> [more.png ...]\n\
             \x20       detect people and their 17 COCO keypoints with a YOLOv8-pose\n\
             \x20       model. Same shape as classify and for the same reasons: one\n\
             \x20       image per argument, the result on stdout and the status block\n\
             \x20       on stderr. --json prints one object per image with every box,\n\
             \x20       every joint, the 12-edge skeleton and the letterbox transform,\n\
             \x20       --text a human summary. --conf/--iou/--kpt/--max-det move the\n\
             \x20       thresholds the container was not given. --pose-dump FILE\n\
             \x20       writes every graph node's fp32 output, for diffing against an\n\
             \x20       independent interpreter.\n\
             \x20       CPU BY DEFAULT, and that is a measurement rather than a\n\
             \x20       preference. --npu-ops conv moves the 72 convolutions to\n\
             \x20       the array and nothing else, needs a container packed with --npu\n\
             \x20       and a design set, and is SLOWER here: 258 ms of network on 16\n\
             \x20       threads against 384 ms dispatched as GEMMs, because N must be a\n\
             \x20       multiple of 128 (2.6x arithmetic padding waste before anything\n\
             \x20       runs), the stem alone is 100 of the 436 dispatches, and a\n\
             \x20       dispatch costs 660 us of which 140 us is the device's GEMM. Per\n\
             \x20       layer 19 of the 72 are faster on the array and an oracle\n\
             \x20       splitting each onto its faster side measured 146 ms -- 2.6%.\n\
             \x20       The status block prints both sides, per image.\n\
             \x20       MEASURED on the CPU path, 640x640 input, i8: 29 ms front end,\n\
             \x20       258 ms network of which 88 ms is the GEMM, 57 ms im2col, 10 ms\n\
             \x20       the weight transpose and 7 ms the output transpose, 1 ms decode.\n\n\
             \x20       The same model over HTTP: `{binary} serve <model>`, which\n\
             \x20       answers POST /v1/pose. Multipart with one `image` part (PNG or\n\
             \x20       JPEG, by magic bytes) and optional conf, iou, kpt, max_det. The\n\
             \x20       answer is byte for byte what `pose --json` prints -- both call\n\
             \x20       npue::pose::result_json -- so the endpoint and the CLI cannot\n\
             \x20       drift. Also GET /health and GET /v1/models. Per-request\n\
             \x20       thresholds apply to that request only.\n\n"
        ));
    }

    if owns(binary, "add") {
        out.push_str(&format!(
            "  {binary} add <org/model> [<sha256>]\n\
             \x20       teach this installation about a model that is not built in --\n\
             \x20       typically a finetune of one that is. Reads the repository's\n\
             \x20       config.json, derives the geometry, checks a design serves it,\n\
             \x20       and writes models/catalog.json. No weights are downloaded until\n\
             \x20       the first serve/embed.\n\
             \x20       WITHOUT a sha256 the weights are NOT verified. That is allowed,\n\
             \x20       and it is warned about on every single run.\n\n"
        ));
    }

    // THE MODEL ARGUMENT, once, for whichever verbs printed above.
    out.push_str(&format!(
        "  <model> in any of this binary's verbs is either a NAME -- looked up\n\
         \x20       in models/, fetched and verified against the catalogue's sha256\n\
         \x20       when the catalogue knows it -- or a PATH to a .npue, used as\n\
         \x20       given and never re-resolved by name:\n\
         \x20           {}\n\
         \x20       The runtime names a path-held model after its file, which is\n\
         \x20       also how it finds runtime/<stem>/artifacts_npu<N>.\n\n",
        usage_example(binary)
    ));

    // THE OPTIONS ARE NOT FILTERED, unlike the verbs above. They are one
    // list annotated with the verb each belongs to (`--language CODE
    // transcribe:`), so a reader can tell in the line itself whether it
    // applies -- and the annotations are a property of the flag, not of the
    // binary, whereas a verb block is a command to type.
    out.push_str(OPTIONS);

    out.push_str(&format!(
        "  The flag form is unchanged and still works:\n\
         \x20   {binary} <root> --model NAME --artifacts DIR --serve [port]\n\
         \x20 and carries the probes and benchmarks; see docs/CURRENT_STATUS.md.\n\n"
    ));
    out
}

/// The verbs a binary owns, as one line.
pub fn verbs(binary: &str) -> &'static str {
    verbs_of(binary)
}

const OPTIONS: &str = concat!(
    "  Options for this binary's verbs:\n",
    "    --port N          listen port (default 8080)\n",
    "    --bind ADDR       interface (default 127.0.0.1, localhost only)\n",
    "    --threads N       host thread budget (default 24 for these)\n",
    "    --pipeline N      concurrent encode lanes (serve/embed pass 4;\n",
    "                      1 lane = no pipelining. A later --pipeline on the\n",
    "                      command line wins, so --pipeline 1 disables it)\n",
    "    --artifacts DIR   override the design set\n",
    "    --npu-ops CODES\n",
    "                      send the listed ops to the ARRAY instead of the\n",
    "                      host, comma-separated. An op not listed runs on\n",
    "                      the host, so there is no inverse flag. The\n",
    "                      default -- nothing listed -- runs every\n",
    "                      op in this process; for most codes that\n",
    "                      is the measured-faster side, and for\n",
    "                      gemm it is not.\n",
    "                        gemm   the four per-layer GEMMs\n",
    "                        gelu   GELU (exact erf; a Whisper\n",
    "                               container declares this one)\n",
    "                        layn   LayerNorm\n",
    "                        softm  softmax\n",
    "                        conv   Whisper's conv1/conv2\n",
    "                        attn   attention, as QK^T and softmax.V GEMMs\n",
    "                        mproj  the slaney mel bank, as a GEMM\n",
    "                        fft    the 400-point transform, as a GEMM\n",
    "                        logit  the tied-embedding projection, in\n",
    "                               eight chunks of the vocabulary\n",
    "                      The same nine codes in every architecture;\n",
    "                      which of them yours can honour is in\n",
    "                      NPU_OPS.md, and a code that cannot is refused\n",
    "                      by name with the reason, never ignored.\n",
    "                      Worth asking for? Measured, per 3 s of\n",
    "                      whisper-tiny on 16 threads: conv 0.94 -> 0.66 s\n",
    "                      wall and 5.5 -> 3.8 s CPU; gemm 1.06 -> 0.17 s\n",
    "                      on an embedder and 0.435 -> 0.241 s on a ViT.\n",
    "                      Every other code is either neutral or slower.\n",
    "                      README, 'What to move to the NPU'.\n",
    "                      The exporter's flag is the SAME STRING and it\n",
    "                      BUILDS what this one selects. gelu/layn/softm\n",
    "                      write a sibling design set each (gelu/,\n",
    "                      layernorm/, softmax/) and cost one extra\n",
    "                      hw_context EACH; gemm/conv/attn/mproj/fft/\n",
    "                      logit are streams inside the GEMM sets and\n",
    "                      cost none.\n",
    "                      A missing design is refused by name, never\n",
    "                      silently run on the host -- and a run that\n",
    "                      would exceed the context budget is refused\n",
    "                      before any context is built unless\n",
    "                      --allow-contention says the contention is\n",
    "                      intended (see below).\n",
    "                      The speech-to-text codes are refused on an\n",
    "                      embedder. Replaces --npu-ops, --npu-eltwise and\n",
    "                      --host-ln/--host-sm/--host-gelu, which are now\n",
    "                      refused by name so a stale command line cannot\n",
    "                      look like it worked.\n",
    "    --backend cpu|vulkan\n",
    "                      where the ARITHMETIC happens. The default is\n",
    "                      the host, and the host's arithmetic is the\n",
    "                      C++'s arithmetic byte for byte. 'vulkan' runs\n",
    "                      the text family's projections, its LayerNorm,\n",
    "                      its GELU and its softmax on the GPU: the same\n",
    "                      network on a different machine, so the vectors\n",
    "                      are the same to about 1e-7 and not bit for bit\n",
    "                      -- a GPU's f32 is not a host's f32. The\n",
    "                      attention stays on the host, because the\n",
    "                      family's score rows are in its own packed\n",
    "                      layout and the shader's attention takes a\n",
    "                      different one. A name that is neither is\n",
    "                      refused by name, and so is 'vulkan' on a\n",
    "                      machine with no Vulkan device, and so is\n",
    "                      'vulkan' for a family that does not run on the\n",
    "                      GPU: the ViT's and the Whisper's LayerNorm and\n",
    "                      GELU are the exact variants, which the\n",
    "                      shaders do not implement.\n",
    "    --allow-contention\n",
    "                      proceed even though another process holds NPU\n",
    "                      hw_contexts. Without it, a run that would exceed\n",
    "                      the device's context budget is refused before any\n",
    "                      context is built, and a timed run refuses to\n",
    "                      report a throughput. ANY NUMBER FROM A CONTENDED\n",
    "                      RUN IS NOT AN NPU PERFORMANCE CLAIM.\n",
    "    --dev npu1|npu2   the NPU generation this process runs on. A\n",
    "                      design set built for the other generation is\n",
    "                      refused, not loaded. Defaults to NPU_DEVICE, or\n",
    "                      NPU2=1, or npu1\n",
    "    --language CODE   transcribe: the language to prime the decoder\n",
    "                      with (en, de, ru, ...). There is no detection\n",
    "                      and no default worth having: an assumed language\n",
    "                      is a fluent transcript of the wrong language, so\n",
    "                      it defaults to en and says so on stderr. A code\n",
    "                      the model's tokenizer has no token for is\n",
    "                      refused, never approximated.\n",
    "    --task NAME      transcribe: 'transcribe' (default) or\n",
    "                      'translate' -- English output either way\n",
    "    --convert        transcribe: ingest through ffmpeg instead of the\n",
    "                      WAV reader. The reader refuses what it cannot\n",
    "                      represent (a second channel, 8-bit, 44.1 kHz)\n",
    "                      by name; this is the way in for those\n",
    "    --max-new N      transcribe: cap on generated tokens per window\n",
    "                      (default: the model's own position bound, 448)\n",
    "    --chunk-seconds N  transcribe: window length, default 30 -- the\n",
    "                      model's own. Longer is refused: the encoder has\n",
    "                      weights for a fixed number of positions and a\n",
    "                      cut window is a transcript of the first 30 s\n",
    "                      presented as the whole recording\n",
    "    --stride-seconds N transcribe: overlap on EACH side of a window,\n",
    "                      default 5 (transformers' own), so windows start\n",
    "                      chunk - 2*stride apart and the merge\n",
    "                      de-duplicates the overlap\n",
    "    --json           transcribe: print {\"text\", \"segments\", ...}\n",
    "    --top-k          classify: print the runners-up to stderr, under\n",
    "                      the image. The whole row is ranked and the top\n",
    "                      ten shown -- a truncated list reads as \"nothing\n",
    "                      else is close\", which is a different claim\n",
    "    --root DIR        override where models/ and the design live\n",
    "    --token VALUE     HuggingFace access token for a GATED model\n",
    "                      (falls back to the HF_TOKEN env var if omitted)\n",
    "    --prefix NAME     task prefix to prepend to every input text, for\n",
    "                      `embed` and `--bench` only. REQUIRED for a model\n",
    "                      whose container carries a prompt table (nomic\n",
    "                      wants 'search_document' or 'search_query';\n",
    "                      EmbeddingGemma has 14) -- there is no default,\n",
    "                      because a wrongly-prefixed embedding is\n",
    "                      correctly shaped and correctly normed, so\n",
    "                      nothing downstream can tell it is wrong. Pass\n",
    "                      --prefix \"\" for no prefix at all. An unknown\n",
    "                      NAME refuses and lists the real options, and so\n",
    "                      does passing one to a model with no table (the\n",
    "                      four BERT models). Always printed on stderr.\n",
    "                      `serve` REJECTS this flag: its prompt is chosen\n",
    "                      per request -- POST \"prompt_name\", and GET\n",
    "                      /health lists what the model accepts.\n",
    "    --allow-truncation\n",
    "                      embed the first `seq` tokens of an input that\n",
    "                      is too long, instead of refusing it. OFF by\n",
    "                      default: a truncated text still returns a\n",
    "                      correctly shaped, correctly normed vector, so\n",
    "                      nothing downstream can tell that the answer is\n",
    "                      wrong -- and inputs sharing a preamble truncate\n",
    "                      to IDENTICAL vectors. Without this flag such an\n",
    "                      input is an error naming its real token count;\n",
    "                      with it, a one-line warning on stderr.\n",
    "                      This build runs at the sequence length its\n",
    "                      design was exported for -- see\n",
    "                      tools/export/export_gemm_rtp.py --seq to build for a\n",
    "                      longer one.\n",
    "\n",
);
