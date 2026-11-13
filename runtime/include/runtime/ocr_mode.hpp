//===- ocr_mode.hpp -------------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the OCR entry point, dispatched from Runtime::run beside the
// arch=1 Gemma mode, the arch=4 STT mode, the arch=5 classify mode, the
// arch=6 YOLOv8-pose mode, the arch=7 hand-landmark mode and the arch=8
// body-pose mode.
//
// WHY A MODE AND NOT A BRANCH
// ---------------------------
// The same reason the other six give: an arch=9 container shares nothing with
// the embedding pipeline -- no batch tiers, no pooling, no golden fixtures, no
// text embeddings -- and this file owns the whole path. Returns -1 for any
// other container, so the embedding path never learns it exists.
//
// WHY IT IS NOT ARCH=6 WITH A FLAG
// --------------------------------
// arch=6 runs one network over one frame and returns boxes and keypoints.
// This runs a DETECTOR over the frame, and then a separate RECOGNISER once
// per box, on a crop of that same frame. The two differ in what they read out
// of the network: a box and a score against a box, a score and a STRING. And
// they differ in what the string means: arch=6's keypoints are a fixed-size
// vector whose length the caller knows, and this one's text is a variable
// length that the CTC decoder decided.
//
// THE TWO STAGES ARE ONE MODEL
// ----------------------------
// The container holds a detector, a recogniser AND the character dictionary,
// because a detector finds text and cannot read it, and a recogniser reads a
// crop and cannot find one. A container holding only one of them answers "no
// text here" to every image, which is a plausible wrong answer rather than an
// error.
//
// SPDX-License-Identifier: Apache-2.0
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <cstddef>
#include <filesystem>
#include <stdexcept>

#include "cli/flags.hpp"
#include "common/app_state.hpp"
#include "common/design_selection.hpp"
#include "common/host_kernels.hpp"   // app::now_s
#include "common/npu_ops_flag.hpp"   // parse_npu_ops, refuse_removed_op_flags
#include "ocr/session.hpp"
#include "runtime/design.hpp"        // prefer_embedded: the container's own set
#include "runtime/device.hpp"        // require_context_budget, survey_contexts
#include "runtime/model.hpp"
#include "runtime/pool.hpp"
#include "vit/image.hpp"

namespace app {

// Returns the process exit code, or -1 when this container is not an arch=9
// model.
inline int maybe_ocr_mode(const std::string &root, int argc, char **argv,
                           const std::string &model_path) {
  namespace fs = std::filesystem;
  npue::File probe(model_path);
  std::string arch;
  try {
    arch = probe.config_string("arch");
  } catch (const std::exception &) {
    return -1;   // a pre-arch container: an embedder, not ours
  }
  if (arch != npue::ocr::kArch) return -1;

  const std::string model_name = fs::path(model_path).stem().string();

  refuse_removed_op_flags(argc, argv);
  refuse_exporter_only_flags(argc, argv);

  auto flag = [&](const char *name) {
    std::string v;
    for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == name) v = argv[i + 1];
    return v;
  };
  auto has = [&](const char *name) {
    for (int i = 1; i < argc; ++i)
      if (std::string(argv[i]) == name) return true;
    return false;
  };

  // --npu-ops: `conv` MOVES THE DENSE CONVOLUTIONS of BOTH graphs onto the
  // array, and the depthwise ones stay on the host whatever it says -- a
  // depthwise filter reduces WITHIN one channel, so there is no [M, N] GEMM
  // in it to dispatch. Any other code is refused by name, because there is no
  // operation by that name in either graph.
  {
    std::string listing;
    for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == "--npu-ops") listing = argv[i + 1];
    const std::set<std::string> codes = parse_npu_ops(listing);
    const bool empty_code = listing == "none" || listing == "";
    if (!codes.empty() && !empty_code) {
      const std::set<std::string> here = {"conv"};
      for (const std::string &c : codes)
        if (!here.count(c))
          throw std::runtime_error(
              "--npu-ops " + c + ": this architecture has no operation named '" +
              c + "'. Its two graphs are {Conv, ConvTranspose, Relu, Sigmoid, "
              "HardSwish, HardSigmoid, Add, Sub, Mul, Div, Sqrt, Concat, "
              "MaxPool, AvgPool, GlobalAveragePool, Resize, Reshape, "
              "Transpose, Slice, Squeeze, Unsqueeze, MatMul, Softmax, "
              "ReduceMean} and the graph already FUSES the activations into "
              "the convolutions and the normalisation into the weights before "
              "it, so 'gelu' has nothing to take over and 'layn' has nothing "
              "to normalise -- there is no BatchNormalization left in either "
              "graph to name. Only 'conv' moves anything: the dense "
              "convolutions, onto a design set built for their padded (K, N) "
              "pairs.");
    }
  }

  // --serve is REFUSED BY NAME, and read through the SHARED reader so that
  // `--serve 9000` and a bare `--serve` are caught the same way they are in
  // the pose and hands modes.
  {
    int serve_port = 0;
    std::string serve_bind;
    if (read_serve(argc, argv, serve_port, serve_bind)) {
      throw std::runtime_error(
          std::string("this architecture has no HTTP endpoint yet, so `serve` "
                      "cannot be honoured for an OCR container. The four that "
                      "exist are /v1/embeddings (arch 1,2,3), "
                      "/v1/audio/transcriptions (arch 4), /v1/classify (arch 5) "
                      "and /v1/pose (arch 6, YOLOv8n-pose -- a DIFFERENT pose "
                      "model); arch 9 is the third container whose arch the "
                      "dispatcher routes here without an endpoint to land on. "
                      "Not refused because a text cannot be found over HTTP -- "
                      "it can -- but because a server that answers from a "
                      "DIFFERENT code path than `npuimage ocr` is exactly the "
                      "drift the pose endpoint documents as a contract. Until "
                      "the endpoint exists, the CLI is the whole answer."));
    }
  }

  // The OTHER architectures' flags are REFUSED rather than ignored, and their
  // own with them. They read as "thresholds for the detection" on a mode
  // whose thresholds are in the CONTAINER, and silently dropping them would
  // produce a run whose text differs from the one that was asked for.
  for (const char *f : {"--conf", "--iou", "--kpt", "--max-det", "--pose",
                        "--pose-dump", "--bo-mode", "--max-hands",
                        "--hands-dump", "--max-people", "--mppose-dump"})
    if (has(f))
      throw std::runtime_error(
          std::string(f) + ": not a flag of `ocr`. The detector's thresholds "
          "and the recogniser's are read from the container (det_thresh, "
          "det_box_thresh, det_unclip_ratio, det_max_candidates, "
          "det_min_size), because they are part of how this checkpoint was "
          "trained and the demo that recorded the golden used the "
          "container's values, not a command line's. For --pose: use `pose`, "
          "which is a different model with different flags; for --max-people: "
          "use `mppose`, whose second stage is a landmark network and not a "
          "recogniser. The dumps are separate because they record different "
          "graphs, and one reader cannot read another's file.");

  // -- the images, given as --ocr (run_ocr() rewrites positionals into it).
  std::vector<std::string> images;
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--ocr") images.push_back(argv[i + 1]);
  if (images.empty())
    throw std::runtime_error(
        "this is an OCR model, so say what to read:\n"
        "    npuimage ocr <model> <image.png> [more.png ...]\n"
        "  (PNG and JPEG; anything else is refused rather than guessed at)\n"
        "  --max-boxes 1      read at most N boxes\n"
        "  --text             a human summary instead of JSON\n"
        "  --ocr-dump FILE    every graph node's output, for the gate\n"
        "  (every threshold comes from the container, not from a flag: they are\n"
        "   part of the checkpoint -- see the refusal above)");

  // --max-boxes: a CAP on how many boxes reach the recogniser, not a score
  // threshold. Every box is an independent crop of the original frame, so
  // there is no way to batch them and the only lever is how many to read.
  int64_t max_boxes = -1;
  {
    const std::string v = flag("--max-boxes");
    if (!v.empty()) {
      char *end = nullptr;
      const long n = std::strtol(v.c_str(), &end, 10);
      if (end == v.c_str() || *end != '\0' || n < 1 || n > 10000)
        throw std::runtime_error("--max-boxes '" + v +
                                 "': expected an integer in 1..10000, or omit "
                                 "it to read every box the detector found");
      max_boxes = n;
    }
  }
  const bool text = has("--text");
  (void)has("--json");   // the default; accepted so it is not mistaken for junk
  (void)root;

  const int threads =
      std::max(1, std::atoi(flag("--threads").empty() ? "16"
                                                     : flag("--threads").c_str()));

  // -- the artifacts. EMPTY BY DEFAULT, and that is the point: the host
  // session needs no device, no design and no directory, so `ocr` runs on a
  // machine with no /dev/accel0. Only --npu-ops conv (or an explicit
  // --artifacts) opens one.
  std::string art;
  const bool conv_on_array =
      parse_npu_ops(flag("--npu-ops")).count("conv") != 0;
  const bool from_cli = !flag("--artifacts").empty();
  const bool self_sufficient =
      npu::prefer_embedded(&probe, "", "gemm_rtp").has("design.json");
  if (!from_cli && self_sufficient) {
    // Nothing to resolve. The Design comes out of the container below.
  } else if (conv_on_array || from_cli) {
    const std::string named = flag("--artifacts");
    std::string streams;
    try {
      streams = probe.config_string("npu_streams");
    } catch (const std::exception &) {
    }
    if (streams.empty())
      throw std::runtime_error(
          "--npu-ops conv needs the array panels, and this container does not "
          "carry any: it was packed without --npu. Repack with "
          "`--ocr-onnx models/paddleocr-onnx --npu --device npu1`, which "
          "stages a pre-tiled bf16 B panel for every DENSE convolution of "
          "BOTH graphs. (A design set is the SECOND requirement, and a "
          "separate one.)");
    if (named.empty()) {
      size_t n = 0;
      for (size_t i = 0; i < streams.size(); ++i)
        if (streams[i] == '{') ++n;
      throw std::runtime_error(
          "--npu-ops conv needs a design set on disk: pass --artifacts <dir> "
          "naming one, or build it with `python tools/export/export_gemm_rtp.py "
          "--target paddleocr-onnx --arch 1 -n 32`. The flag has no default to "
          "fall back on, because this container's " + std::to_string(n) +
          " padded (K, N) designs are its own. If you built one, name it: the "
          "runtime will not go looking for a directory that has no default, "
          "because picking one silently is how a run reports array numbers "
          "from a design set belonging to another model.");
    }
    const std::vector<std::string> cands = artifacts_candidates(root, named);
    auto usable = [](const std::string &c) {
      return std::ifstream(c + "/gemm_rtp/design.json").good();
    };
    art = select_set_for_layout(cands, usable, "");
    if (art.empty()) {
      std::string looked;
      for (size_t i = 0; i < cands.size(); ++i)
        looked += (i ? ", " : "") + cands[i];
      throw std::runtime_error(
          "no ocr design set found for --artifacts '" + named +
          "'; looked for gemm_rtp/design.json under " + looked +
          ". Export one with: python tools/export/export_gemm_rtp.py --target "
          "paddleocr-onnx --arch 1 -n 32");
    }
    if (!npu::require_context_budget(npu::survey_contexts(), 1,
                                     has("--allow-contention"), stderr))
      throw std::runtime_error(
          "NPU context budget: refusing to load 1 hw_context -- see the report "
          "above (close the other process, or pass --allow-contention)");
  }

  const double t0 = app::now_s();
  npue::ocr::Session session(probe, model_name, art, threads, max_boxes,
                              conv_on_array || from_cli);
  const double t_setup = app::now_s() - t0;
  const auto &g = session.geometry();

  // -- the status block --------------------------------------------------------
  std::fprintf(stderr,
               "NpuEmbeddings -- ocr (PaddleOCR text det + rec) on the CPU\n");
  std::fprintf(stderr,
               "  model      %s: detector %lldpx, recogniser %lldx%lld, %lld "
               "classes over %lld characters\n",
               model_name.c_str(),
               static_cast<long long>(g.det_input_size),
               static_cast<long long>(g.rec_input_height),
               static_cast<long long>(g.rec_input_width),
               static_cast<long long>(g.rec_num_classes),
               static_cast<long long>(g.rec_dict_size));
  std::fprintf(stderr,
               "  thresholds thresh %.3f, box %.3f, unclip %.1f, max_candidates "
               "%lld, min_size %lld -- from the container, not a flag\n",
               g.det_thresh, g.det_box_thresh, g.det_unclip_ratio,
               static_cast<long long>(g.det_max_candidates),
               static_cast<long long>(g.det_min_size));
  std::fprintf(stderr,
               "  normalise  detector (x/255 - %.3f)/%.3f, pad %g in the RASTER "
               "(black); recogniser (x/255 - %.3f)/%.3f, pad %g in the "
               "NORMALISED space\n",
               g.det_mean.empty() ? 0.0 : g.det_mean[0],
               g.det_std.empty() ? 1.0 : g.det_std[0], g.det_pad_value,
               g.rec_mean.empty() ? 0.0 : g.rec_mean[0],
               g.rec_std.empty() ? 1.0 : g.rec_std[0], g.rec_pad_value);
  // THE COUNTS, AND THE TRANSPOSED CONVOLUTIONS ARE COUNTED SEPARATELY.
  //
  // They are not dense convolutions that happen to be slow: they are not
  // convolutions the array can run at all, and the line below says so. A
  // count that put them in the "conv" column would claim the array ran 59
  // of the detector's convolutions when it ran 57.
  int64_t det_dense = 0, det_dw = 0, det_tr = 0;
  int64_t rec_dense = 0, rec_dw = 0, rec_tr = 0;
  for (const auto &c : g.det_convs) {
    if (c.transposed) ++det_tr;
    else if (c.depthwise()) ++det_dw;
    else ++det_dense;
  }
  for (const auto &c : g.rec_convs) {
    if (c.transposed) ++rec_tr;
    else if (c.depthwise()) ++rec_dw;
    else ++rec_dense;
  }
  const bool on_array = session.array_placement();
  std::fprintf(stderr, "  ops        (npu = dispatched, host = this process)\n");
  std::fprintf(stderr, "             %-26s %-5s %s\n",
               ((std::to_string(det_dense) + " x conv + " +
                 std::to_string(det_dw) + " x dwconv")
                    .c_str()),
               on_array ? "npu" : "host",
               "text detector, once per frame");
  std::fprintf(stderr, "             %-26s %-5s %s\n",
               ((std::to_string(rec_dense) + " x conv + " +
                 std::to_string(rec_dw) + " x dwconv")
                    .c_str()),
               on_array ? "npu" : "host",
               "text recogniser, once per box");
  std::fprintf(stderr, "             %-26s %-5s %s\n",
               (std::to_string(det_tr + rec_tr) + " x convtranspose").c_str(),
               "host",
               "the DB head's two upsampling convolutions: a sum of SHIFTED "
               "contributions, not a GEMM");
  std::fprintf(stderr, "             %-26s %-5s %s\n", "db, ctc", "host",
               "host-side by construction: a list of boxes and a string");
  std::fprintf(stderr, "  setup      %.3f s (both graphs read, weights "
                       "transposed once)\n",
               t_setup);
  if (session.array_placement())
    std::fprintf(stderr, "  array      %.1f MB of bf16 panels staged, once, at "
                         "load\n",
                 session.array_staged_bytes() / (1024.0 * 1024.0));

  // -- the dump ----------------------------------------------------------------
  //
  // --ocr-dump FILE writes every graph node's output of BOTH graphs, in graph
  // order, as raw fp32 preceded by a reserved JSON header. It exists so the
  // gate can say which node this walk and an independent reading of the ONNX
  // graph first disagree at.
  constexpr size_t kDumpHeader = 1u << 20;   // 1 MiB, and not 64 KiB: the
                                         // detector has 154 nodes and the
                                         // recogniser 334, and the recogniser
                                         // runs ONCE PER BOX -- so a page of
                                         // text can need a hundred thousand
                                         // records before the header is
                                         // written.
  const std::string dump_path = flag("--ocr-dump");
  std::ofstream dump;
  std::vector<std::string> dump_index;
  size_t dump_img = 0;
  if (!dump_path.empty()) {
    dump.open(dump_path, std::ios::binary | std::ios::trunc);
    if (!dump)
      throw std::runtime_error("--ocr-dump " + dump_path +
                               ": cannot open it for writing");
    dump.seekp(0, std::ios::end);
    dump.write(std::string(kDumpHeader, ' ').data(),
               static_cast<std::streamsize>(kDumpHeader));
    std::fprintf(stderr,
                 "  dump       %zu + %zu graph nodes per image -> %s (fp32, "
                 "JSON header in the first MiB)\n",
                 g.det_graph.size(), g.rec_graph.size(), dump_path.c_str());
    auto hook = [&](const char *which) {
      return [&, which](int64_t i, const npue::ocr::Tensor &t) {
        // THE SHAPE IN FULL, and not four of its axes: the recogniser's
        // attention block has rank-5 tensors, and a dump that records four
        // axes of them is a dump whose shapes cannot be checked.
        //
        // The index is -1 for the network's INPUT, which the dump records
        // before the first op: the gate feeds the same tensor to its own
        // reading of the graph, and the input is a resample of a photograph,
        // so the two sides cannot each derive it and agree.
        std::string line =
            "    {\"image\": " + std::to_string(dump_img) + ", \"net\": \"" +
            which + "\", \"i\": " + std::to_string(i) + ", \"rank\": " +
            std::to_string(t.rank()) + ", \"shape\": [";
        for (size_t k = 0; k < t.shape.size(); ++k)
          line += (k ? ", " : "") + std::to_string(t.shape[k]);
        line += "], \"n\": " + std::to_string(t.d.size()) + ", \"off\": " +
                std::to_string(static_cast<unsigned long long>(dump.tellp())) +
                "}";
        dump.write(reinterpret_cast<const char *>(t.d.data()),
                   static_cast<std::streamsize>(t.d.size() * sizeof(float)));
        dump_index.push_back(line);
      };
    };
    session.det_network().on_node = hook("det");
    session.rec_network().on_node = hook("rec");
  }

  // -- run ---------------------------------------------------------------------
  int exit = 0;
  for (const std::string &path : images) {
    npue::ocr::Result r;
    try {
      r = session.read_file(path);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "  %s: %s\n", path.c_str(), e.what());
      exit = 1;
      continue;
    }
    ++dump_img;
    std::fprintf(stderr,
                 "  %-40s %lldx%lld  %lld box%s -> %lld read  %.1f ms "
                 "(front %.1f / det %.1f / db %.1f / crop %.1f / rec %.1f)\n",
                 path.c_str(), static_cast<long long>(r.width),
                 static_cast<long long>(r.height),
                 static_cast<long long>(r.boxes_found),
                 r.boxes_found == 1 ? "" : "es",
                 static_cast<long long>(r.boxes_read), r.total_s * 1e3,
                 r.front_end_s * 1e3, r.det_s * 1e3, r.post_s * 1e3,
                 r.crop_s * 1e3, r.rec_s * 1e3);
    if (session.array_placement())
      std::fprintf(stderr,
                   "             array: %lld dispatches, %.1f ms on the device\n",
                   static_cast<long long>(session.array_dispatches()),
                   1000.0 * session.array_seconds());
    {
      // THE SPLIT, MEASURED, and not the plan.
      //
      // A line that says "the array ran the convolutions" is a statement
      // about the placement and not about the cost: the device's own time is
      // only part of it, because the host still gathers A into the design's
      // padded stride and transposes C back. So the numbers below are the
      // ones a reader needs to decide whether the array is worth it, and the
      // host's share is the first of them.
      //
      // THE SAME BLOCK FOR BOTH PATHS, because the comparison that matters is
      // the host's dense convolutions against the device's, and two blocks
      // with different shapes cannot be compared by eye.
      const auto &dc = r.det_cost;
      const auto &rc = r.rec_cost;
      const bool arr = session.array_placement();
      auto line = [&](const char *who, const npue::ocr::Cost &c,
                      int64_t dense, int64_t dw, int64_t tr) {
        if (arr)
          std::fprintf(stderr,
                       "             split  %s: %lld convs on the device (%.1f ms "
                       "of wall: %.1f ms of GEMM, %.1f ms of repack, %.1f ms of "
                       "transpose back), %lld dwconv + %lld convtranspose on the "
                       "host (%.1f ms), im2col %.1f ms\n",
                       who, static_cast<long long>(dense), 1000.0 * c.t_array,
                       1000.0 * c.t_array_gemm, 1000.0 * c.t_array_repack,
                       1000.0 * c.t_array_transpose,
                       static_cast<long long>(dw), static_cast<long long>(tr),
                       1000.0 * (c.t_dwconv + c.t_transposed),
                       1000.0 * c.t_im2col);
        else
          std::fprintf(stderr,
                       "             split  %s: %lld convs on the host (%.1f ms: "
                       "%.1f ms of GEMM, %.1f ms of transpose back), %lld dwconv "
                       "+ %lld convtranspose on the host (%.1f ms), im2col %.1f "
                       "ms\n",
                       who, static_cast<long long>(dense), 1000.0 * c.t_host,
                       1000.0 * (c.t_host - c.t_transpose),
                       1000.0 * c.t_transpose,
                       static_cast<long long>(dw), static_cast<long long>(tr),
                       1000.0 * (c.t_dwconv + c.t_transposed),
                       1000.0 * c.t_im2col);
      };
      line("det", dc, dc.convs_array ? dc.convs_array : dc.convs_host,
           dc.dwconvs_host, dc.transposed_convs);
      line("rec", rc, rc.convs_array ? rc.convs_array : rc.convs_host,
           rc.dwconvs_host, rc.transposed_convs);
      if (arr)
        std::fprintf(stderr,
                     "                   the device's own time over both graphs "
                     "is %.1f ms; the rest is the host's\n",
                     1000.0 * session.array_seconds());
    }
    if (text) {
      if (r.boxes.empty()) {
        std::printf("%s: no text\n", path.c_str());
        continue;
      }
      for (size_t i = 0; i < r.boxes.size(); ++i) {
        const npue::ocr::TextBox &b = r.boxes[i];
        std::printf("%s: box %zu  score %.3f  conf %.3f  \"%s\"\n",
                    path.c_str(), i, b.box_score, b.conf, b.text.c_str());
      }
      continue;
    }
    std::printf("{\n  \"image\": \"%s\",\n  \"width\": %lld,\n  \"height\": %lld,\n",
                path.c_str(), static_cast<long long>(r.width),
                static_cast<long long>(r.height));
    std::printf(
        "  \"timings_ms\": {\"total\": %.3f, \"front_end\": %.3f, \"detector\": "
        "%.3f, \"db\": %.3f, \"crop\": %.3f, \"recogniser\": %.3f},\n",
        r.total_s * 1e3, r.front_end_s * 1e3, r.det_s * 1e3, r.post_s * 1e3,
        r.crop_s * 1e3, r.rec_s * 1e3);
    std::printf("  \"boxes\": [");
    for (size_t i = 0; i < r.boxes.size(); ++i) {
      const npue::ocr::TextBox &b = r.boxes[i];
      std::printf("%s\n    {\"score\": %.6f, \"conf\": %.6f, \"text\": \"", i ? "," : "",
                  b.box_score, b.conf);
      // The text is JSON-escaped: it comes from a dictionary of characters and
      // a quote in it is a quote in the output, and a raw one makes the whole
      // document unparseable.
      for (char ch : b.text) {
        if (ch == '"' || ch == '\\')
          std::printf("\\%c", ch);
        else if (ch == '\n')
          std::printf("\\n");
        else if (ch == '\t')
          std::printf("\\t");
        else if (static_cast<unsigned char>(ch) < 0x20)
          std::printf("\\u%04x", static_cast<unsigned>(ch));
        else
          std::printf("%c", ch);
      }
      std::printf("\", \"quad\": [");
      for (int k = 0; k < 4; ++k)
        std::printf("%s[%.3f, %.3f]", k ? ", " : "", b.q.x[k], b.q.y[k]);
      std::printf("]}");
    }
    std::printf("%s]}\n", r.boxes.empty() ? "" : "\n  ");
  }

  if (!dump_path.empty()) {
    std::string h = "{\n  \"nodes\": [\n";
    for (size_t i = 0; i < dump_index.size(); ++i)
      h += (i ? ",\n" : "") + dump_index[i];
    h += "\n  ]\n}\n";
    if (h.size() > kDumpHeader)
      throw std::runtime_error(
          "--ocr-dump " + dump_path + ": the header needs " +
          std::to_string(h.size()) + " bytes for " +
          std::to_string(dump_index.size()) +
          " node records and the reserved area is 1048576. Raise kDumpHeader in "
          "runtime/include/runtime/ocr_mode.hpp; do NOT let it overflow, "
          "because the reader would then take tensor bytes for header text.");
    h.append(kDumpHeader - h.size(), ' ');
    dump.seekp(0);
    dump.write(h.data(), static_cast<std::streamsize>(kDumpHeader));
    dump.close();
    std::fprintf(stderr, "             %zu node tensors written\n",
                 dump_index.size());
  }
  return exit;
}

}  // namespace app
