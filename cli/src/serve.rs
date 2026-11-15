//! `serve`: the HTTP listener.
//!
//! The mode is chosen by the CONTAINER'S ARCH, never by a flag. A binary
//! serving a container of another family is refused BY NAME, before the
//! listener opens -- rather than answered by an endpoint this process is not
//! named for.
//!
//! # What the server says about itself
//!
//! The same status block the CLI prints, and then the endpoint's own banner:
//! which path this model answers, and what a request to it looks like. The
//! banner is flushed, because a server that then prints nothing would leave
//! its port in a block buffer, and an operator tailing the log would
//! conclude that the process had not started.
//!
//! # The model id, and the backend
//!
//! The C++'s server hardcodes `"backend":"amd-xdna2-npu"` in all four of
//! its backends -- on a machine whose own status block says `device npu1`.
//! This one reports the device it found. And the model id says `-cpu` when
//! the compute happened on the host, because the C++'s own convention says
//! so: "the model id is what a client sees, and naming a host-only server
//! after the array is exactly the silent mislabel".

use npuembeddings_server::routes::ServerState;
use npuembeddings_server::Result;

/// Serve one model.
///
/// The refusal comes FIRST and the listener does not open until it has
/// passed: a refusal that opens a port and then closes it is a refusal the
/// user can fail to notice.
pub fn serve(
    binary: &'static str,
    model: &str,
    port: u16,
    bind: &str,
    root: &std::path::Path,
    args: &crate::flags::Args,
) -> i32 {
    let path = match resolve_model(model, root) {
        Some(p) => p,
        None => {
            eprintln!("error: the model {model} is not installed.");
            eprintln!("       `serve` downloads and verifies a model it does not have;");
            eprintln!("       give a name from the catalogue, or a path to a .npue.");
            return crate::EXIT_REFUSED;
        }
    };
    let container = match npum_format::Container::open(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {}: {e}", path.display());
            return crate::EXIT_REFUSED;
        }
    };
    let arch = container.config_str("arch").unwrap_or_default().to_string();

    let this_family = match binary {
        "npuaudio" => npum_families::Family::Stt,
        "npuimage" => npum_families::Family::Cls,
        _ => npum_families::Family::Embed,
    };
    let family = npum_families::Family::from_arch(&arch).unwrap_or(npum_families::Family::Embed);
    let belongs = match this_family {
        npum_families::Family::Embed => family == npum_families::Family::Embed,
        npum_families::Family::Stt => family == npum_families::Family::Stt,
        _ => matches!(
            family,
            npum_families::Family::Cls
                | npum_families::Family::Pose
                | npum_families::Family::Hands
                | npum_families::Family::MpPose
                | npum_families::Family::Ocr
        ),
    };
    if !belongs {
        eprint!(
            "{}",
            npum_families::registry::refuse_other_family(model, &arch, family, binary)
        );
        return crate::EXIT_REFUSED;
    }

    // The status block, the same one the CLI prints. Each family has its
    // own: a BERT-shaped model and a ViT are not the same architecture
    // and their blocks do not describe the same things.
    //
    // The speech family's engine is built HERE, inside the block's own
    // branch, because the block reports the setup time and the setup is
    // the engine's construction.
    let mut speech: Option<npuembeddings_server::Transcriber> = None;
    let mut image: Option<npuembeddings_server::Classifier> = None;
    let mut pose: Option<npuembeddings_server::PoseDetector> = None;
    match family {
        npum_families::Family::Embed => {
            let status = crate::status::read_container(&container, &path);
            let artifacts = crate::embed::find_artifacts(root, &status.model_name);
            let design = artifacts
                .as_ref()
                .and_then(|a| crate::status::read_design(a));
            crate::status::print_block(
                &status,
                design.as_ref(),
                &format!(
                    "{} (picked from the container's geometry; no --artifacts given)",
                    artifacts.as_deref().unwrap_or(root).display()
                ),
                // THE SERVER RUNS THE HOST'S OPERATIONS.
                //
                // The server does not take `--npu-ops`, and it does not
                // dispatch to the array -- so every operation in the block
                // is on the host, and the block says so.
                &crate::status::Places::default(),
            );
        }
        npum_families::Family::Cls => {
            let Some(geometry) = crate::vit::read_geometry(&container, &path) else {
                eprintln!(
                    "error: {}: the container does not carry the ViT's geometry",
                    path.display()
                );
                return crate::EXIT_REFUSED;
            };
            let designs = crate::vit::find_designs(root, &geometry.model_name).unwrap_or_else(
                || {
                    root.join("runtime")
                        .join("artifacts")
                        .join(&geometry.model_name)
                },
            );
            let started = std::time::Instant::now();
            let classifier = match npuembeddings_server::Classifier::open(&path) {
                Ok(c) => c,
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
            let setup_s = started.elapsed().as_secs_f64();
            crate::vit::print_block(&geometry, &designs, setup_s);
            image = Some(classifier);
        }
        npum_families::Family::Stt => {
            // The context budget, BEFORE the session builds anything, and the
            // block the CLI prints: a server that described itself
            // differently from the command line would be two runtimes
            // wearing one name.
            if !crate::stt::check_context_budget(args) {
                return crate::EXIT_REFUSED;
            }
            let Some(geometry) = crate::stt::read_geometry(&container, &path) else {
                eprintln!(
                    "error: {}: the container does not carry the speech family's geometry",
                    path.display()
                );
                return crate::EXIT_REFUSED;
            };
            let designs = crate::stt::find_designs(root, &geometry.model_name).unwrap_or_else(
                || {
                    root.join("runtime")
                        .join("artifacts")
                        .join(&geometry.model_name)
                },
            );
            let encoder_design = crate::stt::read_design(&designs, false);
            let decoder_design = crate::stt::read_design(&designs, true);
            let started = std::time::Instant::now();
            let transcriber = match npuembeddings_server::Transcriber::open(
                &path,
                crate::stt::options_from(args),
            ) {
                Ok(t) => t,
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
            let setup_s = started.elapsed().as_secs_f64();
            crate::stt::print_block(
                &geometry,
                &designs,
                encoder_design.as_ref(),
                decoder_design.as_ref(),
            );
            let options = crate::stt::options_from(args);
            crate::stt::print_schedule(setup_s, &options, args.value("--language").is_some());
            speech = Some(transcriber);
        }
        npum_families::Family::Pose => {
            let started = std::time::Instant::now();
            let detector = match npuembeddings_server::PoseDetector::open(&path) {
                Ok(d) => d,
                Err(reason) => {
                    eprintln!("error: {reason}");
                    return crate::EXIT_REFUSED;
                }
            };
            let setup_s = started.elapsed().as_secs_f64();
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let geometry = detector.geometry().clone();
            crate::pose::print_block(&name, &geometry, detector.threads(), setup_s);
            pose = Some(detector);
        }
        npum_families::Family::MpPose => {
            // NOT "not ported": the block IS ported and `npuimage mppose`
            // prints it. The reason there is no server here is the oracle's
            // own -- arch 8 has no endpoint in C++ either -- and saying
            // otherwise would name a gap that does not exist while hiding the
            // one that does.
            eprintln!(
                "error: this architecture has no HTTP endpoint yet, so `serve` cannot be \
                 honoured for a MediaPipe Pose container. `npuimage mppose` is the whole \
                 answer: a server answering from a different code path than the CLI is \
                 exactly the drift the pose endpoint documents as a contract."
            );
            return crate::EXIT_REFUSED;
        }
        npum_families::Family::Ocr => {
            // The oracle's refusal, in the oracle's words: arch 9 has no
            // endpoint in C++ either, and saying "not ported" would name a
            // gap that does not exist -- the CLI reads text -- while hiding
            // the one that does: there is no endpoint to land on.
            eprintln!(
                "error: this architecture has no HTTP endpoint yet, so `serve` cannot be \
                 honoured for an OCR container. The four that exist are /v1/embeddings \
                 (arch 1,2,3), /v1/audio/transcriptions (arch 4), /v1/classify (arch 5) \
                 and /v1/pose (arch 6, YOLOv8n-pose -- a DIFFERENT pose model); arch 9 is \
                 the third container whose arch the dispatcher routes here without an \
                 endpoint to land on. Not refused because a text cannot be found over \
                 HTTP -- it can -- but because a server that answers from a DIFFERENT \
                 code path than `npuimage ocr` is exactly the drift the pose endpoint \
                 documents as a contract. Until the endpoint exists, the CLI is the whole \
                 answer."
            );
            return crate::EXIT_REFUSED;
        }
        _ => {
            // The families whose block is not written yet refuse to start.
            // A server that answered a request it cannot answer would be
            // worse than one that says it is not here.
            eprintln!(
                "error: the {} family's status block is not ported yet, so this \
                 server has nothing honest to say about itself.",
                family.as_str()
            );
            return crate::EXIT_REFUSED;
        }
    }
    let status = crate::status::read_container(&container, &path);

    let mut state = build_state(binary, &path, &arch, family, &container, &status, root);
    if let Some(encoder) = &state.encoder {
        crate::status::print_tokenizer(encoder.vocab_size());
    }
    if let Some(transcriber) = speech {
        // The speech family's own line, and its own word: the C++'s server
        // says "ids" here and the embed server says "tokens", because a
        // Whisper's vocabulary is a BPE's entries and a BERT's is a word
        // piece's.
        println!(
            "  tokenizer  {} ids, from the .npum",
            transcriber.vocab_size()
        );
        state.transcriber = Some(std::sync::Arc::new(transcriber));
    }
    if let Some(classifier) = image {
        state.classifier = Some(std::sync::Arc::new(classifier));
    }
    if let Some(detector) = pose {
        state.pose = Some(std::sync::Arc::new(detector));
    }

    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: the async runtime could not be started: {e}");
            return crate::EXIT_REFUSED;
        }
    };
    runtime.block_on(async move {
        let app = npuembeddings_server::router(state.clone());
        let address = format!("{bind}:{port}");
        let listener = match tokio::net::TcpListener::bind(&address).await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("error: cannot listen on {address}: {e}");
                return crate::EXIT_REFUSED;
            }
        };
        print_banner(binary, &state, &address);
        use std::io::Write;
        let _ = std::io::stdout().flush();

        // The stop path: the C++'s handler sets a flag, the accept loop
        // notices it, and the message goes to stderr on the way out.
        let mut terminate = match tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate(),
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: the signal handler could not be installed: {e}");
                return crate::EXIT_REFUSED;
            }
        };
        let mut interrupt = match tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::interrupt(),
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: the signal handler could not be installed: {e}");
                return crate::EXIT_REFUSED;
            }
        };
        tokio::select! {
            result = axum::serve(listener, app) => {
                if let Err(e) = result {
                    eprintln!("error: the server stopped: {e}");
                    return crate::EXIT_REFUSED;
                }
            }
            _ = terminate.recv() => {}
            _ = interrupt.recv() => {}
        }
        eprintln!("\n  interrupted -- releasing NPU contexts");
        0
    })
}

/// The endpoint's banner: which path it answers, and what a request is.
fn print_banner(binary: &str, state: &ServerState, address: &str) {
    match state.arch.as_str() {
        arch if arch.contains("whisper") => {
            println!(
                "\n  serving http://{address}/v1/audio/transcriptions   (model {})",
                state.model_id
            );
            println!("  POST multipart/form-data: file (required), model, language, task,");
            println!("              response_format (json | verbose_json | text)");
            // The shape of the work, so a client knows what it is asking
            // for: how many mel bands the front end produces and how many
            // encoder positions a window becomes.
            let (mel_bins, positions) = state
                .transcriber
                .as_ref()
                .map(|t| (t.mel_bins(), t.positions()))
                .unwrap_or((0, 0));
            println!(
                "  {mel_bins} mel bins, {positions} encoder positions, greedy, temperature 0, no timestamps"
            );
            println!();
        }
        arch if arch.contains("vit") => {
            println!(
                "\n  serving http://{address}/v1/classify   (model {})",
                state.model_id
            );
            println!("  POST multipart/form-data: image (required, PNG or JPEG), top_k (default 1)");
            let (labels, size) = state
                .classifier
                .as_ref()
                .map(|c| (c.num_labels(), c.image_size()))
                .unwrap_or((0, 0));
            println!("  {labels} labels, {size}px input");
        }
        arch if arch.contains("pose") || arch.contains("mediapipe") => {
            println!(
                "\n  serving http://{address}/v1/pose   (model {})",
                state.model_id
            );
            println!("  POST multipart/form-data: image (required, PNG or JPEG), conf, iou, kpt, max_det");
            let size = state
                .pose
                .as_ref()
                .map(|p| p.geometry().input_size)
                .unwrap_or(0);
            let max_det = npum_cpu::pose_decode::DecodeParams::default().max_det;
            println!("  {size}x{size} input, {max_det} people max, host backend");
        }
        _ => {
            println!(
                "\n  serving http://{address}/v1/embeddings   (model {}, seq {})",
                state.model_id,
                state.encoder.as_ref().map(|e| e.seq()).unwrap_or(0)
            );
            println!("  POST {{\"input\": \"text\" | [\"a\",\"b\"], \"encoding_format\": \"float\"|\"base64\"}}");
            println!();
        }
    }
    let _ = binary;
}

/// Build the server's state: the bodies and the engine.
fn build_state(
    binary: &'static str,
    path: &std::path::Path,
    arch: &str,
    family: npum_families::Family,
    container: &npum_format::Container,
    status: &crate::status::Status,
    root: &std::path::Path,
) -> ServerState {
    let mut state = ServerState::new(&path.display().to_string(), arch, binary);

    // The model id: the container's stem, and -- for the TEXT family only
    // -- where the compute happened. The other families' ids are the bare
    // stem, because that is what the C++'s servers use for them.
    let host_only = container.config_str("gemm_layout").unwrap_or("host") == "host";
    state.model_id = match family {
        npum_families::Family::Embed => format!(
            "{}-{}",
            status.model_name,
            if host_only { "cpu" } else { "npu" }
        ),
        _ => status.model_name.clone(),
    };

    // The backend: the device this process found.
    let generation = npum_xdna::device::Device::detect()
        .map(|d| d.generation())
        .unwrap_or(npum_xdna::device::Generation::Npu1);
    let backend = match generation {
        npum_xdna::device::Generation::Npu1 => "amd-xdna1-npu",
        npum_xdna::device::Generation::Npu2 => "amd-xdna2-npu",
    };

    state.models = format!(
        "{{\"object\":\"list\",\"data\":[{{\"id\":\"{}\",\"object\":\"model\",\"owned_by\":\"npuembeddings\"}}]}}",
        state.model_id
    );

    if arch.contains("whisper") {
        state.health = format!(
            "{{\"status\":\"ok\",\"model\":\"{}\",\"backend\":\"{backend}\",\"kind\":\"stt\",\"language\":\"en\",\"task\":\"transcribe\",\"timestamps\":false,\"chunk_seconds\":30,\"stride_seconds\":5}}",
            state.model_id
        );
        state.not_found = format!(
            "this model serves /v1/audio/transcriptions, not {}. `npuaudio serve <model>` picks the endpoint from the container's arch: text models answer /v1/embeddings, Whisper /v1/audio/transcriptions, a ViT /v1/classify and YOLOv8-pose /v1/pose.",
            "/v1/embeddings"
        );
        return state;
    }
    if arch.contains("vit") {
        let labels = container
            .tensor_bytes("labels.table")
            .map(|b| String::from_utf8_lossy(b).to_string())
            .unwrap_or_default();
        let names: Vec<&str> = labels.lines().filter(|l| !l.is_empty()).collect();
        let n_labels = names.len();
        let mut body = String::from("{\"object\":\"list\",\"data\":[");
        for (i, name) in names.iter().enumerate() {
            if i > 0 {
                body.push_str(", ");
            }
            body.push_str(&format!(
                "{{\"id\": {i}, \"name\": \"{}\"}}",
                npuembeddings_server::routes::json_escape(name)
            ));
        }
        body.push_str("]}");
        state.labels = Some(body);
        state.health = format!(
            "{{\"status\":\"ok\",\"model\":\"{}\",\"backend\":\"{backend}\",\"kind\":\"classify\",\"int8\":false,\"input_size\":224,\"patch_size\":16,\"labels\":{n_labels},\"top_k\":1,\"dispatches_so_far\":0}}",
            state.model_id
        );
        state.not_found = "this model serves /v1/classify, not /v1/embeddings. `npuimage serve <model>` picks the endpoint from the container's arch: text models answer /v1/embeddings, Whisper /v1/audio/transcriptions, a ViT /v1/classify and YOLOv8-pose /v1/pose.".to_string();
        return state;
    }
    if arch.contains("pose") || arch.contains("mediapipe") {
        state.health = format!(
            "{{\"status\":\"ok\",\"model\":\"{}\",\"backend\":\"{backend}\",\"kind\":\"pose\",\"engine\":\"host\",\"input_size\":640,\"conf\":0.25,\"iou\":0.7,\"keypoint\":0.5,\"max_det\":300,\"skeleton\":\"\"}}",
            state.model_id
        );
        state.not_found = "this model serves /v1/pose, not /v1/embeddings. `npuimage serve <model>` picks the endpoint from the container's arch: text models answer /v1/embeddings, Whisper /v1/audio/transcriptions, a ViT /v1/classify and YOLOv8-pose /v1/pose.".to_string();
        return state;
    }

    // The text family: the embedding endpoint.
    let seq = crate::embed::sequence_length(path, container, root, None);
    match npuembeddings_server::Embedder::open(path, seq) {
        Ok(encoder) => {
            state.health = format!(
                "{{\"status\":\"ok\",\"model\":\"{}\",\"backend\":\"{backend}\",\"kind\":\"embeddings\"}}",
                state.model_id
            );
            state.not_found = "this model serves /v1/embeddings. `npuembeddings serve <model>` picks the endpoint from the container's arch: text models answer /v1/embeddings, Whisper /v1/audio/transcriptions, a ViT /v1/classify and YOLOv8-pose /v1/pose.".to_string();
            state.encoder = Some(encoder);
        }
        Err(reason) => {
            eprintln!("error: {reason}");
        }
    }
    state
}

/// Resolve a model's name or path to a container on disk.
///
/// A NAME is looked up in `models/`; a PATH is used as given and never
/// re-resolved by name.
pub fn resolve_model(
    model: &str,
    root: &std::path::Path,
) -> Option<std::path::PathBuf> {
    crate::families::resolve_model(model, root).ok()
}

/// The result type of this module.
pub type ServeResult = Result<i32>;
