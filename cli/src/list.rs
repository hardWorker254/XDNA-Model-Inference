//! `list`: the catalogue table.
//!
//! # The table's SHAPE is the contract
//!
//! The same columns, in the same order, with the same states. What is in
//! the table changes as models are installed; the columns do not.
//!
//! # The states, and what each one means
//!
//! | state | meaning |
//! |---|---|
//! | `ready` | installed, with a matching NPU design -- `serve` runs it |
//! | `available` | not downloaded yet -- `serve` fetches and verifies it |
//! | `cpu` | installed, but the container holds row-major host-side GEMM operands, so it runs entirely on the CPU |
//! | `no design` | installed, but no design set for this geometry is present |
//! | `no encoder` | installed, and a design may match, but this build has no forward pass for the architecture |
//! | `stt` | a speech-to-text model: no pooling, and it needs TWO design sets |
//! | `cls` | an image classifier: no pooling and no text |
//! | `pose`, `hands`, `mppose`, `ocr` | a pixel detector: the state's the VERB that runs it |
//!
//! # The state column names the COMMAND, for the non-embedders
//!
//! A locally packed arch=5 container says `cls`, not `ready`: `ready` there
//! would be read as "ready to embed", and the first thing a user tries is
//! the wrong one.
//!
//! And it reads the `kind` from the CONTAINER rather than from the arch.
//! There are THREE pixel architectures now -- arch=6 pose, arch=7 hands,
//! arch=8 mppose -- and they are three verbs, so a state column derived
//! from the arch could only ever name one of them. It would have labelled
//! a MediaPipe Hands container `pose` and sent the user to a command that
//! dispatches by arch and answers "not a pose container" on a file that
//! finds hands perfectly well.
//!
//! `kind` is the container's own claim and the packer writes it, so this
//! cannot drift from the file the way a list in the binary would.

use npum_families::catalogue::{catalogue, CatalogueEntry};
use npum_format::Container;
use std::path::{Path, PathBuf};

/// One installed container.
#[derive(Debug, Clone)]
pub struct Installed {
    /// The container's path.
    pub path: PathBuf,
    /// The container's stem.
    pub name: String,
    /// The container's architecture.
    pub arch: String,
    /// The container's `kind` -- the verb that runs it.
    pub kind: String,
    /// The pooling, or "n/a".
    pub pooling: String,
    /// The number of layers, or 0.
    pub layers: i64,
    /// The hidden size, or 0.
    pub hidden: i64,
    /// The size in megabytes.
    pub mb: f64,
    /// Whether the container holds its own design set.
    pub carries_design: bool,
    /// Whether the container holds the decoder's design set too.
    pub carries_design_dec: bool,
    /// The layout of the B panels: "host" for a plain row-major container.
    pub gemm_layout: String,
    /// The source repository, for a container that came from one.
    pub repo: String,
    /// The error, for a container that would not open.
    pub error: String,
}

/// The installed containers under a root.
///
/// A container that will not open is LISTED with its error rather than
/// skipped: a model silently missing from the table is a worse failure
/// than one that is visibly broken.
pub fn discover(root: &Path) -> Vec<Installed> {
    let mut out = Vec::new();
    let dir = root.join("models");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return out;
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e == "npue" || e == "npum")
                .unwrap_or(false)
        })
        .collect();
    paths.sort();

    for path in paths {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mb = match npum_format::Container::open(&path) {
            Ok(container) => container.header().data_length as f64 / 1_000_000.0,
            Err(_) => std::fs::metadata(&path)
                .map(|m| m.len() as f64 / 1_000_000.0)
                .unwrap_or(0.0),
        };
        let mut entry = Installed {
            path: path.clone(),
            name,
            arch: String::new(),
            kind: String::new(),
            pooling: "n/a".into(),
            layers: 0,
            hidden: 0,
            mb,
            carries_design: false,
            carries_design_dec: false,
            gemm_layout: String::new(),
            repo: String::new(),
            error: String::new(),
        };
        match Container::open(&path) {
            Err(e) => {
                entry.error = e.to_string();
                out.push(entry);
                continue;
            }
            Ok(container) => {
                let config = container.config();
                entry.arch = config.get_str("arch").unwrap_or_default().to_string();
                entry.kind = config.get_str("kind").unwrap_or_default().to_string();
                entry.pooling = config.get_str("pooling").unwrap_or("n/a").to_string();
                entry.layers = config.get_i64("n_layers").unwrap_or(0);
                entry.hidden = config.get_i64("hidden_size").unwrap_or(0);
                entry.gemm_layout = config.get_str("gemm_layout").unwrap_or_default().to_string();
                entry.repo = config
                    .get_str("source_repo")
                    .or_else(|| config.get_str("repo"))
                    .unwrap_or_default()
                    .to_string();
                entry.carries_design = container
                    .directory()
                    .tensor("design/gemm_rtp/design.json")
                    .is_some();
                entry.carries_design_dec = container
                    .directory()
                    .tensor("design/gemm_rtp_dec/design.json")
                    .is_some();
            }
        }
        out.push(entry);
    }
    out
}

/// Whether the architecture has a forward pass in this build.
///
/// The single source both the `list` table and the refusal read, so that
/// they cannot drift: a new arch must be added here and nowhere else.
pub fn encoder_implemented(arch: &str) -> bool {
    matches!(
        arch,
        "bert_abs_gelu_postln"
            | "nomic_bert_rope_swiglu"
            | "gemma3_mqa_rope_geglu"
            | "gte_new_rope_geglu"
            | "whisper_encdec_gelu"
            | "vit_patch16_prenorm_gelu"
            | "yolov8_pose_c2f_silu_dfl"
            | "mediapipe_hands_palm_ssd_lm_heatmap"
            | "mediapipe_pose_det_ssd_lm_regress"
            | "paddleocr_db_svtr_ctc"
    )
}

/// Whether the architecture is a speech-to-text model.
pub fn is_stt_arch(arch: &str) -> bool {
    arch == "whisper_encdec_gelu"
}

/// Whether the architecture is an image classifier.
pub fn is_vit_arch(arch: &str) -> bool {
    arch == "vit_patch16_prenorm_gelu"
}

/// Whether the architecture is a pixel detector.
///
/// The three of them -- pose, hands and whole-body pose -- share one
/// answer, and they are kept apart from the ViT's because the two answer
/// different questions: the ViT needs a design set and has a pooling
/// column, and a detector needs neither.
pub fn is_pose_arch(arch: &str) -> bool {
    matches!(
        arch,
        "yolov8_pose_c2f_silu_dfl"
            | "mediapipe_hands_palm_ssd_lm_heatmap"
            | "mediapipe_pose_det_ssd_lm_regress"
    )
}

/// The pixel KINDS -- the verbs -- as opposed to the pixel ARCHES.
///
/// The distinction is real and both tables need it: the architectures answer
/// "does this build run the architecture", the kinds answer "which verb".
pub fn is_pixel_kind(kind: &str) -> bool {
    matches!(kind, "pose" | "hands" | "mppose" | "ocr")
}

/// Whether a kind has a SUBCOMMAND.
///
/// An empty answer is a refusal to print a command that does not exist,
/// which is the difference between "this model runs" and "this model runs
/// the way you would have to guess".
pub fn subcommand_for_kind(kind: &str) -> bool {
    is_pixel_kind(kind) || kind == "cls" || kind == "stt"
}

/// The last column for a pixel detector: the command, plus the one thing
/// about the array that is true for all of them and false of everything
/// else in the table.
fn pixel_command_note(kind: &str) -> String {
    if !is_pixel_kind(kind) {
        return String::new();
    }
    format!("npuimage {kind} <name> <image>; CPU only unless packed with --npu")
}

/// Whether a design set for this geometry is on disk.
///
/// The catalogue's rows ask this with the CATALOGUE's geometry, and the
/// installed rows ask it with the CONTAINER's. Both answer the same
/// question -- "is there a design set this model could run on" -- and
/// neither is a claim that the design matches: that is `design_fits()`'s
/// job, and it is a different question with a different answer.
fn have_design_on_disk(root: &Path, name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let bases = [
        root.join("artifacts"),
        root.join("runtime").join("artifacts"),
        root.to_path_buf(),
    ];
    for base in bases {
        for sub in ["", "artifacts_npu1", "artifacts_npu2", "npu1", "npu2"] {
            let dir = if sub.is_empty() {
                base.join(name)
            } else {
                base.join(name).join(sub)
            };
            if dir.join("gemm_rtp").join("design.json").is_file() {
                return true;
            }
        }
    }
    false
}

/// The state of a catalogue row.
fn catalogue_state(root: &Path, _entry: &CatalogueEntry, installed: Option<&Installed>) -> String {
    let Some(m) = installed else {
        return "available".into();
    };
    if !encoder_implemented(&m.arch) {
        return "no encoder".into();
    }
    if is_stt_arch(&m.arch) {
        let both = (m.carries_design && m.carries_design_dec)
            || (have_design_on_disk(root, &m.name)
                && have_design_on_disk(root, &format!("{}_dec", m.name)));
        return if both { "ready" } else { "no design" }.into();
    }
    if is_vit_arch(&m.arch) {
        let have = m.carries_design || have_design_on_disk(root, &m.name);
        return if have { "ready" } else { "no design" }.into();
    }
    if m.gemm_layout == "host" {
        return "cpu".into();
    }
    let have = m.carries_design || have_design_on_disk(root, &m.name);
    if have { "ready" } else { "no design" }.into()
}

/// The state of a locally packed row.
fn installed_state(root: &Path, m: &Installed) -> String {
    if !encoder_implemented(&m.arch) {
        return "no encoder".into();
    }
    if subcommand_for_kind(&m.kind) {
        return m.kind.clone();
    }
    if is_stt_arch(&m.arch) {
        return "stt".into();
    }
    if is_vit_arch(&m.arch) {
        return "cls".into();
    }
    if is_pose_arch(&m.arch) {
        return "pose".into();
    }
    if m.gemm_layout == "host" {
        return "cpu".into();
    }
    let have = m.carries_design || have_design_on_disk(root, &m.name);
    if have { "ready" } else { "no design" }.into()
}

/// The legend under the table.
///
/// It is printed by `list` and by the ambiguity refusal in the model
/// resolver, and the last line names THIS binary -- because the legend is
/// printed by `npuembeddings` and reached by `npuaudio` and `npuimage` too.
/// One literal there would have been a legend offering a program the reader
/// did not run.
const LEGEND: &str = concat!(
    "\n  ready      installed, with a matching NPU design -- `serve` runs it\n",
    "  available  not downloaded yet -- `serve` fetches and verifies it\n",
    "  cpu        installed, but this CONTAINER holds row-major host-side\n",
    "             GEMM operands, so it runs entirely on the CPU. Repack it\n",
    "             (the pre-tiled NPU layout is the default) to use the array\n",
    "  no design  installed, but no design set for this geometry is present\n",
    "  no encoder installed, and a design may match, but this build has no\n",
    "             forward pass for the architecture -- it will refuse rather\n",
    "             than return embeddings for the wrong model\n",
    "  stt        installed, and it is a SPEECH-TO-TEXT model: no pooling,\n",
    "             and it needs TWO design sets -- gemm_rtp for the encoder\n",
    "             stack and gemm_rtp_dec for the decoder. `transcribe` and\n",
    "             `serve` are how it runs; `embed` is not. A whisper row says\n",
    "             `no design` until BOTH are exported, which is one command:\n",
    "               python tools/export/export_gemm_rtp.py --target <name> \\\n",
    "                   --arch 1 --out runtime\n",
    "  cls         installed, and it is an IMAGE CLASSIFIER: no pooling and\n",
    "             no text, and it needs ONE design set -- gemm_rtp, the same\n",
    "             four streams bge-base-en-v1.5 uses, because a ViT's GEMM\n",
    "             shapes ARE bge-base's and its patch embedding is [n,768]x\n",
    "             [768,768], which is attn_out's shape. `classify` is how it\n",
    "             runs; `embed` is not:\n",
    "               npuimage classify <name> <image.png>\n\n",
);

/// The whole table, as a string.
pub fn render(root: &Path, binary: &str) -> String {
    let installed = discover(root);
    let mut out = String::new();

    out.push_str(&format!("\nModels (root {})\n\n", root.display()));
    out.push_str(
        "  model                state     layers hidden  pooling      size  notes\n",
    );

    for entry in catalogue() {
        let m = installed.iter().find(|m| m.name == entry.name);
        let state = catalogue_state(root, entry, m);
        let size = match m {
            Some(m) => format!("{:.0} MB", m.mb),
            None => format!("{:.0} MB dl", entry.download_mb),
        };
        out.push_str(&format!(
            "  {:<20} {:<9} {:>6} {:>6} {:>8} {:>9}  {}\n",
            entry.name, state, entry.layers, entry.hidden, entry.pooling, size, entry.note
        ));
    }

    let mut header = false;
    for m in &installed {
        if catalogue().iter().any(|e| e.name == m.name) {
            continue;
        }
        if !header {
            out.push_str("\n  Locally packed (not in the catalogue):\n");
            header = true;
        }
        if !m.error.is_empty() {
            out.push_str(&format!("  {:<20} UNREADABLE: {}\n", m.name, m.error));
            continue;
        }
        // A pose container's state says "pose" and nothing about design
        // sets, because there is nothing to say: every convolution runs on
        // the host, so the file is runnable as it stands. A "no design"
        // here would tell the user to go and export something, which is
        // both false and the only instruction in this table with no
        // command behind it.
        let (layers, hidden) = if is_pose_arch(&m.arch) {
            ("-".to_string(), "-".to_string())
        } else {
            (m.layers.to_string(), m.hidden.to_string())
        };
        // The last column is the command for a non-embedder and the source
        // repo for the rest, because for an embedder the repo is the thing
        // worth checking and for a locally packed detector the repo is
        // "n/a" -- the file came from a local ONNX and there is nothing to
        // name.
        let note = pixel_command_note(&m.kind);
        let note = if !note.is_empty() {
            note
        } else if is_vit_arch(&m.arch) {
            "npuimage classify <name> <image>".to_string()
        } else if is_stt_arch(&m.arch) {
            "npuaudio transcribe <name> <audio.wav>".to_string()
        } else {
            m.repo.clone()
        };
        out.push_str(&format!(
            "  {:<20} {:<9} {:>6} {:>6} {:>8} {:>6.0} MB  {}\n",
            m.name,
            installed_state(root, m),
            layers,
            hidden,
            m.pooling,
            m.mb,
            note
        ));
    }

    out.push_str(LEGEND);
    out.push_str(&format!("  {binary} serve <model>\n\n"));
    out
}

/// Print the catalogue table.
pub fn print_list(root: &Path, binary: &str) -> i32 {
    print!("{}", render(root, binary));
    0
}
