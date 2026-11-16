//! The ocr family's status block, its per-image lines and its node dump.
//!
//! # The block is the C++'s, row for row
//!
//! `npuimage ocr` printed a block before every run and the golden transcript
//! holds it, so the block is a contract and not a greeting.
//!
//! # Two stages, two rows, and the split is the point
//!
//! The text detector and the text recogniser are reported SEPARATELY,
//! including how many of each one's convolutions are depthwise, because that
//! split is the interesting number on this architecture: 15 of the detector's
//! 74 convolutions and 14 of the recogniser's 38 are depthwise, and no array
//! can take those.
//!
//! # The transposed convolutions get their own row
//!
//! The DB head's two upsampling convolutions are not dense convolutions that
//! happen to be slow: they are not convolutions the array can run at all --
//! a transposed convolution is a sum of SHIFTED contributions and has no
//! `[M, N]` GEMM in it to dispatch -- and the row says so.
//!
//! # The dump is the C++'s format, byte for byte
//!
//! 1 MiB of header, then every node's fp32 in graph order, for BOTH graphs,
//! with the header rewritten once the node count is known. The reader looks
//! for `"i": -1` as the network's INPUT, which is the photograph after the
//! front end: a front-end difference shows up there as a difference in a
//! million pixels rather than as an amplified one ninety nodes later.

use npum_cpu::ocr::Geometry;
use npum_cpu::ocr::Tensor;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

/// The 1 MiB the dump reserves for its header.
///
/// 1 MiB and not 64 KiB: the detector has 154 nodes and the recogniser 334,
/// and the recogniser runs ONCE PER BOX -- so a page of text can need a
/// hundred thousand records before the header is written.
pub const DUMP_HEADER: usize = 1 << 20;

/// The block, printed to stderr before the first image.
pub fn print_block(name: &str, g: &Geometry, setup_s: f64, array: Option<&dyn npum_model::array::ArrayConvs>) {
    eprintln!("NpuEmbeddings -- ocr (PaddleOCR text det + rec) on the CPU");
    eprintln!(
        "  model      {}: detector {}px, recogniser {}x{}, {} classes over {} characters",
        name,
        g.det_input_size,
        g.rec_input_height,
        g.rec_input_width,
        g.rec_num_classes,
        g.rec_dict_size
    );
    eprintln!(
        "  thresholds thresh {:.3}, box {:.3}, unclip {:.1}, max_candidates {}, min_size {} \
         -- from the container, not a flag",
        g.det_thresh, g.det_box_thresh, g.det_unclip_ratio, g.det_max_candidates, g.det_min_size
    );
    eprintln!(
        "  normalise  detector (x/255 - {:.3})/{:.3}, pad {} in the RASTER (black); \
         recogniser (x/255 - {:.3})/{:.3}, pad {} in the NORMALISED space",
        g.det_mean.first().copied().unwrap_or(0.0),
        g.det_std.first().copied().unwrap_or(1.0),
        g.det_pad_value,
        g.rec_mean.first().copied().unwrap_or(0.0),
        g.rec_std.first().copied().unwrap_or(1.0),
        g.rec_pad_value
    );
    eprintln!("  ops        (npu = dispatched, host = this process)");
    // THE CONVOLUTIONS' PLACEMENT, AND IT IS NOT ALWAYS THE HOST'S.
    //
    // The line says where the convolutions ran, and a run that dispatched
    // them to the array and printed "host" would be the one thing the flag
    // exists to prevent.
    let convs_place = if array.is_some() { "npu" } else { "host" };
    eprintln!(
        "             {:<26} {:<5} {}",
        format!("{} x conv + {} x dwconv", g.det_dense(), g.det_depthwise()),
        convs_place,
        "text detector, once per frame"
    );
    eprintln!(
        "             {:<26} {:<5} {}",
        format!("{} x conv + {} x dwconv", g.rec_dense(), g.rec_depthwise()),
        convs_place,
        "text recogniser, once per box"
    );
    eprintln!(
        "             {:<26} {:<5} {}",
        format!("{} x convtranspose", g.det_transposed() + g.rec_transposed()),
        "host",
        "the DB head's two upsampling convolutions: a sum of SHIFTED contributions, not a GEMM"
    );
    eprintln!(
        "             {:<26} {:<5} {}",
        "db, ctc", "host", "host-side by construction: a list of boxes and a string"
    );
    eprintln!(
        "  setup      {:.3} s (both graphs read, weights transposed once)",
        setup_s
    );
    if let Some(array) = array {
        eprintln!(
            "  array      {:.1} MB of bf16 panels staged, once, at load",
            array.staged_bytes() as f64 / (1024.0 * 1024.0)
        );
    }
}

/// The `--ocr-dump` line, once the path is known.
pub fn print_dump_header(path: &Path, g: &Geometry) {
    eprintln!(
        "  dump       {} + {} graph nodes per image -> {} (fp32, JSON header in the first MiB)",
        g.det_graph.len(),
        g.rec_graph.len(),
        path.display()
    );
}

/// One image's line, on stderr.
pub fn print_run(image: &str, r: &npum_cpu::ocr_decode::OcrResult) {
    eprintln!(
        "  {:<40} {}x{}  {} box{} -> {} read  {:.1} ms (front {:.1} / det {:.1} / db {:.1} / \
         crop {:.1} / rec {:.1})",
        image,
        r.width,
        r.height,
        r.boxes_found,
        if r.boxes_found == 1 { "" } else { "es" },
        r.boxes_read,
        r.total_s * 1e3,
        r.front_end_s * 1e3,
        r.det_s * 1e3,
        r.post_s * 1e3,
        r.crop_s * 1e3,
        r.rec_s * 1e3
    );
}

/// The measured split, on stderr, after the image's line.
///
/// THE SAME BLOCK FOR BOTH PATHS, because the comparison that matters is the
/// host's dense convolutions against the device's, and two blocks with
/// different shapes cannot be compared by eye.
pub fn print_split(
    det: &npum_cpu::ocr::Cost,
    rec: &npum_cpu::ocr::Cost,
    det_dw: i64,
    det_tr: i64,
    rec_dw: i64,
    rec_tr: i64,
    array: Option<&dyn npum_model::array::ArrayConvs>,
) {
    let line = |who: &str, c: &npum_cpu::ocr::Cost, dw: i64, tr: i64| {
        eprintln!(
            "             split  {}: {} convs on the host ({:.1} ms: {:.1} ms of GEMM, {:.1} ms \
             of transpose back), {} dwconv + {} convtranspose on the host ({:.1} ms), im2col \
             {:.1} ms",
            who,
            c.convs_host,
            c.t_host * 1e3,
            c.t_gemm * 1e3,
            c.t_transpose * 1e3,
            dw,
            tr,
            (c.t_dwconv + c.t_transposed) * 1e3,
            c.t_im2col * 1e3
        );
    };
    line("det", det, det_dw, det_tr);
    line("rec", rec, rec_dw, rec_tr);
    if let Some(array) = array {
        eprintln!(
            "             array  {} convs on the array ({:.1} ms, {} dispatches, {:.1} ms of it \
             converting to bf16)",
            det.convs_array + rec.convs_array,
            (det.t_array + rec.t_array) * 1e3,
            array.dispatches(),
            array.convert_seconds() * 1e3
        );
    }
}

/// The node-dump writer, installed for the lifetime of the process.
pub struct Dump {
    file: std::fs::File,
    index: Vec<String>,
    /// Which image the records belong to, counting from zero.
    pub image: usize,
}

impl Dump {
    /// Open the dump file and reserve the header.
    pub fn create(path: &Path) -> Result<Dump, String> {
        let mut file = std::fs::File::create(path).map_err(|_| {
            format!(
                "--ocr-dump {}: cannot open it for writing",
                path.display()
            )
        })?;
        file.write_all(&vec![b' '; DUMP_HEADER])
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Dump {
            file,
            index: Vec::new(),
            image: 0,
        })
    }

    /// The hook both networks call: one node's output, in graph order.
    pub fn record(&mut self, net: &str, node: i64, t: &Tensor) {
        let off = self.file.stream_position().unwrap_or(0);
        let shape = t
            .shape
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        self.index.push(format!(
            "    {{\"image\": {}, \"net\": \"{}\", \"i\": {}, \"rank\": {}, \"shape\": [{}], \
             \"n\": {}, \"off\": {}}}",
            self.image,
            net,
            node,
            t.rank(),
            shape,
            t.d.len(),
            off
        ));
        let mut bytes = Vec::with_capacity(t.d.len() * 4);
        for v in &t.d {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let _ = self.file.write_all(&bytes);
    }

    /// Rewrite the reserved header and close.
    ///
    /// A header that does not fit is a REFUSAL, not a truncation: a short
    /// header reads as a short graph, and a graph that quietly lost its last
    /// sixty nodes is the failure this dump exists to catch.
    pub fn finish(mut self, path: &Path) -> Result<usize, String> {
        let mut header = String::from("{\n  \"nodes\": [\n");
        for (i, line) in self.index.iter().enumerate() {
            if i > 0 {
                header.push_str(",\n");
            }
            header.push_str(line);
        }
        header.push_str("\n  ]\n}\n");
        if header.len() > DUMP_HEADER {
            return Err(format!(
                "--ocr-dump {}: the header needs {} bytes for {} node records and the reserved \
                 area is {}. Raise DUMP_HEADER in cli/src/ocr.rs; do NOT let it overflow, \
                 because the reader would then take tensor bytes for header text.",
                path.display(),
                header.len(),
                self.index.len(),
                DUMP_HEADER
            ));
        }
        header.push_str(&" ".repeat(DUMP_HEADER - header.len()));
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        self.file
            .write_all(header.as_bytes())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(self.index.len())
    }
}

/// Whether the run asked for the convolutions to go on the array.
///
/// The flag is a SET of codes, and this family has exactly one of them:
/// `conv`. Anything else is refused by the family's own check, which knows
/// what this architecture has.
pub fn wants_conv(args: &crate::flags::Args) -> bool {
    args.value("--npu-ops")
        .map(|v| v.split(',').any(|c| c.trim() == "conv"))
        .unwrap_or(false)
}

/// Open a container and hand back the session, for the CLI's block.
pub fn open(path: &Path) -> Result<npum_cpu::ocr_session::Ocr, String> {
    npum_cpu::ocr_session::Ocr::open(path, None)
}

/// Open the array's backend for this family, when the run asked for it.
///
/// The family's OWN list of the dense convolutions it runs, and then the
/// shared opening of the board: the panels, the design set and the
/// transport are the same for every family that dispatches convolutions.
pub fn open_array(
    path: &Path,
    geometry: &Geometry,
    args: &crate::flags::Args,
    root: &Path,
) -> Result<Box<dyn npum_model::array::ArrayConvs>, String> {
    let specs = npum_cpu::ocr::array_specs(geometry);
    crate::arrayconv::open(path, &specs, args, root)
}
