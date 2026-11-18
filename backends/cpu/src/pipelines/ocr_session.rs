//! The arch=9 session: one detector, then one recogniser per box.
//!
//! # THE TWO STAGES
//!
//! The DETECTOR runs once per frame and returns boxes; the RECOGNISER runs
//! once per box, on a crop of that frame, and returns text.
//!
//! # THE ORDER OF THE BOXES
//!
//! The boxes come out in the detector's own order and are NOT sorted here.
//! PaddleOCR sorts them top to bottom and left to right before reading them,
//! and the sort is a READING convenience rather than part of the model -- so
//! the caller that wants the reading order sorts them.

use crate::ocr::{Cost, Geometry, Network, Tensor, read_geometry};
use crate::ocr_decode::{
    DbBox, OcrResult, Quad, TextBox, ctc_decode, ctc_logits, db_postprocess, det_input, rec_input,
};
use crate::vit_image::Image;
use npum_format::Container;
use npum_model::array::ArrayConvs;
use std::collections::BTreeMap;
use std::path::Path;

/// How many threads the host convolutions are split over.
///
/// The machine's own count, asked rather than assumed: the oracle's OCR
/// takes a `--threads` flag and defaults to sixteen, and the two split the
/// work the same way -- the GEMM over rows, the depthwise convolutions over
/// channels, the im2col over output rows. None of those splits crosses a
/// reduction, so the answer does not depend on how many threads there are.
pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// The OCR model: both graphs, read out of one container.
pub struct Ocr {
    /// The container's geometry.
    geometry: Geometry,
    /// The detector's graph.
    det: Network,
    /// The recogniser's graph.
    rec: Network,
    /// The array's convolutions, when the run asked for them.
    ///
    /// ONE backend for both graphs: the design is one, its slots are one,
    /// and the panels are one set. The two graphs' convolutions are keyed
    /// by their own names, so a detector's conv 0 and a recogniser's conv 0
    /// are two different panels and not one slot's worth of confusion.
    array: Option<Box<dyn ArrayConvs>>,
}

impl Ocr {
    /// Read both graphs out of a container.
    ///
    /// `array` is the array's backend, when the run asked for the array.
    /// It is taken and not opened here: the design set's source is the
    /// caller's business, and a design set loaded from the wrong place is a
    /// wrong answer rather than an error.
    pub fn open(path: &Path, array: Option<Box<dyn ArrayConvs>>) -> Result<Ocr, String> {
        let container = Container::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let label = path.display().to_string();
        let geometry = read_geometry(&container, &label)?;
        let det = Network::new(
            geometry.det_graph.clone(),
            geometry.det_convs.clone(),
            geometry.consts.clone(),
            default_threads(),
        );
        let rec = Network::new(
            geometry.rec_graph.clone(),
            geometry.rec_convs.clone(),
            geometry.consts.clone(),
            default_threads(),
        );
        Ok(Ocr {
            geometry,
            det,
            rec,
            array,
        })
    }

    /// The container's geometry.
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// The array's backend, when the run asked for the array.
    pub fn array(&self) -> Option<&dyn ArrayConvs> {
        self.array.as_deref()
    }

    /// Give the session the array's backend.
    ///
    /// The backend is taken and not opened here: the design set's source is
    /// the caller's business, and a design set loaded from the wrong place
    /// is a wrong answer rather than an error. It is set after the session
    /// is open because the caller's list of the family's convolutions comes
    /// from the geometry, and the geometry is what the session read.
    pub fn set_array(&mut self, array: Option<Box<dyn ArrayConvs>>) {
        self.array = array;
    }

    /// The detector's graph.
    pub fn det(&self) -> &Network {
        &self.det
    }

    /// The recogniser's graph.
    pub fn rec(&self) -> &Network {
        &self.rec
    }

    /// Read one frame.
    ///
    /// `max_boxes` is a CAP on how many boxes reach the recogniser, not a
    /// score threshold: every box is an independent crop of the original
    /// frame, so there is no way to batch them and the only lever is how many
    /// to read. A negative cap means all of them.
    ///
    /// `on_node` is called once per graph node, in graph order, with the
    /// network's name ("det" or "rec"), the node's index (-1 for the network's
    /// input) and the node's output.
    pub fn read(
        &mut self,
        src: &Image,
        max_boxes: i64,
        on_node: &mut dyn FnMut(&str, i64, &Tensor),
    ) -> Result<OcrResult, String> {
        let mut r = OcrResult::default();
        if src.empty() {
            return Err(format!("the image decoded to nothing"));
        }
        r.width = src.width;
        r.height = src.height;
        self.det.reset_cost();
        self.rec.reset_cost();

        // -- stage 1: the detector
        let t0 = std::time::Instant::now();
        let din = det_input(src, &self.geometry)?;
        r.front_end_s = t0.elapsed().as_secs_f64();

        let det_tensor = Tensor {
            shape: vec![
                1,
                3,
                self.geometry.det_input_size,
                self.geometry.det_input_size,
            ],
            d: din.blob,
        };

        let t1 = std::time::Instant::now();
        let mut det_hook = |i: i64, t: &Tensor| on_node("det", i, t);
        let dnodes = self
            .det
            .run(&det_tensor, &mut det_hook, self.array.as_deref_mut())?;
        r.det_s = t1.elapsed().as_secs_f64();

        // THE PROBABILITY MAP. The detector's output is [1, 1, H, W] -- one
        // channel of probabilities over the whole frame -- and the post-process
        // reads it as a map.
        let prob = dnodes.last().ok_or("the detector produced no output")?;
        if prob.rank() != 4 || prob.shape[0] != 1 || prob.shape[1] != 1 {
            return Err(format!(
                "the detector's output is {} dimensions and the first two are {} and {}. The \
                 DB post-process reads a [1, 1, H, W] probability map.",
                prob.rank(),
                prob.shape[0],
                prob.shape.get(1).copied().unwrap_or(0)
            ));
        }
        let ph = prob.shape[2];
        let pw = prob.shape[3];

        let t2 = std::time::Instant::now();
        let found: Vec<DbBox> = db_postprocess(
            &prob.d,
            ph,
            pw,
            self.geometry.det_thresh,
            self.geometry.det_box_thresh,
            self.geometry.det_unclip_ratio,
            self.geometry.det_max_candidates,
            self.geometry.det_min_size,
            din.scale_x,
            din.scale_y,
            src.width as f64,
            src.height as f64,
        );
        r.post_s = t2.elapsed().as_secs_f64();
        r.boxes_found = found.len() as i64;

        // -- stage 2: one recogniser per box
        let n = found.len() as i64;
        let run = if max_boxes < 0 || n < max_boxes {
            n
        } else {
            max_boxes
        };
        r.boxes_read = run;
        for i in 0..run {
            let b = &found[i as usize];
            let t3 = std::time::Instant::now();
            let rin = rec_input(src, &b.q, &self.geometry)?;
            r.crop_s += t3.elapsed().as_secs_f64();

            let rec_tensor = Tensor {
                shape: vec![
                    1,
                    3,
                    self.geometry.rec_input_height,
                    self.geometry.rec_input_width,
                ],
                d: rin.blob,
            };

            let t4 = std::time::Instant::now();
            let mut rec_hook = |j: i64, t: &Tensor| on_node("rec", j, t);
            let rnodes = self
                .rec
                .run(&rec_tensor, &mut rec_hook, self.array.as_deref_mut())?;
            r.rec_s += t4.elapsed().as_secs_f64();

            // THE CTC HEAD'S OUTPUT: [1, frames, classes]. The frames are the
            // recogniser's time steps -- one per eight columns of the crop --
            // and the classes are the dictionary's characters plus the blank
            // and the space.
            let logits = rnodes.last().ok_or("the recogniser produced no output")?;
            let (frames, classes) = ctc_logits(logits)?;
            let cr = ctc_decode(
                &logits.d,
                frames as i64,
                classes as i64,
                &self.geometry.dict,
            )?;

            r.boxes.push(TextBox {
                q: b.q,
                box_score: b.score,
                text: cr.text,
                conf: cr.conf,
            });
        }
        r.total_s = t0.elapsed().as_secs_f64();
        Ok(r)
    }

    /// The detector's cost, after a run.
    pub fn det_cost(&self) -> &Cost {
        &self.det.cost
    }

    /// The recogniser's cost, after a run.
    pub fn rec_cost(&self) -> &Cost {
        &self.rec.cost
    }
}

/// The named tensors the graphs read, by name.
pub type Consts = BTreeMap<String, crate::ocr::Named>;

/// The corners of a box, as the recogniser wants them.
pub fn quad_of(b: &TextBox) -> Quad {
    b.q
}
