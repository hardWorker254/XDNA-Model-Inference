//! The pose family on the array, against the pose family on the host.
//!
//! # WHAT THIS CLOSES
//!
//! The convolution check beside this one feeds the array a panel and a
//! made-up operand. This one feeds it a FAMILY: the pose network, out of a
//! container that ships the array's own panels, over the real image. It is
//! the check that says the array's path through a graph is the graph -- and
//! that the family's seam (the `ArrayConvs` trait) is wired to the board and
//! not to a stub.
//!
//! # WHAT IT IS NOT
//!
//! It is NOT a parity check against the C++ runtime: that is what the
//! golden transcripts are for, and this crate cannot see them. What it
//! checks is the property the flag exists to make true -- **the array's
//! numbers are the host's numbers, to the width of bf16** -- and it checks
//! the other property too: the run really dispatched, so a flag that did
//! nothing could not pass by being compared against itself.
//!
//! # THE DIFFERENCE IS NOT ZERO, AND IT IS NOT A BUG
//!
//! The array multiplies the weights rounded to bf16 and the host multiplies
//! them in f32, so the two runs are the same network with a coarser weight.
//! The bounds below are MEASURED, not guessed: they are what this model on
//! this machine gives, and they are the numbers a bf16 accumulation over
//! this network's depth can produce.
//!
//! # IT SKIPS, AND SAYS SO, WHEN THERE IS NO BOARD OR NO CONTAINER
//!
//! The array path needs two things a checkout does not have: an NPU, and a
//! container packed with `--npu` (the pre-tiled bf16 panels) that carries a
//! design set. Point `NPUEMBEDDINGS_POSE_NPU` at such a container, and the
//! test runs; without it, the test says it skipped.

use std::path::PathBuf;

use npum_cpu::pose::Network;

/// Whether the board is there at all.
fn board_present() -> bool {
    std::path::Path::new("/dev/accel/accel0").exists()
        || std::path::Path::new("/dev/dri/accel0").exists()
}

/// The container packed with the array's panels.
///
/// The environment variable first -- the reader saying where their own
/// container is -- and then the conventional place on this machine.
fn container() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NPUEMBEDDINGS_POSE_NPU") {
        let p = PathBuf::from(p);
        return if p.is_file() { Some(p) } else { None };
    }
    let home = std::env::var("HOME").ok()?;
    let p = PathBuf::from(home).join("xdna-fixtures/yolov8n-pose.npue");
    if p.is_file() { Some(p) } else { None }
}

/// The image the front end reads.
fn image() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/parity/fixtures/bus.jpg")
}

/// The measured width of the difference between the two runs.
///
/// The class scores are the sharpest of the three: they are what the
/// threshold reads, and a score that moved by more than this would change
/// which people the run found.
const SCORE_TOLERANCE: f32 = 0.05;

/// The measured width of the difference in the boxes, in pixels.
const BOX_TOLERANCE: f32 = 20.0;

/// The measured width of the difference in the keypoints, in pixels.
const KEYPOINT_TOLERANCE: f32 = 30.0;

#[test]
fn the_array_path_is_the_host_path_with_coarser_weights() {
    if !board_present() {
        eprintln!("skipping: no NPU device node");
        return;
    }
    let Some(path) = container() else {
        eprintln!(
            "skipping: no container packed with --npu. Pack one with the packer's --npu \
             and the design set embedded, and point NPUEMBEDDINGS_POSE_NPU at it."
        );
        return;
    };
    let image = image();
    if !image.is_file() {
        eprintln!("skipping: {} is not there", image.display());
        return;
    }

    let network = match Network::open(&path) {
        Ok(n) => n,
        Err(e) => panic!("{}: {e}", path.display()),
    };
    let geometry = network.geometry().clone();

    // THE INPUT, once: the front end is the host's either way, and two runs
    // on two front ends would compare the front ends.
    let decoded = match npum_cpu::vit_image::decode_image(&image) {
        Ok(d) => d,
        Err(e) => panic!("{}: {e}", image.display()),
    };
    let (input, _letterbox) = match npum_cpu::pose_decode::letterbox_normalise(&decoded, &geometry)
    {
        Ok(v) => v,
        Err(e) => panic!("{}: {e}", image.display()),
    };

    // THE HOST.
    let (host, host_timings) = network.run_timed(&input).expect("the host run failed");
    assert_eq!(
        host_timings.convs_array, 0,
        "the host run dispatched to the array, which it must not"
    );

    // THE ARRAY.
    let specs = npum_cpu::pose::array_specs(&geometry);
    assert!(
        !specs.is_empty(),
        "{}: the container names no stream for any convolution, so it was packed \
         without --npu and the array path has nothing to dispatch",
        path.display()
    );
    let transport = npum_xdna::transport::open_selected(0).expect("the board did not open");
    let design = npum_xdna::array::ArrayDesign::open_from_container(&path, "gemm_rtp", transport)
        .expect("the container carries no design set");
    let mut backend =
        npum_xdna::array_convs::ArrayConvs::open(&path, design, &specs).expect("the backend");
    let (array, array_timings) = network
        .run_timed_on_array(&input, Some(&mut backend))
        .expect("the array run failed");

    // IT REALLY DISPATCHED.
    //
    // A run that asked for the array and dispatched nothing would agree with
    // the host exactly, and the comparison below would pass on a lie.
    assert!(
        array_timings.dispatches > 0,
        "the array run dispatched nothing, so the flag did not do what it said"
    );
    assert!(
        array_timings.convs_array > 0,
        "the array run counted no convolution on the array"
    );

    // THE TWO ANSWERS.
    //
    // The head is [4 + nc + nk*3, cells]: the box, the class score and the
    // keypoints, in that order.
    let cells = host.d.len() / host.c as usize;
    let nc = geometry.num_classes as usize;
    let channels = host.c as usize;
    let row = |t: &npum_cpu::pose::Tensor, ch: usize, i: usize| t.d[ch * cells + i];

    let mut worst_score = 0.0f32;
    let mut worst_box = 0.0f32;
    let mut worst_keypoint = 0.0f32;
    for i in 0..cells {
        for ch in 0..channels {
            let (h, a) = (row(&host, ch, i), row(&array, ch, i));
            let d = (h - a).abs();
            if ch < 4 {
                worst_box = worst_box.max(d);
            } else if ch < 4 + nc {
                worst_score = worst_score.max(d);
            } else {
                worst_keypoint = worst_keypoint.max(d);
            }
        }
    }
    assert!(
        worst_score <= SCORE_TOLERANCE,
        "the class scores moved by {worst_score}, and the bound is {SCORE_TOLERANCE}. The \
         array's weights are rounded to bf16 and the host's are not, so the two runs differ \
         -- but not by more than this."
    );
    assert!(
        worst_box <= BOX_TOLERANCE,
        "a box coordinate moved by {worst_box} pixels, and the bound is {BOX_TOLERANCE}"
    );
    assert!(
        worst_keypoint <= KEYPOINT_TOLERANCE,
        "a keypoint moved by {worst_keypoint} pixels, and the bound is {KEYPOINT_TOLERANCE}"
    );

    eprintln!(
        "  pose array  {} dispatches, class score {worst_score:.2e}, box {worst_box:.2} px, \
         keypoint {worst_keypoint:.2} px",
        array_timings.dispatches
    );
}
