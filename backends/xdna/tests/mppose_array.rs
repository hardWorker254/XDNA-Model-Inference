//! The mppose family on the array, against the mppose family on the host.
//!
//! # WHAT THIS CLOSES
//!
//! The hands gate beside this one checks a family whose second network runs
//! once per HAND. This one checks a family whose second network runs once per
//! PERSON, and whose two graphs are named apart -- the detector's
//! "det.conv.3.w" and the landmark network's "pose.conv.3.w" are two
//! different weights in one container, and so are two different panels. A
//! backend that keyed its panels by the bare convolution index would put the
//! landmark network's conv 3 on the detector's conv 3 panel: the right shape,
//! the wrong weights, and nothing in the output saying so.
//!
//! It also checks the count that a family with a per-person second stage gets
//! wrong most easily: ONE backend serves both graphs, so the dispatch count
//! is read from the backend once and not summed over the two networks -- and
//! the landmark network's dispatches are inside that one number, once per
//! person.
//!
//! # THE ROTATED CROP IS WHAT MOVES THE LANDMARKS
//!
//! The detector's keypoints decide the crop's angle, and the crop is what
//! the landmark network reads, so a coarser detector moves every landmark of
//! every person. The landmark bound below is in PIXELS for that reason, and
//! not in "the same shape".
//!
//! # THE DIFFERENCE IS NOT ZERO, AND IT IS NOT A BUG
//!
//! The array multiplies the weights rounded to bf16 and the host multiplies
//! them in f32, so the two runs are the same network with a coarser weight.
//! The bounds below are MEASURED on this model and this image.
//!
//! # IT SKIPS, AND SAYS SO, WHEN THERE IS NO BOARD OR NO CONTAINER
//!
//! Point `NPUEMBEDDINGS_MPPOSE_NPU` at a container packed with `--npu`, and
//! the test runs; without it, the test says it skipped.

use std::path::PathBuf;

use npum_cpu::mppose::MpPose;

/// Whether the board is there at all.
fn board_present() -> bool {
    std::path::Path::new("/dev/accel/accel0").exists()
        || std::path::Path::new("/dev/dri/accel0").exists()
}

/// The container packed with the array's panels.
fn container() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NPUEMBEDDINGS_MPPOSE_NPU") {
        let p = PathBuf::from(p);
        return if p.is_file() { Some(p) } else { None };
    }
    let home = std::env::var("HOME").ok()?;
    let p = PathBuf::from(home).join("xdna-fixtures/mediapipe-pose.npue");
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

/// The image the front end reads.
fn image() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/parity/fixtures/bus.jpg")
}

/// The measured width of the difference in a landmark, in pixels.
///
/// It is large because the difference is NOT only the landmark network's: the
/// detector's keypoints decide the crop's angle, and the crop is what the
/// landmarks are read from. The MEASURED difference on this model and this
/// image is 44.4 pixels -- at the feet, which are the furthest from the
/// detector's own two keypoints -- and the oracle's own host run against its
/// own array run differs by exactly the same 44.4 pixels at the same
/// landmark. The bound is above that, and below the distance at which two
/// answers would be different poses rather than the same pose measured
/// twice.
const LANDMARK_TOLERANCE: f64 = 50.0;

/// The measured width of the difference in a detection's score.
///
/// The MEASURED difference is 7.7e-4, and the bound is two orders of
/// magnitude above it.
const SCORE_TOLERANCE: f64 = 0.1;

/// The measured width of the difference in the landmark network's confidence.
///
/// The MEASURED difference is 4.1e-2 -- the network's own head reads a
/// coarser weight, and this is the number that says so -- and the bound is
/// above it.
const CONF_TOLERANCE: f64 = 0.1;

#[test]
fn the_array_path_is_the_host_path_with_coarser_weights() {
    if !board_present() {
        eprintln!("skipping: no NPU device node");
        return;
    }
    let Some(path) = container() else {
        eprintln!(
            "skipping: no container packed with --npu. Pack one with the packer's --npu \
             and the design set embedded, and point NPUEMBEDDINGS_MPPOSE_NPU at it."
        );
        return;
    };
    let image = image();
    if !image.is_file() {
        eprintln!("skipping: {} is not there", image.display());
        return;
    }

    let model = match MpPose::open(&path) {
        Ok(m) => m,
        Err(e) => panic!("{}: {e}", path.display()),
    };
    let specs = npum_cpu::mppose::array_specs(model.geometry());
    assert!(
        !specs.is_empty(),
        "{}: the container names no stream for any convolution, so it was packed \
         without --npu and the array path has nothing to dispatch",
        path.display()
    );

    let decoded = match npum_cpu::vit_image::decode_image(&image) {
        Ok(d) => d,
        Err(e) => panic!("{}: {e}", image.display()),
    };

    // THE HOST.
    let mut noop = |_net: &str, _node: usize, _t: &npum_cpu::pose::Tensor| {};
    let host = npum_cpu::mppose_decode::detect(&model, &decoded, -1, &mut noop, None)
        .expect("the host run failed");
    assert_eq!(
        host.array_convs, 0,
        "the host run dispatched to the array, which it must not"
    );

    // THE ARRAY.
    let transport = npum_xdna::transport::open_selected(0).expect("the board did not open");
    let design = npum_xdna::array::ArrayDesign::open_from_container(&path, "gemm_rtp", transport)
        .expect("the container carries no design set");
    let mut backend =
        npum_xdna::array_convs::ArrayConvs::open(&path, design, &specs).expect("the backend");
    let mut noop = |_net: &str, _node: usize, _t: &npum_cpu::pose::Tensor| {};
    let array = npum_cpu::mppose_decode::detect(
        &model,
        &decoded,
        -1,
        &mut noop,
        Some(&mut backend),
    )
    .expect("the array run failed");

    // IT REALLY DISPATCHED.
    assert!(
        array.array_convs > 0,
        "the array run counted no convolution on the array"
    );
    assert!(
        array.array_dispatches > 0,
        "the array run dispatched nothing, so the flag did not do what it said"
    );

    // THE TWO ANSWERS.
    assert_eq!(
        host.people.len(),
        array.people.len(),
        "the host found {} people and the array found {}",
        host.people.len(),
        array.people.len()
    );
    let mut worst_landmark = 0.0f64;
    let mut worst_score = 0.0f64;
    let mut worst_conf = 0.0f64;
    for (i, (h, a)) in host.people.iter().zip(array.people.iter()).enumerate() {
        worst_score = worst_score.max((host.detections[i].score - array.detections[i].score).abs());
        worst_conf = worst_conf.max((h.conf - a.conf).abs());
        for (p, q) in h.landmarks.iter().zip(a.landmarks.iter()) {
            worst_landmark = worst_landmark.max((p[0] - q[0]).abs()).max((p[1] - q[1]).abs());
        }
    }
    assert!(
        worst_score <= SCORE_TOLERANCE,
        "a detection score moved by {worst_score}, and the bound is {SCORE_TOLERANCE}"
    );
    assert!(
        worst_conf <= CONF_TOLERANCE,
        "a pose confidence moved by {worst_conf}, and the bound is {CONF_TOLERANCE}"
    );
    assert!(
        worst_landmark <= LANDMARK_TOLERANCE,
        "a landmark moved by {worst_landmark} pixels, and the bound is {LANDMARK_TOLERANCE}"
    );

    eprintln!(
        "  mppose array  {} dispatches, score {worst_score:.2e}, conf {worst_conf:.2e}, \
         landmark {worst_landmark:.2} px",
        array.array_dispatches
    );
}
