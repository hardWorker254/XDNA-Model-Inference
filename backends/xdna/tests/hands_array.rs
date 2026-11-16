//! The hands family on the array, against the hands family on the host.
//!
//! # WHAT THIS CLOSES
//!
//! The pose gate beside this one checks a family with ONE graph. This one
//! checks a family with TWO, and the second one runs once per hand: the
//! palm detector dispatches its dense convolutions, the landmark network
//! dispatches its own, and both share ONE backend -- so the counts must be
//! read from the backend and not summed over the two networks, or every
//! dispatch is counted twice.
//!
//! It also checks the thing the two networks disagree about most: the
//! ROTATED CROP. The palm detector's keypoints decide the crop's angle, and
//! the crop is what the landmark network reads, so a difference in the
//! detector moves every landmark of every hand -- which is why the
//! landmark tolerance is in pixels and not in "the same shape".
//!
//! # THE DIFFERENCE IS NOT ZERO, AND IT IS NOT A BUG
//!
//! The array multiplies the weights rounded to bf16 and the host multiplies
//! them in f32, so the two runs are the same network with a coarser weight.
//! The bounds below are MEASURED on this model and this image.
//!
//! # IT SKIPS, AND SAYS SO, WHEN THERE IS NO BOARD OR NO CONTAINER
//!
//! Point `NPUEMBEDDINGS_HANDS_NPU` at a container packed with `--npu`, and
//! the test runs; without it, the test says it skipped.

use std::path::PathBuf;

use npum_cpu::hands::Hands;

/// Whether the board is there at all.
fn board_present() -> bool {
    std::path::Path::new("/dev/accel/accel0").exists()
        || std::path::Path::new("/dev/dri/accel0").exists()
}

/// The container packed with the array's panels.
fn container() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NPUEMBEDDINGS_HANDS_NPU") {
        let p = PathBuf::from(p);
        return if p.is_file() { Some(p) } else { None };
    }
    let home = std::env::var("HOME").ok()?;
    let p = PathBuf::from(home).join("xdna-fixtures/mediapipe-hands.npue");
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
/// palm detector's keypoints decide the crop's angle, so a coarser detector
/// moves the crop, and the crop is what the landmarks are read from.
const LANDMARK_TOLERANCE: f64 = 25.0;

/// The measured width of the difference in a score.
const SCORE_TOLERANCE: f64 = 0.1;

#[test]
fn the_array_path_is_the_host_path_with_coarser_weights() {
    if !board_present() {
        eprintln!("skipping: no NPU device node");
        return;
    }
    let Some(path) = container() else {
        eprintln!(
            "skipping: no container packed with --npu. Pack one with the packer's --npu \
             and the design set embedded, and point NPUEMBEDDINGS_HANDS_NPU at it."
        );
        return;
    };
    let image = image();
    if !image.is_file() {
        eprintln!("skipping: {} is not there", image.display());
        return;
    }

    let model = match Hands::open(&path) {
        Ok(m) => m,
        Err(e) => panic!("{}: {e}", path.display()),
    };
    let specs = npum_cpu::hands::array_specs(model.geometry());
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
    let host = npum_cpu::hands_decode::detect(&model, &decoded, -1, &mut noop, None)
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
    let array = npum_cpu::hands_decode::detect(
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
        host.hands.len(),
        array.hands.len(),
        "the host found {} hands and the array found {}",
        host.hands.len(),
        array.hands.len()
    );
    let mut worst_landmark = 0.0f64;
    let mut worst_score = 0.0f64;
    for (h, a) in host.hands.iter().zip(array.hands.iter()) {
        worst_score = worst_score.max((h.presence - a.presence).abs());
        for (p, q) in h.landmarks.iter().zip(a.landmarks.iter()) {
            worst_landmark = worst_landmark.max((p[0] - q[0]).abs()).max((p[1] - q[1]).abs());
        }
    }
    assert!(
        worst_score <= SCORE_TOLERANCE,
        "a presence score moved by {worst_score}, and the bound is {SCORE_TOLERANCE}"
    );
    assert!(
        worst_landmark <= LANDMARK_TOLERANCE,
        "a landmark moved by {worst_landmark} pixels, and the bound is {LANDMARK_TOLERANCE}"
    );

    eprintln!(
        "  hands array  {} dispatches, score {worst_score:.2e}, landmark {worst_landmark:.2} px",
        array.array_dispatches
    );
}
