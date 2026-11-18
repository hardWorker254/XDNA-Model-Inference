//! The pose detector, printed as the CLI prints it.
//!
//! The C++ oracle has its own test binary for this
//! (`runtime/tests/test_pose.py` and the `npuimage pose` transcript), and
//! this example is the Rust's counterpart. The two are compared BYTE for
//! byte on the JSON object.
//!
//! Usage:
//!   pose_detect <model.npum> <image.jpg>

use npum_cpu::pose;
use npum_cpu::pose_decode;
use npum_cpu::vit_image;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: pose_detect <model.npum> <image.jpg>");
        std::process::exit(2);
    }
    let model = std::path::Path::new(&args[1]);
    let image = std::path::Path::new(&args[2]);

    let network = pose::Network::open(model).expect("the network's weights load");
    let geometry = network.geometry().clone();
    let params = pose_decode::DecodeParams::default();

    let all = std::time::Instant::now();
    let decoded = vit_image::decode_image(image).expect("the image decodes");
    let front = std::time::Instant::now();
    let (input, letterbox) =
        pose_decode::letterbox_normalise(&decoded, &geometry).expect("the front end runs");
    let front_end_s = front.elapsed().as_secs_f64();

    let net = std::time::Instant::now();
    let head = network.run(&input).expect("the network runs");
    let network_s = net.elapsed().as_secs_f64();

    let dec = std::time::Instant::now();
    let people = pose_decode::decode(&head, &geometry, &letterbox, &params).expect("the decode");
    let decode_s = dec.elapsed().as_secs_f64();

    let result = pose_decode::Detection {
        people,
        front_end_s,
        network_s,
        decode_s,
        total_s: all.elapsed().as_secs_f64(),
        width: letterbox.src_w,
        height: letterbox.src_h,
        scale: letterbox.scale,
        pad_x: letterbox.pad_x,
        pad_y: letterbox.pad_y,
        dispatches: 0,
        convs_host: 72,
    };
    print!(
        "{}",
        pose_decode::result_json(
            &result,
            &geometry,
            &params,
            &image.display().to_string(),
            false
        )
    );
}
