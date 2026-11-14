//! The ViT's classifier, printed as the CLI prints it.
//!
//! The C++ oracle has its own test binary for the image front end
//! (`runtime/tests/test_vit_image.cpp`), and it prints the same lines.
//! This example is the Rust's counterpart, and the two are compared BYTE
//! for byte.
//!
//! Usage:
//!   vit_classify <model.npum> <image.jpg>

use npum_cpu::vit;
use npum_cpu::vit_image;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: vit_classify <model.npum> <image.jpg>");
        std::process::exit(2);
    }
    let model = std::path::Path::new(&args[1]);
    let image = std::path::Path::new(&args[2]);

    let vit = vit::Vit::open(model).expect("the weights load");
    let g = vit.geometry().clone();
    let decoded = vit_image::decode_image(image).expect("the image decodes");
    let patches = vit_image::preprocess(&decoded, &g).expect("the front end runs");
    let hidden = vit.encode(&patches);
    let d = g.hidden as usize;
    let logits = vit.classify(&hidden[..d]);
    let prediction = vit::predict(&logits, vit.labels()).expect("the prediction");

    println!(
        "label {} name {} p {:.9}",
        prediction.label, prediction.name, prediction.p
    );
    println!("logits {}", prediction.logits.len());
}
