//! The pose detector's per-node dump, in the C++'s format.
//!
//! `npuimage pose ... --pose-dump FILE` writes every graph node's output as
//! raw fp32 NCHW behind a JSON header. This example writes the same file,
//! so `tools/verify/diff_pose_dump.py` -- and any reader of the format --
//! can read either side.
//!
//! Usage:
//!   pose_dump <model.npum> <image.jpg> <out.bin>

use npum_cpu::pose;
use npum_cpu::pose_decode;
use npum_cpu::vit_image;
use std::io::{Seek, SeekFrom, Write};

const HEADER: usize = 1 << 16;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: pose_dump <model.npum> <image.jpg> <out.bin>");
        std::process::exit(2);
    }
    let model = std::path::Path::new(&args[1]);
    let image = std::path::Path::new(&args[2]);
    let out_path = std::path::Path::new(&args[3]);

    let network = pose::Network::open(model).expect("the network's weights load");
    let geometry = network.geometry().clone();
    let decoded = vit_image::decode_image(image).expect("the image decodes");
    let (input, _letterbox) =
        pose_decode::letterbox_normalise(&decoded, &geometry).expect("the front end runs");

    let mut file = std::fs::File::create(out_path).expect("the dump opens");
    file.write_all(&vec![b' '; HEADER])
        .expect("the header area is reserved");
    let mut index: Vec<String> = Vec::new();
    {
        let mut hook = |i: usize, t: &pose::Tensor| {
            let off = file.stream_position().expect("the offset");
            index.push(format!(
                "    {{\"image\": 0, \"i\": {}, \"c\": {}, \"h\": {}, \"w\": {}, \
                 \"n\": {}, \"off\": {}}}",
                if i == usize::MAX { -1i64 } else { i as i64 },
                t.c,
                t.h,
                t.w,
                t.d.len(),
                off
            ));
            let bytes: Vec<u8> = t.d.iter().flat_map(|v| v.to_le_bytes()).collect();
            file.write_all(&bytes).expect("the tensor is written");
        };
        network
            .run_with_hook(&input, &mut hook)
            .expect("the network runs");
    }
    let mut header = String::from("{\n  \"nodes\": [\n");
    for (i, line) in index.iter().enumerate() {
        if i > 0 {
            header.push_str(",\n");
        }
        header.push_str(line);
    }
    header.push_str("\n  ]\n}\n");
    assert!(header.len() <= HEADER, "the header needs more than 64 KiB");
    header.push_str(&" ".repeat(HEADER - header.len()));
    file.seek(SeekFrom::Start(0))
        .expect("the header is rewritten");
    file.write_all(header.as_bytes())
        .expect("the header is written");
    eprintln!("{} node tensors written", index.len());
}
