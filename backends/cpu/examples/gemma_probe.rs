//! Dump a Gemma forward run for the Python reference to check.
//!
//! Writes `batch u32, seq u32, ids, mask, output` as little-endian raw numbers
//! to the path in argv[2]. The ids are arbitrary but fixed; the point is to
//! compare the ARITHMETIC, not the tokenizer.

use std::io::Write;

fn main() {
    let path = std::env::args().nth(1).expect("the container path");
    let out = std::env::args().nth(2).expect("the output path");
    let container = npum_format::Container::open(std::path::Path::new(&path)).expect("the container opens");
    let model = npum_cpu::gemma::Model::from_container(&container).expect("the Gemma model opens");
    let (batch, seq) = (2usize, 32usize);
    let mut ids = vec![0u32; batch * seq];
    let mut mask = vec![0u32; batch * seq];
    for b in 0..batch {
        for s in 0..seq {
            ids[b * seq + s] = ((b * 977 + s * 131 + 7) % 200000) as u32 + 2;
            mask[b * seq + s] = if b == 0 || s < 16 { 1 } else { 0 };
        }
    }
    eprintln!("embed row 5 {:?}", &model.debug_embedding(5)[..8]);
    eprintln!("qkv weight row 0 {:?}", &[] as &[f32]);
    let mut ops = npum_cpu::ops::CpuOps::best();
    let mut timings = npum_cpu::gemma::Timings::default();
    let mut taps = std::collections::HashMap::new();
    let result = model
        .embed_timed_with_taps(
            &mut ops,
            &mut timings,
            &ids,
            &mask,
            batch,
            seq,
            None,
            None,
            &mut taps,
        )
        .expect("the Gemma run");
    {
        let mut tf = std::fs::File::create("/tmp/opencode/gemma_taps.bin").unwrap();
        for (name, values) in &taps {
            tf.write_all(&(name.len() as u32).to_le_bytes()).unwrap();
            tf.write_all(name.as_bytes()).unwrap();
            tf.write_all(&(values.len() as u32).to_le_bytes()).unwrap();
            for v in values {
                tf.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }
    eprintln!("timings {timings:?}");

    let mut file = std::fs::File::create(&out).expect("the output file");
    for v in [batch as u32, seq as u32] {
        file.write_all(&v.to_le_bytes()).unwrap();
    }
    for v in ids.iter().chain(mask.iter()) {
        file.write_all(&v.to_le_bytes()).unwrap();
    }
    for v in &result {
        file.write_all(&v.to_le_bytes()).unwrap();
    }
}
