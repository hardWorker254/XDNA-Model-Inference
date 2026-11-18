fn main() {
    match npum_xdna::device::Device::detect() {
        Ok(d) => println!(
            "generation: {:?}  node: {:?}  transport: {}",
            d.generation(),
            d.node(),
            d.transport()
        ),
        Err(e) => println!("error: {e}"),
    }
}
