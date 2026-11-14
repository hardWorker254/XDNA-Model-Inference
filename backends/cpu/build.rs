//! The build script: the image front end's JPEG shim.
//!
//! # WHY THERE IS A C FILE IN A RUST BACKEND
//!
//! The classifier's answer is a probability printed to nine significant
//! digits, and the golden transcript holds those digits. That number is the
//! end of a chain that starts with PIXELS: a different IDCT gives
//! different pixels, a different resize gives different pixels, and a
//! different pixel is a different probability.
//!
//! So the JPEG decode calls the SAME libjpeg the C++ runtime calls, through
//! a shim that keeps libjpeg's struct layout on the C side where libjpeg's
//! own headers define it. The Rust side sees three integers and a byte
//! buffer.
//!
//! # WHAT THE SHIM DOES NOT CHANGE
//!
//! libjpeg's defaults are libjpeg's defaults: the "fancy" upsampling, the
//! slow integer IDCT, the YCbCr -> RGB transform libjpeg ships. The one
//! thing the C++ runtime sets is `out_color_space = JCS_RGB`, so a
//! grayscale JPEG becomes three channels, and the shim sets the same one.

use std::process::Command;

/// The C compiler to use: `CC` if the environment names one, else `cc`.
fn compiler() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

/// The flags the shim is compiled with.
///
/// The optimisation level does not change the OUTPUT: libjpeg does the
/// arithmetic, and this file only moves bytes. `-O2` is here so the copy
/// out of libjpeg's row buffer is not the slowest part of the decode.
const FLAGS: [&str; 3] = ["-O2", "-fPIC", "-std=c11"];

fn main() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo");
    let cc = compiler();

    // The two shims, and the libraries they need.
    //
    // JPEG is the classifier's front end and PNG is the OCR family's, and the
    // reason both are C is the same: the answer is the end of a chain that
    // starts with PIXELS, so the decode has to be the SAME bytes and not a
    // close enough copy.
    let shims: [(&str, &str, &str); 2] = [
        (
            "jpeg_shim.c",
            "npum_jpeg_shim",
            "The image front end decodes JPEG through libjpeg, the same library \
             the C++ runtime uses, because the classifier's probability is the \
             end of a chain that starts with pixels and a different IDCT gives \
             a different answer.\n\
             Install libjpeg's development files (libjpeg-dev, libjpeg-devel) \
             or set CC to a compiler that can find jpeglib.h.",
        ),
        (
            "png_shim.c",
            "npum_png_shim",
            "The OCR family's front end decodes PNG through libpng, the same \
             library the C++ runtime uses, because the transcription is the end \
             of a chain that starts with pixels and a different inflate gives \
             a different character.\n\
             Install libpng's development files (libpng-dev, libpng-devel) \
             or set CC to a compiler that can find png.h.",
        ),
    ];

    for (file, archive_name, hint) in shims {
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join(file);
        let object = std::path::Path::new(&out_dir).join(format!("{archive_name}.o"));

        let mut command = Command::new(&cc);
        command.args(FLAGS).arg("-c").arg(&source).arg("-o").arg(&object);
        // The header is in the default include path on every system that has
        // the library's development files; a caller who keeps them elsewhere
        // can point at them with CFLAGS, which cargo passes through.
        if let Ok(extra) = std::env::var("CFLAGS") {
            for flag in extra.split_whitespace() {
                command.arg(flag);
            }
        }
        let status = command
            .status()
            .unwrap_or_else(|e| panic!("{cc} could not be run: {e}"));
        if !status.success() {
            panic!("{cc} failed to compile {archive_name}.\n{hint}");
        }

        // The object goes into an archive, because the linker wants a LIBRARY
        // and an object file is not one.
        let archive = std::path::Path::new(&out_dir).join(format!("lib{archive_name}.a"));
        let _ = std::fs::remove_file(&archive);
        let ar_status = Command::new("ar")
            .arg("rcs")
            .arg(&archive)
            .arg(&object)
            .status()
            .unwrap_or_else(|e| panic!("ar could not be run: {e}"));
        if !ar_status.success() {
            panic!("ar failed to archive {archive_name}");
        }

        println!("cargo:rustc-link-search=native={out_dir}");
        println!("cargo:rustc-link-lib=static={archive_name}");
        println!("cargo:rerun-if-changed=src/{file}");
    }

    println!("cargo:rustc-link-lib=jpeg");
    println!("cargo:rustc-link-lib=png");
}
