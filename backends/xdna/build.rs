//! The build of the XRT shim.
//!
//! # Why the shim is built HERE and not by the `cc` crate
//!
//! The `cc` crate is the usual way to compile C, and this build does not use
//! it for one reason: the build has to work OFFLINE. The `cc` crate resolves
//! its own compiler and its own include paths, and on a machine with no
//! registry access that is a build that cannot start. This build calls `g++`
//! directly, finds XRT where XRT is, and if XRT is not there it says so and
//! builds the crate WITHOUT the shim -- which is what a machine that only
//! wants the CPU and Vulkan backends needs.
//!
//! # What is compiled and what is linked
//!
//! One translation unit (`shim/xrt_shim.cpp`) becomes one static archive, and
//! the archive plus XRT's own library are what the crate links against. XRT
//! is a shared library here, and the loader already knows where it lives
//! (`/opt/xilinx/xrt/lib` is in the loader's cache), so the rpath is a
//! convenience rather than a requirement.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=XILINX_XRT");
    println!("cargo:rerun-if-env-changed=XRT_ROOT");
    println!("cargo:rerun-if-changed=shim/xrt_shim.cpp");
    println!("cargo:rerun-if-changed=shim/xrt_shim.h");
    // The cfg the Rust side asks about, declared so that a build with an
    // older toolchain does not warn about an unexpected cfg.
    println!("cargo:rustc-check-cfg=cfg(xrt_shim)");

    if !cfg!(feature = "xrt") {
        return;
    }

    let Some(root) = xrt_root() else {
        println!(
            "cargo:warning=the `xrt` feature is on but XRT was not found; \
             the array path will report that it has no transport. Set \
             XILINX_XRT or XRT_ROOT to point at the installation."
        );
        return;
    };

    let include = root.join("include");
    if !include.join("xrt/xrt_device.h").exists() {
        println!(
            "cargo:warning=XRT at {} has no xrt/xrt_device.h; the array path \
             will report that it has no transport.",
            root.display()
        );
        return;
    }
    let lib_dir = root.join("lib");
    if !lib_dir.join("libxrt_coreutil.so").exists() && !lib_dir.join("libxrt_coreutil.a").exists() {
        println!(
            "cargo:warning=XRT at {} has no libxrt_coreutil; the array path \
             will report that it has no transport.",
            root.display()
        );
        return;
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let obj = out_dir.join("xrt_shim.o");
    let archive = out_dir.join("libnpum_xrt_shim.a");

    let cxx = std::env::var("CXX").unwrap_or_else(|_| "g++".to_string());
    let status = Command::new(&cxx)
        .args(["-std=c++17", "-O2", "-fPIC", "-c"])
        .arg("shim/xrt_shim.cpp")
        .arg("-I")
        .arg(&include)
        .arg("-o")
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| panic!("cannot run {cxx}: {e}"));
    assert!(status.success(), "{cxx} failed to compile the XRT shim");

    let ar = std::env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let status = Command::new(&ar)
        .arg("crs")
        .arg(&archive)
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| panic!("cannot run {ar}: {e}"));
    assert!(status.success(), "{ar} failed to archive the XRT shim");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=npum_xrt_shim");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
    // THE C++ RUNTIME, because the shim is C++.
    //
    // The shim catches XRT's exceptions and reports them as a message, so it
    // uses the C++ runtime's own machinery -- and a static archive that
    // references it has to be linked against it. On Windows the C++ runtime
    // is part of the toolchain's default, and there is no separate library
    // to name.
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    }
    // The rpath is a convenience: the loader already knows this directory on a
    // machine with XRT installed, and a runtime that cannot find its own
    // library at start-up is a runtime that fails before it says why.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    println!("cargo:rustc-cfg=xrt_shim");
}

/// Where XRT is installed.
///
/// The same order the C++ runtime's CMake uses: the environment first, the
/// usual installation second. An environment variable that points at
/// nothing is not silently ignored -- it is a build that says so.
fn xrt_root() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    for name in ["XILINX_XRT", "XRT_ROOT"] {
        if let Ok(value) = std::env::var(name) {
            let path = PathBuf::from(value);
            if path.is_dir() {
                return Some(path);
            }
            println!(
                "cargo:warning={name} points at {}, which is not a directory",
                path.display()
            );
        }
    }
    candidates.push(PathBuf::from("/opt/xilinx/xrt"));
    candidates.into_iter().find(|p| p.is_dir())
}
