//! The image binary: classification, pose, hands, whole-body pose and OCR.
//!
//! One line says which family this is, and everything else is shared with
//! the other two.

fn main() -> std::process::ExitCode {
    npuembeddings_cli::run("npuimage")
}
