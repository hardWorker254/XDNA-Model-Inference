//! The speech-to-text binary.
//!
//! One line says which family this is, and everything else is shared with
//! the other two.

fn main() -> std::process::ExitCode {
    npuembeddings_cli::run("npuaudio")
}
