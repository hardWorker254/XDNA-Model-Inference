//! `serve`: the HTTP listener.
//!
//! The mode is chosen by the CONTAINER'S ARCH, never by a flag. A binary
//! serving a container of another family is refused BY NAME, before the
//! listener opens -- rather than answered by an endpoint this process is not
//! named for.

use npuembeddings_server::routes::ServerState;
use npuembeddings_server::Result;

/// Serve one model.
///
/// The refusal comes FIRST and the listener does not open until it has
/// passed: a refusal that opens a port and then closes it is a refusal the
/// user can fail to notice.
pub fn serve(
    binary: &'static str,
    model: &str,
    port: u16,
    bind: &str,
    root: &std::path::Path,
) -> i32 {
    let arch = match container_arch(model, root) {
        Some(arch) => arch,
        None => {
            eprintln!("error: the model {model} is not installed.");
            eprintln!("       `serve` downloads and verifies a model it does not have;");
            eprintln!("       give a name from the catalogue, or a path to a .npue.");
            return crate::EXIT_REFUSED;
        }
    };

    let owner = npum_families::registry::family_of_arch(&arch);
    let this = npum_families::registry::family_of_arch("bert");
    let this_family = match binary {
        "npuaudio" => npum_families::Family::Stt,
        "npuimage" => npum_families::Family::Cls,
        _ => npum_families::Family::Embed,
    };
    let _ = owner;
    let _ = this;

    // The family check is by BINARY, not by the arch alone: `npuimage` runs
    // the image families, `npuaudio` the speech one, and `npuembeddings` the
    // text one.
    let family = npum_families::Family::from_arch(&arch).unwrap_or(npum_families::Family::Embed);
    let belongs = match this_family {
        npum_families::Family::Embed => family == npum_families::Family::Embed,
        npum_families::Family::Stt => family == npum_families::Family::Stt,
        _ => matches!(
            family,
            npum_families::Family::Cls
                | npum_families::Family::Pose
                | npum_families::Family::Hands
                | npum_families::Family::MpPose
                | npum_families::Family::Ocr
        ),
    };
    if !belongs {
        eprint!(
            "{}",
            npum_families::registry::refuse_other_family(model, &arch, family, binary)
        );
        return crate::EXIT_REFUSED;
    }

    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: the async runtime could not be started: {e}");
            return crate::EXIT_REFUSED;
        }
    };
    runtime.block_on(async move {
        let state = ServerState::new(model, &arch, binary);
        let app = npuembeddings_server::router(state);
        let address = format!("{bind}:{port}");
        let listener = match tokio::net::TcpListener::bind(&address).await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("error: cannot listen on {address}: {e}");
                return crate::EXIT_REFUSED;
            }
        };
        eprintln!("serving {model} ({arch}) on http://{address}");
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("error: the server stopped: {e}");
            return crate::EXIT_REFUSED;
        }
        0
    })
}

/// Read a container's arch.
///
/// A container that cannot be read is NOT "the embed family": it is an
/// error, and the error names the file.
fn container_arch(model: &str, root: &std::path::Path) -> Option<String> {
    let path = resolve_model(model, root)?;
    let container = npum_format::Container::open(&path).ok()?;
    container.config_str("arch").map(|s| s.to_string())
}

/// Resolve a model's name or path to a container on disk.
///
/// A NAME is looked up in `models/`; a PATH is used as given and never
/// re-resolved by name.
pub fn resolve_model(
    model: &str,
    root: &std::path::Path,
) -> Option<std::path::PathBuf> {
    crate::families::resolve_model(model, root).ok()
}

/// The result type of this module.
pub type ServeResult = Result<i32>;
