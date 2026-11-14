//! Where the models and the design sets live.
//!
//! # The search, and why it is in this order
//!
//! The C++ runtime had this function and it was the source of a quiet bug:
//! a release staged or unzipped INSIDE the source tree
//! (`dist\npuembeddings-0.2.0\`) climbed past its own directory, found the
//! repository's `models/` and `runtime/`, and served the repo's containers
//! while claiming to be the release. Everything worked and everything was
//! wrong -- which is the failure shape this project keeps meeting.
//!
//! So: **the executable's own directory wins**, whenever it holds anything
//! of ours. The upward search only starts when there is nothing here.
//!
//! # How deep, and why the cap
//!
//! The layouts this has to recognise, shallowest first:
//!
//! ```text
//! <d>/gemm_rtp                                    a bare design set
//! <d>/<set>/gemm_rtp                              one level down
//! <d>/<model>/artifacts_npu<N>/gemm_rtp            the pre-2026 layout
//! <d>/artifacts/<model>/artifacts_npu<N>/gemm_rtp  the current one
//! ```
//!
//! so three levels is the deepest, and the walk stops there. The cap is not
//! only tidiness: this runs on whatever directory the process was launched
//! from, and when that is the repository root the alternative is walking
//! `.git/objects`. DOT-DIRECTORIES ARE SKIPPED for the same reason -- nothing
//! this looks for is ever inside one, and `.git` has ~256 fan-out at two
//! levels.
//!
//! # The two upward searches do not interleave
//!
//! The source tree is recognised by BOTH `models/` and `runtime/`, because a
//! design alone would stop at `runtime/`, which carries the design sets but
//! not the models. The C++ code had exactly that bug while fixing this
//! function: checking both conditions at each level made `runtime/` win over
//! the repository root.
//!
//! Each search runs to completion before the other starts.

use std::path::{Path, PathBuf};

/// The deepest level the design search walks.
const MAX_DESIGN_DEPTH: usize = 3;

/// How many levels up the upward search walks.
const MAX_UP: usize = 5;

/// Whether a directory holds a design set, at any of the known depths.
pub fn has_design(dir: &Path) -> bool {
    let mut level = vec![dir.to_path_buf()];
    for depth in 0..=MAX_DESIGN_DEPTH {
        if level.is_empty() {
            return false;
        }
        let mut next = Vec::new();
        for p in &level {
            if p.join("gemm_rtp").exists() {
                return true;
            }
            if depth == MAX_DESIGN_DEPTH {
                continue;
            }
            let Ok(entries) = std::fs::read_dir(p) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.starts_with('.') {
                    continue;
                }
                next.push(path);
            }
        }
        level = next;
    }
    false
}

/// The root, when nobody says which one.
///
/// `argv0` is the path the program was invoked with.
pub fn default_root(argv0: &str) -> PathBuf {
    let start = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .or_else(|| {
            std::fs::canonicalize(argv0)
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        })
        .unwrap_or_else(|| PathBuf::from("."));

    // The executable's own directory wins, whenever it holds anything of
    // ours.
    if has_design(&start) || start.join("models").exists() {
        return start;
    }

    // Nothing here, so this is a build directory. Search upwards -- and the
    // two searches must each run to completion before the other starts.
    if let Some(found) = walk_up(&start, |d| {
        d.join("models").exists() && d.join("runtime").exists()
    }) {
        return found;
    }
    if let Some(found) = walk_up(&start, has_design) {
        return found;
    }
    PathBuf::from("..")
}

/// Walk up, five levels, looking for a directory that matches.
fn walk_up<F>(start: &Path, matches: F) -> Option<PathBuf>
where
    F: Fn(&Path) -> bool,
{
    let mut dir = start.parent()?.to_path_buf();
    for _ in 0..MAX_UP {
        if matches(&dir) {
            return Some(dir);
        }
        let next = dir.parent()?.to_path_buf();
        if next == dir {
            break;
        }
        dir = next;
    }
    None
}
