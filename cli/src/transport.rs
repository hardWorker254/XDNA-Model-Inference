//! The transport this build uses.
//!
//! The rule -- the direct transport when the build has it, XRT otherwise --
//! lives in `npum_xdna::transport`, because it is not the CLI's rule: the
//! tests and any future tool have to speak through the same transport, and a
//! copy of the rule per dependent is a copy that will disagree.
//!
//! What is the CLI's own is the message: a run that cannot open the board
//! hears about the board, not about the transport.

pub type Transport = npum_xdna::transport::SelectedTransport;

/// Open the board.
pub fn open() -> Result<Transport, String> {
    npum_xdna::transport::open_selected(0).map_err(|e| format!("cannot open the NPU board: {e}"))
}
