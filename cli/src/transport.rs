//! The transport this build uses.
//!
//! # THE TRANSPORT IS A PROPERTY OF THE BUILD
//!
//! There are two ways to reach the array: through XRT's C++ library, or
//! through the `amdxdna` driver's ioctls. They are two transports to the
//! same board, and everything above them -- the designs, the packing, the
//! op kernels, the families -- is written against one interface and not
//! against either.
//!
//! Which one is in use is decided when the binary is built, and the rule
//! is the same one the device module uses: the direct transport, when the
//! build has it, is the one that speaks; the XRT transport is what a build
//! without it uses. A build can have both -- the direct transport does not
//! need XRT to be absent, only to be unused -- and the direct one wins,
//! because it is the one the build was asked for.
//!
//! # WHY NOT A FLAG
//!
//! A `--transport` flag would let a run choose, and a run that chooses a
//! transport this build does not have is a refusal to write and a trap to
//! fall into. The build says what it can do; the run uses it.

#[cfg(feature = "direct")]
pub type Transport = npum_xdna::direct::DirectTransport;

#[cfg(all(feature = "xrt", not(feature = "direct")))]
pub type Transport = npum_xdna::xrt::XrtTransport;

/// Open the board.
pub fn open() -> Result<Transport, String> {
    use npum_xdna::transport::Transport as _;
    Transport::open(0).map_err(|e| format!("cannot open the NPU board: {e}"))
}
