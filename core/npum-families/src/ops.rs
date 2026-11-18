//! What each architecture HAS, and the refusal for what it does not.
//!
//! # The flag says where an operation runs
//!
//! `--npu-ops` names the operations that go on the array, and everything
//! else stays on the host. The flag is one rule for six architectures, and
//! the six do not have the same operations: a Whisper container has a
//! mel filter bank and a vocabulary projection, and an OCR container has
//! neither -- its graph has already fused the activations into the
//! convolutions and the normalisation into the weights.
//!
//! So each architecture refuses the codes it does not have, BY NAME, and
//! the refusal says what the architecture does have. A code that was
//! accepted and quietly ignored would leave an operation on the host
//! while the command line said it was on the array, and that is the
//! failure the whole flag exists to prevent.
//!
//! # The texts are the reference's, word for word
//!
//! They are not paraphrases. A refusal is read by someone whose command
//! line did not work, and the second one they read has to say the same
//! thing as the first -- or they will wonder which of the two runtimes
//! they are talking to.
//!
//! # Which code is refused first
//!
//! The codes are a SET, so they are in alphabetical order, and the first
//! one that the architecture does not have is the one that is named. The
//! listing itself is echoed unchanged where the reference's echoes it: what
//! the user typed is evidence, and a message that rewrites it is a
//! message that hides what was asked.

use crate::Family;

/// The refusal for an operation code this architecture does not have.
///
/// `listing` is the flag's value as the user typed it, and `codes` are
/// the codes it parsed to, in alphabetical order -- the order the reference's
/// `std::set` gives.
///
/// `None` when the architecture has every code, and when there is
/// nothing to refuse: an empty listing asks for nothing.
pub fn refuse_op_not_in_arch(family: Family, listing: &str, codes: &[&str]) -> Option<String> {
    if codes.is_empty() {
        return None;
    }
    match family {
        Family::Ocr => first_missing(codes, &["conv"], OCR_TAIL),
        Family::Hands => first_missing(codes, &["conv"], HANDS_TAIL),
        Family::MpPose => first_missing(codes, &["conv"], MPPOSE_TAIL),
        Family::Pose => pose_refusal(listing, codes),
        Family::Cls => vit_refusal(codes),
        // The text family and the speech one have every code this build
        // knows, so there is nothing to refuse.
        Family::Embed | Family::Stt => None,
    }
}

/// The refusal for the first code the architecture does not have.
fn first_missing(codes: &[&str], here: &[&str], tail: &str) -> Option<String> {
    for code in codes {
        if !here.contains(code) {
            return Some(format!(
                "--npu-ops {code}: this architecture has no operation named '{code}'. {tail}"
            ));
        }
    }
    None
}

/// The pose family's refusal.
///
/// A pose model has exactly ONE array-able operation, and this refusal
/// says so -- and it is a different shape from the other families',
/// because the pose flag was never a list of choices: it is one
/// operation, and asking for anything else is a question about the
/// network's activations.
fn pose_refusal(listing: &str, codes: &[&str]) -> Option<String> {
    if codes.len() == 1 && codes[0] == "conv" {
        return None;
    }
    let names: Vec<&str> = codes.iter().copied().filter(|c| *c != "conv").collect();
    Some(format!(
        "--npu-ops {listing}: a pose model has exactly one array-able op, `conv`, \
         which moves the 72 dispatched convolutions of the network onto the array \
         as dispatched GEMMs (the 73rd weight, the head's [1,16,1,1] DFL, is \
         folded analytically into the decoder and never dispatched). The codes \
         {} name Whisper's or BERT's eltwise and projection passes, and this \
         architecture has none of them: the activations here are SiLU, which is \
         fused into the convolution's epilogue because it is elementwise on the \
         GEMM's own output, and there is no LayerNorm, no attention and no \
         vocabulary. Re-run with `--npu-ops conv`.",
        names.join(", ")
    ))
}

/// The classifier's refusal.
///
/// Three codes are DEAD here -- they name Whisper's front end and its
/// vocabulary projection, and this front end is decode/resize/normalise/
/// im2col -- and the refusal says which of the others this architecture
/// does honour, so the reader can fix the command line instead of
/// guessing.
fn vit_refusal(codes: &[&str]) -> Option<String> {
    const DEAD: &[&str] = &["mproj", "fft", "logit"];
    const KEPT: &[&str] = &["layn", "gelu", "attn", "softm", "conv", "gemm"];
    let refused: Vec<&str> = codes.iter().copied().filter(|c| DEAD.contains(c)).collect();
    if refused.is_empty() {
        return None;
    }
    let keep: Vec<&str> = KEPT.iter().copied().filter(|c| codes.contains(c)).collect();
    let tail = if keep.is_empty() {
        " Drop it.".to_string()
    } else {
        format!(
            " Keep --npu-ops {}, which this architecture does honour.",
            keep.join(", ")
        )
    };
    Some(format!(
        "--npu-ops {}: this architecture has no such host/array choice to make. \
         mproj and fft name Whisper's mel bank and transform, and this front end \
         is decode/resize/normalise/im2col, so neither exists here. logit names \
         the vocabulary projection; the nearest thing here is the classification \
         head, which cannot run on this array at all because 1000 is not a \
         multiple of tile_n*cols = 192 and no legal B panel of that width \
         exists.{tail}",
        refused.join(", ")
    ))
}

/// The OCR family's tail: what its two graphs contain.
const OCR_TAIL: &str = "Its two graphs are {Conv, ConvTranspose, Relu, Sigmoid, \
     HardSwish, HardSigmoid, Add, Sub, Mul, Div, Sqrt, Concat, MaxPool, AvgPool, \
     GlobalAveragePool, Resize, Reshape, Transpose, Slice, Squeeze, Unsqueeze, \
     MatMul, Softmax, ReduceMean} and the graph already FUSES the activations \
     into the convolutions and the normalisation into the weights before it, so \
     'gelu' has nothing to take over and 'layn' has nothing to normalise -- there \
     is no BatchNormalization left in either graph to name. Only 'conv' moves \
     anything: the dense convolutions, onto a design set built for their padded \
     (K, N) pairs.";

/// The hands family's tail: what its two graphs contain.
const HANDS_TAIL: &str = "Its two graphs are {Conv, DwConv, Add, MaxPool, Pad, \
     Resize} and the graph already FUSES the activations into the convolutions' \
     epilogues, so 'gelu' has nothing to take over and 'layn' has nothing to \
     normalise -- there is no BatchNormalization or InstanceNormalization in \
     either graph to name. Only 'conv' moves anything: the 61 dense \
     convolutions, onto a design set built for their twelve padded (K, N) pairs.";

/// The MediaPipe pose family's tail: what its two graphs contain.
const MPPOSE_TAIL: &str = "Its two graphs are {Conv, DwConv, Add, MaxPool, Resize, \
     DepthToSpace} and the graph already FUSES the activations into the \
     convolutions' epilogues, so 'gelu' has nothing to take over and 'layn' has \
     nothing to normalise -- there is no BatchNormalization or \
     InstanceNormalization in either graph to name. Only 'conv' moves anything: \
     the 99 dense convolutions, onto a design set built for their 22 padded (K, \
     N) pairs.";
