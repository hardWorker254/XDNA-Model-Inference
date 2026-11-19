//! Tensors: a dtype, a shape, and the bytes.

use std::fmt;

/// The dtypes a tensor can hold.
///
/// The list is short because the list of DATAPATHS is short. The array runs
/// bf16 and int8; the host runs f32. There is no f16 and no f8, because
/// neither board has a datapath that could run one, and a container claiming
/// a dtype no design can be built for would fail only after the work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit float.
    F32,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 16-bit unsigned integer.
    U16,
    /// Brain float, stored as raw `u16`.
    ///
    /// BF16 has no native dtype and the array's panel is a `u16` array, so it
    /// travels as raw bits and is widened on read.
    Bf16,
    /// 8-bit signed integer: a per-output-channel symmetrically quantised
    /// operand.
    I8,
    /// Raw bytes: a design set's files, which are not numbers.
    U8,
}

impl DType {
    /// The tag as it appears in a container's directory.
    pub fn tag(self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::U16 => "U16",
            DType::Bf16 => "BF16",
            DType::I8 => "I8",
            DType::U8 => "U8",
        }
    }

    /// Parse a directory's dtype tag.
    pub fn parse(tag: &str) -> Option<Self> {
        match tag {
            "F32" => Some(DType::F32),
            "I32" => Some(DType::I32),
            "I64" => Some(DType::I64),
            "U16" => Some(DType::U16),
            "BF16" => Some(DType::Bf16),
            "I8" => Some(DType::I8),
            "U8" => Some(DType::U8),
            _ => None,
        }
    }

    /// How many bytes one element of this dtype is.
    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::I64 => 8,
            DType::U16 | DType::Bf16 => 2,
            DType::I8 | DType::U8 => 1,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.tag())
    }
}

/// What can go wrong with a tensor.
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("expected {expected} elements, found {found}")]
    WrongLength { expected: usize, found: usize },

    #[error("expected {expected} dimensions, found {found}")]
    WrongRank { expected: usize, found: usize },

    #[error("shape {shape:?} does not match the data length {len}")]
    ShapeMismatch { shape: Vec<i64>, len: usize },

    #[error("cannot widen {from} into {to}")]
    BadDType { from: DType, to: DType },
}

/// A tensor: a dtype, a shape, and the bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    dtype: DType,
    shape: Vec<i64>,
    data: Vec<u8>,
}

impl Tensor {
    /// A tensor of f32, from a shape and the values.
    pub fn f32(shape: Vec<i64>, values: Vec<f32>) -> Result<Self, TensorError> {
        let expected: usize = shape.iter().map(|d| *d as usize).product();
        if expected != values.len() {
            return Err(TensorError::WrongLength {
                expected,
                found: values.len(),
            });
        }
        let mut data = Vec::with_capacity(expected * 4);
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Ok(Self {
            dtype: DType::F32,
            shape,
            data,
        })
    }

    /// A tensor of raw bytes.
    pub fn raw(dtype: DType, shape: Vec<i64>, data: Vec<u8>) -> Result<Self, TensorError> {
        let expected: usize = shape.iter().map(|d| *d as usize).product::<usize>() * dtype.size();
        if expected != data.len() {
            return Err(TensorError::ShapeMismatch {
                shape,
                len: data.len(),
            });
        }
        Ok(Self { dtype, shape, data })
    }

    /// The dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The shape.
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// The raw bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// The raw bytes, mutable.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// How many elements the tensor holds.
    pub fn len(&self) -> usize {
        self.shape.iter().map(|d| *d as usize).product()
    }

    /// Whether the tensor holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The tensor's values as f32, widening bf16 and int8.
    ///
    /// BF16 widens by shifting: `bf16_to_f32` is exact, because bf16 IS the
    /// top half of an f32. Int8 is NOT exact -- it carries a per-channel
    /// scale that lives elsewhere -- so a widened int8 is the QUANTISED
    /// value, and any arithmetic on it is the arithmetic of a dequantised
    /// number. That is why the int8 path has its own tolerances and does not
    /// share the f32 ones.
    pub fn to_f32(&self) -> Vec<f32> {
        let mut out = Vec::new();
        self.to_f32_into(&mut out);
        out
    }

    /// Widening into a caller-owned `Vec`, reusing its capacity.
    ///
    /// `to_f32` allocates a fresh vector on every call, and the generic
    /// `Backend` path calls it for every operand of every op. The decode
    /// loop here is the SAME as `to_f32`'s, element by element and in order,
    /// so reusing the buffer cannot change a single bit.
    pub fn to_f32_into(&self, out: &mut Vec<f32>) {
        out.clear();
        out.reserve(self.len());
        match self.dtype {
            DType::F32 => out.extend(
                self.data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            ),
            DType::Bf16 => {
                // The block helper widens the same little-endian bits with the
                // same zero-extend-and-shift as `bf16_to_f32`, but 8 lanes at a
                // time when AVX2 is available. Bit-identical by construction:
                // a bf16 value IS the top half of its f32.
                let n = self.len();
                out.resize(n, 0.0);
                npum_format::bf16::bf16_to_f32_bytes(&self.data, &mut out[..n]);
            }
            DType::I32 => out.extend(
                self.data
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32),
            ),
            DType::I8 => out.extend(self.data.iter().map(|b| (*b as i8) as f32)),
            DType::I64 => out.extend(self.data.chunks_exact(8).map(|c| {
                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
            })),
            DType::U16 | DType::U8 => out.extend(self.data.iter().map(|b| *b as f32)),
        }
    }
}
