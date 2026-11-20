//! Malformed shapes must be refused, not panicked on or silently widened.
//! SPDX-License-Identifier: Apache-2.0

use npum_model::{DType, Tensor};

#[test]
fn negative_or_overflowing_dimensions_are_refused() {
    assert!(Tensor::raw(DType::F32, vec![-1], vec![0u8; 4]).is_err());
    assert!(Tensor::f32(vec![usize::MAX as i64, 2], vec![0.0]).is_err());
}

#[test]
fn u16_widens_as_words_not_as_bytes() {
    let tensor = Tensor::raw(DType::U16, vec![2], vec![1, 2, 3, 4]).unwrap();
    assert_eq!(tensor.to_f32(), vec![513.0, 1027.0]);
}
