//! Reading a GEMM operand out of a container, in either storage form.
//!
//! A container packed for the array keeps its dense operands as pre-tiled
//! BF16 panels; one packed for the host keeps row-major F32. The host's
//! forward pass has to work with BOTH, because the same model is served by
//! the same binary whether or not this run dispatched to the board. Reading
//! a BF16 panel as F32 gives a right-looking operand of half the length --
//! and a GEMM that walks off the end of it.

use npum_format::Container;

/// The operand as a row-major `[K, N]` f32 matrix, de-tiled when needed.
pub fn read_operand(container: &Container, name: &str) -> Result<Vec<f32>, String> {
    let entry = container
        .directory()
        .tensor(name)
        .ok_or_else(|| format!("the container has no tensor named {name}"))?;
    let bytes = container
        .tensor_bytes(name)
        .map_err(|e| format!("{name}: {e}"))?;
    match entry.dtype.as_str() {
        "F32" => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "BF16" => {
            let shape: Vec<i64> = entry.padded_shape.clone();
            if shape.len() != 2 {
                return Err(format!(
                    "{name}: a GEMM operand has a {}-dimensional padded shape; a dense operand is [K, N]",
                    shape.len()
                ));
            }
            if !entry.logical_shape.is_empty()
                && (entry.logical_shape[0] != shape[0] || entry.logical_shape[1] != shape[1])
            {
                return Err(format!(
                    "{name}: logical [{},{}] but padded [{},{}]. The host builds [K,N] from the padded shape and the caller indexes it with the logical one, so the two must agree.",
                    entry.logical_shape[0],
                    entry.logical_shape[1],
                    shape[0],
                    shape[1]
                ));
            }
            let (k, n) = (shape[0] as usize, shape[1] as usize);
            if k == 0 || n == 0 {
                return Err(format!("{name}: a GEMM operand with a zero dimension"));
            }
            let layout = entry.layout.clone().ok_or_else(|| {
                format!("{name}: the payload carries no layout. The layout is what says where each element sits, so without it these bytes are a right count in an unknown order.")
            })?;
            let layout: npum_format::layout::BLayout =
                serde_json::from_value(layout).map_err(|e| {
                    format!("{name}: the recorded layout is unreadable: {e}")
                })?;
            if layout.kind != "block_panel"
                || layout.tile_k <= 0
                || layout.tile_n <= 0
                || layout.mac_s <= 0
                || layout.mac_t <= 0
            {
                return Err(format!(
                    "{name}: the payload carries no usable block_panel layout (kind '{}').",
                    layout.kind
                ));
            }
            let tiled = npum_format::tiling::untile_b(
                bytes,
                k,
                n,
                layout.tile_k as usize,
                layout.tile_n as usize,
                2,
                Some(layout.mac_s as usize),
                Some(layout.mac_t as usize),
                npum_format::tiling::TileOrder::Kn,
            )
            .map_err(|e| format!("{name}: the panel does not de-tile: {}", e.message))?;
            let mut values = vec![0f32; k * n];
            npum_format::bf16::bf16_to_f32_bytes(&tiled, &mut values);
            Ok(values)
        }
        other => Err(format!(
            "{name}: an operand of dtype {other} is not a host GEMM operand"
        )),
    }
}
