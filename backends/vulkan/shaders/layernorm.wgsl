// Layer normalisation over the last axis.
//
// The mean is removed BEFORE the variance is taken, and the epsilon is added
// to the variance before the square root. That order is the reference's
// order, and the epsilon is `npum_cpu::kernels::LAYERNORM_EPS` (1e-12) and
// not the container's `layer_norm_eps`: the CPU kernel makes that choice and
// the GPU has to make the same one or the two disagree on every row.
//
// gamma and beta are OPTIONAL. A container without them passes a
// one-element zero buffer and the `has_gamma`/`has_beta` flags, so the
// shader never reads past the buffer it was given.

struct Params {
    width: u32,
    rows: u32,
    has_gamma: u32,
    has_beta: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const EPS: f32 = 1e-12;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= params.rows) {
        return;
    }
    let base = row * params.width;

    var sum: f32 = 0.0;
    for (var i = lid.x; i < params.width; i = i + 64u) {
        sum = sum + x[base + i];
    }
    partial[lid.x] = sum;
    workgroupBarrier();

    var total: f32 = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        total = total + partial[i];
    }
    let mean = total / f32(params.width);
    workgroupBarrier();

    var var_sum: f32 = 0.0;
    for (var i = lid.x; i < params.width; i = i + 64u) {
        let d = x[base + i] - mean;
        var_sum = var_sum + d * d;
    }
    partial[lid.x] = var_sum;
    workgroupBarrier();

    var v_total: f32 = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        v_total = v_total + partial[i];
    }
    let variance = v_total / f32(params.width);
    let inv = inverseSqrt(variance + EPS);
    workgroupBarrier();

    for (var i = lid.x; i < params.width; i = i + 64u) {
        var g: f32 = 1.0;
        var b: f32 = 0.0;
        if (params.has_gamma == 1u) {
            g = gamma[i];
        }
        if (params.has_beta == 1u) {
            b = beta[i];
        }
        x[base + i] = (x[base + i] - mean) * inv * g + b;
    }
}
