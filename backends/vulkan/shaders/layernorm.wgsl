// Layer normalisation over the last axis.
//
// The mean is removed BEFORE the variance is taken, and the epsilon is added
// to the variance before the square root. That order is the reference's
// order, and the shortcut -- sum of squares -- is what the array's kernel
// does: the two differ in the last bits, and the last bits are what a
// bit-for-bit gate would catch.

struct Params {
    width: u32,
    rows: u32,
    _pad: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
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
    let inv = inverseSqrt(variance + 1e-5);

    for (var i = lid.x; i < params.width; i = i + 64u) {
        let g = select(1.0, gamma[i], i < arrayLength(&gamma));
        let b = select(0.0, beta[i], i < arrayLength(&beta));
        x[base + i] = (x[base + i] - mean) * inv * g + b;
    }
}
