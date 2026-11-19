// Softmax over the last axis, with the three clamps.
//
// The clamps are the array's clamps, and they are here for the same reason:
// a mask value that is finite in f32 narrows to -inf in a narrower type,
// and exp(-inf) times a zero sum is a NaN that destroys the whole row.
//
//   1. on load, in the narrower type, before the widening
//   2. on the difference:            d = max(x - m, -100)
//   3. on the base-2 argument:       arg = max(d * log2e, -120)
//
// The third one is the subtle one: a negative exponent below about -127 is a
// negative exponent FIELD, which is a NaN bit pattern.
//
// BOTH reductions are trees: six steps of 64 -> 1 instead of every lane
// reading all 64 partials (O(64) work per lane). The tree's barriers double
// as the availability barrier before the next pass over the row.

struct Params {
    width: u32,
    rows: u32,
    stride: u32, // 0 means `width`: rows may be padded wider than the row
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
var<immediate> params: Params;

const FLOOR_LOAD: f32 = -1e30;
const FLOOR_DIFF: f32 = -100.0;
const FLOOR_ARG: f32 = -120.0;
const LOG2E: f32 = 1.4426950408889634;
const NEG_INF: f32 = -3.402823e38;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= params.rows) {
        return;
    }
    let width = params.width;
    let stride = select(width, params.stride, params.stride != 0u);
    let base = row * stride;

    var m: f32 = NEG_INF;
    for (var i = lid.x; i < width; i = i + 64u) {
        let v = max(x[base + i], FLOOR_LOAD);
        x[base + i] = v;
        m = max(m, v);
    }
    partial[lid.x] = m;
    workgroupBarrier();
    for (var step = 32u; step > 0u; step = step >> 1u) {
        if (lid.x < step) {
            partial[lid.x] = max(partial[lid.x], partial[lid.x + step]);
        }
        workgroupBarrier();
    }
    let mx = partial[0];

    var sum: f32 = 0.0;
    for (var i = lid.x; i < width; i = i + 64u) {
        let d = max(x[base + i] - mx, FLOOR_DIFF);
        let arg = max(d * LOG2E, FLOOR_ARG);
        let e = exp2(arg);
        x[base + i] = e;
        sum = sum + e;
    }
    partial[lid.x] = sum;
    workgroupBarrier();
    for (var step = 32u; step > 0u; step = step >> 1u) {
        if (lid.x < step) {
            partial[lid.x] = partial[lid.x] + partial[lid.x + step];
        }
        workgroupBarrier();
    }
    let inv = select(0.0, 1.0 / partial[0], partial[0] > 0.0);

    for (var i = lid.x; i < width; i = i + 64u) {
        x[base + i] = x[base + i] * inv;
    }
}
