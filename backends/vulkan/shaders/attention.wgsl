// Scaled dot-product attention.
//
// The scores go through the same softmax the other path uses, clamps
// included: an attention mask is exactly the case the clamps exist for.

struct Params {
    q_len: u32,
    kv_len: u32,
    head_dim: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const FLOOR_LOAD: f32 = -1e30;
const FLOOR_DIFF: f32 = -100.0;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if (qi >= params.q_len) {
        return;
    }
    let d = params.head_dim;
    let scale = inverseSqrt(f32(d));

    var scores: array<f32, 1024>;
    var m: f32 = -3.402823e38;
    for (var j = 0u; j < params.kv_len; j = j + 1u) {
        var sum: f32 = 0.0;
        for (var p = 0u; p < d; p = p + 1u) {
            sum = sum + q[qi * d + p] * k[j * d + p];
        }
        let s = sum * scale;
        scores[j] = s;
        m = max(m, s);
    }

    var total: f32 = 0.0;
    for (var j = 0u; j < params.kv_len; j = j + 1u) {
        let e = exp(max(scores[j] - m, FLOOR_DIFF));
        scores[j] = e;
        total = total + e;
    }
    let inv = select(0.0, 1.0 / total, total > 0.0);

    for (var p = 0u; p < d; p = p + 1u) {
        var acc: f32 = 0.0;
        for (var j = 0u; j < params.kv_len; j = j + 1u) {
            acc = acc + scores[j] * inv * v[j * d + p];
        }
        out[qi * d + p] = acc;
    }
}
