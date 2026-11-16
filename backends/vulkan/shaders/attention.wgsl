// Scaled dot-product attention: softmax(q k^T / sqrt(d)) v.
//
// The scores go through the same softmax the other path uses, clamps
// included: an attention mask is exactly the case the clamps exist for.
//
// ONE INVOCATION PER QUERY ROW, and the score row is a private array of
// MAX_KV. That is the whole softmax row per invocation, which is why the
// kernel is bounded: a longer key sequence is a refusal, and the backend
// falls back to the host rather than quietly computing a truncated softmax.

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
@group(0) @binding(5) var<uniform> params: Params;

const MAX_KV: u32 = 1024u;
const FLOOR_LOAD: f32 = -1e30;
const FLOOR_DIFF: f32 = -100.0;
const FLOOR_ARG: f32 = -120.0;
const LOG2E: f32 = 1.4426950408889634;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if (qi >= params.q_len || params.kv_len > MAX_KV) {
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
        let s = max(sum * scale, FLOOR_LOAD);
        scores[j] = s;
        m = max(m, s);
    }

    var total: f32 = 0.0;
    for (var j = 0u; j < params.kv_len; j = j + 1u) {
        let dl = max(scores[j] - m, FLOOR_DIFF);
        let arg = max(dl * LOG2E, FLOOR_ARG);
        let e = exp2(arg);
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
