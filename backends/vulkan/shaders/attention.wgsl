// Scaled dot-product attention: softmax(q k^T / sqrt(d)) v.
//
// ONE WORKGROUP PER QUERY ROW. The 64 lanes of the workgroup cooperate on
// the row instead of each lane owning a whole row: the score row lives in
// workgroup shared memory (1 KB), not as a `array<f32, 1024>` private copy
// per invocation (4 KB, which spilled), and the v access is coalesced --
// consecutive lanes read consecutive output columns of the same key.
//
// THE KEYS ARE STREAMED in 256-key chunks with an online softmax. There is
// no MAX_KV bound: a longer sequence is another chunk, with the running
// max and sum rescaled the standard way. All three clamps of the host's
// softmax are kept, in the same order.
//
// `inv` is multiplied into the running accumulator ONCE per chunk, not once
// per (key, column) pair.

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
var<immediate> params: Params;

const CHUNK: u32 = 256u;
const LANES: u32 = 64u;
const FLOOR_LOAD: f32 = -1e30;
const FLOOR_DIFF: f32 = -100.0;
const FLOOR_ARG: f32 = -120.0;
const LOG2E: f32 = 1.4426950408889634;
const NEG_INF: f32 = -3.402823e38;

var<workgroup> scores: array<f32, 256>;
var<workgroup> partial_m: array<f32, 64>;
var<workgroup> partial_l: array<f32, 64>;
var<workgroup> m_shared: f32;
var<workgroup> l_shared: f32;
var<workgroup> scale_shared: f32;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let qi = wid.x;
    let lane = lid.x;
    let d = params.head_dim;
    if (qi >= params.q_len || d == 0u) {
        return;
    }
    let scale = inverseSqrt(f32(d));

    // A lane owns output columns `lane`, `lane + 64`, ... up to `d`.
    // Four slots cover head dimensions up to 256; wider heads are rare and
    // would need a larger workgroup, which the backend would have to choose.
    var acc: array<f32, 4>;
    for (var c = 0u; c < 4u; c = c + 1u) {
        acc[c] = 0.0;
    }
    var m_run: f32 = NEG_INF;
    var l_run: f32 = 0.0;
    // Whether an EARLIER chunk contributed. `m_run` alone cannot answer this:
    // a whole clamped chunk sets it to FLOOR_LOAD, which is also the value the
    // first chunk can produce. Resetting then would discard every earlier key.
    var has_prev: bool = false;

    for (var chunk = 0u; chunk < params.kv_len; chunk = chunk + CHUNK) {
        let cnt = min(CHUNK, params.kv_len - chunk);

        // QK^T for this chunk: 64 lanes, up to 4 keys each.
        for (var j = lane; j < cnt; j = j + LANES) {
            var dot: f32 = 0.0;
            for (var p = 0u; p < d; p = p + 1u) {
                dot = dot + q[qi * d + p] * k[(chunk + j) * d + p];
            }
            scores[j] = max(dot * scale, FLOOR_LOAD);
        }
        workgroupBarrier();

        // Chunk maximum: lane-local maximum, then a tree reduction.
        var local_max: f32 = NEG_INF;
        for (var j = lane; j < cnt; j = j + LANES) {
            local_max = max(local_max, scores[j]);
        }
        partial_m[lane] = local_max;
        workgroupBarrier();
        for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
            if (lane < stride) {
                partial_m[lane] = max(partial_m[lane], partial_m[lane + stride]);
            }
            workgroupBarrier();
        }
        let m_new = max(m_run, partial_m[0]);
        // Rescale what came before by exp2(m_old - m_new), in base 2 so
        // exp2 is exact on a power-of-two difference.
        let scale_old = select(
            0.0,
            exp2(max((m_run - m_new) * LOG2E, FLOOR_ARG)),
            has_prev,
        );
        if (lane == 0u) {
            m_shared = m_new;
            scale_shared = scale_old;
        }
        workgroupBarrier();

        // Exponentials and the chunk's sum of exponentials.
        var local_l: f32 = 0.0;
        for (var j = lane; j < cnt; j = j + LANES) {
            let dl = max(scores[j] - m_shared, FLOOR_DIFF);
            let e = exp2(max(dl * LOG2E, FLOOR_ARG));
            scores[j] = e;
            local_l = local_l + e;
        }
        partial_l[lane] = local_l;
        workgroupBarrier();
        for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
            if (lane < stride) {
                partial_l[lane] = partial_l[lane] + partial_l[lane + stride];
            }
            workgroupBarrier();
        }
        if (lane == 0u) {
            l_run = l_run * scale_shared + partial_l[0];
            l_shared = l_run;
        }
        workgroupBarrier();

        // softmax.V for this chunk. Lane p reads v[j * d + p], consecutive
        // lanes read consecutive columns: coalesced.
        for (var c = 0u; c < 4u; c = c + 1u) {
            let p = lane + c * LANES;
            if (p < d) {
                var value = acc[c] * scale_shared;
                for (var j = 0u; j < cnt; j = j + 1u) {
                    value = value + scores[j] * v[(chunk + j) * d + p];
                }
                acc[c] = value;
            }
        }
        m_run = m_shared;
        has_prev = true;
    }

    if (params.kv_len == 0u || l_shared <= 0.0) {
        for (var c = 0u; c < 4u; c = c + 1u) {
            let p = lane + c * LANES;
            if (p < d) {
                out[qi * d + p] = 0.0;
            }
        }
        return;
    }
    for (var c = 0u; c < 4u; c = c + 1u) {
        let p = lane + c * LANES;
        if (p < d) {
            out[qi * d + p] = acc[c] / l_shared;
        }
    }
}
