// GELU: the SAME degree-8 minimax polynomial the CPU kernel uses.
//
// Not the exact erf and not the tanh approximation. The CPU's kernel and the
// array's fused kernel both use this polynomial, and a shader that computed
// a different function would be a tolerance to explain rather than a
// backend.

struct Params {
    count: u32,
    _pad: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn gelu(v: f32) -> f32 {
    let u = min(abs(v), 4.0);
    var pl = -7.2340282171e-05;
    pl = pl * u + 1.8179518005e-03;
    pl = pl * u + -1.7707383379e-02;
    pl = pl * u + 8.4577147641e-02;
    pl = pl * u + -1.9228671834e-01;
    pl = pl * u + 9.8431124458e-02;
    pl = pl * u + 3.6137852062e-01;
    pl = pl * u + -4.9454128936e-01;
    pl = pl * u + -1.3007010117e-04;
    return max(v, 0.0) + pl;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.count) {
        x[i] = gelu(x[i]);
    }
}
