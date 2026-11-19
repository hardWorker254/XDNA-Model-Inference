// Elementwise multiplication: z = x * y.

struct Params {
    count: u32,
    _pad: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> z: array<f32>;
var<immediate> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.count) {
        z[i] = x[i] * y[i];
    }
}
