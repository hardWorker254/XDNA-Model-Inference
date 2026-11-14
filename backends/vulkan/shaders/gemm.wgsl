// The matrix multiply, tiled 16x16 per workgroup.
//
// THE TILE, and why 16: the array's panels are 64 wide and the GPU's
// workgroup is 16 wide, so a GEMM that is right on one is not right on the
// other. The two are separate backends with separate tolerances, and this
// comment says so rather than leaving the reader to assume the tile carries
// across.

struct Params {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 16u;

var<workgroup> a_tile: array<f32, 256>;
var<workgroup> b_tile: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;

    var sum: f32 = 0.0;
    let tiles = (params.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let a_col = t * TILE + lid.x;
        let b_row = t * TILE + lid.y;
        if (row < params.m && a_col < params.k) {
            a_tile[lid.y * TILE + lid.x] = a[row * params.k + a_col];
        } else {
            a_tile[lid.y * TILE + lid.x] = 0.0;
        }
        if (b_row < params.k && col < params.n) {
            b_tile[lid.y * TILE + lid.x] = b[b_row * params.n + col];
        } else {
            b_tile[lid.y * TILE + lid.x] = 0.0;
        }
        workgroupBarrier();

        for (var p = 0u; p < TILE; p = p + 1u) {
            sum = sum + a_tile[lid.y * TILE + p] * b_tile[lid.x * TILE + p];
        }
        workgroupBarrier();
    }

    if (row < params.m && col < params.n) {
        c[row * params.n + col] = sum;
    }
}
