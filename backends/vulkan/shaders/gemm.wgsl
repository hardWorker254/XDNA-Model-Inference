// The matrix multiply, tiled 16x16 per workgroup.
//
// THE TILE, and why 16: the array's panels are 64 wide and the GPU's
// workgroup is 16 wide, so a GEMM that is right on one is not right on the
// other. The two are separate backends with separate tolerances, and this
// comment says so rather than leaving the reader to assume the tile carries
// across.
//
// a is [m, k] row-major, b is [k, n] row-major, c is [m, n] -- the same
// contract as `npum_cpu::kernels::gemm`.
//
// # No early return, and that is not a style choice
//
// `workgroupBarrier()` must be reached by every invocation of the workgroup.
// An invocation for a row past `m` or a column past `n` that RETURNED before
// the barrier would leave the reduction reading a tile nobody loaded -- the
// kind of bug that gives a plausible answer for a 2x2 and a wrong one for
// an 8x16. So out-of-range invocations do the work (loading zeros) and
// simply do not store.
//
// # Why B is staged transposed
//
// The inner product below reads `b_tile[lid.x * TILE + p]`, so B's tile is
// stored column-major: `b_tile[lid.x * TILE + lid.y]` is `b[t*TILE + lid.y,
// col]`. Storing it like A looks symmetric and computes the right answer only
// when the matrix happens to be symmetric.

struct Params {
    m: u32,
    k: u32,
    n: u32,
    has_bias: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
// The bias is per output COLUMN and added after the whole reduction, in the
// same order as the host's `value += bias[col]`. Binding 3 is otherwise unused
// by this shader, so the epilogue costs no descriptor layout change.
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const TILE: u32 = 16u;

var<workgroup> a_tile: array<f32, 256>;
var<workgroup> b_tile: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;
    let row_ok = row < params.m;
    let col_ok = col < params.n;

    var sum: f32 = 0.0;
    let tiles = (params.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let a_col = t * TILE + lid.x;
        let b_row = t * TILE + lid.y;

        var av: f32 = 0.0;
        if (row_ok && a_col < params.k) {
            av = a[row * params.k + a_col];
        }
        a_tile[lid.y * TILE + lid.x] = av;

        var bv: f32 = 0.0;
        if (col_ok && b_row < params.k) {
            bv = b[b_row * params.n + col];
        }
        b_tile[lid.x * TILE + lid.y] = bv;

        workgroupBarrier();

        for (var p = 0u; p < TILE; p = p + 1u) {
            sum = sum + a_tile[lid.y * TILE + p] * b_tile[lid.x * TILE + p];
        }
        workgroupBarrier();
    }

    if (row_ok && col_ok) {
        var value = sum;
        if (params.has_bias != 0u) {
            value = value + bias[col];
        }
        c[row * params.n + col] = value;
    }
}
