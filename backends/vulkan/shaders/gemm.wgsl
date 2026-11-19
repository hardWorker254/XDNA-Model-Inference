// The matrix multiply, 8x32 output tile per workgroup, 32-wide K tile.
//
// a is [m, k] row-major, b is [k, n] row-major, c is [m, n] -- the same
// contract as `npum_cpu::kernels::gemm`.
//
// # Why B is NOT stored transposed
//
// The audit of this shader suspected a non-coalesced `b[b_row*n+col]` load.
// It is the opposite: the 32 lanes of a workgroup map to output COLUMNS, so
// they read 32 consecutive B elements. Storing B as [n, k] in the weight
// cache would make those 32 lanes stride by `k` and lose that coalescing.
// The real hazard was in shared memory, addressed by the 33-wide strides
// below. B therefore keeps its [k, n] layout in the cache.
//
// # The 33-wide shared strides
//
// a_tile[lid.y*33 + p] and b_tile[p*33 + lid.x]. Without the padding both
// arrays have 32-wide rows, and 8 (respectively 32) threads addressing the
// same p land in the same bank. 33 is coprime with 32, so each row shifts
// the bank by one. The cost is 1/32 of shared memory; the saving is the
// 8-way conflict on every iteration of the inner product.
//
// # No early return, and that is not a style choice
//
// `workgroupBarrier()` must be reached by every invocation of the workgroup.
// An invocation for a row past `m` or a column past `n` that RETURNED before
// the barrier would leave the reduction reading a tile nobody loaded. So
// out-of-range lanes load zeros and simply do not store.
//
// # Double buffering
//
// Two tile pairs alternate. The barrier after the reduction of tile t
// protects the buffers that tile t+2 will overwrite, so loads of the next
// tile can start while the current one is still being read. There are two
// barriers per K tile, but the tile is twice as wide as before: half as
// many K tiles, half as many barriers per GEMM.

struct Params {
    m: u32,
    k: u32,
    n: u32,
    flags: u32, // bit 0: add bias[col], bit 2: add acc[row, col]
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> acc: array<f32>;
var<immediate> params: Params;

const TM: u32 = 8u;
const TN: u32 = 32u;
const TK: u32 = 32u;
const PAD_A: u32 = 33u;
const PAD_B: u32 = 33u;

var<workgroup> a0: array<f32, 8u * 33u>;
var<workgroup> b0: array<f32, 32u * 33u>;
var<workgroup> a1: array<f32, 8u * 33u>;
var<workgroup> b1: array<f32, 32u * 33u>;

@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.y * TM + lid.y;
    let col = wid.x * TN + lid.x;
    let row_ok = row < params.m;
    let col_ok = col < params.n;

    var sum: f32 = 0.0;
    let tiles = (params.k + TK - 1u) / TK;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let kbase = t * TK;
        let ak = kbase + lid.x;
        var av: f32 = 0.0;
        if (row_ok && ak < params.k) {
            av = a[row * params.k + ak];
        }

        // Four B values per thread: the K tile has 32 rows and the
        // workgroup has only 8 lanes in y.
        var bv: array<f32, 4>;
        for (var q = 0u; q < 4u; q = q + 1u) {
            let bk = kbase + lid.y + q * 8u;
            bv[q] = 0.0;
            if (col_ok && bk < params.k) {
                bv[q] = b[bk * params.n + col];
            }
        }

        if ((t & 1u) == 0u) {
            a0[lid.y * PAD_A + lid.x] = av;
            for (var q = 0u; q < 4u; q = q + 1u) {
                b0[(lid.y + q * 8u) * PAD_B + lid.x] = bv[q];
            }
        } else {
            a1[lid.y * PAD_A + lid.x] = av;
            for (var q = 0u; q < 4u; q = q + 1u) {
                b1[(lid.y + q * 8u) * PAD_B + lid.x] = bv[q];
            }
        }
        workgroupBarrier();

        if ((t & 1u) == 0u) {
            for (var p = 0u; p < TK; p = p + 1u) {
                sum = sum + a0[lid.y * PAD_A + p] * b0[p * PAD_B + lid.x];
            }
        } else {
            for (var p = 0u; p < TK; p = p + 1u) {
                sum = sum + a1[lid.y * PAD_A + p] * b1[p * PAD_B + lid.x];
            }
        }
        workgroupBarrier();
    }

    if (row_ok && col_ok) {
        var value = sum;
        if ((params.flags & 1u) != 0u) {
            value = value + bias[col];
        }
        if ((params.flags & 4u) != 0u) {
            value = value + acc[row * params.n + col];
        }
        c[row * params.n + col] = value;
    }
}
