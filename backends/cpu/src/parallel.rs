//! The shared parallelism policy and the reusable worker pool.
//!
//! # Why this module exists
//!
//! A host inference walks a model as a long chain of small, independent
//! operations. Splitting each one over freshly spawned threads costs more than
//! the work itself for all but the largest shapes, and spawning on every call
//! pays the thread-creation cost again and again. Both problems have the same
//! answer: ONE pool, kept alive between calls, and ONE policy that decides
//! whether a split is worth making at all.
//!
//! The policy is deliberately conservative. A split is made only when there
//! is enough total work to amortise the barrier AND enough rows per worker
//! that every worker gets real work. Small GEMMs stay serial; that is faster
//! than parallelising them, and it keeps the reference's numbers unchanged.
//!
//! # The split crosses no reduction
//!
//! Every caller splits along an axis whose elements are computed
//! independently -- GEMM output rows, LayerNorm/softmax rows, im2col output
//! rows, output channels, NCHW planes. A row's accumulation is over K in the
//! same order whichever thread computes it. That is what makes the split
//! bit-identical to the serial path, and why the ISA gate can keep comparing
//! bytes even when many threads are running.

use rayon::prelude::*;
use std::sync::OnceLock;

/// The total work below which a thread split is not worth its barrier.
///
/// This is the reference runtime's own `par_items` number, 65 536 elements.
/// With a persistent pool the barrier is a condition variable and is cheap,
/// but the policy stays the same: below the threshold the serial answer is
/// the cheap answer.
pub const PAR_MIN_ITEMS: i64 = 65_536;

/// The fewest rows a worker should get before we split at all.
///
/// A work threshold alone is not enough: four rows of a very wide GEMM can
/// clear 65 536 items while still giving a sixteen-core machine four tasks
/// and twelve idle workers. Requiring a minimum number of rows per worker
/// keeps such a GEMM on the serial path.
pub const PAR_MIN_ROWS_PER_WORKER: usize = 8;

/// The fewest OUTPUT COLUMNS a worker should get in a column split.
///
/// Row splitting is impossible when `m < 8` -- decode runs its projections
/// at `m = 1` -- but those GEMMs are often very wide (`1 x 768 x 2304`).
/// Their output columns are independent, so they can be split across the
/// pool instead. One eight-lane vector is the smallest useful column chunk:
/// narrower than that and masked tails start to dominate.
pub const PAR_MIN_COLS_PER_WORKER: usize = 8;

/// The defaults above are deliberate, but they are machine-dependent.
///
/// A many-core host with large batches can profit from a lower threshold or
/// a smaller rows-per-worker; a small host can lose time on the barrier. The
/// knobs are read ONCE, at first use, so a process has one stable policy:
///
/// | variable | meaning | default |
/// |---|---|---|
/// | `NPU_CPU_THREADS` | workers in the shared pool | `available_parallelism()` |
/// | `NPU_PAR_MIN_ITEMS` | work below which the split is skipped | [`PAR_MIN_ITEMS`] |
/// | `NPU_PAR_MIN_ROWS_PER_WORKER` | rows a worker must get | [`PAR_MIN_ROWS_PER_WORKER`] |
///
/// A bad value is ignored rather than fatal: a profiler run with a typo in
/// an environment variable must not change a numerical answer or abort a
/// server. The counts only change WHICH thread computes an independent row,
/// never the order of the reduction inside a row, so the gates stay byte-exact.
struct Policy {
    threads: Option<usize>,
    min_items: i64,
    min_rows: usize,
}

fn parse_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()?
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|v| *v > 0)
}

static POLICY: OnceLock<Policy> = OnceLock::new();

fn policy() -> &'static Policy {
    POLICY.get_or_init(|| Policy {
        threads: parse_usize("NPU_CPU_THREADS"),
        min_items: std::env::var("NPU_PAR_MIN_ITEMS")
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok())
            .filter(|v| *v >= 0)
            .unwrap_or(PAR_MIN_ITEMS),
        min_rows: parse_usize("NPU_PAR_MIN_ROWS_PER_WORKER").unwrap_or(PAR_MIN_ROWS_PER_WORKER),
    })
}

static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// The one pool every parallel kernel in this crate shares.
///
/// Built on first use and kept for the life of the process. Threads are NOT
/// created per call: a pipeline runs hundreds of GEMMs per inference, and a
/// spawn per GEMM is a tax on every one of them.
pub fn pool() -> &'static rayon::ThreadPool {
    POOL.get_or_init(|| {
        let threads = policy()
            .threads
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1)
            .max(1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("npum-cpu-{i}"))
            .build()
            .expect("npum-cpu: could not create the worker pool")
    })
}

/// The number of workers in the shared pool.
pub fn threads() -> usize {
    pool().current_num_threads()
}

/// Decide whether `rows` independent rows doing `work` items should be split.
///
/// Returns the number of rows in each contiguous chunk when a split is
/// worthwhile, and `None` when the caller should take the serial path. The
/// chunk count is at most `threads`, and never more than `rows` divided by
/// [`PAR_MIN_ROWS_PER_WORKER`].
#[inline]
pub fn chunk_rows(rows: usize, work: i64, requested: usize) -> Option<usize> {
    let policy = policy();
    if rows == 0 || work < policy.min_items {
        return None;
    }
    let cap = (rows / policy.min_rows).max(1);
    let workers = requested.min(threads()).min(cap).max(1);
    if workers <= 1 {
        return None;
    }
    Some(rows.div_ceil(workers))
}

/// Decide whether `cols` independent output columns doing `work` items
/// should be split across workers, and return the chunk width.
///
/// This is the decode counterpart of [`chunk_rows`]. Every output column is
/// reduced over K in the SAME order whichever worker computes it, so a
/// column split crosses no reduction and stays bit-identical to the serial
/// path. The minimum chunk keeps each worker at least one vector wide.
#[inline]
pub fn chunk_cols(cols: usize, work: i64, requested: usize) -> Option<usize> {
    let policy = policy();
    if cols == 0 || work < policy.min_items {
        return None;
    }
    let cap = (cols / PAR_MIN_COLS_PER_WORKER).max(1);
    let workers = requested.min(threads()).min(cap).max(1);
    if workers <= 1 {
        return None;
    }
    Some(cols.div_ceil(workers))
}

/// Hand each worker a disjoint set of output columns.
///
/// `data` is a row-major `rows x cols` matrix. For every column chunk the
/// closure receives the chunk's index, its starting column, and one mutable
/// window per row -- the slice `data[row][j0..j0+nc]`. The windows are
/// disjoint by construction (a row is split at `j0`, a chunk never overlaps
/// another), so rayon can run the closures in parallel without an atomic.
///
/// `f` is `Fn`, not `FnMut`, because each worker only mutates the slices it
/// was handed. No scratch is allocated in the kernel: the windows borrow
/// the caller's matrix directly.
#[inline]
pub fn for_each_col_chunk<T, F>(data: &mut [T], cols: usize, chunk_cols: usize, f: F)
where
    T: Send,
    F: Fn(usize, usize, Vec<&mut [T]>) + Sync + Send,
{
    debug_assert!(cols > 0 && data.len() % cols == 0 && chunk_cols > 0);
    let rows = data.len() / cols;
    let nchunks = cols.div_ceil(chunk_cols);
    let mut row_slices: Vec<&mut [T]> = data.chunks_mut(cols).collect();
    let mut windows: Vec<Vec<&mut [T]>> = (0..nchunks).map(|_| Vec::with_capacity(rows)).collect();
    // Split each row ONCE into all of its column windows. Re-splitting the
    // same row for every chunk would hand out the same bytes twice (and set
    // off an aliasing bug) -- a row is consumed left to right here.
    for row in row_slices.iter_mut() {
        let mut rest = std::mem::replace(row, &mut []);
        for ci in 0..nchunks {
            let nc = (cols - ci * chunk_cols).min(chunk_cols);
            let (window, tail) = rest.split_at_mut(nc);
            windows[ci].push(window);
            rest = tail;
        }
    }
    pool().install(|| {
        windows
            .into_par_iter()
            .enumerate()
            .for_each(|(ci, wins)| f(ci, ci * chunk_cols, wins));
    });
}

/// Run `f` once per contiguous chunk of `data`, in parallel, on the pool.
///
/// `f` receives the chunk's index and the chunk itself. The chunks are
/// disjoint mutable slices, so two workers can never write the same element;
/// the closure is `Fn + Sync` because rayon runs the same closure object on
/// several workers. Each worker's scratch (the GEMM accumulator included) is
/// thread-local and reused across calls.
#[inline]
pub fn for_each_chunk_mut<T, F>(data: &mut [T], chunk_items: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    pool().install(|| {
        data.par_chunks_mut(chunk_items)
            .enumerate()
            .for_each(|(i, chunk)| f(i, chunk));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_policy_defaults_are_the_reference_numbers() {
        let p = policy();
        assert_eq!(p.min_items, PAR_MIN_ITEMS);
        assert_eq!(p.min_rows, PAR_MIN_ROWS_PER_WORKER);
    }

    #[test]
    fn chunk_rows_keeps_small_work_and_short_row_sets_serial() {
        // Below the work threshold: no split, however many rows.
        assert_eq!(chunk_rows(1024, PAR_MIN_ITEMS - 1, 8), None);
        // Enough work, but fewer than eight rows per worker: no split.
        assert_eq!(chunk_rows(8, PAR_MIN_ITEMS, 8), None);
        assert_eq!(chunk_rows(0, PAR_MIN_ITEMS * 8, 8), None);
        assert_eq!(chunk_rows(1, PAR_MIN_ITEMS * 8, 8), None);
    }

    #[test]
    fn chunk_rows_never_returns_a_chunk_smaller_than_the_policy() {
        let rows = PAR_MIN_ROWS_PER_WORKER * threads() * 4;
        let work = (rows as i64) * 64;
        if let Some(chunk) = chunk_rows(rows, work, threads()) {
            assert!(chunk >= PAR_MIN_ROWS_PER_WORKER);
            assert!(chunk * threads() >= rows);
        }
    }

    #[test]
    fn chunk_cols_handles_the_decode_shape_and_skips_small_work() {
        // m = 1 cannot be row-split, but 768*2304 items clears the threshold.
        let work = 768i64 * 2304;
        let chunk = chunk_cols(2304, work, 16).expect("decode qkv should split by columns");
        assert!(chunk >= PAR_MIN_COLS_PER_WORKER);
        assert!(chunk * 16 >= 2304);
        // Too few columns to give every worker a vector, or too little work.
        assert_eq!(chunk_cols(4, work, 16), None);
        assert_eq!(chunk_cols(2304, PAR_MIN_ITEMS - 1, 16), None);
    }

    #[test]
    fn for_each_col_chunk_hands_out_disjoint_windows() {
        let rows = 3;
        let cols = 37;
        let mut data: Vec<usize> = (0..rows * cols).collect();
        let chunk = 8;
        for_each_col_chunk(&mut data, cols, chunk, |ci, j0, mut wins| {
            assert_eq!(wins.len(), rows);
            let nc = (cols - j0).min(chunk);
            for (r, window) in wins.iter_mut().enumerate() {
                assert_eq!(window.len(), nc);
                for (j, slot) in window.iter_mut().enumerate() {
                    assert_eq!(*slot, r * cols + j0 + j, "chunk {ci}");
                    *slot += 1000;
                }
            }
        });
        for (r, slot) in data.iter().enumerate() {
            assert_eq!(*slot, r + 1000);
        }
    }
}
