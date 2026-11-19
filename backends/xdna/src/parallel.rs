//! Small, conservative host-side parallelism for the array backend.
//!
//! The array's hot paths convert, transpose and widen whole panels on the
//! host before/after a dispatch. Those loops are embarrassingly parallel over
//! ROWS: each row reads and writes a disjoint slice, so splitting them across
//! threads gives bit-identical results to the serial loop -- the same property
//! the numerics gates rely on.
//!
//! The pool is lazy and persistent: creating it once amortises thread startup
//! across an inference run. Work below the thresholds stays serial, because
//! for a short panel the scheduling cost is larger than the saved cycles and
//! because the pipeline already overlaps host work with a dispatch.

use rayon::prelude::*;
use std::sync::OnceLock;

/// Do not split for fewer than this many elements.
const PAR_MIN_ITEMS: usize = 65_536;
/// Do not split into fewer than this many rows per worker.
const PAR_MIN_ROWS_PER_WORKER: usize = 8;

static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

fn pool() -> &'static rayon::ThreadPool {
    POOL.get_or_init(|| {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("npum-xdna-{i}"))
            .build()
            .expect("could not build the xdna host work pool")
    })
}

/// The number of threads the pool will use.
pub fn threads() -> usize {
    pool().current_num_threads()
}

/// Run `f` over disjoint row slices of `data`.
///
/// `f(row_index, row)` sees exactly the same slice, in the same order, as a
/// serial `for (i, row) in data.chunks_mut(row_len).enumerate()` loop. The
/// parallel path is only taken when there is enough independent work for it
/// to pay for itself; below that it is the serial loop verbatim.
pub fn for_each_row<T, F>(data: &mut [T], row_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    if row_len == 0 || data.len() < PAR_MIN_ITEMS {
        for (i, row) in data.chunks_mut(row_len.max(1)).enumerate() {
            f(i, row);
        }
        return;
    }
    let rows = data.len() / row_len;
    let workers = threads();
    if rows < PAR_MIN_ROWS_PER_WORKER * workers {
        for (i, row) in data.chunks_mut(row_len).enumerate() {
            f(i, row);
        }
        return;
    }
    pool().install(|| {
        data.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(i, row)| f(i, row));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_independent_and_in_order() {
        let mut data = vec![0u32; 4096 * 32];
        for_each_row(&mut data, 32, |i, row| {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (i * 31 + j) as u32;
            }
        });
        for i in 0..4096 {
            for j in 0..32 {
                assert_eq!(data[i * 32 + j], (i * 31 + j) as u32);
            }
        }
    }
}
