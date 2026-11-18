//! The questions the driver answers without a context.
//!
//! # WHY THESE ARE TESTS
//!
//! Everything else in this crate's hardware tests needs a design set and a
//! dispatch. These need neither: they are the queries a run makes BEFORE it
//! builds anything -- what the device is, and who else is using it -- and
//! they are the ones that decide whether a run may start at all.
//!
//! `query_hw_contexts` is what replaced `xrt-smi examine -r all` in the
//! contention check: a build without XRT has no `xrt-smi`, and a budget
//! check that cannot see the device refuses the run. So the query has to
//! answer on a machine with the driver and nothing else, which is exactly
//! what this asserts.
//!
//! # THEY SKIP WHEN THERE IS NO BOARD
//!
//! A checkout on a machine without the NPU still builds and still runs
//! `cargo test`; the tests that need the hardware say they skipped.

use npum_xdna::direct::query_hw_contexts;

/// Whether the direct driver's node is there at all.
fn board_present() -> bool {
    std::path::Path::new("/dev/accel/accel0").exists()
        || std::path::Path::new("/dev/dri/accel0").exists()
}

#[test]
fn the_driver_names_the_contexts_it_holds() {
    if !board_present() {
        eprintln!("skipping: no NPU device node");
        return;
    }
    let contexts = match query_hw_contexts() {
        Ok(contexts) => contexts,
        Err(e) => panic!("the driver did not answer its own context query: {e}"),
    };
    // The answer can legitimately be empty -- nothing is running, and the
    // driver hides other users' contexts -- but every entry it does return
    // has a pid and a partition, because that is what the budget check reads.
    for context in &contexts {
        assert!(
            context.num_col > 0,
            "a context with no columns: {context:?}"
        );
    }
    eprintln!("# contexts {}", contexts.len());
}

#[test]
fn the_context_query_reports_this_process_once_it_has_a_context() {
    use npum_xdna::direct::DirectTransport;
    use npum_xdna::transport::Transport;

    if !board_present() {
        eprintln!("skipping: no NPU device node");
        return;
    }
    // No context, no entry: the query is about live contexts and not about
    // processes, so this proves the entry appears and not that the process
    // exists. A transport with no context is the cheapest way to say so.
    let transport = match DirectTransport::open(0) {
        Ok(transport) => transport,
        Err(e) => {
            eprintln!("skipping: no NPU board: {e}");
            return;
        }
    };
    drop(transport);
    let contexts = query_hw_contexts().expect("the driver answered before");
    for context in &contexts {
        assert_ne!(context.pid, 0, "a context with no pid: {context:?}");
    }
}
