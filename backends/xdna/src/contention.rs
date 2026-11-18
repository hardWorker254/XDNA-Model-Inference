//! Is anything else using the array right now?
//!
//! # WHAT THIS IS FOR
//!
//! Every design set is one xclbin and one hw_context, and the npu1 driver
//! allows six of them. A run that asks for more than the device has fails
//! with a raw driver error (`DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed
//! (err=-22)`), which says nothing about which of the two design sets was
//! the problem. So the budget is checked BEFORE anything is built, and the
//! answer names the number.
//!
//! # THE PARSE
//!
//! `xrt-smi examine -r all` prints each hw_context as a FOUR-line record in
//! a pipe-delimited table, with the fields for one context spread down the
//! rows rather than across them:
//!
//! ```text
//! |PID                 |Ctx ID     |Submissions |Migrations  |Err  |Priority |
//! |Process Name        |Status     |Completions |Suspensions |     |GOPS     |
//! |Memory Usage        |Instr BO   |            |            |     |FPS      |
//! ...
//! |63804               |144        |7233        |0           |0    |Normal   |
//! |npuembeddings.exe        |Active     |7232        |0           |     |N/A      |
//! |1032 MB             |608 KB     |            |            |     |N/A      |
//! |                    |           |            |            |     |N/A      |
//! ```
//!
//! So: a row whose first cell is all digits starts a record, the next row's
//! first two cells are the process name and the status, and the row after
//! that carries the memory. The header and the separator rows are not
//! special-cased: anything that is not digits-then-name-then-memory is
//! skipped, and if the whole parse yields nothing that is reported as a
//! FAILURE and not as "no contention". A format change must fail closed
//! too.
//!
//! # THE DRIVER FIRST, `xrt-smi` SECOND
//!
//! The table above is XRT's, and a build without XRT cannot ask for it --
//! which made the budget check a check that a direct-transport run could
//! only pass with `--allow-contention`. The driver answers the same
//! question by itself (`QUERY_HW_CONTEXTS`: every context, its pid, its
//! partition, its submissions and its errors), so that is asked first and
//! the table is the fallback.
//!
//! # THE LIMIT IS A LABELLED CONSTANT
//!
//! Measured on this NPU1 (Ryzen 7 PRO 8845HS, amdxdna 2.26 on Linux): a
//! SEVENTH concurrent hw_context fails to construct, so the budget is six.
//! The driver's `hwctx_limit` is also 6, but XRT exposes no query for
//! either number, so this is a constant that says where it came from
//! instead of pretending it read it.

use std::process::Command;

/// One hw_context another process holds.
#[derive(Debug, Clone)]
pub struct ForeignContext {
    /// The process that holds it.
    pub pid: u32,
    /// The process's name.
    pub process: String,
    /// The context's status.
    pub status: String,
    /// The memory it holds.
    pub memory: String,
}

/// What the survey found.
#[derive(Debug, Clone, Default)]
pub struct ContentionReport {
    /// Whether the survey tool ran at all.
    pub tool_ran: bool,
    /// Why it did not, when it did not.
    pub failure: String,
    /// The contexts other processes hold.
    pub foreign: Vec<ForeignContext>,
}

/// The device's hw_context budget, and where the number came from.
#[derive(Debug, Clone)]
pub struct ContextBudget {
    /// How many concurrent contexts the device allows; 0 when unknown.
    pub limit: usize,
    /// Where the number came from, in the C++'s words.
    pub source: String,
}

/// The device's context budget.
///
/// `NPU_HWCTX_LIMIT` overrides it, and it says so in the source: a number
/// that came from the environment is not a measurement.
pub fn context_budget() -> ContextBudget {
    if let Ok(value) = std::env::var("NPU_HWCTX_LIMIT") {
        if let Ok(limit) = value.trim().parse::<usize>() {
            if limit > 0 {
                return ContextBudget {
                    limit,
                    source: "NPU_HWCTX_LIMIT environment".to_string(),
                };
            }
        }
    }
    let device = running_device();
    if device.is_empty() || device == "npu1" {
        return ContextBudget {
            limit: 6,
            // The WORDS are the oracle's: this line is printed before a run
            // and compared byte for byte against the C++ runtime's.
            source: "measured on NPU1 (six concurrent hw_contexts; XRT exposes no \
                     usable-count query)"
                .to_string(),
        };
    }
    ContextBudget {
        limit: 0,
        source: format!("unknown for device '{device}' -- XRT exposes no usable-count query"),
    }
}

/// The device this build is running on: `npu1`, `npu2` or "".
///
/// From the ENVIRONMENT and not from the hardware: the environment says
/// which generation this run is FOR, and a run that overrides it is saying
/// something about the design set it will load.
pub fn running_device() -> String {
    if let Ok(device) = std::env::var("NPU_DEVICE") {
        if !device.is_empty() {
            return device;
        }
    }
    if let Ok(npu2) = std::env::var("NPU2") {
        if !npu2.is_empty() && npu2 != "0" {
            return "npu2".to_string();
        }
    }
    "npu1".to_string()
}

/// Survey the contexts other processes hold.
///
/// The driver first: it answers without XRT, and it answers for the whole
/// device. `xrt-smi` is the fallback, for the machines where the driver's
/// query is refused (`QUERY_HW_CONTEXTS` is newer than the node).
pub fn survey_contexts() -> ContentionReport {
    if let Some(report) = from_driver() {
        return report;
    }
    from_xrt_smi()
}

/// The survey the driver answers.
fn from_driver() -> Option<ContentionReport> {
    let contexts = crate::direct::query_hw_contexts().ok()?;
    let own = std::process::id();
    let mut report = ContentionReport {
        tool_ran: true,
        failure: String::new(),
        foreign: Vec::new(),
    };
    for context in contexts.iter().filter(|context| context.pid != own) {
        let busy = context.submissions > context.completions;
        report.foreign.push(ForeignContext {
            pid: context.pid,
            // The driver's per-context entry carries no process name; the
            // name is in the array query, and a name is not needed to say
            // whether the device has room.
            process: format!("ctx {} col {}", context.start_col, context.num_col),
            status: if busy { "Active" } else { "Idle" }.to_string(),
            memory: String::new(),
        });
    }
    Some(report)
}

/// The survey `xrt-smi` answers.
fn from_xrt_smi() -> ContentionReport {
    let mut report = ContentionReport::default();
    let Some(text) = run_capture() else {
        report.failure =
            "neither the driver's GET_INFO context query nor `xrt-smi examine -r all` \
             answered"
                .to_string();
        return report;
    };
    report.tool_ran = true;
    let own = std::process::id();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i + 2 < lines.len() {
        let Some(row) = cells(lines[i]) else {
            i += 1;
            continue;
        };
        if row.is_empty() || !all_digits(&row[0]) {
            i += 1;
            continue;
        }
        let pid: u32 = match row[0].parse() {
            Ok(p) => p,
            Err(_) => {
                i += 1;
                continue;
            }
        };
        let name = cells(lines[i + 1]).map(|c| c.first().cloned().unwrap_or_default());
        let status = cells(lines[i + 1]).and_then(|c| c.get(1).cloned());
        let memory = cells(lines[i + 2]).map(|c| c.first().cloned().unwrap_or_default());
        if pid != own {
            report.foreign.push(ForeignContext {
                pid,
                process: name.unwrap_or_default(),
                status: status.unwrap_or_default(),
                memory: memory.unwrap_or_default(),
            });
        }
        i += 3;
    }
    report
}

/// Check the budget and print the report.
///
/// Returns false when the run must not proceed.
pub fn require_context_budget(
    report: &ContentionReport,
    requested: usize,
    allow_override: bool,
) -> bool {
    let budget = context_budget();
    let foreign = report.foreign.len();

    // A missing data source is not a negative reading: if we cannot see the
    // live contexts, we cannot claim there is room. Refuse the run rather
    // than collide with a leftover process and surface a raw driver error.
    if !report.tool_ran {
        eprintln!();
        eprintln!(
            "  !! CANNOT VERIFY THE NPU CONTEXT BUDGET: {}",
            report.failure
        );
        eprintln!("     An absent data source is not a negative reading (tasks/0040).");
        if budget.limit > 0 {
            eprintln!(
                "     This run needs {requested} concurrent hw_context(s) of a {}-context budget ({}).",
                budget.limit, budget.source
            );
        } else {
            eprintln!(
                "     This run needs {requested} concurrent hw_context(s); the budget is {}.",
                budget.source
            );
        }
        if allow_override {
            eprintln!("  !! --allow-contention given: continuing without being able to check.");
            eprintln!();
            return true;
        }
        eprintln!();
        eprintln!(
            "Refusing to build {requested} hw_context(s) without knowing the device has room. Make `xrt-smi examine -r all` work, close other NPU"
        );
        eprintln!("  processes, or pass --allow-contention to proceed anyway.");
        eprintln!();
        return false;
    }

    // The tool ran. A generation with no known budget (npu2) cannot be
    // compared, so report the state and proceed rather than invent a limit.
    if budget.limit == 0 {
        eprintln!();
        eprintln!(
            "  npu        {requested} requested context(s), {foreign} foreign; budget {}",
            budget.source
        );
        return true;
    }

    let total = requested + foreign;
    eprintln!();
    eprintln!(
        "  npu        context budget -- {requested} requested + {foreign} foreign = {total} of {} ({})",
        budget.limit, budget.source
    );
    if total <= budget.limit {
        return true;
    }

    eprintln!(
        "  !! NPU CONTEXT BUDGET EXCEEDED -- this run needs {requested} hw_context(s), the device allows {} ({}), and {foreign} are already held by other processes:",
        budget.limit, budget.source
    );
    for context in &report.foreign {
        eprintln!(
            "       pid {:<8} {:<24} {:<8} {}",
            context.pid, context.process, context.status, context.memory
        );
    }
    if allow_override {
        eprintln!(
            "  !! --allow-contention given: continuing anyway. A context construction may still fail with a raw driver error."
        );
        eprintln!();
        return true;
    }
    eprintln!();
    eprintln!("  Refusing BEFORE any hw_context is built. Close the other process(es), or pass");
    eprintln!("  --allow-contention to proceed anyway.");
    eprintln!();
    false
}

/// Run `xrt-smi examine -r all` and return its output.
fn run_capture() -> Option<String> {
    let candidates = ["/opt/xilinx/xrt/bin/xrt-smi", "xrt-smi"];
    for exe in candidates {
        let Ok(output) = Command::new(exe).args(["examine", "-r", "all"]).output() else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        if !text.contains('|') {
            continue;
        }
        return Some(text);
    }
    None
}

/// Split a `|a |b |c |` row into its cells. Empty for a non-row.
///
/// The table is INDENTED under "HW Contexts:", so the leading whitespace
/// has to go first.
fn cells(raw: &str) -> Option<Vec<String>> {
    let line = raw.trim();
    if line.is_empty() || !line.starts_with('|') {
        return None;
    }
    let mut out = Vec::new();
    let mut rest = &line[1..];
    while let Some(end) = rest.find('|') {
        out.push(rest[..end].trim().to_string());
        rest = &rest[end + 1..];
    }
    Some(out)
}

/// Whether a cell is all digits.
fn all_digits(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}
