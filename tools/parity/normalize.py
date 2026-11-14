# SPDX-License-Identifier: Apache-2.0
"""The volatile-field rules: what the comparison is allowed to ignore.

A golden transcript is only useful if it is COMPARABLE. Two things get in
the way, and they are different kinds of thing:

1. **Noise.** A duration, a timestamp, a port number, a temporary path. These
   differ between two runs of the SAME binary, so they say nothing about
   whether the new binary matches the old one.

2. **Renames.** The container's extension changed from `.npue` to `.npum`, and
   the runtime lives in a different directory. These differ between the
   ORACLE and the CANDIDATE for a reason that is known and recorded, so they
   are normalised -- and the normalisation is EXPLICIT, one named rule per
   thing, so that a reader can see what is being forgiven.

Everything else is compared exactly. A rule that is not on this list is not
applied, and a difference that is not on this list is a failure.

THE RULES ARE DATA, and they are printed with the comparison result. A
normaliser that changed silently would make every gate weaker without anyone
noticing, and the whole point of the golden harness is that a passing gate
means something.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    """One normalisation rule.

    `name` is what the comparison report calls it, and it is printed when the
    rule fires -- so a report that says "12 differences, 12 of them timing"
    is a report a reader can judge.
    """

    name: str
    pattern: re.Pattern[str]
    replacement: str
    why: str
    #: When true, the rule applies only to the named stream ("stdout",
    #: "stderr", "body"). Empty means it applies to all of them.
    streams: tuple[str, ...] = field(default_factory=tuple)


def _rule(name, pattern, replacement, why, streams=()):
    return Rule(
        name=name,
        pattern=re.compile(pattern),
        replacement=replacement,
        why=why,
        streams=tuple(streams),
    )


#: The rules, in the order they are applied.
#
# The order matters: the ROOT rule runs before the CONTAINER rule, because
# the ROOT rule would otherwise swallow the container's extension inside a
# path and leave `.npue` behind as a fragment of a replaced path.
RULES: list[Rule] = [
    _rule(
        "field-padding",
        re.compile(r"  +(?=\d+(?:\.\d+)? (?:ms|s)\b|\d+\.\d+ ?%|\d+\.\d+(?= x\b))"),
        " ",
        "the padding of a fixed-width numeric field, in front of a DURATION, "
        "a SHARE or a RATIO. The status block prints those with %8.1f and "
        "%5.1f, so the padding depends on how many digits the value has -- "
        "and the value is a measurement, which differs between two runs of "
        "the same binary. The lookahead is what keeps this rule off the "
        "aligned TABLES, whose columns are a contract of their own: a table "
        "cell is padded too, and padding it here would flatten the table",
    ),
    _rule(
        "root",
        re.compile(r"(?m)^(\s*(?:Models \(root |root ))(/[^\s)]*)"),
        r"\1<ROOT>",
        "the repository's absolute path differs between the oracle's tree and "
        "the candidate's, and the path itself is not the contract",
    ),
    _rule(
        "container-extension",
        re.compile(r"\.npue\b"),
        ".npum",
        "the container format was renamed from .npue to .npum; the magic "
        "bytes changed and the reader accepts both, so the extension in a "
        "message is a rename and not a difference",
    ),
    _rule(
        "container-extension-path",
        re.compile(r"\.npue\b"),
        ".npum",
        "the container format was renamed from .npue to .npum",
    ),
    _rule(
        "timing-ms",
        re.compile(r"\b\d+(?:\.\d+)? ms\b"),
        "<MS> ms",
        "a duration; it differs between two runs of the same binary. The "
        "integer form is here because a whole number of milliseconds is "
        "still a duration: 'wall was 132 ms' moves just as much as "
        "'0.13 s' does. TWO OR MORE SPACES in front of it are absorbed: the "
        "status block prints its numbers with %8.1f, so the padding depends "
        "on how many digits the value has, and the value is a measurement",
    ),
    _rule(
        "timing-s",
        re.compile(r"\b\d+\.\d+ s\b"),
        "<S> s",
        "a duration; it differs between two runs of the same binary. Two or "
        "more spaces in front of it are absorbed, for the reason the "
        "timing-ms rule gives",
    ),
    _rule(
        "timing-split",
        re.compile(r"\b\d+\.\d+(?= / )"),
        "<MS>",
        "the per-stage split of a duration; same reason as timing-ms",
    ),
    _rule(
        "split-last-stage",
        re.compile(r"(?<=\w )\d+\.\d+(?=\))"),
        "<MS>",
        "the LAST stage of a split: it has no ' / ' after it, because the "
        "closing bracket is there instead, and it moves just as much as the "
        "others. The look-behind wants a WORD before the space, and that is "
        "what keeps it off a number that is not a stage -- the normalisation "
        "constants, '(x/255 - 0.485)/0.229', are preceded by a minus and not "
        "by a word",
    ),
    _rule(
        "reference-timing",
        re.compile(r"(torch's )\d+\.\d+"),
        r"\1<MS>",
        "the reference timing a comparison quotes -- '28.5x torch's 20.5'. It "
        "is a measurement of another implementation on this machine, and it "
        "moves every run",
    ),
    _rule(
        "timings-json",
        re.compile(r'("(?:total|front_end|front_end_s|encoder_s|head_s|'
                   r'network|decode|detector|db|crop|recogniser|setup|'
                   r'palm|nms|landmarks|postprocess)"'
                   r':\s*)\d+\.\d+'),
        r"\1<MS>",
        "the timings block of a JSON answer; same reason as timing-ms. The "
        "key list is the WHOLE set of measured fields the answers carry: a "
        "field that is a measurement and is not on the list is a golden that "
        "cannot be reproduced by the binary that recorded it",
    ),
    _rule(
        "throughput",
        re.compile(r"\b\d+\.\d+(?= seq/s)"),
        "<RATE>",
        "a rate: texts per second. Same reason as timing-ms -- it is a "
        "duration divided by something, and it moves every run",
    ),
    _rule(
        "percentage",
        re.compile(r"\b\d+\.\d+ ?%"),
        "<PCT>%",
        "a share of a duration; same reason as timing-ms, and it moves "
        "whenever the durations it is a share of move. Two or more spaces "
        "in front of it are absorbed: %5.1f pads the share to a fixed width "
        "and the padding depends on the value's digits",
    ),
    _rule(
        "ratio",
        re.compile(r"\b\d+\.\d+(?= ?x\b)"),
        "<RATIO>",
        "a ratio of two durations -- the pipelining achieved. Same reason as "
        "timing-ms, padding included",
    ),
    _rule(
        "port",
        re.compile(r"(?m)(\b(?:127\.0\.0\.1|localhost):)\d{2,5}\b"),
        r"\1<PORT>",
        "the port the harness picked; it is free when the transcript is "
        "recorded and occupied when it is replayed",
    ),
    _rule(
        "tmp-path",
        re.compile(r"/tmp/[A-Za-z0-9._-]+"),
        "<TMP>",
        "a temporary path; the harness names one per run",
    ),
    _rule(
        "buffer-alignment",
        re.compile(r"(?m)(bo-align\s+last data buffer aligned to )\d+( B)(?: \(>= 2 MB\))?"),
        r"\1<ALIGN>\2",
        "the alignment of a buffer's ADDRESS. It is not a number the runtime "
        "computed about the model: it is where the allocator happened to put "
        "a buffer, and the ORACLE'S OWN RUNS disagree with each other -- "
        "4096, 8192, 16384 and 2097152 bytes for the same command, in the "
        "same tree, on the same machine. The '(>= 2 MB)' that follows it is "
        "part of the value: it says the value is at least two megabytes, and "
        "it goes with the value",
    ),
    _rule(
        "runtime-name",
        re.compile(r"(?m)^(NpuEmbeddings )(?:C\+\+|Rust) runtime"),
        r"\1runtime",
        "the implementation word in the runtime's own banner. The C++ tree "
        "says 'C++ runtime' and this tree says 'Rust runtime', because a "
        "runtime that calls itself C++ is lying about what it is. The word "
        "names the implementation and not the contract",
    ),
    _rule(
        "pid",
        re.compile(r"\bpid[= ]\d+\b"),
        "pid=<PID>",
        "a process id",
    ),
]


#: The absolute paths that are replaced by a placeholder.
#:
#: A LIST of pairs and not a map, because one placeholder can stand for
#: several paths -- `<ROOT>` is both trees, the oracle's and the
#: candidate's -- and a map would keep only the last one.
#:
#: The ORDER matters and it is not incidental: the fixtures live INSIDE the
#: repository, so the repository's path is a PREFIX of the fixtures' path.
#: Replacing the shorter one first would leave the fixtures' path as
#: `<ROOT>/tools/parity/fixtures/in.txt`, and the two runtimes would then
#: print different placeholders for the same file. The pairs are therefore
#: sorted longest-first at substitution time, so a caller does not have to
#: get the order right.
PATHS: list[tuple[str, str]] = []


def register_path(placeholder: str, path: str | Path) -> None:
    """Register one absolute path and the placeholder that replaces it."""
    if not path:
        return
    PATHS.append((placeholder, str(path)))


def substitute_paths(text: str) -> str:
    """Replace the registered paths, longest first."""
    for placeholder, path in sorted(PATHS, key=lambda kv: len(kv[1]), reverse=True):
        if path:
            text = text.replace(path, placeholder)
    return text


def normalise(text: str, stream: str = "stdout") -> str:
    """Apply every rule that applies to `stream`."""
    text = substitute_paths(text)
    for rule in RULES:
        if rule.streams and stream not in rule.streams:
            continue
        text = rule.pattern.sub(rule.replacement, text)
    return text


def rules_fired(before: str, after: str, stream: str = "stdout") -> list[str]:
    """Which rules fired, by name.

    The comparison report prints this, so a reader can see that a difference
    was forgiven rather than missed.
    """
    fired = []
    for rule in RULES:
        if rule.streams and stream not in rule.streams:
            continue
        if rule.pattern.search(before) and rule.pattern.sub(rule.replacement, before) != before:
            fired.append(rule.name)
    return fired


def rules_documentation() -> str:
    """The rules, as a table a reader can check against the intent."""
    lines = ["| rule | what it forgives |", "|---|---|"]
    for rule in RULES:
        lines.append(f"| `{rule.name}` | {rule.why} |")
    return "\n".join(lines)
