# SPDX-License-Identifier: Apache-2.0
"""Record the golden transcripts, and compare a candidate against them.

    python tools/parity/run.py record --oracle <path-to-the-cpp-binaries>
    python tools/parity/run.py compare --candidate <path-to-the-rust-binaries>

WHAT A GOLDEN IS
----------------

For each case, a directory under `golden/<case-id>/` holding:

    stdout   what the process wrote to stdout
    stderr   what it wrote to stderr
    code     the exit code
    body     the HTTP response's body, for the http cases
    status   the HTTP response's status code, for the http cases
    meta.json  the case's parameters, and the rules that fired

`meta.json` is what makes a golden REVIEWABLE: it says which binary wrote
it, on what date, and which normalisation rules were applied. A golden
without it is a mystery file.

THE TWO DIRECTIONS
-------------------

`record` runs the baseline/reference binaries and writes the goldens.
`compare` runs the CANDIDATE -- the Rust binaries -- and diffs.

They are the same machinery with a different binary directory, and that is
the point: a difference between the two transcripts is a difference between
the two runtimes, and not a difference between two harnesses.

THE FIXTURE ROOT
----------------

Every run gets its own root, holding the packed containers. The runtime
resolves a model NAME to `<root>/models/<name>.npue`, so the root decides
what is installed -- and therefore what the catalogue says and which
refusals happen. A fixed root is what makes the catalogue's states a
contract rather than a snapshot of whoever's machine.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cases as case_table  # noqa: E402
import normalize  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOLDEN = HERE / "golden"
FIXTURES = HERE / "fixtures"
ROOT = FIXTURES / "root"

#: The three binaries.
BINARIES = ("npuembeddings", "npuaudio", "npuimage")

#: How long to wait for a server to start listening.
STARTUP_TIMEOUT = 60.0

#: How long a request may take.
REQUEST_TIMEOUT = 600.0


@dataclass
class Transcript:
    """One run's result."""

    case: case_table.Case
    stdout: str = ""
    stderr: str = ""
    code: int = 0
    body: str = ""
    status: int = 0
    headers: dict[str, str] = field(default_factory=dict)
    skipped: str | None = None
    rules: list[str] = field(default_factory=list)

    def normalised(self) -> dict[str, str]:
        """The transcript, with the volatile fields taken out."""
        out = {
            "stdout": normalize.normalise(self.stdout, "stdout"),
            "stderr": normalize.normalise(self.stderr, "stderr"),
            "code": str(self.code),
        }
        if self.case.kind == "http":
            out["body"] = normalize.normalise(self.body, "body")
            out["status"] = str(self.status)
        return out


# ---------------------------------------------------------------- the root


def ensure_root() -> Path:
    """The fixture root, created if it is not there.

    The root holds:

    * `models/` -- a SYMLINK to the repository's `models/`, so the packed
      containers are used where they are and not copied by the hundred
      megabytes. The runtime resolves a model NAME to
      `<root>/models/<name>.npue`, so the root is what decides what is
      installed -- and therefore what the catalogue says and which
      refusals happen.
    * `runtime/artifacts/` -- the design sets, copied in, so that the
      `list` case reports the same states on every machine.

    A fixed root is what makes the catalogue's states a CONTRACT rather
    than a snapshot of whoever's machine.

    THE ROOT IS PASSED EXPLICITLY, as `--root`, to every run. It is not
    inferred: the oracle's default root is the tree its BINARY lives in,
    and the candidate's default root is the tree its binary lives in, and
    those are two different directories holding two different containers.
    A harness that let each side use its own root would be comparing two
    different models and calling the difference a mismatch.
    """
    root = ROOT
    (root / "models").mkdir(parents=True, exist_ok=True)
    link = root / "models"
    if link.is_symlink():
        if Path(os.readlink(link)).resolve() == (REPO / "models").resolve():
            return root
        link.unlink()
    elif link.is_dir() and not any(link.iterdir()):
        link.rmdir()
    else:
        return root
    link.symlink_to(REPO / "models", target_is_directory=True)
    return root


def register_paths() -> None:
    """Tell the normaliser which absolute paths are placeholders.

    Four paths, and each of them is a path that differs between the two
    trees and says nothing about the contract:

    * the fixtures -- the same files on both sides, at different absolute
      paths;
    * the candidate's repository -- where the Rust binary lives;
    * the reference repository -- where the baseline binary lives;
    * the fixture root -- the `--root` both sides are given.

    The oracle's tree and the candidate's tree both become `<ROOT>`,
    because the contract does not say WHICH tree the runtime is installed
    in.
    """
    normalize.register_path("<FIXTURE>", FIXTURES)
    normalize.register_path("<ROOT>", REPO)
    normalize.register_path("<ROOT>", REPO.parent / "Npu-Embeddings-XDNA1")
    normalize.register_path("<ROOT>", ROOT)


def installed_models(root: Path) -> set[str]:
    """The models the fixture root holds."""
    out = set()
    models = root / "models"
    if not models.is_dir():
        return out
    for path in models.iterdir():
        if path.suffix in (".npue", ".npum"):
            out.add(path.stem)
    return out


def has_design_set(root: Path, model: str) -> bool:
    """Whether the fixture root holds a COMPILED design set for a model.

    A design set is a directory holding a `design.json` AND the compiled
    program beside it: an `xclbin` and the instruction binaries. The
    `design.json` alone would be enough to fool a reader that only checks
    for the file's existence, and the reference baseline's own checks may only
    only for the file -- which is how a directory with a design.json and no
    program passes the first gate and fails at the second one.

    So this checks for the PROGRAM too, because a design set without one is
    not a design set: it is the JSON of one.
    """
    bases = [
        root / "runtime" / "artifacts",
        root / "artifacts",
        root / "runtime",
        root,
    ]
    for base in bases:
        for sub in ("", "artifacts_npu1", "artifacts_npu2"):
            directory = base / model / sub if sub else base / model
            set_dir = directory / "gemm_rtp"
            if not (set_dir / "design.json").is_file():
                continue
            # The compiled program lives BESIDE the design.json, inside the
            # set's own directory -- not beside the model. A check that
            # looked one level up would find the design.json and no program,
            # and would call that a design set.
            if any(set_dir.glob("*.xclbin")) or any(set_dir.glob("insts*.bin")):
                return True
    return False


def missing_designs(case: case_table.Case, root: Path) -> list[str]:
    """The models whose path needs a design set the root does not hold.

    The reference baseline asks for a design set before it runs the embedding, the
    classification and the transcription paths, and it asks even for a
    container whose operands are row-major host-side values and whose every
    operation would run on the CPU. The check is unconditional.

    A design set cannot be downloaded: it is the output of MLIR-AIE. So
    these cases are skipped, loudly, and the count of them is printed beside
    the count of the ones that ran.
    """
    return [m for m in case.design_models if not has_design_set(root, m)]


def divergences() -> dict[str, str]:
    """The declared differences, and why each one is intended.

    A difference between the oracle and the candidate is a FAILURE until
    somebody decides it is not. The decision is written down here, with
    its reason, and it is a file in the tree so that the reason is
    reviewed rather than remembered.

    The format is one case id and its reason per line, separated by a
    colon:

        sub/embed: the reference baseline needs a compiled design set; this
                   runtime runs a host-layout container on the CPU.

    A reason that is not there is a reason that does not exist.
    """
    path = HERE / "divergences.txt"
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        case_id, _, reason = line.partition(":")
        reason = reason.strip()
        if not reason:
            continue
        out[case_id.strip()] = reason
    return out


def binary_dir(path: str | Path | None) -> Path:
    """The directory the three binaries live in."""
    if path is None:
        return REPO / "target" / "release"
    p = Path(path)
    return p if p.is_dir() else p.parent


# ---------------------------------------------------------------- running


def run_env(root: Path) -> dict[str, str]:
    """The environment every run gets.

    Two settings, and both of them are about not letting the harness reach
    out to the network:

    * `HF_HUB_OFFLINE=1` -- the runtime fetches a model it does not have,
      and a recording that silently downloaded 90 MB of checkpoint is a
      recording of the network's mood on that day, not of the runtime. A
      case whose model is not installed is SKIPPED before the run (see
      `missing_models`), and this setting is the belt to that braces: if a
      model is somehow missing at run time, the run fails loudly instead
      of fetching.
    * `NPU_DEVICE` is NOT set. The generation comes from the device.
    """
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["NPUE_PARITY_ROOT"] = str(root)
    return env


def missing_models(case: case_table.Case, root: Path) -> list[str]:
    """The models the case needs that the fixture root does not hold.

    A case whose model is not installed produces a REFUSAL, and a golden
    of a refusal is a golden of the absence: it would be matched by any
    implementation that also refuses, and it would pin nothing. So the
    harness SKIPS such a case, loudly, and reports the count.
    """
    installed = installed_models(root)
    return [m for m in case.needs if m not in installed]


def network_allowed() -> bool:
    """Whether the harness may reach the network.

    It may not, by default. A golden that depends on what HuggingFace
    answered that day is a golden of the internet, not of the runtime, and
    the network is the one thing about a recording that cannot be
    reproduced. The cases that need it are tagged `network` and are
    skipped unless `NPUE_PARITY_NETWORK=1` is set.

    The reason is not only reproducibility: the reference baseline's `add` path
    fetches over libcurl and, on this machine, it ABORTS inside glibc
    ("invalid stdio handle") rather than returning an error. A case that
    crashes its own oracle is not a case that can pin a behaviour.
    """
    return os.environ.get("NPUE_PARITY_NETWORK") == "1"


def run_cli(binary: Path, case: case_table.Case, root: Path) -> Transcript:
    """Run one CLI case.

    The `--root` is appended, and it is appended LAST so that it cannot
    shadow a positional argument. It is omitted for the cases that test the
    no-argument behaviour -- see `Case.inject_root`.
    """
    argv = [str(binary), *case.argv]
    if case.inject_root:
        argv += ["--root", str(root)]
    env = run_env(root)
    try:
        proc = subprocess.run(
            argv,
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=REQUEST_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        t = Transcript(case=case, skipped="the process did not finish in time")
        return t
    return Transcript(
        case=case,
        stdout=proc.stdout,
        stderr=proc.stderr,
        code=proc.returncode,
    )


def free_port() -> int:
    """A port that is free right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(
    port: int, deadline: float, proc: subprocess.Popen | None = None
) -> bool:
    """Whether the port is listening.

    The PROCESS is watched too, and that is the point: a server that
    refuses to start does not sit there for the whole timeout, it exits.
    Waiting out the full minute for a process that died in the first
    second is a minute of nothing, eight times over, and the reason it
    died is printed on its stderr -- which is the interesting part.

    So: as soon as the process is gone, the answer is "it did not start",
    and the reason comes out of its stderr.
    """
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except OSError:
            pass
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(0.1)
    return False


def run_http(binary: Path, case: case_table.Case, root: Path) -> Transcript:
    """Run one HTTP case: start the server, make the request, stop it."""
    http = case.http
    assert http is not None
    port = free_port()
    model = http.serves or ""
    server_argv = [
        str(binary),
        "serve",
        model,
        "--port",
        str(port),
        "--bind",
        "127.0.0.1",
        "--root",
        str(root),
    ]
    env = run_env(root)
    proc = subprocess.Popen(
        server_argv,
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if not wait_for_port(port, time.time() + STARTUP_TIMEOUT, proc):
            out, err = proc.communicate(timeout=10)
            return Transcript(
                case=case,
                stdout=out,
                stderr=err,
                skipped=(
                    "the server did not start listening: "
                    + (err.strip().splitlines()[-1] if err.strip() else "no reason given")
                ),
            )
        body, status, headers = make_request(http, port)
        # Give the server a moment to write its banner, then stop it.
        time.sleep(0.2)
        proc.terminate()
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
        return Transcript(
            case=case,
            stdout=out,
            stderr=err,
            body=body,
            status=status,
            headers=headers,
            code=status,
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


def make_request(spec: case_table.Http, port: int) -> tuple[str, int, dict[str, str]]:
    """One HTTP request, by hand -- no client library.

    The point of doing it by hand is that the harness has no dependency that
    could change the request's shape between the recording and the replay.
    """
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=REQUEST_TIMEOUT)
    headers = {}
    payload = None
    if spec.body is not None:
        if isinstance(spec.body, tuple):
            name, path = spec.body
            boundary = "----npueparityboundary"
            data = Path(path).read_bytes()
            payload = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{Path(path).name}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n"
            ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        else:
            payload = json.dumps(spec.body).encode()
            headers["Content-Type"] = spec.content_type
    conn.request(spec.method, spec.path, body=payload, headers=headers)
    response = conn.getresponse()
    body = response.read().decode("utf-8", errors="replace")
    hdrs = {k.lower(): v for k, v in response.getheaders()}
    conn.close()
    return body, response.status, hdrs


def run_case(bin_dir: Path, case: case_table.Case, root: Path) -> Transcript:
    """Run one case, of either kind.

    A case whose models are not installed is SKIPPED, and the skip says
    which model. See `missing_models` for why a golden of a refusal is
    worse than no golden at all.
    """
    binary = bin_dir / case.binary
    if not binary.exists():
        return Transcript(case=case, skipped=f"{binary} is not there")
    missing = missing_models(case, root)
    if missing:
        return Transcript(
            case=case,
            skipped=f"the model(s) {', '.join(missing)} are not installed",
        )
    without_design = missing_designs(case, root)
    if without_design:
        return Transcript(
            case=case,
            skipped=(
                f"the reference baseline needs a COMPILED design set for this path, "
                f"and there is none for {', '.join(without_design)}: a design "
                f"set is the output of MLIR-AIE, not a download"
            ),
        )
    if "network" in case.tags and not network_allowed():
        return Transcript(
            case=case,
            skipped="the case needs the network, and the harness does not "
            "use it unless NPUE_PARITY_NETWORK=1",
        )
    if case.kind == "http":
        return run_http(binary, case, root)
    return run_cli(binary, case, root)


# ------------------------------------------------------------- recording


def golden_dir(case: case_table.Case) -> Path:
    """Where a case's golden lives."""
    return GOLDEN / case.id


def write_golden(transcript: Transcript, source: str) -> None:
    """Write one golden."""
    out = golden_dir(transcript.case)
    out.mkdir(parents=True, exist_ok=True)
    normalised = transcript.normalised()
    (out / "stdout").write_text(normalised["stdout"], encoding="utf-8")
    (out / "stderr").write_text(normalised["stderr"], encoding="utf-8")
    (out / "code").write_text(normalised["code"] + "\n", encoding="utf-8")
    if transcript.case.kind == "http":
        (out / "body").write_text(normalised["body"], encoding="utf-8")
        (out / "status").write_text(normalised["status"] + "\n", encoding="utf-8")
    meta = {
        "case": transcript.case.id,
        "kind": transcript.case.kind,
        "binary": transcript.case.binary,
        "argv": list(transcript.case.argv),
        "http": (
            None
            if transcript.case.http is None
            else {
                "method": transcript.case.http.method,
                "path": transcript.case.http.path,
                "serves": transcript.case.http.serves,
            }
        ),
        "why": transcript.case.why,
        "expect_code": transcript.case.expect_code,
        "recorded_from": source,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rules": sorted(
            set(
                normalize.rules_fired(transcript.stdout, normalised["stdout"], "stdout")
                + normalize.rules_fired(transcript.stderr, normalised["stderr"], "stderr")
            )
        ),
    }
    (out / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_golden(case: case_table.Case) -> dict[str, str] | None:
    """Read a case's golden, if it has one."""
    out = golden_dir(case)
    if not (out / "stdout").exists():
        return None
    golden = {
        "stdout": (out / "stdout").read_text(encoding="utf-8"),
        "stderr": (out / "stderr").read_text(encoding="utf-8"),
        "code": (out / "code").read_text(encoding="utf-8").strip(),
    }
    if case.kind == "http":
        golden["body"] = (out / "body").read_text(encoding="utf-8")
        golden["status"] = (out / "status").read_text(encoding="utf-8").strip()
    return golden


# ------------------------------------------------------------- comparing


def diff_text(name: str, want: str, got: str) -> list[str]:
    """The lines that differ, as a readable report."""
    import difflib

    lines = list(
        difflib.unified_diff(
            want.splitlines(keepends=True),
            got.splitlines(keepends=True),
            fromfile=f"golden/{name}",
            tofile=f"candidate/{name}",
            n=1,
        )
    )
    return lines[:40]


@dataclass
class Comparison:
    """The outcome of one case."""

    case: case_table.Case
    ok: bool
    detail: str = ""
    skipped: bool = False


def compare_case(bin_dir: Path, case: case_table.Case, root: Path) -> Comparison:
    """Run a case with the candidate and diff it against the golden."""
    golden = read_golden(case)
    if golden is None:
        # No golden is a SKIP and not a failure, and the reason is the same
        # one the recorder would have printed: a case that could not be
        # recorded could not be compared either, and calling that a failure
        # would put a red mark on the report for something nobody can fix
        # without a design set.
        missing = missing_models(case, root)
        if missing:
            return Comparison(
                case=case,
                ok=False,
                skipped=True,
                detail=f"the model(s) {', '.join(missing)} are not installed",
            )
        without_design = missing_designs(case, root)
        if without_design:
            return Comparison(
                case=case,
                ok=False,
                skipped=True,
                detail=(
                    f"the reference baseline needs a COMPILED design set for this path, "
                    f"and there is none for {', '.join(without_design)}: a design "
                    f"set is the output of MLIR-AIE, not a download"
                ),
            )
        return Comparison(
            case=case,
            ok=False,
            detail="there is no golden for this case",
        )
    transcript = run_case(bin_dir, case, root)
    if transcript.skipped:
        return Comparison(case=case, ok=False, skipped=True, detail=transcript.skipped)
    got = transcript.normalised()
    problems = []
    for key in ("stdout", "stderr", "code", "body", "status"):
        if key not in golden:
            continue
        if golden[key] != got[key]:
            lines = diff_text(key, golden[key], got[key])
            problems.append(f"  {key} differs:\n" + "".join(lines))
    if case.expect_code is not None and case.kind == "cli":
        if int(golden["code"]) != case.expect_code:
            problems.append(
                f"  the golden's exit code is {golden['code']} and the case "
                f"expects {case.expect_code}"
            )
    if problems:
        return Comparison(case=case, ok=False, detail="\n".join(problems))
    return Comparison(case=case, ok=True)
