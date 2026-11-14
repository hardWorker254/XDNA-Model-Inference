#!/usr/bin/env python3
"""The oracle's EXACT vector: the float32 BYTES, not a decimal rendering.

The golden transcripts hold seven significant digits, which is enough to
catch a wrong arithmetic and not enough to prove a right one. The C++'s HTTP
endpoint also speaks `encoding_format: "base64"`, and that carries the float32
bytes themselves -- so this script is the bit-exactness oracle.

    python tools/parity/exact_vector.py --bin <the C++ binaries> --root <the fixture root>

It prints the base64 of one vector, and the Rust's own test compares against
those bytes.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import socket
import subprocess
import sys
import time
from pathlib import Path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, deadline: float, proc) -> bool:
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="the directory with the C++ binaries")
    ap.add_argument("--root", default=str(here / "fixtures" / "root"))
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--text", default="a golden transcript is a contract")
    ap.add_argument("--out", default="", help="write the base64 here instead of stdout")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    binary = Path(args.bin).resolve() / "npuembeddings"
    port = free_port()
    proc = subprocess.Popen(
        [
            str(binary),
            "serve",
            args.model,
            "--port",
            str(port),
            "--bind",
            "127.0.0.1",
            "--root",
            str(root),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_for_port(port, time.time() + 60.0, proc):
            print("the oracle's server did not start", file=sys.stderr)
            return 1
        body = json.dumps(
            {"input": args.text, "encoding_format": "base64"}
        ).encode()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        conn.request(
            "POST",
            "/v1/embeddings",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = response.read().decode()
        if response.status != 200:
            print(f"the oracle answered {response.status}: {payload}", file=sys.stderr)
            return 1
        vector = json.loads(payload)["data"][0]["embedding"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    if args.out:
        Path(args.out).write_text(vector)
    else:
        print(vector)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
