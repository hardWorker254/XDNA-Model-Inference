# SPDX-License-Identifier: Apache-2.0
"""The case table: what the golden transcripts cover.

This file IS the contract. Every case names one behaviour of the runtime
that must not change when the runtime moves from C++ to Rust, and says why
it matters. A case with no `why` is a case that should be deleted.

THE COVERAGE, and what it is for
---------------------------------

The plan asks for three things: the eight HTTP endpoints, the eleven
subcommands, and the error modes. The table below has all three, and each
group says which of the three it belongs to so that a gap is visible.

    * 11 subcommands: one case each, plus the second binary's spelling of
      the same verb where the binaries differ.
    * 8 endpoints: one case each, plus the wrong-family 404 for the two
      endpoints whose family is not the container's.
    * error modes: the refusals, the unrecognised options, the missing
      arguments, and the two flag semantics that cost a bug report.

THE MODELS
----------

One model per family, and the smallest one that runs:

    all-MiniLM-L6-v2      embed, tokenize
    vit-base-patch16-224  classify
    whisper-tiny          transcribe
    yolov8n-pose          pose
    mediapipe-hands       hands
    mediapipe-pose        mppose
    paddleocr-onnx        ocr
    nomic-embed-text-v1.5 the prompt table

A model that is NOT installed produces a refusal, and a transcript of a
refusal-because-the-model-is-missing is a transcript of the absence. Every
case that needs a model names it, and the harness SKIPS a case whose model
is not installed -- loudly, and never silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Http:
    """One HTTP request.

    `body` is either `None` (no body), a `dict` (sent as JSON), or a
    `tuple` of (field name, file path) for a multipart upload.
    """

    method: str
    path: str
    body: object = None
    content_type: str = "application/json"
    #: The model that must be SERVING for this case. The harness starts the
    #: right server for it.
    serves: str | None = None
    #: Whether the response's body is expected to be JSON.
    json_body: bool = True


@dataclass(frozen=True)
class Case:
    """One transcript."""

    #: The case's name. The golden's path is derived from it.
    id: str
    #: `cli` or `http`.
    kind: str
    #: The binary that runs it.
    binary: str
    #: The arguments, WITHOUT the program name.
    argv: tuple[str, ...] = ()
    #: The HTTP request, for the `http` cases.
    http: Http | None = None
    #: The models the case needs installed.
    needs: tuple[str, ...] = ()
    #: The models for whose path this case needs a COMPILED DESIGN SET.
    #:
    #: The C++ runtime asks for one before it will run the embedding, the
    #: classification and the transcription paths -- and it asks even when
    #: the container holds row-major host-side operands and every operation
    #: would run on the CPU. The check is unconditional, so a CPU-only
    #: container cannot be run without a design set that the container will
    #: not use.
    #:
    #: A design set is the OUTPUT of MLIR-AIE, not a download: it is a
    #: compiled AIE program with an xclbin and a set of instruction
    #: binaries beside a design.json. There is no way to fetch one, and
    #: there is no way to make one without the toolchain.
    #:
    #: So these cases are SKIPPED, with the reason printed, rather than
    #: recorded as a golden of a refusal: a refusal that says "there is no
    #: design set" is a transcript of this machine's inventory, not of the
    #: runtime.
    design_models: tuple[str, ...] = ()
    #: What this case pins down.
    why: str = ""
    #: Whether the case is expected to fail, and with which exit code.
    expect_code: int | None = None
    #: Whether the harness adds `--root <the fixture root>` to the command
    #: line.
    #:
    #: It does for every case but the ones that test the no-argument
    #: behaviour, and the reason is not tidiness: the C++ runtime prints the
    #: help when it is given NO arguments at all -- `argc == 1` -- and the
    #: harness's `--root` would make that `argc == 3`, which is a different
    #: command with a different meaning. That command fell through to the
    #: embedding path, where an empty model name and exactly one installed
    #: container resolve to that container -- and the refusal that came back
    #: was about the container, not about the no-argument behaviour the case
    #: was written to pin.
    inject_root: bool = True
    #: Tags, for the runner's selection.
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def family(self) -> str:
        """The family this case belongs to."""
        return self.id.split("/", 1)[0]


#: The fixtures directory.
FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: The text input the embed and tokenize cases use.
TEXT_INPUT = str(FIXTURES / "in.txt")

#: The image the vision cases use.
IMAGE = str(FIXTURES / "bus.jpg")

#: The image the OCR case uses.
OCR_IMAGE = str(FIXTURES / "ocr_text.png")

#: The audio the transcribe case uses.
AUDIO = str(FIXTURES / "tone.wav")

#: Where the embed case writes its vector.
EMBED_OUT = str(FIXTURES / "out.f32")


CASES: list[Case] = [
    # ---------------------------------------------------------------- help
    Case(
        id="help/npuembeddings",
        kind="cli",
        binary="npuembeddings",
        argv=("--help",),
        why="the help text names every flag and every verb this binary owns; "
            "a reader who typed the wrong binary learns it here",
        tags=("help",),
    ),
    Case(
        id="help/npuaudio",
        kind="cli",
        binary="npuaudio",
        argv=("--help",),
        why="the same help, with the audio verbs and without the others",
        tags=("help",),
    ),
    Case(
        id="help/npuimage",
        kind="cli",
        binary="npuimage",
        argv=("--help",),
        why="the same help, with the image verbs and without the others",
        tags=("help",),
    ),
    Case(
        id="help/no-args-npuembeddings",
        kind="cli",
        binary="npuembeddings",
        argv=(),
        inject_root=False,
        why="no verb at all prints the help on stdout; a reader who typed "
            "the program's name gets the help and not an error",
        tags=("help",),
    ),

    # ------------------------------------------------------------ subcommands
    Case(
        id="sub/list",
        kind="cli",
        binary="npuembeddings",
        argv=("list",),
        why="the catalogue table: its columns, its states and its notes are "
            "what a reader checks before anything else. NOTE: the C++ "
            "runtime's `list` IGNORES `--root` and reports the tree its own "
            "binary lives in, so this case compares the two runtimes only "
            "when the oracle's default root holds the same containers as the "
            "fixture root. The Rust runtime honours `--root`, which is the "
            "fix",
        tags=("subcommand",),
    ),
    Case(
        id="sub/embed",
        kind="cli",
        binary="npuembeddings",
        argv=("embed", "all-MiniLM-L6-v2", TEXT_INPUT),
        needs=("all-MiniLM-L6-v2",),
        design_models=("all-MiniLM-L6-v2",),
        why="the embedding path: the vector, its length and its norm",
        tags=("subcommand", "numeric"),
    ),
    Case(
        id="sub/embed-out",
        kind="cli",
        binary="npuembeddings",
        argv=("embed", "all-MiniLM-L6-v2", TEXT_INPUT, EMBED_OUT),
        needs=("all-MiniLM-L6-v2",),
        design_models=("all-MiniLM-L6-v2",),
        why="the same embedding, written to a file: the bytes are the "
            "contract and not the printed shape",
        tags=("subcommand", "numeric"),
    ),
    Case(
        id="sub/tokenize",
        kind="cli",
        binary="npuembeddings",
        argv=("tokenize", "all-MiniLM-L6-v2", TEXT_INPUT),
        needs=("all-MiniLM-L6-v2",),
        why="the tokenizer's ids: a table that disagrees with the packer's "
            "produces a correctly shaped vector of the wrong text",
        tags=("subcommand", "numeric"),
    ),
    Case(
        id="sub/transcribe",
        kind="cli",
        binary="npuaudio",
        argv=("transcribe", "whisper-tiny", AUDIO),
        needs=("whisper-tiny",),
        design_models=("whisper-tiny",),
        why="the transcript, and the status block that goes to stderr",
        tags=("subcommand", "numeric"),
    ),
    Case(
        id="sub/transcribe-json",
        kind="cli",
        binary="npuaudio",
        argv=("transcribe", "whisper-tiny", AUDIO, "--json"),
        needs=("whisper-tiny",),
        design_models=("whisper-tiny",),
        why="the OpenAI-shaped object: the body the server also returns",
        tags=("subcommand", "numeric"),
    ),
    Case(
        id="sub/classify",
        kind="cli",
        binary="npuimage",
        argv=("classify", "vit-base-patch16-224", IMAGE),
        needs=("vit-base-patch16-224",),
        design_models=("vit-base-patch16-224",),
        why="the label and the score",
        tags=("subcommand", "numeric"),
    ),
    Case(
        id="sub/classify-json",
        kind="cli",
        binary="npuimage",
        argv=("classify", "vit-base-patch16-224", IMAGE, "--json"),
        needs=("vit-base-patch16-224",),
        design_models=("vit-base-patch16-224",),
        why="the object the server's /v1/classify returns: the endpoint and "
            "the CLI must not drift",
        tags=("subcommand", "numeric"),
    ),
    Case(
        id="sub/pose",
        kind="cli",
        binary="npuimage",
        argv=("pose", "yolov8n-pose", IMAGE),
        needs=("yolov8n-pose",),
        why="the boxes, the joints and the scores",
        tags=("subcommand", "numeric"),
    ),
    Case(
        id="sub/pose-json",
        kind="cli",
        binary="npuimage",
        argv=("pose", "yolov8n-pose", IMAGE, "--json"),
        needs=("yolov8n-pose",),
        why="the object the server's /v1/pose returns",
        tags=("subcommand", "numeric"),
    ),
    Case(
        id="sub/hands",
        kind="cli",
        binary="npuimage",
        argv=("hands", "mediapipe-hands", IMAGE),
        needs=("mediapipe-hands",),
        why="the hands and their landmarks",
        tags=("subcommand", "numeric"),
    ),
    Case(
        id="sub/mppose",
        kind="cli",
        binary="npuimage",
        argv=("mppose", "mediapipe-pose", IMAGE),
        needs=("mediapipe-pose",),
        why="the people and their 33 keypoints",
        tags=("subcommand", "numeric"),
    ),
    Case(
        id="sub/ocr",
        kind="cli",
        binary="npuimage",
        argv=("ocr", "paddleocr-onnx", OCR_IMAGE),
        needs=("paddleocr-onnx",),
        why="the boxes, the text and the confidences",
        tags=("subcommand", "numeric"),
    ),
    Case(
        id="sub/add-bad-sha",
        kind="cli",
        binary="npuembeddings",
        argv=("add", "org/model", "deadbeef"),
        why="a sha256 that is not 64 hex characters is refused BEFORE any "
            "network call: a pin that can never match is worse than no pin, "
            "because it looks like a pin",
        expect_code=2,
        tags=("subcommand", "error"),
    ),
    Case(
        id="sub/add",
        kind="cli",
        binary="npuembeddings",
        argv=("add", "org/model"),
        why="the refusal for a repository that does not exist, and the "
            "warning that a missing sha256 means the weights are unverified",
        tags=("subcommand", "error", "network"),
    ),

    # --------------------------------------------------------------- errors
    Case(
        id="error/unknown-option",
        kind="cli",
        binary="npuembeddings",
        argv=("--no-such-flag",),
        why="an unrecognised option is refused BY NAME; the C++ parser used "
            "to fall through silently, and a flag that did nothing looked "
            "exactly like a flag that worked",
        expect_code=2,
        tags=("error",),
    ),
    Case(
        id="error/refused-by-name-host-gelu",
        kind="cli",
        binary="npuembeddings",
        argv=("embed", "all-MiniLM-L6-v2", TEXT_INPUT, "--host-gelu"),
        needs=("all-MiniLM-L6-v2",),
        why="the removed flags are refused by name, with the replacement "
            "named; a stale command line must not look like it worked",
        expect_code=2,
        tags=("error",),
    ),
    Case(
        id="error/refused-by-name-extra-ops",
        kind="cli",
        binary="npuembeddings",
        argv=("embed", "all-MiniLM-L6-v2", TEXT_INPUT, "--extra-ops"),
        needs=("all-MiniLM-L6-v2",),
        why="the same, for the other removed spelling",
        expect_code=2,
        tags=("error",),
    ),
    Case(
        id="error/verb-wrong-binary-transcribe",
        kind="cli",
        binary="npuembeddings",
        argv=("transcribe", "whisper-tiny", AUDIO),
        why="`transcribe` belongs to npuaudio; the refusal names the "
            "program to run and prints the roster",
        expect_code=2,
        tags=("error",),
    ),
    Case(
        id="error/verb-wrong-binary-embed",
        kind="cli",
        binary="npuaudio",
        argv=("embed", "all-MiniLM-L6-v2", TEXT_INPUT),
        why="`embed` belongs to npuembeddings",
        expect_code=2,
        tags=("error",),
    ),
    Case(
        id="error/verb-wrong-binary-classify",
        kind="cli",
        binary="npuembeddings",
        argv=("classify", "vit-base-patch16-224", IMAGE),
        why="`classify` belongs to npuimage",
        expect_code=2,
        tags=("error",),
    ),
    Case(
        id="error/unknown-model",
        kind="cli",
        binary="npuembeddings",
        argv=("embed", "no-such-model", TEXT_INPUT),
        why="a model that is not installed is named as missing, and the "
            "refusal says where it would look",
        expect_code=2,
        tags=("error",),
    ),
    Case(
        id="error/missing-argument-embed",
        kind="cli",
        binary="npuembeddings",
        argv=("embed",),
        why="a verb with too few arguments says what it needs",
        expect_code=2,
        tags=("error",),
    ),
    Case(
        id="error/missing-argument-serve",
        kind="cli",
        binary="npuembeddings",
        argv=("serve",),
        why="`serve` needs a model; the refusal comes before the listener "
            "opens",
        expect_code=2,
        tags=("error",),
    ),
    Case(
        id="error/wrong-family-container",
        kind="cli",
        binary="npuembeddings",
        argv=("embed", "whisper-tiny", TEXT_INPUT),
        needs=("whisper-tiny",),
        why="a Whisper container handed to the embedder is refused by name, "
            "and the refusal says which binary runs it",
        expect_code=2,
        tags=("error",),
    ),
    Case(
        id="error/serve-equals-port",
        kind="cli",
        binary="npuembeddings",
        argv=("serve", "all-MiniLM-L6-v2", "--serve=8080"),
        needs=("all-MiniLM-L6-v2",),
        why="`--serve=8080` is NOT accepted: the grammar has always been "
            "`--serve [PORT]` with a separate argument, and a new spelling "
            "that nobody asked for is a new spelling",
        expect_code=2,
        tags=("error",),
    ),
    Case(
        id="error/dev-unknown-generation",
        kind="cli",
        binary="npuembeddings",
        argv=("embed", "all-MiniLM-L6-v2", TEXT_INPUT, "--dev", "npu3"),
        needs=("all-MiniLM-L6-v2",),
        design_models=("all-MiniLM-L6-v2",),
        why="the generation is CARRIED into the design lookup, not defaulted: "
            "the refusal names `device npu3`, where a default would have "
            "named npu1 and a misspelling would have looked like a working "
            "command",
        expect_code=2,
        tags=("error",),
    ),
    Case(
        id="error/prefix-unknown-name",
        kind="cli",
        binary="npuembeddings",
        argv=("embed", "nomic-embed-text-v1.5", TEXT_INPUT, "--prefix", "no_such_prefix"),
        needs=("nomic-embed-text-v1.5",),
        design_models=("nomic-embed-text-v1.5",),
        why="an unknown task prefix is refused and the real options are "
            "listed; a wrongly-prefixed embedding is correctly shaped and "
            "correctly normed, so nothing downstream can tell it is wrong",
        expect_code=2,
        tags=("error",),
    ),
    Case(
        id="error/prefix-on-model-without-table",
        kind="cli",
        binary="npuembeddings",
        argv=("embed", "all-MiniLM-L6-v2", TEXT_INPUT, "--prefix", "search_document"),
        needs=("all-MiniLM-L6-v2",),
        design_models=("all-MiniLM-L6-v2",),
        why="a prefix for a model with no prompt table is refused too -- the "
            "same mistake in the other direction",
        expect_code=2,
        tags=("error",),
    ),

    # ------------------------------------------------------- flag semantics
    #
    # THESE RUN ON THE PATHS THAT NEED NO DESIGN SET, and that is not a
    # detail of the case table -- it is the whole reason they are written
    # this way.
    #
    # The first version of these three cases ran `embed`, and every one of
    # them recorded the SAME transcript: the refusal that says there is no
    # design set for this geometry. Which is to say, three cases that said
    # "the flag was honoured" recorded three copies of "the flag was never
    # read", and the harness could not tell them from three real passes.
    #
    # A case that cannot distinguish its own subject pins nothing. So the
    # flags are exercised where the answer is VISIBLE:
    #
    # * `tokenize` prints the token ids, padded to `--max-len`, and says so
    #   when a line was truncated -- so the flag's value is in the output;
    # * `ocr` reads `--npu-ops` BEFORE it needs anything from the array,
    #   and refuses an unknown code and an impossible one by name.
    Case(
        id="flags/last-occurrence-wins",
        kind="cli",
        binary="npuembeddings",
        argv=(
            "tokenize",
            "all-MiniLM-L6-v2",
            TEXT_INPUT,
            "--max-len",
            "8",
            "--max-len",
            "64",
        ),
        needs=("all-MiniLM-L6-v2",),
        why="the LAST occurrence of a flag wins, not the first: the output is "
            "padded to 64 tokens and nothing was truncated, where the first "
            "occurrence would have given 8 tokens and a note saying so. The "
            "C++ tree got this wrong once and `serve <whisper> --threads 8` "
            "answered with 24 threads",
        tags=("flags",),
    ),
    Case(
        id="flags/after-positionals",
        kind="cli",
        binary="npuembeddings",
        argv=("tokenize", "all-MiniLM-L6-v2", TEXT_INPUT, "--max-len", "8"),
        needs=("all-MiniLM-L6-v2",),
        why="a flag may follow the positional arguments: the output is "
            "truncated to 8 tokens and says how many lines were truncated",
        tags=("flags",),
    ),
    Case(
        id="flags/npu-ops-unknown-code",
        kind="cli",
        binary="npuimage",
        argv=("ocr", "paddleocr-onnx", OCR_IMAGE, "--npu-ops", "gemm,nonsense"),
        needs=("paddleocr-onnx",),
        why="an operation code that does not exist is refused by name, and "
            "the valid codes are listed; a misspelled code that was silently "
            "ignored would leave the operation on the host and the user "
            "would not know",
        expect_code=2,
        tags=("flags",),
    ),
    Case(
        id="flags/npu-ops-arch-refusal",
        kind="cli",
        binary="npuimage",
        argv=("ocr", "paddleocr-onnx", OCR_IMAGE, "--npu-ops", "gelu"),
        needs=("paddleocr-onnx",),
        why="a code that EXISTS but that this architecture has no operation "
            "for is refused by name too, and the refusal says what the two "
            "graphs do contain -- otherwise `--npu-ops gelu` would be read as "
            "a request that was honoured",
        expect_code=2,
        tags=("flags",),
    ),

    # ------------------------------------------------------------- endpoints
    Case(
        id="http/root",
        kind="http",
        binary="npuembeddings",
        http=Http("GET", "/", serves="all-MiniLM-L6-v2"),
        needs=("all-MiniLM-L6-v2",),
        why="the root: what the server says it is",
        tags=("endpoint",),
    ),
    Case(
        id="http/health",
        kind="http",
        binary="npuembeddings",
        http=Http("GET", "/health", serves="all-MiniLM-L6-v2"),
        needs=("all-MiniLM-L6-v2",),
        why="liveness, and what the model accepts",
        tags=("endpoint",),
    ),
    Case(
        id="http/models",
        kind="http",
        binary="npuembeddings",
        http=Http("GET", "/v1/models", serves="all-MiniLM-L6-v2"),
        needs=("all-MiniLM-L6-v2",),
        why="the catalogue over HTTP",
        tags=("endpoint",),
    ),
    Case(
        id="http/labels",
        kind="http",
        binary="npuimage",
        http=Http("GET", "/v1/labels", serves="vit-base-patch16-224"),
        needs=("vit-base-patch16-224",),
        why="the classifier's whole vocabulary, so a client can turn an id "
            "into a name without the container",
        tags=("endpoint",),
    ),
    Case(
        id="http/embeddings",
        kind="http",
        binary="npuembeddings",
        http=Http(
            "POST",
            "/v1/embeddings",
            body={"input": ["a golden transcript is a contract"]},
            serves="all-MiniLM-L6-v2",
        ),
        needs=("all-MiniLM-L6-v2",),
        why="the embedding endpoint's body, 1:1 with the CLI",
        tags=("endpoint", "numeric"),
    ),
    Case(
        id="http/classify",
        kind="http",
        binary="npuimage",
        http=Http(
            "POST",
            "/v1/classify",
            body=("image", IMAGE),
            content_type="multipart/form-data",
            serves="vit-base-patch16-224",
        ),
        needs=("vit-base-patch16-224",),
        why="the classification endpoint's body, 1:1 with `classify --json`",
        tags=("endpoint", "numeric"),
    ),
    Case(
        id="http/pose",
        kind="http",
        binary="npuimage",
        http=Http(
            "POST",
            "/v1/pose",
            body=("image", IMAGE),
            content_type="multipart/form-data",
            serves="yolov8n-pose",
        ),
        needs=("yolov8n-pose",),
        why="the pose endpoint's body, 1:1 with `pose --json`",
        tags=("endpoint", "numeric"),
    ),
    Case(
        id="http/transcriptions",
        kind="http",
        binary="npuaudio",
        http=Http(
            "POST",
            "/v1/audio/transcriptions",
            body={"text": "a golden transcript is a contract"},
            serves="whisper-tiny",
        ),
        needs=("whisper-tiny",),
        why="the speech endpoint's body",
        tags=("endpoint", "numeric"),
    ),
    Case(
        id="http/wrong-family-embeddings",
        kind="http",
        binary="npuaudio",
        http=Http(
            "POST",
            "/v1/embeddings",
            body={"input": "x"},
            serves="whisper-tiny",
        ),
        needs=("whisper-tiny",),
        why="a wrong path is a 404 that NAMES THE RIGHT ONE, and never "
            "another endpoint's answer",
        expect_code=404,
        tags=("endpoint", "error"),
    ),
    Case(
        id="http/wrong-family-labels",
        kind="http",
        binary="npuembeddings",
        http=Http("GET", "/v1/labels", serves="all-MiniLM-L6-v2"),
        needs=("all-MiniLM-L6-v2",),
        why="the same, for the labels endpoint on an embedder",
        expect_code=404,
        tags=("endpoint", "error"),
    ),
]


def by_id() -> dict[str, Case]:
    """Every case, by id."""
    return {case.id: case for case in CASES}


def with_tag(tag: str) -> list[Case]:
    """The cases carrying a tag."""
    return [case for case in CASES if tag in case.tags]
