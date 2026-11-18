//! The backend choice: where a run's arithmetic happens.
//!
//! # ONE FLAG, TWO NAMES, AND A REFUSAL FOR ANYTHING ELSE
//!
//! `--backend cpu` and `--backend vulkan` are the two this build knows.
//! A third name is refused BY NAME, with the two real ones listed --
//! because a flag that is accepted and then ignored is the failure this
//! runtime refuses everywhere else: an ignored flag changes what gets
//! built without saying so.
//!
//! # THE DEFAULT IS THE HOST
//!
//! The host's arithmetic is the reference's arithmetic, byte for byte, and that
//! is the contract the golden transcripts check. So `--backend` is not
//! needed to get the reference's numbers: it is needed to ask for DIFFERENT
//! ones, on a different machine, and to be told when that machine is not
//! there.
//!
//! # WHAT THE GPU RUNS, AND WHAT IT DOES NOT
//!
//! The GPU backend's arithmetic is the TEXT family's: the projections,
//! the LayerNorm, the GELU and the softmax, in the host's f32. The
//! attention's two halves -- QK^T and softmax.V -- stay on the host,
//! because the family's score rows are laid out in the family's own
//! packed layout and the shader's attention takes a different one.
//!
//! And the ViT and the Whisper are NOT this arithmetic: their LayerNorm
//! accumulates in double and takes the container's epsilon, and their
//! GELU is the exact erf rather than the degree-8 polynomial the shaders
//! implement. Those families refuse the flag by name.

use crate::flags::Args;
use npum_model::Ops;

/// The backend a run's arithmetic goes through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// The host's own kernels: the reference, and the reference's numbers.
    Cpu,
    /// The GPU.
    Vulkan,
}

impl Backend {
    /// The name the flag's value carries.
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Vulkan => "vulkan",
        }
    }

    /// The names this build knows.
    pub const NAMES: [&'static str; 2] = ["cpu", "vulkan"];

    /// A backend by name, or nothing.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "cpu" => Some(Backend::Cpu),
            "vulkan" => Some(Backend::Vulkan),
            _ => None,
        }
    }
}

/// The arithmetic a run uses.
pub struct Choice {
    /// Which backend the command line asked for.
    pub backend: Backend,
    /// The arithmetic itself.
    ///
    /// A box of the trait, because the family does not care WHICH
    /// implementation it is talking to -- only that the four operations
    /// come back with numbers.
    pub ops: Box<dyn Ops>,
}

/// The choice the command line made, or the refusal if it cannot be had.
///
/// The GPU is opened HERE, once: the choice is made before the run
/// starts, and a machine with no Vulkan device is refused before it
/// prints a status block that would claim the GPU.
///
/// `--vk-ops` selects which operations go to the GPU (like `--npu-ops`
/// selects operations for the array) and implies the Vulkan backend.
/// Given alone it offloads the named ops; the rest stay on the host.
pub fn choose(args: &Args) -> Result<Choice, String> {
    let vk_ops = args.value("--vk-ops");
    let backend_name = args.value("--backend");

    if let Some(list) = vk_ops {
        if backend_name == Some("cpu") {
            return Err(
                "--vk-ops selects operations for the Vulkan backend, but --backend cpu was \
                 given. Drop --backend cpu, or drop --vk-ops to run everything on the host."
                    .to_string(),
            );
        }
        if let Some(name) = backend_name {
            if Backend::parse(name) != Some(Backend::Vulkan) {
                return Err(refuse_unknown(name));
            }
        }
        let codes = crate::flags::parse_vk_ops(list)?;
        if codes.is_empty() {
            return Err(
                "--vk-ops: no Vulkan op named. Valid codes: [gemm, layn, gelu, softm]. \
                 Name at least one op to offload, or drop --vk-ops to run on the host."
                    .to_string(),
            );
        }
        let gemm = codes.contains(&"gemm");
        let layn = codes.contains(&"layn");
        let gelu = codes.contains(&"gelu");
        let softm = codes.contains(&"softm");
        let ops = npum_vulkan::ops::VulkanOps::with_ops(gemm, layn, gelu, softm);
        if !ops.has_gpu() {
            return Err(refuse_no_gpu());
        }
        return Ok(Choice {
            backend: Backend::Vulkan,
            ops: Box::new(ops),
        });
    }

    let Some(name) = backend_name else {
        return Ok(Choice {
            backend: Backend::Cpu,
            ops: Box::new(npum_cpu::ops::CpuOps::best()),
        });
    };
    let Some(backend) = Backend::parse(name) else {
        return Err(refuse_unknown(name));
    };
    match backend {
        Backend::Cpu => Ok(Choice {
            backend,
            ops: Box::new(npum_cpu::ops::CpuOps::best()),
        }),
        Backend::Vulkan => {
            let ops = npum_vulkan::ops::VulkanOps::new();
            if !ops.has_gpu() {
                return Err(refuse_no_gpu());
            }
            Ok(Choice {
                backend,
                ops: Box::new(ops),
            })
        }
    }
}

/// The refusal for a `--backend` value that is not a backend.
///
/// It is checked in ONE place -- before any verb runs -- so that a
/// misspelled backend is refused the same way for every verb, and so
/// that it is refused before a model is resolved or a device is opened.
pub fn refuse_unknown(name: &str) -> String {
    format!(
        "--backend '{name}' is not a backend this build knows. The backends are: {}. A name \
         that is not one of them is refused rather than run on the host, because a command line \
         that says one backend about a run that used another is a command line that lies.",
        Backend::NAMES.join(", ")
    )
}

/// The refusal for a GPU that is not on this machine.
pub fn refuse_no_gpu() -> String {
    "--backend vulkan: no Vulkan device was opened on this machine, so the GPU backend has \
     nothing to run on. The host's arithmetic is the same network computed by the host's own \
     kernels -- and the GPU's f32 is not the host's f32, so the run would be a DIFFERENT set of \
     numbers rather than the same one. Refusing is the honest answer: a run that said 'vulkan' \
     and computed on the host is a run that lied. Check the driver with `vulkaninfo --summary`."
        .to_string()
}

/// The refusal for a family whose arithmetic does not go through the seam.
///
/// The GPU backend computes the TEXT family's arithmetic: the projections,
/// the LayerNorm, the GELU and the softmax, in the host's f32. The other
/// families do their own arithmetic -- the ViT's LayerNorm accumulates in
/// double and takes the container's epsilon, the Whisper's GELU is the
/// exact erf -- and the GPU's shaders do not implement those.
///
/// So those families refuse the flag by name. Accepting it and running
/// the host would make the command line say something the run did not do.
pub fn refuse_family(family: npum_families::Family) -> String {
    format!(
        "--backend vulkan: the {} family does not run on the GPU in this runtime. The GPU \
         backend's arithmetic is the TEXT family's -- the projections, the LayerNorm, the GELU \
         and the softmax -- and this family's LayerNorm and GELU are the exact variants, which \
         the GPU's shaders do not implement. Accepting the flag and running the host would make \
         the command line say something the run did not do.",
        family.as_str()
    )
}

/// The refusal for a family without a GPU path, if the flag named one.
///
/// The check is by NAME: `--backend vulkan` or `--vk-ops` on a family
/// that cannot run on the GPU is refused, and the default host path says
/// nothing.
pub fn refuse_if_vulkan(args: &Args, family: npum_families::Family) -> Option<String> {
    if args.value("--backend") == Some("vulkan") || args.value("--vk-ops").is_some() {
        Some(refuse_family(family))
    } else {
        None
    }
}

/// The refusal for a server family whose engine cannot run on the GPU.
///
/// The embedding endpoint's server engine can use the Vulkan Ops seam;
/// other families' server engines run on the host.
pub fn refuse_server() -> String {
    "--backend vulkan/--vk-ops: this family's HTTP server engine runs on the host. The GPU's \
     arithmetic is not wired into this server path yet -- a server that answered with the \
     GPU's vectors while its engine ran on the host would be a server that lied. (The \
     embedding endpoint supports --vk-ops/--backend vulkan.)"
        .to_string()
}
