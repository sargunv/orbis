//! D3D12's shader compiler: DXC, never FXC by accident.
//!
//! wgpu translates each pipeline's naga module to HLSL and compiles it
//! with the instance's compiler. Its `Auto` choice falls back to FXC
//! (`d3dcompiler_47.dll`, on every Windows) when no `dxcompiler.dll` loads,
//! and FXC can't compile the symbol shader in a minute on an Adreno X1
//! (#141): the watchdog then loses the device. So Orbis finds DXC itself,
//! in this order, and says which it took:
//!
//! 1. `WGPU_DX12_COMPILER`, as wgpu reads it: `dxc` (wgpu loads
//!    `dxcompiler.dll` by name) or `fxc`. Other values (`auto`, and
//!    `staticdxc`, which this build doesn't link) fall through to the rest.
//! 2. `dxcompiler.dll` beside the executable, with its `dxil.dll`, from
//!    DXC's release (`bin/<arch>/`): `cargo xtask dxc` puts the pinned
//!    release's beside the binaries in `target/`.
//! 3. The newest Windows SDK's (`Windows Kits\10\bin\<version>\<arch>`),
//!    on a machine that builds Orbis.
//! 4. A `dxcompiler.dll` on `PATH`.
//!
//! A candidate counts only if it loads: one of another architecture (the
//! x64 DXC the Vulkan SDK puts on `PATH`, in an ARM64 process) would fail
//! wgpu's D3D12 instance and leave no adapter at all. Only without any
//! that loads does it compile with FXC, with a warning.
//!
//! Compiling DXIL ahead of time, as Orbis ships naga modules, would leave
//! no compiler to ship: DXIL doesn't depend on the GPU. wgpu 30 takes DXIL
//! through its passthrough shader modules, but then skips its own binding
//! translation: the DXIL's registers must match the root signature
//! wgpu-hal builds privately from each pipeline layout (samplers through
//! an index buffer into one heap, a special-constants root parameter, a
//! space per group), override constants aren't applied, and the vertex
//! stage's outputs must be cut to its fragment stage's inputs, so every
//! pipeline's variant would be enumerated at build time against a copy of
//! wgpu-hal's internals that no backend check catches drifting. A missed
//! variant would still need a compiler. DXC's DLLs cost 25.8 MB on ARM64
//! (24.3 MB on x64; 10.2 MB and 10.6 MB deflated) and take none of that.

use std::path::{Path, PathBuf};

/// The DLLs DXC needs: the compiler, and the validator that signs DXIL.
const DXC: &str = "dxcompiler.dll";
const DXIL: &str = "dxil.dll";

/// Why FXC is worth a warning.
const SLOW: &str = "FXC is slow enough on some drivers (Adreno X1) that pipelines miss the \
                    compile watchdog and the map stops.";

/// The compiler [`choose`] set.
pub(crate) struct Choice {
    /// Which it is and where it came from, for the adapter's description.
    pub(crate) described: String,
    /// What to say when it is FXC, once the adapter is known to be D3D12's.
    pub(crate) warning: Option<String>,
}

/// Sets the compiler `descriptor` uses, unless the environment chose one.
pub(crate) fn choose(descriptor: &mut wgpu::InstanceDescriptor) -> Choice {
    let compiler = &mut descriptor.backend_options.dx12.shader_compiler;
    match wgpu::Dx12Compiler::from_env() {
        Some(wgpu::Dx12Compiler::Fxc) => {
            *compiler = wgpu::Dx12Compiler::Fxc;
            return Choice {
                described: "FXC (WGPU_DX12_COMPILER)".into(),
                warning: Some(format!(
                    "orbis: WARNING: D3D12 is compiling shaders with FXC (d3dcompiler_47.dll), \
                     as WGPU_DX12_COMPILER asks. {SLOW}"
                )),
            };
        }
        Some(choice @ wgpu::Dx12Compiler::DynamicDxc { .. }) => {
            *compiler = choice;
            return Choice {
                described: format!("DXC (WGPU_DX12_COMPILER), {DXC} by name"),
                warning: None,
            };
        }
        Some(wgpu::Dx12Compiler::StaticDxc) => eprintln!(
            "orbis: WGPU_DX12_COMPILER asks for static DXC, which this build doesn't link; \
             looking for {DXC} instead"
        ),
        Some(wgpu::Dx12Compiler::Auto) | None => {}
    }
    let found = beside_executable()
        .map(|dir| (dir, "beside the executable"))
        .or_else(|| sdk().map(|dir| (dir, "from the Windows SDK")))
        .or_else(|| on_path().map(|dir| (dir, "on PATH")));
    match found {
        Some((dir, whence)) => {
            preload_validator(&dir);
            let dll = dir.join(DXC);
            let described = format!("DXC {whence}, {}", dll.display());
            *compiler = wgpu::Dx12Compiler::DynamicDxc {
                dxc_path: dll.to_string_lossy().into_owned(),
            };
            Choice {
                described,
                warning: None,
            }
        }
        None => {
            *compiler = wgpu::Dx12Compiler::Fxc;
            Choice {
                described: "FXC".into(),
                warning: Some(format!(
                    "orbis: WARNING: D3D12 is compiling shaders with FXC (d3dcompiler_47.dll): \
                     no {DXC} for this architecture loads from beside the executable, a \
                     Windows SDK or PATH. {SLOW} Put {DXC} and {DXIL} from DXC's release \
                     beside the executable (`cargo xtask dxc` does, in target/)."
                )),
            }
        }
    }
}

/// Whether `dir` has a DXC this process loads. One that loads stays
/// loaded, and wgpu's load of the same file finds it.
#[allow(unsafe_code)]
fn has_dxc(dir: &Path) -> bool {
    let dll = dir.join(DXC);
    // SAFETY: dxcompiler.dll is DXC, whose initialisation runs no code
    // that could conflict with this process's; it stays loaded.
    dll.is_file()
        && unsafe { libloading::Library::new(&dll) }
            .map(std::mem::forget)
            .is_ok()
}

fn beside_executable() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    // Test and example binaries run from `deps/` and `examples/`, a level
    // below the binaries the DLLs are put beside.
    [Some(dir), dir.parent()]
        .into_iter()
        .flatten()
        .find(|d| has_dxc(d))
        .map(Path::to_path_buf)
}

/// This process's architecture as the SDK names its directories.
fn sdk_arch() -> Option<&'static str> {
    match std::env::consts::ARCH {
        "aarch64" => Some("arm64"),
        "x86_64" => Some("x64"),
        "x86" => Some("x86"),
        _ => None,
    }
}

/// The newest installed Windows 10/11 SDK's DXC for this architecture.
fn sdk() -> Option<PathBuf> {
    let arch = sdk_arch()?;
    let kits = ["ProgramFiles(x86)", "ProgramFiles"]
        .into_iter()
        .filter_map(std::env::var_os)
        .map(|p| PathBuf::from(p).join(r"Windows Kits\10\bin"))
        .find(|p| p.is_dir())?;
    let mut versions: Vec<(Vec<u32>, PathBuf)> = std::fs::read_dir(&kits)
        .ok()?
        .filter_map(|e| {
            let path = e.ok()?.path();
            let name = path.file_name()?.to_str()?;
            let version = name
                .split('.')
                .map(str::parse)
                .collect::<Result<Vec<u32>, _>>()
                .ok()?;
            Some((version, path.join(arch)))
        })
        .collect();
    // Newest first, loading only the one taken.
    versions.sort();
    versions
        .into_iter()
        .rev()
        .map(|(_, dir)| dir)
        .find(|d| has_dxc(d))
}

fn on_path() -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?).find(|d| has_dxc(d))
}

/// Loads `dir`'s `dxil.dll`, if it has one, for the life of the process:
/// DXC loads its validator by name, which Windows doesn't look for beside
/// `dxcompiler.dll` but finds among the modules already loaded.
#[allow(unsafe_code)]
fn preload_validator(dir: &Path) {
    let dll = dir.join(DXIL);
    if !dll.is_file() {
        return;
    }
    // SAFETY: dxil.dll is DXC's validator, whose initialisation runs no
    // code that could conflict with this process's; it stays loaded.
    if let Ok(library) = unsafe { libloading::Library::new(&dll) } {
        std::mem::forget(library);
    }
}
