//! D3D12's shader compiler: DXC, never FXC by accident.
//!
//! wgpu translates each pipeline's naga module to HLSL and compiles it
//! with the instance's compiler. Its `Auto` choice falls back to FXC
//! (`d3dcompiler_47.dll`, on every Windows) when no `dxcompiler.dll` loads,
//! and FXC can't compile the symbol shader in a minute on an Adreno X1
//! (#141): the watchdog then loses the device. So Orbis finds DXC itself,
//! in this order, and says which it took:
//!
//! 1. `WGPU_DX12_COMPILER`, as wgpu reads it (`dxc`, `fxc`, `static-dxc`).
//! 2. `dxcompiler.dll` beside the executable, where a Windows build ships
//!    it with its `dxil.dll` (DXC's release, `bin/<arch>/`).
//! 3. The newest Windows SDK's (`Windows Kits\10\bin\<version>\<arch>`),
//!    on a machine that builds Orbis.
//! 4. A `dxcompiler.dll` on `PATH`.
//!
//! Only without any of them does it compile with FXC, with a warning.
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

/// The compiler `descriptor` uses, set unless the environment chose one;
/// what it is, for the adapter's description.
pub(crate) fn choose(descriptor: &mut wgpu::InstanceDescriptor) -> String {
    if let Some(choice) = std::env::var_os("WGPU_DX12_COMPILER") {
        return format!("{} (WGPU_DX12_COMPILER)", choice.to_string_lossy());
    }
    let found = beside_executable()
        .map(|dir| (dir, "beside the executable"))
        .or_else(|| sdk().map(|dir| (dir, "from the Windows SDK")))
        .or_else(|| on_path().map(|dir| (dir, "on PATH")));
    let compiler = &mut descriptor.backend_options.dx12.shader_compiler;
    match found {
        Some((dir, whence)) => {
            preload_validator(&dir);
            let dll = dir.join(DXC);
            let described = format!("DXC {whence}, {}", dll.display());
            *compiler = wgpu::Dx12Compiler::DynamicDxc {
                dxc_path: dll.to_string_lossy().into_owned(),
            };
            described
        }
        None => {
            *compiler = wgpu::Dx12Compiler::Fxc;
            "FXC".into()
        }
    }
}

/// Says loudly that D3D12 compiles with FXC, once the adapter is known to
/// be D3D12's.
pub(crate) fn warn_if_fxc(compiler: &str) {
    if compiler.starts_with("FXC") || compiler.starts_with("fxc") {
        eprintln!(
            "orbis: WARNING: D3D12 is compiling shaders with FXC (d3dcompiler_47.dll): no \
             {DXC} beside the executable, in a Windows SDK or on PATH. FXC is slow enough \
             on some drivers (Adreno X1) that pipelines miss the compile watchdog and the \
             map stops. Ship {DXC} and {DXIL} from DXC's release beside the executable."
        );
    }
}

fn has_dxc(dir: &Path) -> bool {
    dir.join(DXC).is_file()
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
            let dir = path.join(arch);
            has_dxc(&dir).then_some((version, dir))
        })
        .collect();
    versions.sort();
    versions.pop().map(|(_, dir)| dir)
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
