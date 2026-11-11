//! `cargo xtask dxc`: DXC's two DLLs, from its pinned release, beside the
//! binaries in `target/debug` and `target/release`, where D3D12 builds
//! look for them first (`orb-hal-wgpu`'s `src/dx12.rs`).

use std::path::Path;
use std::process::Command;

use sha2::{Digest, Sha256};

const VERSION: &str = "v1.9.2609";
const ZIP: &str = "dxc_2026_09_29.zip";
/// The release asset's digest, as GitHub reports it.
const ZIP_SHA256: &str = "ad31b1fc8443175d204f77a611fdb3ef2ec42759bdc2f1167368de24a4a7e7f1";
pub(crate) const DLLS: [&str; 2] = ["dxcompiler.dll", "dxil.dll"];

const HELP: &str = "\
cargo xtask dxc [--arch arm64|x64]

  Fetches DXC's release (pinned) into target/dxc and copies its
  dxcompiler.dll and dxil.dll for <arch> (default this machine's) into
  target/debug and target/release, beside the binaries built there.
";

pub(crate) fn run(args: &[String]) -> Result<(), String> {
    let mut arch = match std::env::consts::ARCH {
        "aarch64" => "arm64",
        _ => "x64",
    }
    .to_string();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--arch" => {
                arch.clone_from(args.next().ok_or("dxc: --arch takes arm64 or x64")?);
                if !["arm64", "x64"].contains(&arch.as_str()) {
                    return Err(format!("dxc: unknown arch {arch}\n\n{HELP}"));
                }
            }
            "help" | "--help" => {
                eprint!("{HELP}");
                return Ok(());
            }
            other => return Err(format!("dxc: unknown argument {other}\n\n{HELP}")),
        }
    }
    // Not canonicalized: Windows's canonical `\\?\D:\…` paths are lost on
    // tar.
    let target = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("dxc: xtask sits in the repository")?
        .join("target");
    let cache = target.join("dxc").join(VERSION);
    std::fs::create_dir_all(&cache).map_err(|e| format!("{}: {e}", cache.display()))?;
    let zip = cache.join(ZIP);
    if sha256(&zip).as_deref() != Some(ZIP_SHA256) {
        let url = format!(
            "https://github.com/microsoft/DirectXShaderCompiler/releases/download/{VERSION}/{ZIP}"
        );
        eprintln!("dxc: fetching {url}");
        let bytes = crate::fixtures::get(&url)?;
        let hash = crate::fixtures::hex(&Sha256::digest(&bytes));
        if hash != ZIP_SHA256 {
            return Err(format!(
                "dxc: {ZIP} hashes to {hash}, not the pinned {ZIP_SHA256}"
            ));
        }
        std::fs::write(&zip, bytes).map_err(|e| format!("{}: {e}", zip.display()))?;
    }
    // bsdtar reads zips: macOS's tar, and Windows's own tar.exe.
    let members: Vec<String> = DLLS.iter().map(|d| format!("bin/{arch}/{d}")).collect();
    crate::run(
        Command::new("tar")
            .arg("-xf")
            .arg(&zip)
            .arg("-C")
            .arg(&cache)
            .args(&members),
    )
    .map_err(|e| format!("dxc: extracting {members:?} from {ZIP}: {e}"))?;
    for profile in ["debug", "release"] {
        let dir = target.join(profile);
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for (member, dll) in members.iter().zip(DLLS) {
            let to = dir.join(dll);
            std::fs::copy(cache.join(member), &to).map_err(|e| format!("{}: {e}", to.display()))?;
        }
        eprintln!("dxc: DXC {VERSION} ({arch}) in {}", dir.display());
    }
    Ok(())
}

fn sha256(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(crate::fixtures::hex(&Sha256::digest(&bytes)))
}
