//! `cargo xtask pages`: publishes the web demo to the public GitHub Pages
//! site (the `gh-pages` branch, https://code.sargunv.dev/orbis/).

use std::path::Path;
use std::process::Command;

const BRANCH: &str = "gh-pages";

pub(crate) fn publish() -> Result<(), String> {
    crate::web::build()?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let status = Command::new("pnpm")
        .current_dir(root.join("web"))
        .arg("build")
        .status()
        .map_err(|e| format!("failed to run pnpm build: {e}"))?;
    if !status.success() {
        return Err("pnpm build failed".into());
    }
    let site = std::env::temp_dir().join(format!("orbis-pages-{}", std::process::id()));
    git(&root, &["fetch", "--quiet", "origin", BRANCH])?;
    git(
        &root,
        &[
            "worktree",
            "add",
            "--quiet",
            "--detach",
            &site.to_string_lossy(),
            &format!("origin/{BRANCH}"),
        ],
    )?;
    let result = (|| {
        // Old HTML and open tabs must still be able to fetch their assets.
        // Overlay the new build, retaining immutable hashes and the frozen
        // fixed-name assets from before the Vite migration.
        copy_tree(&root.join("web/dist"), &site)?;
        std::fs::write(site.join(".nojekyll"), "").map_err(|e| e.to_string())?;

        let head = Command::new("git")
            .current_dir(&root)
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .map_err(|e| e.to_string())?;
        let head = String::from_utf8_lossy(&head.stdout).trim().to_string();
        git(&site, &["add", "--all"])?;
        let message = format!("Publish the web demo from {head}");
        let diff = Command::new("git")
            .current_dir(&site)
            .args(["diff", "--cached", "--quiet", "--exit-code"])
            .status()
            .map_err(|e| format!("failed to run git diff: {e}"))?;
        match diff.code() {
            Some(0) => {
                eprintln!("pages: nothing changed");
                return Ok(());
            }
            Some(1) => git(&site, &["commit", "--quiet", "-m", &message])?,
            _ => return Err("git diff failed".into()),
        }
        git(
            &site,
            &["push", "--quiet", "origin", &format!("HEAD:{BRANCH}")],
        )?;
        wait_for_deploy(&site)?;
        eprintln!("pages: published {head} to https://code.sargunv.dev/orbis/");
        Ok(())
    })();
    git(
        &root,
        &["worktree", "remove", "--force", &site.to_string_lossy()],
    )?;
    result
}

/// Waits for GitHub's Pages deployment of the pushed commit: a failed
/// deployment leaves the previous build live.
fn wait_for_deploy(site: &Path) -> Result<(), String> {
    let output = |args: &[&str]| -> Result<String, String> {
        let out = Command::new(args[0])
            .current_dir(site)
            .args(&args[1..])
            .output()
            .map_err(|e| format!("failed to run {}: {e}", args[0]))?;
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let sha = output(&["git", "rev-parse", "HEAD"])?;
    let mut run = String::new();
    for _ in 0..60 {
        run = output(&[
            "gh",
            "run",
            "list",
            "--commit",
            &sha,
            "--json",
            "databaseId",
            "--jq",
            ".[0].databaseId",
        ])?;
        if !run.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
    if run.is_empty() {
        return Err(format!("no Pages deployment run started for {sha}"));
    }
    eprintln!("pages: waiting for deployment run {run}");
    let status = Command::new("gh")
        .current_dir(site)
        .args(["run", "watch", &run, "--exit-status", "--interval", "10"])
        .stdout(std::process::Stdio::null())
        .status()
        .map_err(|e| format!("failed to run gh: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "Pages deployment run {run} failed; the previous build is still live (rerun with `gh run rerun {run}`)"
        ))
    }
}

fn copy_tree(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::create_dir_all(to).map_err(|e| format!("{}: {e}", to.display()))?;
    for entry in std::fs::read_dir(from).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        let dest = to.join(path.file_name().expect("directory entry"));
        if path.is_dir() {
            copy_tree(&path, &dest)?;
        } else {
            std::fs::copy(&path, &dest).map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    Ok(())
}

fn git(dir: &Path, args: &[&str]) -> Result<(), String> {
    let status = Command::new("git")
        .current_dir(dir)
        .args(args)
        .status()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`git {}` failed", args.join(" ")))
    }
}
