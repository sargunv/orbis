//! Platform orchestration, report aggregation and native/wasm identity.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use serde_json::{Value, json};

use crate::{Counts, Suite, images::Image, native, pixelmatch, pixelmatch_cpp, write_json};

pub(crate) fn run(root: &Path, filter: &str, platform: &str, suite: Suite) -> Result<(), String> {
    let output = match suite {
        Suite::Render => root.join("docs/scratch/rt"),
        Suite::NativeRender => root.join("docs/scratch/rt").join(suite.name()),
    };
    // Only the container sets this; its suite writes its own report without
    // recursively launching Docker or replacing the host's combined report.
    if std::env::var_os("ORB_FIXTURES_LAVAPIPE").is_some() {
        suite.run(root, filter, "lavapipe")?;
        return Ok(());
    }
    // A CPU driver platform's own process, its device made to take it.
    if std::env::var_os(CHILD).is_some() {
        suite.run(root, filter, platform)?;
        return Ok(());
    }
    if let Some(driver) = cpu_driver(platform) {
        return run_cpu_driver(root, filter, platform, driver, suite);
    }
    std::fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let combined = output.join("summary.json");
    if combined.exists() {
        std::fs::remove_file(&combined).map_err(|e| e.to_string())?;
    }
    let started = Instant::now();
    let mut reports = BTreeMap::new();
    // Chrome always gets a fresh native baseline from the same invocation.
    if ["native", "chrome", "all"].contains(&platform) {
        reports.insert("native", suite.run(root, filter, "native")?);
    }
    if ["chrome", "all"].contains(&platform) {
        reports.insert("chrome", suite.run(root, filter, "chrome")?);
    }
    if ["lavapipe", "all"].contains(&platform) {
        let start = Instant::now();
        container(root, filter, suite, &output)?;
        let path = output.join("lavapipe/summary.json");
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let mut report: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        report["container_seconds"] = json!(start.elapsed().as_secs_f64());
        reports.insert("lavapipe", report);
    }
    let identity =
        if let (Some(native), Some(chrome)) = (reports.get("native"), reports.get("chrome")) {
            identity(&output, native, chrome, suite)?
        } else {
            Value::Null
        };
    println!("\nplatform  prefix          pass  fail  skip  in-scope pass rate");
    for (platform, report) in &reports {
        for (prefix, counts) in report["prefixes"].as_object().expect("prefixes") {
            print!("{platform:<9} ");
            counts_from(counts).print(prefix);
        }
        print!("{platform:<9} ");
        counts_from(&report["totals"]).print("TOTAL");
        if let Some(ignored) = report.get("native_ignored") {
            println!(
                "  Native ignores {} of them: {} pass, {} fail, {} skip",
                ignored["pass"].as_u64().unwrap_or(0)
                    + ignored["fail"].as_u64().unwrap_or(0)
                    + ignored["skip"].as_u64().unwrap_or(0),
                ignored["pass"],
                ignored["fail"],
                ignored["skip"]
            );
        }
        if let Some(gl_js) = report.get("gl_js") {
            println!(
                "  shared with GL JS: {} ({} with the same style); expected images comparable \
                 in {}, agreeing in {}; where they disagree Orbis matches {}; \
                 Orbis matches GL JS's image in {}",
                gl_js["shared"],
                gl_js["same_style"],
                gl_js["comparable"],
                gl_js["references_agree"],
                gl_js["references_disagree"],
                gl_js["matches_gl_js"]
            );
        }
        println!(
            "  suite {:.3} s{}",
            report["total_seconds"].as_f64().unwrap(),
            report["container_seconds"]
                .as_f64()
                .map_or(String::new(), |s| format!("; Docker build + run {s:.3} s"))
        );
    }
    if !identity.is_null() {
        println!(
            "\nChrome/native identity: {} match, {} differ, {} unavailable ({} native passes)",
            identity["match"], identity["different"], identity["unavailable"], identity["eligible"]
        );
        for test in identity["tests"].as_array().expect("identity tests") {
            if test["status"] != "match" {
                println!(
                    "  {}: {} (difference {}, allowed {})",
                    test["id"], test["status"], test["difference"], test["allowed"]
                );
            }
        }
    }
    let summary = json!({"filter":filter, "total_seconds":started.elapsed().as_secs_f64(), "platforms":reports,"identity":identity});
    write_json(&combined, &summary)?;
    println!("\nReport: {}", combined.display());
    Ok(())
}

fn counts_from(v: &Value) -> Counts {
    Counts {
        pass: v["pass"].as_u64().unwrap() as usize,
        fail: v["fail"].as_u64().unwrap() as usize,
        skip: v["skip"].as_u64().unwrap() as usize,
    }
}

/// Runs the suite in the software-rendering container: Mesa's lavapipe
/// (Vulkan). Its report goes to `<output>/lavapipe`.
fn container(root: &Path, filter: &str, suite: Suite, output: &Path) -> Result<(), String> {
    let software = "lavapipe";
    let dockerfile = root.join("crates/orb-fixtures/lavapipe/Dockerfile");
    run_command(
        Command::new("docker")
            .args(["build", "--tag", "orbis-rt-lavapipe", "--file"])
            .arg(dockerfile)
            .arg(root.join("crates/orb-fixtures/lavapipe")),
    )?;
    let output = &output.join(software);
    let target = output
        .strip_prefix(root)
        .map_err(|e| e.to_string())?
        .to_string_lossy()
        .into_owned();
    // No old report can be mistaken for this run's result.
    if output.exists() {
        std::fs::remove_dir_all(output).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir_all(output).map_err(|e| e.to_string())?;
    // Every worktree builds into the one target volume, and cargo takes a
    // path crate whose files are older than its last build as fresh: after
    // another tree's build, the binary would mix both trees' crates. Under
    // the volume's lock, a run from another root than the last first
    // cleans the workspace's own packages; dependencies stay built.
    let clean: String = workspace_packages(root)?
        .iter()
        .map(|p| format!(" -p {p}"))
        .collect();
    let script = format!(
        "exec 9>/work/target/.orbis-lock && flock 9 && \
         if [ \"$(cat /work/target/.orbis-root 2>/dev/null)\" != \"$ORBIS_ROOT\" ]; then \
         cargo clean --release{clean} && printf %s \"$ORBIS_ROOT\" > /work/target/.orbis-root; fi && \
         exec cargo run --locked --release -p orb-fixtures -- --suite {} --filter \"$ORBIS_FILTER\"",
        suite.name()
    );
    run_command(
        Command::new("docker")
            .args(["run", "--rm", "--init", "--mount"])
            .arg(format!(
                "type=bind,source={},target=/work,readonly",
                root.display()
            ))
            .args(["--mount"])
            .arg(format!(
                "type=bind,source={},target=/work/{target}",
                output.display()
            ))
            .args(["--env", "ORB_FIXTURES_LAVAPIPE=1", "--env"])
            .arg(format!("ORBIS_ROOT={}", root.display()))
            .args(["--env"])
            .arg(format!("ORBIS_FILTER={filter}"))
            .args([
                "--mount",
                "type=volume,source=orbis-rt-target,target=/work/target",
                "--mount",
                "type=volume,source=orbis-rt-registry,target=/usr/local/cargo/registry",
                "--mount",
                "type=volume,source=orbis-rt-git,target=/usr/local/cargo/git",
                "orbis-rt-lavapipe",
                "sh",
                "-c",
                &script,
            ]),
    )
}

/// The workspace's packages, by name.
fn workspace_packages(root: &Path) -> Result<Vec<String>, String> {
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(root)
        .output()
        .map_err(|e| format!("cargo metadata: {e}"))?;
    if !output.status.success() {
        return Err(format!("cargo metadata failed: {}", output.status));
    }
    let metadata: Value = serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())?;
    Ok(metadata["packages"]
        .as_array()
        .ok_or("cargo metadata: no packages")?
        .iter()
        .filter_map(|p| p["name"].as_str().map(str::to_owned))
        .collect())
}

fn run_command(command: &mut Command) -> Result<(), String> {
    let status = command
        .status()
        .map_err(|e| format!("{}: {e}", command.get_program().to_string_lossy()))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "{} failed: {status}",
            command.get_program().to_string_lossy()
        ))
    }
}

/// Compares native and Chrome images of every native pass, with the suite's
/// comparison: GL JS's pixelmatch and the test's threshold for GL JS's
/// suite, Native's pixelmatch on premultiplied pixels for Native's.
fn identity(output: &Path, native: &Value, chrome: &Value, suite: Suite) -> Result<Value, String> {
    let dir = output.join("identity");
    if dir.exists() {
        std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let chrome: BTreeMap<_, _> = chrome["tests"]
        .as_array()
        .expect("tests")
        .iter()
        .map(|t| (t["id"].as_str().unwrap(), t))
        .collect();
    let mut tests = Vec::new();
    let mut matches = 0;
    let mut different = 0;
    let mut unavailable = 0;
    for test in native["tests"]
        .as_array()
        .expect("tests")
        .iter()
        .filter(|t| t["status"] == "pass")
    {
        let id = test["id"].as_str().unwrap();
        let metadata = &test["metadata"];
        let allowed = metadata["allowed"].as_f64().unwrap();
        let threshold = metadata["threshold"].as_f64().unwrap_or(native::THRESHOLD);
        let mut result = json!({"id":id,"allowed":allowed,"threshold":threshold});
        if let Some(other) = chrome.get(id).filter(|t| t["image"].is_string()) {
            let a = Image::read(&output.join("native").join(test["image"].as_str().unwrap()))?;
            let b = Image::read(&output.join("chrome").join(other["image"].as_str().unwrap()))?;
            if (a.width, a.height) != (b.width, b.height) {
                return Err(format!("{id}: identity dimensions differ"));
            }
            let (count, diff) = match suite {
                Suite::Render => pixelmatch::compare(
                    &a.pixels,
                    &b.pixels,
                    a.width as usize,
                    a.height as usize,
                    threshold,
                ),
                Suite::NativeRender => {
                    let (mut a, mut b) = (a.clone(), b.clone());
                    a.premultiply_native();
                    b.premultiply_native();
                    pixelmatch_cpp::compare(
                        &a.pixels,
                        &b.pixels,
                        a.width as usize,
                        a.height as usize,
                        native::THRESHOLD,
                    )
                }
            };
            let fraction = count as f64 / (f64::from(a.width) * f64::from(a.height));
            result["difference"] = json!(fraction);
            result["different_pixels"] = json!(count);
            if fraction <= allowed {
                matches += 1;
                result["status"] = json!("match");
            } else {
                different += 1;
                result["status"] = json!("different");
                let target = dir.join(id);
                std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
                a.write(&target.join("native.png"))?;
                b.write(&target.join("chrome.png"))?;
                Image {
                    width: a.width,
                    height: a.height,
                    pixels: diff,
                }
                .write(&target.join("diff.png"))?;
                result["diff"] = json!(format!("identity/{id}/diff.png"));
            }
        } else {
            unavailable += 1;
            result["status"] = json!("unavailable");
            result["chrome_status"] = chrome.get(id).map_or(Value::Null, |t| t["status"].clone());
        }
        tests.push(result);
    }
    Ok(
        json!({"eligible":tests.len(),"match":matches,"different":different,"unavailable":unavailable,"tests":tests}),
    )
}

/// Set in a CPU driver platform's process.
const CHILD: &str = "ORB_FIXTURES_CPU_DRIVER";

/// The `ORB_DRIVER` of a platform that runs the native suite on WebGL
/// 2's frame driver (#140): `native-cpu` with the tables in storage
/// buffers, `native-cpu-tex` in data textures.
pub(crate) fn cpu_driver(platform: &str) -> Option<&'static str> {
    match platform {
        "native-cpu" => Some("cpu"),
        "native-cpu-tex" => Some("cpu-tex"),
        _ => None,
    }
}

/// Runs the suite natively, then in a process of its own with the CPU
/// frame driver (`ORB_DRIVER`), and compares the two's images byte for
/// byte: they draw the same frames.
fn run_cpu_driver(
    root: &Path,
    filter: &str,
    platform: &str,
    driver: &str,
    suite: Suite,
) -> Result<(), String> {
    let output = match suite {
        Suite::Render => root.join("docs/scratch/rt"),
        Suite::NativeRender => root.join("docs/scratch/rt").join(suite.name()),
    };
    let native = suite.run(root, filter, "native")?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    run_command(
        Command::new(exe)
            .args([
                "--suite",
                suite.name(),
                "--filter",
                filter,
                "--platform",
                platform,
            ])
            .env("ORB_DRIVER", driver)
            .env(CHILD, "1"),
    )?;
    let path = output.join(platform).join("summary.json");
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let other: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    println!("\nplatform        pass  fail  skip");
    for (name, report) in [("native", &native), (platform, &other)] {
        let c = counts_from(&report["totals"]);
        println!("{name:<15} {:>4}  {:>4}  {:>4}", c.pass, c.fail, c.skip);
    }
    let identity = byte_identity(&output, &native, &other, platform)?;
    println!(
        "\n{platform}/native: {} identical ({} of them terrain within one in a channel, #174), {} differ, {} without both images ({} tests)",
        identity["identical"],
        identity["terrain_within_one"],
        identity["different"],
        identity["unavailable"],
        identity["tests"]
    );
    for test in identity["differing"].as_array().expect("differing") {
        println!(
            "  {}: {} pixels differ{}",
            test["id"].as_str().unwrap_or_default(),
            test["pixels"],
            test["status"]
                .as_str()
                .map_or(String::new(), |s| format!(" ({s})"))
        );
    }
    let combined = output.join(format!("{platform}-identity.json"));
    write_json(&combined, &identity)?;
    println!("\nReport: {}", combined.display());
    Ok(())
}

/// Compares `other`'s image of every test with native's, byte for byte;
/// writes each pair that differs, and a mask of where, under
/// `<output>/<platform>-identity`.
///
/// A test drawing 3D terrain whose images differ by at most one in any
/// channel counts as identical: the drape's anisotropic samples depend on
/// where in its atlas a tile's slot lies, which depends on the order its
/// tiles arrived in, so two runs of one driver differ that much too (on
/// Apple M-series, a handful of pixels; #174).
fn byte_identity(
    output: &Path,
    native: &Value,
    other: &Value,
    platform: &str,
) -> Result<Value, String> {
    let dir = output.join(format!("{platform}-identity"));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
    }
    let others: BTreeMap<_, _> = other["tests"]
        .as_array()
        .expect("tests")
        .iter()
        .map(|t| (t["id"].as_str().unwrap_or_default(), t))
        .collect();
    let (mut identical, mut unavailable, mut differing) = (0, 0, Vec::new());
    let mut jittered = 0;
    let tests = native["tests"].as_array().expect("tests");
    for test in tests {
        let id = test["id"].as_str().unwrap_or_default();
        let theirs = others.get(id);
        let status = |t: &Value| t["status"].as_str().unwrap_or_default().to_string();
        let statuses = theirs
            .map(|t| status(t))
            .filter(|s| *s != status(test))
            .map(|s| format!("{} natively, {s} here", status(test)));
        let images = test["image"]
            .as_str()
            .zip(theirs.and_then(|t| t["image"].as_str()));
        let Some((a, b)) = images else {
            unavailable += 1;
            if let Some(statuses) = statuses {
                differing.push(json!({"id":id,"pixels":Value::Null,"status":statuses}));
            }
            continue;
        };
        let a = Image::read(&output.join("native").join(a))?;
        let b = Image::read(&output.join(platform).join(b))?;
        let same_size = (a.width, a.height) == (b.width, b.height);
        let pixels = if same_size {
            a.pixels
                .chunks(4)
                .zip(b.pixels.chunks(4))
                .filter(|(p, q)| p != q)
                .count()
        } else {
            (a.width * a.height) as usize
        };
        let drape_jitter = test["terrain"] == true
            && same_size
            && a.pixels
                .iter()
                .zip(&b.pixels)
                .all(|(p, q)| p.abs_diff(*q) <= 1);
        if (pixels == 0 || drape_jitter) && statuses.is_none() {
            identical += 1;
            jittered += usize::from(pixels > 0);
            continue;
        }
        if pixels > 0 && (a.width, a.height) == (b.width, b.height) {
            let target = dir.join(id);
            std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
            a.write(&target.join("native.png"))?;
            b.write(&target.join(format!("{platform}.png")))?;
            let mask = a
                .pixels
                .chunks(4)
                .zip(b.pixels.chunks(4))
                .flat_map(|(p, q)| {
                    if p == q {
                        [p[0] / 4 + 160, p[1] / 4 + 160, p[2] / 4 + 160, 255]
                    } else {
                        [255, 0, 0, 255]
                    }
                })
                .collect();
            Image {
                width: a.width,
                height: a.height,
                pixels: mask,
            }
            .write(&target.join("diff.png"))?;
        }
        differing.push(json!({"id":id,"pixels":pixels,"status":statuses}));
    }
    Ok(json!({
        "tests": tests.len(),
        "identical": identical,
        "terrain_within_one": jittered,
        "different": differing.len(),
        "unavailable": unavailable,
        "differing": differing,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_uses_only_native_passes_and_reports_missing_and_different_images() {
        let output = std::env::temp_dir().join(format!("orb-identity-{}", std::process::id()));
        if output.exists() {
            std::fs::remove_dir_all(&output).unwrap();
        }
        for platform in ["native", "chrome"] {
            std::fs::create_dir_all(output.join(platform)).unwrap();
        }
        let a = Image {
            width: 16,
            height: 16,
            pixels: [0, 0, 0, 255].repeat(256),
        };
        let b = Image {
            width: 16,
            height: 16,
            pixels: [255, 255, 255, 255].repeat(256),
        };
        a.write(&output.join("native/a.png")).unwrap();
        b.write(&output.join("chrome/b.png")).unwrap();
        let test = |id: &str, status: &str, allowed: f64| json!({"id":id,"status":status,"image":"a.png", "metadata":{"allowed":allowed,"threshold":0.1285}});
        let native = json!({"tests":[test("exact", "pass", 0.0),test("different", "pass", 0.0),test("allowed", "pass", 1.0),test("missing", "pass", 0.0),test("failed", "fail", 0.0),test("skipped", "skip", 0.0)]});
        let chrome = json!({"tests":[{"id":"exact","image":"../native/a.png"},{"id":"different","image":"b.png"},{"id":"allowed","image":"b.png"},{"id":"missing","status":"skip"}]});
        let report = identity(&output, &native, &chrome, Suite::Render).unwrap();
        assert_eq!(report["eligible"], 4);
        assert_eq!(report["match"], 2);
        assert_eq!(report["different"], 1);
        assert_eq!(report["unavailable"], 1);
        assert_eq!(report["tests"][1]["different_pixels"], 256);
        assert!(output.join("identity/different/diff.png").exists());
        assert!(!output.join("identity/exact").exists());
        std::fs::remove_dir_all(output).unwrap();
    }
}
