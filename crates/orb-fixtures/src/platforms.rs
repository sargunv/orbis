//! Platform orchestration, report aggregation and native/wasm identity.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use serde_json::{Value, json};

use crate::tally::Counts;
use crate::{Suite, images::Image, native, pixelmatch, pixelmatch_cpp, write_json};

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
    // A beside-native platform's own process, its device made to take it.
    if std::env::var_os(CHILD).is_some() {
        suite.run(root, filter, platform)?;
        return Ok(());
    }
    if let Some(env) = beside_native(platform) {
        return run_beside_native(root, filter, platform, &env?, suite);
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
    // WebGL 2 beside WebGPU in the same Chrome: what GL alone fails.
    if platform == "chrome-webgl2" {
        reports.insert("chrome", suite.run(root, filter, "chrome")?);
        reports.insert("chrome-webgl2", suite.run(root, filter, "chrome-webgl2")?);
    }
    // A phone's Chrome beside this machine's on the same backend, and
    // Firefox or WebKit beside Chrome: what the one alone fails.
    if let Some(reference) = beside_chrome(platform) {
        reports.insert(reference, suite.run(root, filter, reference)?);
        reports.insert(platform, suite.run(root, filter, platform)?);
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
    let width = reports.keys().map(|p| p.len()).max().unwrap_or(0).max(8);
    println!(
        "\n{:<width$}  prefix          pass  fail  skip  in-scope pass rate",
        "platform"
    );
    for (platform, report) in &reports {
        for (prefix, counts) in report["prefixes"].as_object().expect("prefixes") {
            print!("{platform:<width$}  ");
            Counts::from_json(counts).print(prefix);
        }
        print!("{platform:<width$}  ");
        Counts::from_json(&report["totals"]).print("TOTAL");
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
    let webgl2 = match (reports.get("chrome"), reports.get("chrome-webgl2")) {
        (Some(webgpu), Some(gl)) => lost_report("WebGPU", webgpu, "GL", gl),
        _ => Value::Null,
    };
    let beside = match beside_chrome(platform) {
        Some(reference) => {
            lost_report(reference, &reports[reference], platform, &reports[platform])
        }
        None => Value::Null,
    };
    let summary = json!({"filter":filter, "total_seconds":started.elapsed().as_secs_f64(), "platforms":reports,"identity":identity,"webgl2":webgl2,"beside":beside});
    write_json(&combined, &summary)?;
    println!("\nReport: {}", combined.display());
    Ok(())
}

/// The Chrome a browser platform runs beside, for what it alone fails: a
/// phone's beside this machine's on the same backend, Firefox and WebKit
/// beside Chrome on WebGPU.
fn beside_chrome(platform: &str) -> Option<&'static str> {
    match platform {
        "android-chrome" | "firefox" | "webkit" => Some("chrome"),
        "android-chrome-webgl2" => Some("chrome-webgl2"),
        _ => None,
    }
}

/// The tests `first` passes and `second` doesn't, by cause: a gap the map
/// reported (`degraded:<gap>`), or one the renderer didn't, which is a bug
/// of the second's path (WebGL 2's against WebGPU, #140; a phone's against
/// this machine; Firefox's or WebKit's against Chrome). Printed, and the
/// summary's `webgl2` or `beside`.
fn lost_report(first_name: &str, webgpu: &Value, second_name: &str, gl: &Value) -> Value {
    let status = |report: &Value| -> BTreeMap<String, Value> {
        (report["tests"].as_array().into_iter().flatten())
            .map(|t| (t["id"].as_str().unwrap_or_default().to_string(), t.clone()))
            .collect()
    };
    let (webgpu, gl) = (status(webgpu), status(gl));
    let mut causes: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (id, test) in &webgpu {
        let Some(theirs) = gl.get(id) else { continue };
        if test["status"] != "pass" || theirs["status"] == "pass" {
            continue;
        }
        let cause = match (theirs["status"].as_str(), theirs["failure_reason"].as_str()) {
            (Some("fail"), Some(reason)) => reason.to_string(),
            (Some(status), _) => status.to_string(),
            _ => "missing".to_string(),
        };
        causes.entry(cause).or_default().push(id.clone());
    }
    let lost: usize = causes.values().map(Vec::len).sum();
    println!("\n{lost} tests {first_name} passes fail on {second_name}, by cause:");
    for (cause, ids) in &causes {
        println!("  {cause}: {}", ids.len());
        if !cause.starts_with("degraded:") {
            for id in ids {
                println!("    {id}");
            }
        }
    }
    json!({"lost": lost, "causes": causes})
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

/// Set in a beside-native platform's process.
const CHILD: &str = "ORB_FIXTURES_CPU_DRIVER";

/// The environment of a platform that runs the native suite again in a
/// process of its own, beside native: WebGL 2's frame driver (#140,
/// `ORB_DRIVER`), `native-cpu` with the tables in storage buffers,
/// `native-cpu-tex` in data textures, `native-webgl2` with everything
/// else WebGL 2 lacks too (compute: drawn mips; its gaps); or `native-gl`,
/// wgpu's GLES back end itself (`ORB_BACKEND=gl`), through ANGLE on macOS,
/// where GL's own quirks show without a browser.
pub(crate) fn beside_native(platform: &str) -> Option<Result<Vec<(&'static str, String)>, String>> {
    let driver = match platform {
        "native-cpu" => "cpu",
        "native-cpu-tex" => "cpu-tex",
        "native-webgl2" => "webgl2",
        "native-gl" => return Some(gl_env()),
        _ => return None,
    };
    Some(Ok(vec![("ORB_DRIVER", driver.to_string())]))
}

/// `native-gl`'s environment: GLES, and on macOS ANGLE's libraries from
/// `ORB_ANGLE_DIR` or the newest Playwright headless Chromium that ships
/// them whole (some ship 736-byte stubs).
fn gl_env() -> Result<Vec<(&'static str, String)>, String> {
    if !cfg!(feature = "gl") {
        return Err("native-gl needs orb-fixtures' `gl` feature (`cargo xtask rt` adds it)".into());
    }
    let mut env = vec![("ORB_BACKEND", "gl".to_string())];
    if cfg!(target_os = "macos") {
        let whole = |dir: &Path| {
            ["libEGL.dylib", "libGLESv2.dylib"]
                .iter()
                .all(|lib| std::fs::metadata(dir.join(lib)).is_ok_and(|m| m.len() > 64 << 10))
        };
        let dir = match std::env::var_os("ORB_ANGLE_DIR") {
            Some(dir) => Some(std::path::PathBuf::from(dir)),
            None => {
                let cache = Path::new(&std::env::var_os("HOME").unwrap_or_default())
                    .join("Library/Caches/ms-playwright");
                let mut shells: Vec<_> = std::fs::read_dir(&cache)
                    .into_iter()
                    .flatten()
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with("chromium_headless_shell-"))
                    })
                    .flat_map(|p| std::fs::read_dir(p).into_iter().flatten().flatten())
                    .map(|e| e.path())
                    .collect();
                shells.sort();
                shells.into_iter().rev().find(|dir| whole(dir))
            }
        }
        .filter(|dir| whole(dir))
        .ok_or(
            "native-gl on macOS needs ANGLE: a Playwright headless Chromium \
             (`pnpm -C web install`) or ORB_ANGLE_DIR at libEGL.dylib and libGLESv2.dylib",
        )?;
        env.push(("DYLD_LIBRARY_PATH", dir.display().to_string()));
    }
    Ok(env)
}

/// Runs the suite natively, then in a process of its own in `env`, and
/// compares the two's images byte for byte: the CPU drivers draw the same
/// frames, and GL lists what it draws otherwise.
fn run_beside_native(
    root: &Path,
    filter: &str,
    platform: &str,
    env: &[(&'static str, String)],
    suite: Suite,
) -> Result<(), String> {
    let output = match suite {
        Suite::Render => root.join("docs/scratch/rt"),
        Suite::NativeRender => root.join("docs/scratch/rt").join(suite.name()),
    };
    let native = suite.run(root, filter, "native")?;
    for (key, value) in env {
        println!("{platform}: {key}={value}");
    }
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
            .envs(env.iter().map(|(k, v)| (k, v)))
            .env(CHILD, "1"),
    )?;
    let path = output.join(platform).join("summary.json");
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let other: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    println!("\nplatform        pass  fail  skip");
    for (name, report) in [("native", &native), (platform, &other)] {
        let c = Counts::from_json(&report["totals"]);
        println!("{name:<15} {:>4}  {:>4}  {:>4}", c.pass, c.fail, c.skip);
    }
    let identity = byte_identity(&output, &native, &other, platform)?;
    println!(
        "\n{platform}/native: {} identical, {} differ, {} without both images ({} tests)",
        identity["identical"], identity["different"], identity["unavailable"], identity["tests"]
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
        if pixels == 0 && statuses.is_none() {
            identical += 1;
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
