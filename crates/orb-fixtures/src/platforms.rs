//! Platform orchestration, report aggregation and native/wasm identity.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use serde_json::{Value, json};

use crate::{Counts, images::Image, pixelmatch, run_suite, write_json};

pub(crate) fn run(root: &Path, filter: &str, platform: &str) -> Result<(), String> {
    let output = root.join("docs/scratch/rt");
    // Only the container sets this; its suite writes its own report without
    // recursively launching Docker or replacing the host's combined report.
    if std::env::var_os("ORB_FIXTURES_LAVAPIPE").is_some() {
        run_suite(root, filter, "lavapipe")?;
        return Ok(());
    }
    if std::env::var_os("ORB_FIXTURES_LLVMPIPE").is_some() {
        run_suite(root, filter, "llvmpipe")?;
        return Ok(());
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
        reports.insert("native", run_suite(root, filter, "native")?);
    }
    if ["chrome", "all"].contains(&platform) {
        reports.insert("chrome", run_suite(root, filter, "chrome")?);
    }
    for compat in ["gl", "chrome-webgl2"] {
        if platform == compat {
            reports.insert(compat, run_suite(root, filter, compat)?);
        }
    }
    for software in ["lavapipe", "llvmpipe"] {
        if platform == software || (platform == "all" && software == "lavapipe") {
            let start = Instant::now();
            container(root, filter, software)?;
            let path = output.join(software).join("summary.json");
            let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
            let mut report: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            report["container_seconds"] = json!(start.elapsed().as_secs_f64());
            reports.insert(software, report);
        }
    }
    let identity =
        if let (Some(native), Some(chrome)) = (reports.get("native"), reports.get("chrome")) {
            identity(&output, native, chrome)?
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
/// (Vulkan, Modern) or llvmpipe (GLES 3 through EGL, Compat).
fn container(root: &Path, filter: &str, software: &str) -> Result<(), String> {
    let dockerfile = root.join("crates/orb-fixtures/lavapipe/Dockerfile");
    run_command(
        Command::new("docker")
            .args(["build", "--tag", "orbis-rt-lavapipe", "--file"])
            .arg(dockerfile)
            .arg(root.join("crates/orb-fixtures/lavapipe")),
    )?;
    let output = root.join("docs/scratch/rt").join(software);
    // No old report can be mistaken for this run's result.
    if output.exists() {
        std::fs::remove_dir_all(&output).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    run_command(
        Command::new("docker")
            .args(["run", "--rm", "--init", "--mount"])
            .arg(format!(
                "type=bind,source={},target=/work,readonly",
                root.display()
            ))
            .args(["--mount"])
            .arg(format!(
                "type=bind,source={},target=/work/docs/scratch/rt/{software}",
                output.display()
            ))
            .args([
                "--env",
                &format!("ORB_FIXTURES_{}=1", software.to_uppercase()),
            ])
            .args([
                "--mount",
                "type=volume,source=orbis-rt-target,target=/work/target",
                "--mount",
                "type=volume,source=orbis-rt-registry,target=/usr/local/cargo/registry",
                "--mount",
                "type=volume,source=orbis-rt-git,target=/usr/local/cargo/git",
                "orbis-rt-lavapipe",
                "cargo",
                "run",
                "--locked",
                "--release",
                "-p",
                "orb-fixtures",
                "--",
                "--filter",
                filter,
            ]),
    )
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

fn identity(output: &Path, native: &Value, chrome: &Value) -> Result<Value, String> {
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
        let mut result = json!({"id":id,"allowed":allowed,"threshold":metadata["threshold"]});
        if let Some(other) = chrome.get(id).filter(|t| t["image"].is_string()) {
            let a = Image::read(&output.join("native").join(test["image"].as_str().unwrap()))?;
            let b = Image::read(&output.join("chrome").join(other["image"].as_str().unwrap()))?;
            if (a.width, a.height) != (b.width, b.height) {
                return Err(format!("{id}: identity dimensions differ"));
            }
            let (count, diff) = pixelmatch::compare(
                &a.pixels,
                &b.pixels,
                a.width as usize,
                a.height as usize,
                metadata["threshold"].as_f64().unwrap(),
            );
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
        let report = identity(&output, &native, &chrome).unwrap();
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
