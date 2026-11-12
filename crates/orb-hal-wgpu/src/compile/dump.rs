//! `ORB_DUMP_PIPELINES=<dir>`: each pipeline a run makes, written down as
//! it compiles, for `examples/pipeline_probe.rs` to compile again alone. A
//! line of `<dir>/pipelines.tsv` per pipeline, once across runs (written
//! before the driver sees it, so a compile that loses the device is there
//! too), and each module's WGSL beside it, named by its label and a hash
//! of its text.
//!
//! The line's tab-separated fields: `render`, the label, then the vertex
//! and fragment stages (each its WGSL file, entry point and `k=v,…`
//! override constants), the targets (`format:blend|…`), the depth
//! (`format:write` or `-`), the topology, back-face culling, the sample
//! count and WebGL 2's attribute zero; or `compute`, the label and its
//! stage.

use std::collections::HashSet;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use super::{Job, Stage, convert};

/// Where pipelines are written down, and the lines written so far.
static DUMP: OnceLock<Option<(PathBuf, Mutex<HashSet<String>>)>> = OnceLock::new();

/// Writes `job` down where `ORB_DUMP_PIPELINES` says, unless it was.
pub(super) fn job(job: &Job) {
    let dump = DUMP.get_or_init(|| {
        let dir = PathBuf::from(std::env::var_os("ORB_DUMP_PIPELINES")?);
        std::fs::create_dir_all(&dir).ok()?;
        // Lines earlier runs wrote there aren't written again.
        let earlier = std::fs::read_to_string(dir.join("pipelines.tsv")).unwrap_or_default();
        let written = earlier.lines().map(|l| format!("{l}\n")).collect();
        Some((dir, Mutex::new(written)))
    });
    let Some((dir, written)) = dump else {
        return;
    };
    let stage = |s: &Stage| {
        let name: String = (s.module.label.chars())
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        // A module's variants (`orb_shaders::specialise`) are its WGSL's.
        let mut hasher = std::hash::DefaultHasher::new();
        std::hash::Hash::hash(&s.module.source, &mut hasher);
        let file = format!(
            "{name}-{:08x}.wgsl",
            std::hash::Hasher::finish(&hasher) as u32
        );
        let path = dir.join(&file);
        if !path.exists() {
            let _ = std::fs::write(&path, &s.module.source);
        }
        // Those folded into the module too: the probe's `--specialise`
        // writes them in again.
        let folded = (s.module.fold.iter())
            .filter(|(k, _)| s.module.source.contains(&format!("override {k}:")));
        let constants: Vec<String> = (s.constants.iter().chain(folded))
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        format!("{file}\t{}\t{}", s.entry_point, constants.join(","))
    };
    let line = match job {
        Job::Render(_, j) => {
            let targets: Vec<String> = (j.targets.iter())
                .map(|t| format!("{:?}:{:?}", convert::texture_format(t.format), t.blend))
                .collect();
            let depth = (j.depth).map_or("-".to_string(), |d| {
                format!("{:?}:{}", convert::texture_format(d.format), d.write)
            });
            format!(
                "render\t{}\t{}\t{}\t{}\t{depth}\t{:?}\t{}\t{}\t{}\n",
                j.label,
                stage(&j.vertex),
                stage(&j.fragment),
                targets.join("|"),
                j.topology,
                j.cull_back,
                j.sample_count,
                j.attribute_zero,
            )
        }
        Job::Compute(_, j) => format!("compute\t{}\t{}\n", j.label, stage(&j.compute)),
    };
    let Ok(mut written) = written.lock() else {
        return;
    };
    if written.insert(line.clone())
        && let Ok(mut file) =
            (std::fs::OpenOptions::new().create(true).append(true)).open(dir.join("pipelines.tsv"))
    {
        let _ = file.write_all(line.as_bytes());
    }
}
