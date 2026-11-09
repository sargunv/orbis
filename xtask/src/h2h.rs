//! `cargo xtask h2h`: Orbis against mln-rs (a Rust port of MapLibre
//! Native, at a pinned commit) on the same machine, data and cameras.
//!
//! - `stills`: one process per still (`orbis render`, `mlnstill`) per view
//!   of mln-rs's `tools/liberty-compare/views.txt`: wall time, CPU,
//!   peak RSS, instructions retired (macOS) and requests.
//! - `tiles`: one process per pass over `xtask/h2h/tiles.txt`'s 4×4 tile
//!   blocks (`orbis tiles --list --times`, `mlntiles`), as Still-mode views
//!   of each tile and in Tile mode: per-set wall and CPU totals.
//! - `frames`: whole nudged frames read back, per view (`orb-bench
//!   --nudge`, `mlnframes`: mln-rs's own still bench).
//! - `size`: both still renderers in the `dist` profile.
//! - `web`: `pnpm -C web load` against the mirror: Orbis, GL JS and
//!   mln-rs's web build in Chromium, from page script to first idle.
//! - `mirror`: just the mirror, for the web pages or a manual run.
//!
//! Data comes from a caching mirror of `tiles.openfreemap.org` on
//! 127.0.0.1 that rewrites the style and TileJSON to itself, so every
//! request stays local; it fills from upstream on a miss (`--offline`
//! answers 404 instead) and keeps upstream's 404s. Once filled, the cache
//! is the pinned snapshot: its TileJSON names the planet build it holds,
//! and nothing refetches. `--network cold` uses the live server, and
//! `--network rtt=<ms>` delays every mirror answer.
//!
//! Runs interleave the engines (A, B, A, B) after an untimed pass of each,
//! so drift and caches hit both alike; tables give medians over runs. The
//! cache (mirror, mln-rs's build, about 2 GB) is in `$ORB_H2H_CACHE`, or
//! `~/Library/Caches/orbis-h2h` (`~/.cache/orbis-h2h` elsewhere), shared
//! by worktrees; raw results go to `target/h2h/`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const UPSTREAM: &str = "https://tiles.openfreemap.org";
/// The mln-rs commit `xtask/h2h/Cargo.toml` pins.
const MLN_RS: &str = "louwers/mln-rs@38cc1a17";

const HELP: &str = "\
cargo xtask h2h <stills|tiles|frames|size|all|web|mirror> [options]

  --runs N             timed runs per engine (default 5; 3 for frames)
  --filter <s>         views (stills, frames) whose name contains <s>
  --views <file>       cameras in mln-rs's views.txt format (default: its own)
  --style <name>       the OpenFreeMap style the views name (default liberty)
  --network warm|cold|rtt=<ms>  the local mirror (default), the live server,
                       or the mirror with every answer delayed
  --offline            the mirror answers misses with 404 instead of filling
  --bin <name>=<path>  another Orbis build as an engine: an `orbis` binary,
                       `orb-bench` beside it for frames (copy both aside)
  --engines a,b        engines and their order (default orbis, every --bin,
                       mln-rs)
  --frames N           frames per view for `frames` (default 40)
  --port N             the mirror's port for `mirror` (default 18731)
";

pub(crate) fn run(args: &[String]) -> Result<(), String> {
    let options = Options::parse(args)?;
    let cache = cache_dir()?;
    let delay = match options.network {
        Network::Rtt(ms) => Duration::from_millis(ms),
        _ => Duration::ZERO,
    };
    let port = if options.block == "mirror" {
        options.port
    } else {
        0
    };
    let mirror = Mirror::start(&cache.join("mirror"), options.offline, delay, port)?;
    if options.block == "mirror" {
        eprintln!(
            "h2h: mirror of {UPSTREAM} on {} ({}), cache {}; /__stats, /__reset; Ctrl-C ends it",
            mirror.base,
            if options.offline {
                "offline"
            } else {
                "filling"
            },
            cache.join("mirror").display()
        );
        loop {
            std::thread::park();
        }
    }
    if options.block == "web" {
        return web(&options, &mirror);
    }
    let blocks: &[&str] = match options.block.as_str() {
        "all" => &["stills", "tiles", "frames"],
        "stills" => &["stills"],
        "tiles" => &["tiles"],
        "frames" => &["frames"],
        "size" => &["size"],
        other => return Err(format!("h2h: unknown block `{other}`\n\n{HELP}")),
    };
    let h2h = H2h::new(options, cache, mirror)?;
    // Each block prints as it ends, so a later one's failure loses nothing.
    print!("{}", h2h.about());
    for block in blocks {
        let load = uptime();
        let report = match *block {
            "stills" => h2h.stills()?,
            "tiles" => h2h.tiles()?,
            "frames" => h2h.frames()?,
            _ => h2h.size()?,
        };
        println!(
            "\n{report}\nLoad average: {load} before, {} after.",
            uptime()
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Network {
    Warm,
    Cold,
    Rtt(u64),
}

#[derive(Debug)]
struct Options {
    block: String,
    runs: Option<usize>,
    filter: Option<String>,
    views: Option<PathBuf>,
    style: String,
    network: Network,
    offline: bool,
    bins: Vec<(String, PathBuf)>,
    engines: Option<Vec<String>>,
    frames: usize,
    port: u16,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut options = Self {
            block: String::new(),
            runs: None,
            filter: None,
            views: None,
            style: "liberty".into(),
            network: Network::Warm,
            offline: false,
            bins: Vec::new(),
            engines: None,
            frames: 40,
            port: 18731,
        };
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            let mut value = || {
                args.next()
                    .cloned()
                    .ok_or_else(|| format!("h2h: {arg} needs a value"))
            };
            let number = |v: String| {
                v.parse::<usize>()
                    .map_err(|e| format!("h2h: {arg} {v}: {e}"))
            };
            match arg.as_str() {
                "--runs" => options.runs = Some(number(value()?)?.max(1)),
                "--frames" => options.frames = number(value()?)?.max(2),
                "--port" => options.port = number(value()?)? as u16,
                "--filter" => options.filter = Some(value()?),
                "--views" => options.views = Some(PathBuf::from(value()?)),
                "--style" => options.style = value()?,
                "--offline" => options.offline = true,
                "--network" => {
                    let v = value()?;
                    options.network = match v.as_str() {
                        "warm" => Network::Warm,
                        "cold" => Network::Cold,
                        _ => match v.strip_prefix("rtt=").map(str::parse) {
                            Some(Ok(ms)) => Network::Rtt(ms),
                            _ => return Err(format!("h2h: --network {v}\n\n{HELP}")),
                        },
                    }
                }
                "--bin" => {
                    let v = value()?;
                    let (name, path) = v
                        .split_once('=')
                        .ok_or("h2h: --bin takes <name>=<path to orbis>")?;
                    let path = Path::new(path)
                        .canonicalize()
                        .map_err(|e| format!("h2h: {path}: {e}"))?;
                    options.bins.push((name.to_string(), path));
                }
                "--engines" => {
                    options.engines = Some(value()?.split(',').map(str::to_string).collect())
                }
                "help" | "--help" => return Err(HELP.to_string()),
                block if options.block.is_empty() && !block.starts_with('-') => {
                    options.block = block.to_string()
                }
                other => return Err(format!("h2h: unknown argument `{other}`\n\n{HELP}")),
            }
        }
        if options.block.is_empty() {
            return Err(HELP.to_string());
        }
        Ok(options)
    }
}

/// The shared cache: the mirror's files and mln-rs's build.
fn cache_dir() -> Result<PathBuf, String> {
    let dir = match std::env::var_os("ORB_H2H_CACHE") {
        Some(dir) => PathBuf::from(dir),
        None => {
            let home = PathBuf::from(std::env::var_os("HOME").ok_or("h2h: HOME is unset")?);
            if cfg!(target_os = "macos") {
                home.join("Library/Caches/orbis-h2h")
            } else {
                std::env::var_os("XDG_CACHE_HOME")
                    .map_or_else(|| home.join(".cache"), PathBuf::from)
                    .join("orbis-h2h")
            }
        }
    };
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    Ok(dir)
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the workspace root")
        .to_path_buf()
}

/// An engine: an Orbis build (its `orbis`, with `orb-bench` beside it) or
/// the mln-rs harness.
#[derive(Clone, Debug)]
enum Engine {
    Orbis { name: String, orbis: PathBuf },
    Mln,
}

impl Engine {
    fn name(&self) -> &str {
        match self {
            Self::Orbis { name, .. } => name,
            Self::Mln => "mln-rs",
        }
    }
}

struct H2h {
    options: Options,
    engines: Vec<Engine>,
    mirror: Mirror,
    /// mln-rs's binaries.
    mln: PathBuf,
    views: PathBuf,
    out: PathBuf,
}

impl H2h {
    fn new(options: Options, cache: PathBuf, mirror: Mirror) -> Result<Self, String> {
        let root = root();
        let mut known: Vec<Engine> = vec![Engine::Orbis {
            name: "orbis".into(),
            orbis: root.join("target/release/orbis"),
        }];
        for (name, orbis) in &options.bins {
            known.push(Engine::Orbis {
                name: name.clone(),
                orbis: orbis.clone(),
            });
        }
        known.push(Engine::Mln);
        let engines = match &options.engines {
            None => known,
            Some(names) => names
                .iter()
                .map(|n| {
                    known
                        .iter()
                        .find(|e| e.name() == n)
                        .cloned()
                        .ok_or_else(|| format!("h2h: no engine `{n}` (give it with --bin)"))
                })
                .collect::<Result<_, _>>()?,
        };
        let manifest = root.join("xtask/h2h/Cargo.toml");
        let target = cache.join("target");
        let uses = |n: &str| engines.iter().any(|e| e.name() == n);
        let profile = if options.block == "size" {
            "dist"
        } else {
            "release"
        };
        if uses("orbis") {
            crate::cargo(&[
                "build",
                "--profile",
                profile,
                "-p",
                "orb-cli",
                "-p",
                "orb-bench",
            ])?;
        }
        // mln-rs builds even when it isn't an engine: its checkout has the
        // views.
        eprintln!(
            "$ CARGO_TARGET_DIR={} cargo build --profile {profile} --manifest-path {}",
            target.display(),
            manifest.display()
        );
        let status = Command::new(env!("CARGO"))
            .args(["build", "--profile", profile, "--manifest-path"])
            .arg(&manifest)
            .env("CARGO_TARGET_DIR", &target)
            .status()
            .map_err(|e| format!("cargo: {e}"))?;
        if !status.success() {
            return Err("h2h: building mln-rs's harness failed".into());
        }
        let views = match &options.views {
            Some(views) => views.clone(),
            None => mln_checkout(&manifest)?.join("tools/liberty-compare/views.txt"),
        };
        let out = root.join("target/h2h");
        std::fs::create_dir_all(out.join("png")).map_err(|e| format!("{}: {e}", out.display()))?;
        Ok(Self {
            options,
            engines,
            mirror,
            mln: target.join(profile),
            views,
            out,
        })
    }

    fn runs(&self, default: usize) -> usize {
        self.options.runs.unwrap_or(default)
    }

    fn style_url(&self) -> String {
        let base = match self.options.network {
            Network::Cold => UPSTREAM,
            _ => &self.mirror.base,
        };
        format!("{base}/styles/{}", self.options.style)
    }

    /// The report's head: machine, builds, data, method.
    fn about(&self) -> String {
        let mut out = String::from("## Orbis head to head\n\n");
        let cpu = output(Command::new("sysctl").args(["-n", "machdep.cpu.brand_string"]));
        let rev = output(Command::new("git").args(["rev-parse", "--short", "HEAD"]));
        let planet = std::fs::read_to_string(self.mirror.cache.join("planet.__f"))
            .ok()
            .and_then(|t| {
                let at = t.find("/planet/")? + "/planet/".len();
                Some(t[at..].split('/').next()?.to_string())
            });
        let lines = [
            format!(
                "machine: {}, {} logical CPUs",
                cpu.as_deref().unwrap_or("unknown"),
                std::thread::available_parallelism().map_or(0, |n| n.get())
            ),
            format!(
                "Orbis {} (working tree), mln-rs {MLN_RS}",
                rev.as_deref().unwrap_or("?")
            ),
            format!(
                "engines: {}",
                self.engines
                    .iter()
                    .map(Engine::name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            match self.options.network {
                Network::Cold => format!("network: live {UPSTREAM}, no engine cache"),
                network => format!(
                    "network: local mirror of {UPSTREAM} ({}{}), planet {}",
                    if self.options.offline {
                        "offline"
                    } else {
                        "filling on a miss"
                    },
                    match network {
                        Network::Rtt(ms) => format!(", {ms} ms added to every answer"),
                        _ => String::new(),
                    },
                    planet.as_deref().unwrap_or("not cached yet"),
                ),
            },
            format!("views: {} ({})", self.views.display(), self.options.style),
            "runs interleave the engines after an untimed pass of each; medians over runs".into(),
        ];
        for line in lines {
            writeln!(out, "- {line}").expect("a String");
        }
        out
    }

    /// The views of the style, as `(name, [lon, lat, zoom, bearing, pitch,
    /// ratio, width, height])`.
    fn views(&self) -> Result<Vec<(String, [String; 8])>, String> {
        let text = std::fs::read_to_string(&self.views)
            .map_err(|e| format!("{}: {e}", self.views.display()))?;
        let mut views = Vec::new();
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or_default();
            let f: Vec<&str> = line.split_whitespace().collect();
            let [style, name, rest @ ..] = &f[..] else {
                continue;
            };
            if *style != self.options.style
                || self
                    .options
                    .filter
                    .as_ref()
                    .is_some_and(|s| !name.contains(s.as_str()))
            {
                continue;
            }
            let numbers: [String; 8] = rest
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| format!("{}: {line}: want 10 columns", self.views.display()))?;
            views.push((name.to_string(), numbers));
        }
        if views.is_empty() {
            return Err(format!(
                "h2h: no {} view in {}",
                self.options.style,
                self.views.display()
            ));
        }
        Ok(views)
    }

    fn stills(&self) -> Result<String, String> {
        let views = self.views()?;
        let style = self.style_url();
        let command = |engine: &Engine, name: &str, v: &[String; 8], png: &Path| {
            let at = format!("{},{},{},{},{}", v[0], v[1], v[2], v[3], v[4]);
            let size = format!("{}x{}", v[6], v[7]);
            let png = png.join(format!("{}-{name}.png", engine.name()));
            let png = png.display().to_string();
            match engine {
                Engine::Orbis { orbis, .. } => vec![
                    orbis.display().to_string(),
                    "render".into(),
                    style.clone(),
                    "-o".into(),
                    png,
                    "--at".into(),
                    at,
                    "--size".into(),
                    size,
                    "--scale".into(),
                    v[5].clone(),
                ],
                Engine::Mln => vec![
                    self.mln.join("mlnstill").display().to_string(),
                    style.clone(),
                    "-o".into(),
                    png,
                    "--at".into(),
                    at,
                    "--size".into(),
                    size,
                    "--ratio".into(),
                    v[5].clone(),
                ],
            }
        };
        let png = self.out.join("png");
        // Untimed: fills the mirror, the OS's file and shader caches.
        for (name, v) in &views {
            for engine in &self.engines {
                eprintln!("h2h: stills, untimed: {} {name}", engine.name());
                timed(&command(engine, name, v, &png))?;
            }
        }
        let mut rows: Vec<StillRow> = Vec::new();
        let runs = self.runs(5);
        for run in 0..runs {
            for (name, v) in &views {
                for engine in &self.engines {
                    eprintln!(
                        "h2h: stills, run {} of {runs}: {} {name}",
                        run + 1,
                        engine.name()
                    );
                    self.mirror.reset();
                    let t = timed(&command(engine, name, v, &png))?;
                    rows.push(StillRow {
                        engine: engine.name().to_string(),
                        view: name.clone(),
                        run,
                        t,
                        net: self.mirror.stats(),
                    });
                }
            }
        }
        let mut csv = String::from(
            "engine,view,run,wall_ms,cpu_ms,rss_mb,ginstr,internal_ms,requests,kbytes,misses,not_found,failed,fetched\n",
        );
        for r in &rows {
            let opt = |v: Option<f64>| v.map_or(String::new(), |v| format!("{v:.3}"));
            writeln!(
                csv,
                "{},{},{},{:.2},{},{},{},{},{},{:.1},{},{},{},{}",
                r.engine,
                r.view,
                r.run,
                ms(r.t.wall),
                opt(r.t.cpu_ms),
                opt(r.t.rss_mb),
                opt(r.t.ginstr),
                opt(r.t.internal_ms),
                r.net.requests,
                r.net.bytes as f64 / 1024.0,
                r.net.misses,
                r.net.not_found,
                r.net.failed,
                r.net.fetched
            )
            .expect("a String");
        }
        let csv_path = self.out.join(format!("stills-{}.csv", self.network_tag()));
        std::fs::write(&csv_path, csv).map_err(|e| format!("{}: {e}", csv_path.display()))?;

        let names: Vec<&str> = self.engines.iter().map(Engine::name).collect();
        let median_of = |engine: &str, view: &str, f: &dyn Fn(&StillRow) -> Option<f64>| {
            let mut v: Vec<f64> = (rows.iter())
                .filter(|r| r.engine == engine && r.view == view)
                .filter_map(f)
                .collect();
            median(&mut v)
        };
        let metrics: [Metric; 5] = [
            ("wall ms", &|r| Some(ms(r.t.wall)), 0),
            ("CPU ms", &|r| r.t.cpu_ms, 0),
            ("RSS MB", &|r| r.t.rss_mb, 0),
            ("G instr", &|r| r.t.ginstr, 2),
            ("requests", &|r| Some(r.net.requests as f64), 0),
        ];
        let mut out = format!(
            "### Stills: one process per view ({} views × {runs} runs, 512 px unless the view says otherwise)\n\n| view |",
            views.len()
        );
        for (metric, ..) in &metrics {
            for name in &names {
                write!(out, " {metric} {name} |").expect("a String");
            }
        }
        writeln!(out, "\n|---|{}", "---|".repeat(metrics.len() * names.len())).expect("a String");
        let mut sums = vec![0.0; metrics.len() * names.len()];
        for (view, _) in &views {
            write!(out, "| {view} |").expect("a String");
            let mut i = 0;
            for (_, f, digits) in &metrics {
                for name in &names {
                    let m = median_of(name, view, *f);
                    sums[i] += m.unwrap_or(0.0);
                    i += 1;
                    write!(
                        out,
                        " {} |",
                        m.map_or("—".into(), |m| format!("{m:.digits$}"))
                    )
                    .expect("a String");
                }
            }
            out.push('\n');
        }
        write!(out, "| **sum** |").expect("a String");
        for (i, s) in sums.iter().enumerate() {
            let digits = metrics[i / names.len()].2;
            write!(out, " **{s:.digits$}** |").expect("a String");
        }
        out.push('\n');
        // Each engine against the last (mln-rs by default), view by view.
        if let [first @ .., last] = &names[..] {
            for name in first {
                for (metric, f, _) in &metrics[..2] {
                    let (mut better, mut tie, mut worse) = (0, 0, 0);
                    for (view, _) in &views {
                        let (Some(a), Some(b)) =
                            (median_of(name, view, *f), median_of(last, view, *f))
                        else {
                            continue;
                        };
                        match a / b {
                            r if r < 0.95 => better += 1,
                            r if r > 1.05 => worse += 1,
                            _ => tie += 1,
                        }
                    }
                    writeln!(out, "\n{name} against {last}, {metric}: lower on {better} views, within 5% on {tie}, higher on {worse}.").expect("a String");
                }
            }
        }
        let mut net = Stats::default();
        for r in &rows {
            net.add(&r.net);
        }
        writeln!(
            out,
            "\nWall: exec to exit; CPU: user + system, and instructions retired, from `/usr/bin/time`; requests: what the engine asked the mirror for. Timed runs: {}. Raw rows: {}.",
            net.answers(),
            csv_path.display()
        )
        .expect("a String");
        Ok(out)
    }

    fn tiles(&self) -> Result<String, String> {
        let list = root().join("xtask/h2h/tiles.txt");
        let style = self.style_url();
        let runs = self.runs(5);
        let mut out = String::new();
        for still in [true, false] {
            let mode = if still {
                "Still-mode views of each tile"
            } else {
                "Tile mode"
            };
            let command = |engine: &Engine| {
                let mut c = match engine {
                    Engine::Orbis { orbis, .. } => vec![
                        orbis.display().to_string(),
                        "tiles".into(),
                        style.clone(),
                        "--list".into(),
                        list.display().to_string(),
                        "--times".into(),
                    ],
                    Engine::Mln => vec![
                        self.mln.join("mlntiles").display().to_string(),
                        style.clone(),
                        "--list".into(),
                        list.display().to_string(),
                    ],
                };
                if still {
                    c.push("--still".into());
                }
                c
            };
            for engine in &self.engines {
                eprintln!("h2h: tiles, untimed: {} ({mode})", engine.name());
                timed(&command(engine))?;
            }
            // Per engine and set, each run's totals; and each run's requests.
            let mut totals: SetTotals = BTreeMap::new();
            let mut requests: BTreeMap<String, Vec<f64>> = BTreeMap::new();
            let mut net = Stats::default();
            let mut sets: Vec<String> = Vec::new();
            for run in 0..runs {
                for engine in &self.engines {
                    eprintln!(
                        "h2h: tiles, run {} of {runs}: {} ({mode})",
                        run + 1,
                        engine.name()
                    );
                    self.mirror.reset();
                    let t = timed(&command(engine))?;
                    let stats = self.mirror.stats();
                    net.add(&stats);
                    requests
                        .entry(engine.name().into())
                        .or_default()
                        .push(stats.requests as f64);
                    let mut run_totals: BTreeMap<String, (f64, f64, usize)> = BTreeMap::new();
                    for line in t.stdout.lines() {
                        let f: Vec<&str> = line.split_whitespace().collect();
                        let ["tile", set, _, _, _, wall, cpu] = f[..] else {
                            continue;
                        };
                        if !sets.iter().any(|s| s == set) {
                            sets.push(set.to_string());
                        }
                        let e = run_totals.entry(set.to_string()).or_default();
                        e.0 += wall.parse::<f64>().unwrap_or(f64::NAN);
                        e.1 += cpu.parse::<f64>().unwrap_or(f64::NAN);
                        e.2 += 1;
                    }
                    for (set, total) in run_totals {
                        totals
                            .entry((engine.name().into(), set))
                            .or_default()
                            .push(total);
                    }
                }
            }
            write!(
                out,
                "### Tiles: {mode} ({runs} runs, one process each)\n\n| set | tiles |"
            )
            .expect("a String");
            let names: Vec<&str> = self.engines.iter().map(Engine::name).collect();
            for name in &names {
                write!(out, " wall {name} | CPU {name} |").expect("a String");
            }
            if names.len() > 1 {
                write!(out, " CPU {} / {} |", names[0], names[names.len() - 1]).expect("a String");
            }
            writeln!(
                out,
                "\n|---|---|{}",
                "---|".repeat(2 * names.len() + usize::from(names.len() > 1))
            )
            .expect("a String");
            let column = |name: &str, set: &str, i: usize| {
                let mut v: Vec<f64> = (totals
                    .get(&(name.to_string(), set.to_string()))
                    .into_iter()
                    .flatten())
                .map(|t| if i == 0 { t.0 } else { t.1 })
                .collect();
                median(&mut v)
            };
            for set in &sets {
                let count = totals
                    .get(&(names[0].to_string(), set.clone()))
                    .and_then(|t| t.first())
                    .map_or(0, |t| t.2);
                write!(out, "| {set} | {count} |").expect("a String");
                for name in &names {
                    let show = |v: Option<f64>| v.map_or("—".into(), |v| format!("{v:.0}"));
                    write!(
                        out,
                        " {} | {} |",
                        show(column(name, set, 0)),
                        show(column(name, set, 1))
                    )
                    .expect("a String");
                }
                if names.len() > 1 {
                    let ratio = column(names[0], set, 1)
                        .zip(column(names[names.len() - 1], set, 1))
                        .map(|(a, b)| a / b);
                    write!(
                        out,
                        " {} |",
                        ratio.map_or("—".into(), |r| format!("{r:.2}"))
                    )
                    .expect("a String");
                }
                out.push('\n');
            }
            write!(out, "\nRequests per pass:").expect("a String");
            for name in &names {
                let r = median(requests.get_mut(*name).map_or(&mut Vec::new(), |v| v));
                write!(out, " {name} {:.0};", r.unwrap_or(0.0)).expect("a String");
            }
            writeln!(out, " times in ms, medians of per-set totals; the `startup` set is the first tile, with the style's load. Timed runs: {}.\n", net.answers()).expect("a String");
        }
        Ok(out)
    }

    fn frames(&self) -> Result<String, String> {
        let style = self.style_url();
        let runs = self.runs(3);
        let frames = self.options.frames.to_string();
        let warmup = "10".to_string();
        let views = self.views.display().to_string();
        let command = |engine: &Engine| {
            let mut c = match engine {
                Engine::Orbis { orbis, .. } => vec![
                    orbis.with_file_name("orb-bench").display().to_string(),
                    style.clone(),
                    "--views".into(),
                    views.clone(),
                    "--nudge".into(),
                    "--frames".into(),
                    frames.clone(),
                    "--warmup".into(),
                    warmup.clone(),
                    "--rounds".into(),
                    "1".into(),
                ],
                Engine::Mln => vec![
                    self.mln.join("mlnframes").display().to_string(),
                    style.clone(),
                    "--views".into(),
                    views.clone(),
                    "--frames".into(),
                    frames.clone(),
                    "--warmup".into(),
                    warmup.clone(),
                ],
            };
            if let Some(filter) = &self.options.filter {
                c.extend(["--filter".into(), filter.clone()]);
            }
            c
        };
        // Per engine and view, each run's median frame (and Orbis's GPU).
        let mut walls: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
        let mut gpu: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
        let mut order: Vec<String> = Vec::new();
        for run in 0..runs {
            for engine in &self.engines {
                eprintln!("h2h: frames, run {} of {runs}: {}", run + 1, engine.name());
                let t = timed(&command(engine))?;
                for line in t.stdout.lines() {
                    let (view, wall, g) = match engine {
                        // `| view | frame wall | p90 | GPU | ...`
                        Engine::Orbis { .. } => {
                            let cells: Vec<&str> = line.split('|').map(str::trim).collect();
                            match cells[..] {
                                ["", view, wall, _, g, ..] if wall.parse::<f64>().is_ok() => {
                                    (view, wall, g.parse::<f64>().ok())
                                }
                                _ => continue,
                            }
                        }
                        Engine::Mln => match line.split_whitespace().collect::<Vec<_>>()[..] {
                            ["frame", view, wall, ..] => (view, wall, None),
                            _ => continue,
                        },
                    };
                    if !order.iter().any(|v| v == view) {
                        order.push(view.to_string());
                    }
                    let key = (engine.name().to_string(), view.to_string());
                    walls
                        .entry(key.clone())
                        .or_default()
                        .extend(wall.parse::<f64>().ok());
                    gpu.entry(key).or_default().extend(g);
                }
            }
        }
        let names: Vec<&str> = self.engines.iter().map(Engine::name).collect();
        let mut out = format!(
            "### Frames: whole frames read back, the zoom nudged by 0.001 a frame ({frames} frames a view after 10, {runs} runs)\n\n| view |"
        );
        for name in &names {
            write!(out, " {name} |").expect("a String");
        }
        if names.len() > 1 {
            write!(out, " {} / {} |", names[0], names[names.len() - 1]).expect("a String");
        }
        let orbis: Vec<&str> = (self.engines.iter())
            .filter(|e| matches!(e, Engine::Orbis { .. }))
            .map(Engine::name)
            .collect();
        for name in &orbis {
            write!(out, " GPU {name} |").expect("a String");
        }
        writeln!(
            out,
            "\n|---|{}",
            "---|".repeat(names.len() + usize::from(names.len() > 1) + orbis.len())
        )
        .expect("a String");
        let get = |map: &BTreeMap<(String, String), Vec<f64>>, name: &str, view: &str| {
            median(
                &mut map
                    .get(&(name.to_string(), view.to_string()))
                    .cloned()
                    .unwrap_or_default(),
            )
        };
        let mut sums = vec![0.0; names.len()];
        for view in &order {
            write!(out, "| {view} |").expect("a String");
            for (i, name) in names.iter().enumerate() {
                let m = get(&walls, name, view);
                sums[i] += m.unwrap_or(0.0);
                write!(out, " {} |", m.map_or("—".into(), |m| format!("{m:.2}")))
                    .expect("a String");
            }
            if names.len() > 1 {
                let r = get(&walls, names[0], view).zip(get(&walls, names[names.len() - 1], view));
                write!(
                    out,
                    " {} |",
                    r.map_or("—".into(), |(a, b)| format!("{:.2}", a / b))
                )
                .expect("a String");
            }
            for name in &orbis {
                write!(
                    out,
                    " {} |",
                    get(&gpu, name, view).map_or("—".into(), |m| format!("{m:.2}"))
                )
                .expect("a String");
            }
            out.push('\n');
        }
        write!(out, "| **sum** |").expect("a String");
        for s in &sums {
            write!(out, " **{s:.1}** |").expect("a String");
        }
        out.push_str(&" |".repeat(usize::from(names.len() > 1) + orbis.len()));
        writeln!(
            out,
            "\n\nMedian ms a frame: Orbis renders continuously into a texture and reads each frame back (`orb-bench --nudge`); mln-rs renders each frame as a Static-mode still and reads it back (its own still bench)."
        )
        .expect("a String");
        Ok(out)
    }

    fn size(&self) -> Result<String, String> {
        let mut out = String::from(
            "### Size: still renderers, `dist` profile (LTO, one codegen unit, abort on panic, stripped)\n\n| binary | bytes | gzip -9 |\n|---|---|---|\n",
        );
        let mut binaries = Vec::new();
        for engine in &self.engines {
            match engine {
                Engine::Orbis { name, .. } if name == "orbis" => {
                    binaries.push(("orbis".to_string(), root().join("target/dist/orbis")));
                }
                Engine::Orbis { name, orbis } => binaries.push((name.clone(), orbis.clone())),
                Engine::Mln => binaries.push(("mlnstill".into(), self.mln.join("mlnstill"))),
            }
        }
        for (name, path) in binaries {
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
            gz.write_all(&bytes).map_err(|e| e.to_string())?;
            let gz = gz.finish().map_err(|e| e.to_string())?;
            writeln!(out, "| {name} | {} | {} |", bytes.len(), gz.len()).expect("a String");
        }
        Ok(out)
    }

    fn network_tag(&self) -> String {
        match self.options.network {
            Network::Warm => "warm".into(),
            Network::Cold => "cold".into(),
            Network::Rtt(ms) => format!("rtt{ms}"),
        }
    }
}

/// The `web` block: core.wasm, then `pnpm -C web load` against the mirror.
fn web(options: &Options, mirror: &Mirror) -> Result<(), String> {
    crate::web::build()?;
    let base = match options.network {
        Network::Cold => UPSTREAM,
        _ => &mirror.base,
    };
    let load = uptime();
    let mut command = Command::new("pnpm");
    command
        .current_dir(root().join("web"))
        .args(["load", &format!("--style={base}/styles/{}", options.style)])
        .arg(format!("--runs={}", options.runs.unwrap_or(5)))
        .stderr(std::process::Stdio::inherit());
    let output = command
        .output()
        .map_err(|e| format!("pnpm ({e}); `mise install` provides it"))?;
    if !output.status.success() {
        return Err("h2h: pnpm load failed".into());
    }
    print!("{}", String::from_utf8_lossy(&output.stdout));
    println!("\nLoad average: {load} before, {} after.", uptime());
    Ok(())
}

/// mln-rs's checkout, where Cargo keeps the pinned commit.
fn mln_checkout(manifest: &Path) -> Result<PathBuf, String> {
    let out = Command::new(env!("CARGO"))
        .args(["metadata", "--format-version", "1", "--manifest-path"])
        .arg(manifest)
        .output()
        .map_err(|e| format!("cargo metadata: {e}"))?;
    let metadata: serde_json::Value =
        serde_json::from_slice(&out.stdout).map_err(|e| format!("cargo metadata: {e}"))?;
    let package = (metadata["packages"].as_array().into_iter().flatten())
        .find(|p| p["name"] == "mln-rs")
        .ok_or("h2h: cargo metadata lists no mln-rs")?;
    let manifest = package["manifest_path"]
        .as_str()
        .ok_or("h2h: mln-rs has no manifest")?;
    Ok(Path::new(manifest)
        .parent()
        .expect("a directory")
        .to_path_buf())
}

/// A table column: its name, a row's value, and the digits shown.
type Metric<'a> = (&'a str, &'a dyn Fn(&StillRow) -> Option<f64>, usize);

/// Per engine and tile set, each run's wall and CPU totals and tile count.
type SetTotals = BTreeMap<(String, String), Vec<(f64, f64, usize)>>;

struct StillRow {
    engine: String,
    view: String,
    run: usize,
    t: Timed,
    net: Stats,
}

/// A process's run.
#[derive(Debug)]
struct Timed {
    wall: Duration,
    cpu_ms: Option<f64>,
    rss_mb: Option<f64>,
    ginstr: Option<f64>,
    /// What the engine reports: `in N ms`.
    internal_ms: Option<f64>,
    stdout: String,
}

/// Runs a command under `/usr/bin/time` (`-l` on macOS, `-v` elsewhere),
/// failing if it fails.
fn timed(command: &[String]) -> Result<Timed, String> {
    let time = Path::new("/usr/bin/time");
    let mut c = if time.exists() {
        let mut c = Command::new(time);
        c.arg(if cfg!(target_os = "macos") {
            "-l"
        } else {
            "-v"
        });
        c.args(command);
        c
    } else {
        let mut c = Command::new(&command[0]);
        c.args(&command[1..]);
        c
    };
    let started = Instant::now();
    let output = c.output().map_err(|e| format!("{}: {e}", command[0]))?;
    let wall = started.elapsed();
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        let tail: String = stderr
            .chars()
            .rev()
            .take(2000)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        return Err(format!("`{}` failed:\n{tail}", command.join(" ")));
    }
    let mut t = Timed {
        wall,
        cpu_ms: None,
        rss_mb: None,
        ginstr: None,
        internal_ms: None,
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
    };
    let (mut user, mut sys) = (None, None);
    for line in stderr.lines() {
        let line = line.trim();
        let f: Vec<&str> = line.split_whitespace().collect();
        let first = || f.first().and_then(|v| v.parse::<f64>().ok());
        let last = || f.last().and_then(|v| v.parse::<f64>().ok());
        if let [_, "real", u, "user", s, "sys"] = f[..] {
            (user, sys) = (u.parse::<f64>().ok(), s.parse::<f64>().ok());
        } else if line.ends_with("maximum resident set size") {
            t.rss_mb = first().map(|b| b / 1e6);
        } else if line.ends_with("instructions retired") {
            t.ginstr = first().map(|n| n / 1e9);
        } else if line.starts_with("User time (seconds):") {
            user = last();
        } else if line.starts_with("System time (seconds):") {
            sys = last();
        } else if line.starts_with("Maximum resident set size (kbytes):") {
            t.rss_mb = last().map(|kb| kb * 1024.0 / 1e6);
        } else if let Some(at) = line.find(" in ").filter(|_| line.contains(" ms")) {
            t.internal_ms = line[at + 4..]
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok());
        }
    }
    t.cpu_ms = user.zip(sys).map(|(u, s)| (u + s) * 1000.0);
    Ok(t)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn median(v: &mut [f64]) -> Option<f64> {
    v.sort_by(f64::total_cmp);
    match v.len() {
        0 => None,
        n if n % 2 == 1 => Some(v[n / 2]),
        n => Some((v[n / 2 - 1] + v[n / 2]) / 2.0),
    }
}

fn output(command: &mut Command) -> Option<String> {
    let out = command.output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The load averages, from `uptime`.
fn uptime() -> String {
    output(&mut Command::new("uptime"))
        .and_then(|u| {
            Some(
                u.split("load average")
                    .nth(1)?
                    .trim_start_matches(['s', ':', ' '])
                    .to_string(),
            )
        })
        .unwrap_or_else(|| "unknown".into())
}

/// Requests the mirror answered since the last reset.
#[derive(Clone, Copy, Debug, Default)]
struct Stats {
    requests: u64,
    bytes: u64,
    /// Answered 404 because the cache lacked them (offline).
    misses: u64,
    /// Answered 404 because upstream did (kept in the cache): a few are
    /// normal, many mean the cached TileJSON names a planet build upstream
    /// has dropped (delete the mirror's cache to take the current one).
    not_found: u64,
    /// Answered 502: upstream failed or answered something else.
    failed: u64,
    /// Fetched upstream.
    fetched: u64,
}

impl Stats {
    fn add(&mut self, other: &Self) {
        self.requests += other.requests;
        self.bytes += other.bytes;
        self.misses += other.misses;
        self.not_found += other.not_found;
        self.failed += other.failed;
        self.fetched += other.fetched;
    }

    /// What the mirror did besides answering from its cache.
    fn answers(&self) -> String {
        format!(
            "{} fetched upstream, {} missed offline, {} kept upstream 404s, {} failed",
            self.fetched, self.misses, self.not_found, self.failed
        )
    }
}

/// A caching mirror of [`UPSTREAM`] on 127.0.0.1, a thread per
/// connection. A cached file is `<cache>/<path>.__f`, its content type
/// beside it in `.__type` and an upstream 404 or 204 as `.__status`.
struct Mirror {
    base: String,
    cache: PathBuf,
    shared: Arc<Shared>,
}

struct Shared {
    base: String,
    cache: PathBuf,
    offline: bool,
    delay: Duration,
    stats: Mutex<Stats>,
    agent: ureq::Agent,
    temp: AtomicU64,
}

impl Mirror {
    fn start(cache: &Path, offline: bool, delay: Duration, port: u16) -> Result<Self, String> {
        std::fs::create_dir_all(cache).map_err(|e| format!("{}: {e}", cache.display()))?;
        let listener = TcpListener::bind(("127.0.0.1", port))
            .map_err(|e| format!("h2h: the mirror's port {port}: {e}"))?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        let base = format!("http://127.0.0.1:{port}");
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .build()
            .into();
        let shared = Arc::new(Shared {
            base: base.clone(),
            cache: cache.to_path_buf(),
            offline,
            delay,
            stats: Mutex::default(),
            agent,
            temp: AtomicU64::new(0),
        });
        let accept = Arc::clone(&shared);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let shared = Arc::clone(&accept);
                std::thread::spawn(move || {
                    let _ = connection(stream, &shared);
                });
            }
        });
        Ok(Self {
            base,
            cache: cache.to_path_buf(),
            shared,
        })
    }

    fn reset(&self) {
        *self.shared.stats.lock().expect("stats") = Stats::default();
    }

    fn stats(&self) -> Stats {
        *self.shared.stats.lock().expect("stats")
    }
}

/// Answers a connection's requests until it closes.
fn connection(stream: TcpStream, shared: &Shared) -> std::io::Result<()> {
    stream.set_nodelay(true)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Ok(());
        }
        let mut parts = line.split_whitespace();
        let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or("/"));
        let (mut close, mut length) = (false, 0usize);
        loop {
            let mut header = String::new();
            if reader.read_line(&mut header)? == 0 || header.trim().is_empty() {
                break;
            }
            let lower = header.to_ascii_lowercase();
            if lower.starts_with("connection:") && lower.contains("close") {
                close = true;
            } else if let Some(v) = lower.strip_prefix("content-length:") {
                length = v.trim().parse().unwrap_or(0);
            }
        }
        std::io::copy(&mut (&mut reader).take(length as u64), &mut std::io::sink())?;
        let (status, kind, body) = match method {
            "GET" | "HEAD" => shared.answer(path),
            "OPTIONS" => (204, String::new(), Vec::new()),
            _ => (405, String::new(), Vec::new()),
        };
        let reason = match status {
            200 => "OK",
            204 => "No Content",
            404 => "Not Found",
            405 => "Method Not Allowed",
            _ => "Bad Gateway",
        };
        let mut head = format!(
            "HTTP/1.1 {status} {reason}\r\ncontent-length: {}\r\naccess-control-allow-origin: *\r\naccess-control-allow-headers: *\r\ncross-origin-resource-policy: cross-origin\r\ncache-control: no-store\r\n",
            body.len()
        );
        if !kind.is_empty() {
            write!(head, "content-type: {kind}\r\n").expect("a String");
        }
        head.push_str(if close {
            "connection: close\r\n\r\n"
        } else {
            "\r\n"
        });
        writer.write_all(head.as_bytes())?;
        if method != "HEAD" {
            writer.write_all(&body)?;
        }
        if close {
            return Ok(());
        }
    }
}

impl Shared {
    /// A request's status, content type and body.
    fn answer(&self, path: &str) -> (u16, String, Vec<u8>) {
        match path {
            "/__stats" => {
                let s = *self.stats.lock().expect("stats");
                let json = format!(
                    "{{\"requests\":{},\"bytes\":{},\"misses\":{},\"not_found\":{},\"failed\":{},\"fetched\":{}}}",
                    s.requests, s.bytes, s.misses, s.not_found, s.failed, s.fetched
                );
                return (200, "application/json".into(), json.into_bytes());
            }
            "/__reset" => {
                *self.stats.lock().expect("stats") = Stats::default();
                return (200, "text/plain".into(), b"ok".to_vec());
            }
            _ => {}
        }
        self.stats.lock().expect("stats").requests += 1;
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        let file = self.cache_path(path);
        let with = |suffix: &str| {
            let mut name = file.clone().into_os_string();
            name.push(suffix);
            PathBuf::from(name)
        };
        let mut missed = false;
        let (mut status, mut kind, mut body) =
            if let Ok(status) = std::fs::read_to_string(with(".__status")) {
                (
                    status.trim().parse().unwrap_or(404),
                    String::new(),
                    Vec::new(),
                )
            } else if let Ok(body) = std::fs::read(with(".__f")) {
                (
                    200,
                    std::fs::read_to_string(with(".__type")).unwrap_or_default(),
                    body,
                )
            } else if self.offline {
                missed = true;
                (404, String::new(), Vec::new())
            } else {
                match self.fetch(path, &with(".__f"), &with(".__type"), &with(".__status")) {
                    Ok(answer) => answer,
                    Err(e) => (502, "text/plain".into(), e.into_bytes()),
                }
            };
        if status == 200 && is_json(path, &kind) {
            body = String::from_utf8_lossy(&body)
                .replace(UPSTREAM, &self.base)
                .into_bytes();
        }
        if status == 200 && kind.is_empty() {
            kind = "application/octet-stream".into();
        }
        if status == 204 {
            body.clear();
        } else if status != 200 && status != 404 {
            status = 502;
        }
        let mut stats = self.stats.lock().expect("stats");
        stats.bytes += body.len() as u64;
        match status {
            404 if missed => stats.misses += 1,
            404 => stats.not_found += 1,
            502 => stats.failed += 1,
            _ => {}
        }
        (status, kind, body)
    }

    /// Fetches `path` upstream into the cache.
    fn fetch(
        &self,
        path: &str,
        body_file: &Path,
        type_file: &Path,
        status_file: &Path,
    ) -> Result<(u16, String, Vec<u8>), String> {
        let url = format!("{UPSTREAM}{path}");
        let mut response = self
            .agent
            .get(&url)
            .call()
            .map_err(|e| format!("{url}: {e}"))?;
        self.stats.lock().expect("stats").fetched += 1;
        let status = response.status().as_u16();
        if let Some(dir) = body_file.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        if status == 404 || status == 204 {
            self.write(status_file, status.to_string().as_bytes())?;
            return Ok((status, String::new(), Vec::new()));
        }
        if status != 200 {
            return Err(format!("{url}: HTTP {status}"));
        }
        let kind = (response.headers().get("content-type"))
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let mut body = Vec::new();
        (response.body_mut().with_config().limit(u64::MAX).reader())
            .read_to_end(&mut body)
            .map_err(|e| format!("{url}: {e}"))?;
        self.write(type_file, kind.as_bytes())?;
        self.write(body_file, &body)?;
        Ok((200, kind, body))
    }

    /// Writes a cache file whole: other runs read the cache meanwhile.
    fn write(&self, path: &Path, bytes: &[u8]) -> Result<(), String> {
        let n = self.temp.fetch_add(1, Ordering::Relaxed);
        let mut temp = path.to_path_buf().into_os_string();
        temp.push(format!(".partial-{}-{n}", std::process::id()));
        std::fs::write(&temp, bytes)
            .and_then(|()| std::fs::rename(&temp, path))
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Where a request path's answer is kept: its path, percent-decoded,
    /// without the query or `..`.
    fn cache_path(&self, path: &str) -> PathBuf {
        let path = path.split(['?', '#']).next().unwrap_or_default();
        let decoded = percent_decode(path).replace("..", "_");
        self.cache.join(decoded.trim_start_matches('/'))
    }
}

fn is_json(path: &str, kind: &str) -> bool {
    kind.contains("json")
        || path.ends_with(".json")
        || path.starts_with("/styles/")
        || path.trim_end_matches('/') == "/planet"
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| (b as char).to_digit(16);
        match (
            bytes[i],
            bytes.get(i + 1).copied().and_then(hex),
            bytes.get(i + 2).copied().and_then(hex),
        ) {
            (b'%', Some(h), Some(l)) => {
                out.push((h * 16 + l) as u8);
                i += 3;
            }
            (b, ..) => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_paths_decode_and_stay_inside() {
        let shared = Shared {
            base: String::new(),
            cache: PathBuf::from("/c"),
            offline: true,
            delay: Duration::ZERO,
            stats: Mutex::default(),
            agent: ureq::Agent::new_with_defaults(),
            temp: AtomicU64::new(0),
        };
        assert_eq!(
            shared.cache_path("/fonts/Noto%20Sans%20Regular/0-255.pbf?x=1"),
            PathBuf::from("/c/fonts/Noto Sans Regular/0-255.pbf")
        );
        assert_eq!(
            shared.cache_path("/../etc/passwd"),
            PathBuf::from("/c/_/etc/passwd")
        );
        assert!(
            is_json("/planet", "")
                && is_json("/styles/liberty", "")
                && !is_json("/planet/1/0/0.pbf", "")
        );
    }

    #[test]
    fn time_output() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(median(&mut []), None);
    }
}
