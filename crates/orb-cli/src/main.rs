//! `orbis`: a thin driver over `orb-api`'s still output.
//!
//! ```text
//! orbis render <style> -o out.png [--at lon,lat,zoom[,bearing,pitch]] [--size WxH] [--scale s]
//! orbis tiles <style> [-o dir] [--tms <id|tms.json>] (--zoom z[-z] [--bbox w,s,e,n] | --list file)
//!     [--size px] [--scale s] [--still] [--times]
//! ```
//!
//! `<style>` is a v8 style: a URL (`https://`, `file://`, `pmtiles://`)
//! or a path. `render` draws one picture in Still
//! mode (`--at`'s zoom is Web Mercator's, as the style's own plane is known
//! only once it loads). `tiles` draws tiles of a tile matrix set (default
//! `WebMercatorQuad`) on the style's plane in Tile mode, into
//! `dir/z/col/row.png` when `-o` is given, rows in the set's own order:
//! every tile of `--zoom`, those `--bbox` (degrees) touches, or the
//! `[name] z col row` lines of a `--list` file (`#` comments). `--still`
//! draws each Web Mercator tile as a Still-mode picture of the tile's own
//! view instead, as a host showing just that tile would. `--times` prints
//! `tile <name> <z> <col> <row> <wall ms> <cpu ms>` per tile, CPU being the
//! whole process's (job threads too).
//!
//! Both commands take `--state-dir <dir>`, the runtime's state directory
//! (compiled pipelines where the platform keeps none, the HTTP cache), so a
//! later run starts warm; `--previews <names|all>`, the comma-separated
//! previews (style inputs beyond the approved spec) the style may use; and
//! `--missing-tiles <parent|empty>`, what a vector tile that 404s draws:
//! its nearest ancestor (the default) or nothing, as GL JS and MapLibre
//! Native draw it.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use orb_api::{
    CameraState, Geodetic, Image, Map, MapOptions, MissingTiles, Preview, Previews, Runtime,
    RuntimeOptions, TileMatrixSet, TileRequest,
};
use orb_crs::TileIndex;

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("orbis: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Flags that take no value.
const SWITCHES: &[&str] = &["--still", "--times"];

/// Flags and the positional arguments.
struct Args {
    positional: Vec<String>,
    flags: Vec<(String, String)>,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut positional = Vec::new();
        let mut flags = Vec::new();
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            if SWITCHES.contains(&arg.as_str()) {
                flags.push((arg, String::new()));
            } else if arg.starts_with('-') {
                let value = args.next().ok_or(format!("{arg} needs a value"))?;
                flags.push((arg, value));
            } else {
                positional.push(arg);
            }
        }
        Ok(Self { positional, flags })
    }

    fn has(&self, name: &str) -> bool {
        self.flags.iter().any(|(f, _)| f == name)
    }

    fn get(&self, names: &[&str]) -> Option<&str> {
        self.flags
            .iter()
            .rev()
            .find(|(f, _)| names.contains(&f.as_str()))
            .map(|(_, v)| v.as_str())
    }

    /// A flag's numbers, split at `separator`.
    fn numbers(&self, name: &str, separator: char) -> Result<Option<Vec<f64>>, String> {
        self.get(&[name])
            .map(|v| {
                v.split(separator)
                    .map(|n| n.trim().parse::<f64>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| format!("{name} {v}: {e}"))
            })
            .transpose()
    }

    fn number(&self, name: &str, default: f64) -> Result<f64, String> {
        match self.get(&[name]) {
            Some(v) => v.parse().map_err(|e| format!("{name} {v}: {e}")),
            None => Ok(default),
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let args = Args::parse(args)?;
    let [command, style] = args.positional.as_slice() else {
        return Err("usage: orbis render|tiles <style> -o <output> [options]".into());
    };
    let output = args.get(&["-o", "--output"]).map(PathBuf::from);
    let style = style_url(style)?;
    let scale = args.number("--scale", 1.0)?;
    let options = RuntimeOptions {
        state_dir: args.get(&["--state-dir"]).map(PathBuf::from),
        ..RuntimeOptions::default()
    };
    let runtime = Runtime::new(&options).map_err(|e| e.to_string())?;
    match command.as_str() {
        "render" => {
            let output = output.ok_or("-o is required")?;
            render(&runtime, &style, &args, &output, scale)
        }
        "tiles" => tiles(&runtime, &style, &args, output.as_deref(), scale),
        other => Err(format!("unknown command {other}")),
    }
}

/// `--previews`: the previews to enable, comma-separated names or `all`.
fn previews(args: &Args) -> Result<Previews, String> {
    let Some(names) = args.get(&["--previews"]) else {
        return Ok(Previews::NONE);
    };
    match Previews::from_names(names.split(',').map(str::trim)) {
        (previews, unknown) if unknown.is_empty() => Ok(previews),
        (_, unknown) => Err(format!(
            "unknown previews {unknown:?}; known: all, {}",
            Preview::ALL.map(Preview::name).join(", ")
        )),
    }
}

/// `--missing-tiles`: what a vector tile that 404s draws.
fn missing_tiles(args: &Args) -> Result<MissingTiles, String> {
    match args.get(&["--missing-tiles"]) {
        None | Some("parent") => Ok(MissingTiles::Parent),
        Some("empty") => Ok(MissingTiles::Empty),
        Some(other) => Err(format!(
            "--missing-tiles takes parent or empty, not {other}"
        )),
    }
}

/// A URL as given, or a path as a `file://` URL.
fn style_url(style: &str) -> Result<String, String> {
    if style.contains("://") {
        return Ok(style.to_string());
    }
    let path = Path::new(style)
        .canonicalize()
        .map_err(|e| format!("{style}: {e}"))?;
    Ok(format!("file://{}", path.display()))
}

fn render(
    runtime: &Runtime,
    style: &str,
    args: &Args,
    output: &Path,
    scale: f64,
) -> Result<(), String> {
    let camera = match args.numbers("--at", ',')?.as_deref() {
        None => None,
        Some(&[lon, lat, zoom, ref rest @ ..]) if rest.len() <= 2 => {
            let mercator = *TileMatrixSet::by_id("WebMercatorQuad")
                .expect("built in")
                .projection();
            Some(CameraState {
                bearing: rest.first().copied().unwrap_or(0.0),
                pitch: rest.get(1).copied().unwrap_or(0.0),
                ..CameraState::new(Geodetic::new(lon, lat), zoom, &mercator)
            })
        }
        Some(_) => return Err("--at takes lon,lat,zoom[,bearing,pitch]".into()),
    };
    let (width, height) = match args.numbers("--size", 'x')?.as_deref() {
        None => (1024.0, 768.0),
        Some(&[w, h]) => (w, h),
        Some(_) => return Err("--size takes WxH".into()),
    };
    let mut map = Map::new(
        runtime,
        MapOptions {
            camera,
            previews: previews(args)?,
            missing_tiles: missing_tiles(args)?,
            ..MapOptions::default()
        },
    );
    map.set_viewport(width, height);
    map.load_style_url(style);
    let started = Instant::now();
    let image = map
        .snapshot(width as u32, height as u32, scale)
        .map_err(|e| e.to_string())?;
    warn(&mut map);
    write_png(output, image)?;
    let stats = runtime.stats();
    eprintln!(
        "orbis: {} in {:.0} ms; {} pipelines in {:.0} ms of compile threads",
        output.display(),
        started.elapsed().as_secs_f64() * 1000.0,
        stats.pipelines_created,
        stats.pipeline_time.as_secs_f64() * 1000.0,
    );
    Ok(())
}

fn tiles(
    runtime: &Runtime,
    style: &str,
    args: &Args,
    output: Option<&Path>,
    scale: f64,
) -> Result<(), String> {
    let tms = match args.get(&["--tms"]).unwrap_or("WebMercatorQuad") {
        tms if tms.ends_with(".json") => {
            let text = std::fs::read_to_string(tms).map_err(|e| format!("{tms}: {e}"))?;
            let json = orb_expr::json::parse(&text).map_err(|e| format!("{tms}: {e:?}"))?;
            let def = orb_style::v8::tms_definition(&json).map_err(|e| format!("{tms}: {e}"))?;
            TileMatrixSet::from_definition(&def)
        }
        tms => TileMatrixSet::by_id(tms),
    }
    .map_err(|e| e.to_string())?;
    let list = match args.get(&["--list"]) {
        Some(file) => tile_list(&tms, file)?,
        None => tile_range(&tms, args)?,
    };
    let size = args.number("--size", 512.0)? as u32;
    let still = args.has("--still");
    if still && tms.id() != "WebMercatorQuad" {
        return Err("--still draws WebMercatorQuad tiles".into());
    }
    let times = args.has("--times");
    let mut map = Map::new(
        runtime,
        MapOptions {
            previews: previews(args)?,
            missing_tiles: missing_tiles(args)?,
            ..MapOptions::default()
        },
    );
    map.load_style_url(style);
    let started = Instant::now();
    for (name, index) in &list {
        let (wall, cpu) = (Instant::now(), cpu_time());
        let z = index.z;
        let image = if still {
            map.jump_to_state(tile_view(&tms, *index, size));
            map.snapshot(size, size, scale)
        } else {
            map.render_tile(&TileRequest {
                tile_matrix_set: tms.clone(),
                z,
                col: index.col,
                row: index.row,
                size,
                scale,
            })
        }
        .map_err(|e| format!("{z}/{}/{}: {e}", index.col, index.row))?;
        let wall = wall.elapsed();
        let cpu = cpu_time().zip(cpu).map(|(after, before)| after - before);
        warn(&mut map);
        let row = tms.native_row(*index);
        if times {
            let ms = |d: Duration| d.as_secs_f64() * 1000.0;
            let cpu = cpu.map_or("-".to_string(), |c| format!("{:.2}", ms(c)));
            println!("tile {name} {z} {} {row} {:.2} {cpu}", index.col, ms(wall));
        }
        if let Some(output) = output {
            let dir = output.join(z.to_string()).join(index.col.to_string());
            std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            write_png(&dir.join(format!("{row}.png")), image)?;
        }
    }
    eprintln!(
        "orbis: {} tiles of {} in {:.1} s",
        list.len(),
        tms.id(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// The tiles `--zoom` and `--bbox` cover, named by their zoom.
fn tile_range(tms: &TileMatrixSet, args: &Args) -> Result<Vec<(String, TileIndex)>, String> {
    let zooms = match args.numbers("--zoom", '-')?.as_deref() {
        Some(&[z]) => (z as u8, z as u8),
        Some(&[a, b]) => (a as u8, b as u8),
        _ => return Err("--zoom takes z or z-z (or give --list)".into()),
    };
    let rect = match args.numbers("--bbox", ',')?.as_deref() {
        None => tms.projection().world_bounds(),
        Some(&[w, s, e, n]) => world_rect(tms, [w, s, e, n]),
        Some(_) => return Err("--bbox takes w,s,e,n".into()),
    };
    Ok((zooms.0.max(tms.min_zoom())..=zooms.1.min(tms.max_zoom()))
        .flat_map(|z| tms.tiles_in(rect, z).map(move |t| (format!("z{z}"), t)))
        .collect())
}

/// `--list`'s tiles: `[name] z col row` lines, rows in the set's order.
fn tile_list(tms: &TileMatrixSet, file: &str) -> Result<Vec<(String, TileIndex)>, String> {
    let text = std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
    let mut tiles = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let at = || format!("{file}:{}", n + 1);
        let line = line.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        let (name, numbers) = match fields[..] {
            [z, col, row] => (None, [z, col, row]),
            [name, z, col, row] => (Some(name), [z, col, row]),
            _ => return Err(format!("{}: want [name] z col row", at())),
        };
        let [z, col, row] = numbers.map(|v| v.parse::<u32>());
        let (z, col, row) = match (z, col, row) {
            (Ok(z), Ok(col), Ok(row)) if z <= 30 => (z as u8, col, row),
            _ => return Err(format!("{}: want [name] z col row", at())),
        };
        // A native row has the same range as a top-down one, so check
        // before flipping it; `native_row` is its own inverse.
        let native = TileIndex::new(z, col, row);
        if z > tms.max_zoom() || !tms.contains(native) {
            return Err(format!("{}: {z}/{col}/{row} is outside {}", at(), tms.id()));
        }
        let index = TileIndex::new(z, col, tms.native_row(native));
        tiles.push((name.map_or_else(|| format!("z{z}"), str::to_string), index));
    }
    Ok(tiles)
}

/// The Still-mode camera that shows exactly a Web Mercator tile in a
/// `size`-pixel square.
fn tile_view(tms: &TileMatrixSet, tile: TileIndex, size: u32) -> CameraState {
    let projection = tms.projection();
    let centre = projection.inverse(tms.tile_bounds(tile).center());
    let zoom = f64::from(tile.z) + (f64::from(size) / 512.0).log2();
    CameraState::new(centre, zoom, projection)
}

/// The process's CPU time so far, every thread's.
fn cpu_time() -> Option<Duration> {
    #[cfg(unix)]
    {
        let t = rustix::time::clock_gettime(rustix::time::ClockId::ProcessCPUTime);
        Some(Duration::new(t.tv_sec as u64, t.tv_nsec as u32))
    }
    #[cfg(not(unix))]
    None
}

/// The world rectangle around a geodetic box, sampled along a grid.
fn world_rect(tms: &TileMatrixSet, [w, s, e, n]: [f64; 4]) -> orb_crs::WorldRect {
    let projection = tms.projection();
    let steps = 32;
    let mut points = (0..=steps).flat_map(|i| {
        (0..=steps).map(move |j| {
            let lon = w + (e - w) * f64::from(i) / f64::from(steps);
            let lat = s + (n - s) * f64::from(j) / f64::from(steps);
            projection.forward(Geodetic::new(lon, lat))
        })
    });
    let first = points.next().expect("a point");
    let (min, max) = points.fold((first, first), |(min, max), p| (min.min(p), max.max(p)));
    orb_crs::WorldRect { min, max }
}

fn warn(map: &mut Map) {
    while let Some(event) = map.poll_event() {
        if let orb_api::Event::Map(
            orb_api::MapEvent::Warning { message } | orb_api::MapEvent::Error { message },
        ) = event
        {
            eprintln!("orbis: {message}");
        }
    }
}

/// Writes a picture as a PNG of straight alpha.
fn write_png(path: &Path, image: Image) -> Result<(), String> {
    let (width, height) = (image.width, image.height);
    let pixels = image.into_straight();
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .and_then(|mut w| w.write_image_data(&pixels))
        .map_err(|e| format!("{}: {e}", path.display()))
}
