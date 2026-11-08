//! `orbis`: a thin driver over `orb-api`'s still output.
//!
//! ```text
//! orbis render <style> -o out.png [--at lon,lat,zoom[,bearing,pitch]] [--size WxH] [--scale s]
//! orbis tiles <style> -o dir --tms <id|tms.json> --zoom z[-z] [--bbox w,s,e,n] [--size px] [--scale s]
//! ```
//!
//! `<style>` is a v8 style: a URL (`https://`, `file://`, `pmtiles://`)
//! or a path. `render` draws one picture in Still
//! mode (`--at`'s zoom is Web Mercator's, as the style's own plane is known
//! only once it loads). `tiles` draws every tile of a tile matrix set on the
//! style's plane in Tile mode, `dir/z/col/row.png`, rows in the set's own
//! order; `--bbox` (degrees) limits them to the tiles it touches.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use orb_api::{
    CameraState, Geodetic, Image, Map, MapOptions, Runtime, RuntimeOptions, TileMatrixSet,
    TileRequest,
};

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("orbis: {e}");
            ExitCode::FAILURE
        }
    }
}

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
            if arg.starts_with('-') {
                let value = args.next().ok_or(format!("{arg} needs a value"))?;
                flags.push((arg, value));
            } else {
                positional.push(arg);
            }
        }
        Ok(Self { positional, flags })
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
    let output = PathBuf::from(args.get(&["-o", "--output"]).ok_or("-o is required")?);
    let style = style_url(style)?;
    let scale = args.number("--scale", 1.0)?;
    let runtime = Runtime::new(&RuntimeOptions::default()).map_err(|e| e.to_string())?;
    match command.as_str() {
        "render" => render(&runtime, &style, &args, &output, scale),
        "tiles" => tiles(&runtime, &style, &args, &output, scale),
        other => Err(format!("unknown command {other}")),
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
            ..MapOptions::default()
        },
    );
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
    output: &Path,
    scale: f64,
) -> Result<(), String> {
    let tms = args.get(&["--tms"]).ok_or("--tms is required")?;
    let tms = if tms.ends_with(".json") {
        let text = std::fs::read_to_string(tms).map_err(|e| format!("{tms}: {e}"))?;
        let json = orb_expr::json::parse(&text).map_err(|e| format!("{tms}: {e:?}"))?;
        let def = orb_style::v8::tms_definition(&json).map_err(|e| format!("{tms}: {e}"))?;
        TileMatrixSet::from_definition(&def)
    } else {
        TileMatrixSet::by_id(tms)
    }
    .map_err(|e| e.to_string())?;
    let zooms = match args.numbers("--zoom", '-')?.as_deref() {
        Some(&[z]) => (z as u8, z as u8),
        Some(&[a, b]) => (a as u8, b as u8),
        _ => return Err("--zoom takes z or z-z".into()),
    };
    let rect = match args.numbers("--bbox", ',')?.as_deref() {
        None => tms.projection().world_bounds(),
        Some(&[w, s, e, n]) => world_rect(&tms, [w, s, e, n]),
        Some(_) => return Err("--bbox takes w,s,e,n".into()),
    };
    let size = args.number("--size", 512.0)? as u32;
    let mut map = Map::new(runtime, MapOptions::default());
    map.load_style_url(style);
    let started = Instant::now();
    let mut count = 0;
    for z in zooms.0.max(tms.min_zoom())..=zooms.1.min(tms.max_zoom()) {
        for index in tms.tiles_in(rect, z) {
            let image = map
                .render_tile(&TileRequest {
                    tile_matrix_set: tms.clone(),
                    z,
                    col: index.col,
                    row: index.row,
                    size,
                    scale,
                })
                .map_err(|e| format!("{z}/{}/{}: {e}", index.col, index.row))?;
            warn(&mut map);
            let dir = output.join(z.to_string()).join(index.col.to_string());
            std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            let path = dir.join(format!("{}.png", tms.native_row(index)));
            write_png(&path, image)?;
            count += 1;
        }
    }
    eprintln!(
        "orbis: {count} tiles of {} into {} in {:.1} s",
        tms.id(),
        output.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
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
    let Image {
        width,
        height,
        mut pixels,
    } = image;
    for p in pixels.as_chunks_mut::<4>().0 {
        let a = u32::from(p[3]);
        if a != 0 && a != 255 {
            for c in &mut p[..3] {
                *c = ((u32::from(*c) * 255 + a / 2) / a).min(255) as u8;
            }
        }
    }
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .and_then(|mut w| w.write_image_data(&pixels))
        .map_err(|e| format!("{}: {e}", path.display()))
}
