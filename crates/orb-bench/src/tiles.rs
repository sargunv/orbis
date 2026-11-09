//! The tile bench: every tile of the committed Tokyo extract
//! (`web/demo/data`, 21 OpenMapTiles tiles at z0–14, MVT and MLT) through
//! a tile job's stages on one thread, with no fetching: decode the bytes,
//! lay the tile out with the reduced Liberty style, and pack the result as
//! the web's job workers send it (`orb_runtime::stages`).
//!
//! A first round fetches the glyph ranges labels ask for from the
//! extract's subsets and is not timed; each timed round then starts from a
//! new worker, so nothing decoded or laid out carries over. Ideographs lay
//! out from the system's fonts, as a native map's do; the sprite's icons
//! are not loaded.

use std::fmt::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use orb_crs::TileIndex;
use orb_runtime::stages::Stages;
use orb_source::pmtiles::{Entry, Header};
use orb_style::Encoding;

use crate::report::{Summary, ms};
use crate::{Settings, clock};

/// The extract's only source: the style's first.
const SOURCE: u32 = 1;

/// Both encodings' tiles.
#[derive(Debug)]
pub struct TilesReport {
    about: Vec<String>,
    samples: usize,
    encodings: Vec<(&'static str, Vec<TileTimes>)>,
}

/// One tile's stages over the timed rounds.
#[derive(Debug)]
struct TileTimes {
    tile: TileIndex,
    /// Bytes decoded (after the archive's compression).
    input: usize,
    packed: usize,
    decode: Vec<Duration>,
    layout: Vec<Duration>,
    pack: Vec<Duration>,
    total: Vec<Duration>,
}

pub(crate) fn run(
    settings: &Settings,
    about: Vec<String>,
    data: &Path,
    mut progress: impl FnMut(&str),
) -> Result<TilesReport, String> {
    let mut encodings = Vec::new();
    for (name, encoding) in [("mvt", Encoding::Mvt), ("mlt", Encoding::Mlt)] {
        progress(&format!("{name}: reading the extract"));
        let tiles = read_archive(&data.join(format!("tokyo-{name}.pmtiles")))?;
        let style_path = data.join(format!("tokyo-{name}.json"));
        let style = std::fs::read_to_string(&style_path)
            .map_err(|e| format!("{}: {e}", style_path.display()))?
            .replace("\"./", &format!("\"file://{}/", data.display()));
        let local: Arc<dyn orb_runtime::LocalGlyphs> = orb_host_native::SystemFonts::shared();
        let stages = || {
            Stages::new(
                &style,
                encoding,
                Some(Arc::clone(&local)),
                Some("sans-serif".into()),
            )
        };

        // The untimed round: the glyph ranges every tile's labels need.
        let mut resources: Vec<(String, Result<Vec<u8>, String>)> = Vec::new();
        let first = stages()?;
        for (tile, bytes) in &tiles {
            first.decode(SOURCE, *tile, bytes)?;
            while let Err(urls) = first.layout(SOURCE, *tile)? {
                for url in urls {
                    let got = read_url(&url);
                    first.add_resource(&url, got.as_deref().map_err(Clone::clone));
                    resources.push((url, got));
                }
            }
        }

        let mut times: Vec<TileTimes> = (tiles.iter())
            .map(|(tile, bytes)| TileTimes {
                tile: *tile,
                input: bytes.len(),
                packed: 0,
                decode: Vec::new(),
                layout: Vec::new(),
                pack: Vec::new(),
                total: Vec::new(),
            })
            .collect();
        for sample in 0..settings.samples {
            progress(&format!(
                "{name}: round {} of {}",
                sample + 1,
                settings.samples
            ));
            let stages = stages()?;
            for (url, got) in &resources {
                stages.add_resource(url, got.as_deref().map_err(Clone::clone));
            }
            for ((tile, bytes), t) in tiles.iter().zip(&mut times) {
                let started = clock::thread();
                stages.decode(SOURCE, *tile, bytes)?;
                let decoded = clock::thread();
                let build = stages
                    .layout(SOURCE, *tile)?
                    .map_err(|urls| format!("{tile:?} asked for {urls:?} after the first round"))?;
                let laid_out = clock::thread();
                let packed = orb_runtime::pack(build);
                let done = clock::thread();
                t.decode.push(decoded - started);
                t.layout.push(laid_out - decoded);
                t.pack.push(done - laid_out);
                t.total.push(done - started);
                t.packed = packed.len();
            }
        }
        encodings.push((name, times));
    }
    Ok(TilesReport {
        about,
        samples: settings.samples,
        encodings,
    })
}

/// Every tile of a PMTiles archive with a root directory alone, by zoom,
/// with its bytes decompressed.
fn read_archive(path: &Path) -> Result<Vec<(TileIndex, Vec<u8>)>, String> {
    let archive = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let at = |offset: u64, length: u64| {
        archive
            .get(offset as usize..(offset + length) as usize)
            .ok_or_else(|| format!("{}: truncated", path.display()))
    };
    let header = Header::parse(at(0, 127)?)?;
    let root = at(header.root.offset, header.root.length)?;
    let entries = orb_source::pmtiles::directory(&header.internal_compression.decode(root)?)?;
    let find = |id: u64| {
        entries
            .iter()
            .find(|e: &&Entry| e.run > 0 && (e.id..e.id + e.run).contains(&id))
    };
    // Every tile at the archive's zooms that touches its bounds.
    let [w, s, e, n] = header.bounds;
    let mut tiles = Vec::new();
    for z in header.minzoom..=header.maxzoom {
        let (x0, y0) = tile_of(w, n, z);
        let (x1, y1) = tile_of(e, s, z);
        for row in y0..=y1 {
            for col in x0..=x1 {
                let tile = TileIndex::new(z, col, row);
                let Some(entry) = find(orb_source::pmtiles::tile_id(tile)?) else {
                    continue;
                };
                let bytes = at(header.tiles_offset + entry.offset, entry.length)?;
                tiles.push((tile, header.tile_compression.decode(bytes)?));
            }
        }
    }
    if tiles.is_empty() {
        return Err(format!("{}: no tiles in its bounds", path.display()));
    }
    Ok(tiles)
}

/// The Web Mercator tile holding a point at zoom `z`.
fn tile_of(lon: f64, lat: f64, z: u8) -> (u32, u32) {
    let n = f64::from(1u32 << z);
    let x = (lon + 180.0) / 360.0 * n;
    let lat = lat.to_radians();
    let y = (1.0 - lat.tan().asinh() / std::f64::consts::PI) / 2.0 * n;
    let clamp = |v: f64| (v.floor().max(0.0) as u32).min((1u32 << z) - 1);
    (clamp(x), clamp(y))
}

/// A `file://` URL's bytes.
fn read_url(url: &str) -> Result<Vec<u8>, String> {
    let path = url
        .strip_prefix("file://")
        .ok_or_else(|| format!("{url}: not a file URL"))?
        .replace("%20", " ");
    std::fs::read(&path).map_err(|e| format!("{path}: {e}"))
}

impl TilesReport {
    pub(crate) fn write(&self, out: &mut String) -> std::fmt::Result {
        writeln!(out, "### Tile bench")?;
        writeln!(out)?;
        for line in &self.about {
            writeln!(out, "- {line}")?;
        }
        writeln!(
            out,
            "- the committed Tokyo extract's tiles through decode, layout (reduced Liberty, \
             ideographs from system fonts, no icons) and pack on one thread, timed by its CPU clock; {} timed rounds \
             after an untimed one, each from a new worker. The CPU clock runs 1.5–2.5 times as long on an efficiency \
             core as on a performance one, so compare runs interleaved in one session; across sessions, \
             compare instructions retired (`/usr/bin/time -l`)",
            self.samples
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "| encoding | tiles | KiB in | decode | layout | pack | total | KiB packed |"
        )?;
        writeln!(out, "|---|---|---|---|---|---|---|---|")?;
        let median = |v: &[Duration]| Summary::of(&mut v.to_vec()).median;
        let kib = |b: usize| format!("{:.0}", b as f64 / 1024.0);
        for (name, tiles) in &self.encodings {
            let sum = |f: fn(&TileTimes) -> &[Duration]| -> Duration {
                tiles.iter().map(|t| median(f(t))).sum()
            };
            writeln!(
                out,
                "| {name} | {} | {} | {} | {} | {} | **{}** | {} |",
                tiles.len(),
                kib(tiles.iter().map(|t| t.input).sum()),
                ms(sum(|t| &t.decode)),
                ms(sum(|t| &t.layout)),
                ms(sum(|t| &t.pack)),
                ms(sum(|t| &t.total)),
                kib(tiles.iter().map(|t| t.packed).sum()),
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "Sums of per-tile medians, in ms. Per tile (median, and p90 of the total):"
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "| tile | encoding | KiB in | decode | layout | pack | total | p90 | KiB packed |"
        )?;
        writeln!(out, "|---|---|---|---|---|---|---|---|---|")?;
        for (name, tiles) in &self.encodings {
            for t in tiles {
                let total = Summary::of(&mut t.total.clone());
                writeln!(
                    out,
                    "| {}/{}/{} | {name} | {} | {} | {} | {} | {} | {} | {} |",
                    t.tile.z,
                    t.tile.col,
                    t.tile.row,
                    kib(t.input),
                    ms(median(&t.decode)),
                    ms(median(&t.layout)),
                    ms(median(&t.pack)),
                    ms(total.median),
                    ms(total.p90),
                    kib(t.packed),
                )?;
            }
        }
        Ok(())
    }
}
