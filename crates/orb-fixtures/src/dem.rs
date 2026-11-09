//! A plausibility lint for the fixtures' DEM tiles: each `raster-dem`
//! source's tiles near its test's camera, decoded with the encoding the
//! style declares. A tile outside Earth's elevations was encoded otherwise
//! (GL JS's expected images draw the same wrong mountains), so the test
//! can't say whether terrain is right; it skips as `fixture:implausible-dem`.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde_json::Value;

use crate::images::Image;

/// The skip reason of a test whose DEM tiles are implausible.
pub(crate) const IMPLAUSIBLE: &str = "fixture:implausible-dem";
/// Earth's elevations with a margin: below the Challenger Deep, above
/// Everest.
const PLAUSIBLE: std::ops::RangeInclusive<f64> = -12_000.0..=9_000.0;
const MAPBOX: [f64; 4] = [6553.6, 25.6, 0.1, 10000.0];
const TERRARIUM: [f64; 4] = [256.0, 1.0, 1.0 / 256.0, 32768.0];

/// A DEM tile's decoded range under an unpack vector.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Decoded {
    min: f64,
    max: f64,
}

/// What the lint found for a test: why it can't check terrain, one line per
/// implausible tile.
pub(crate) fn lint(style: &Value, width: u32, height: u32, assets: &Path) -> Vec<String> {
    let Some(sources) = style["sources"].as_object() else {
        return Vec::new();
    };
    let camera = Camera {
        lon: style["center"][0].as_f64().unwrap_or(0.0),
        lat: style["center"][1].as_f64().unwrap_or(0.0),
        zoom: style["zoom"].as_f64().unwrap_or(0.0),
        pitch: style["pitch"].as_f64().unwrap_or(0.0),
        size: f64::from(width.max(height)),
    };
    let mut found = Vec::new();
    for (id, source) in sources {
        if source["type"] != "raster-dem" {
            continue;
        }
        let (name, unpack) = encoding(source);
        for template in templates(source, assets) {
            for (tile, path) in tiles(&template, assets) {
                if !camera.reads(tile, f64::from(tile_size(source))) {
                    continue;
                }
                let Some(d) = decode(&path, unpack) else {
                    continue;
                };
                // All no data, or Earth's elevations.
                if d.min > d.max || (PLAUSIBLE.contains(&d.min) && PLAUSIBLE.contains(&d.max)) {
                    continue;
                }
                let others: Vec<String> = [("mapbox", MAPBOX), ("terrarium", TERRARIUM)]
                    .into_iter()
                    .filter(|(other, _)| *other != name)
                    .filter_map(|(other, u)| {
                        let d = decode(&path, u)?;
                        Some(format!("{other} reads {}", range(d)))
                    })
                    .collect();
                found.push(format!(
                    "source {id:?}: {} decodes to {} under {name}; {}",
                    path.strip_prefix(assets).unwrap_or(&path).display(),
                    range(d),
                    others.join(", ")
                ));
            }
        }
    }
    found
}

fn range(d: Decoded) -> String {
    let m = |v: f64| {
        if v.abs() >= 100_000.0 {
            format!("{:.0} km", v / 1000.0)
        } else {
            format!("{v:.0} m")
        }
    };
    format!("{}..{}", m(d.min), m(d.max))
}

/// The source's encoding's name and GL JS's unpack vector for it.
fn encoding(source: &Value) -> (&str, [f64; 4]) {
    match source["encoding"].as_str() {
        Some("terrarium") => ("terrarium", TERRARIUM),
        Some("custom") => {
            let f = |key: &str, default: f64| source[key].as_f64().unwrap_or(default);
            (
                "custom",
                [
                    f("redFactor", 1.0),
                    f("greenFactor", 1.0),
                    f("blueFactor", 1.0),
                    f("baseShift", 0.0),
                ],
            )
        }
        _ => ("mapbox", MAPBOX),
    }
}

fn tile_size(source: &Value) -> u32 {
    source["tileSize"].as_u64().map_or(512, |s| s as u32)
}

/// The source's local tile templates, relative to the assets, from its
/// `tiles` or its local TileJSON's.
fn templates(source: &Value, assets: &Path) -> Vec<String> {
    let tilejson;
    let tiles = match source["url"].as_str().and_then(local) {
        Some(url) => {
            tilejson = std::fs::read_to_string(assets.join(url))
                .ok()
                .and_then(|text| serde_json::from_str::<Value>(&text).ok());
            tilejson.as_ref().map(|t| &t["tiles"])
        }
        None => Some(&source["tiles"]),
    };
    (tiles.and_then(Value::as_array).into_iter().flatten())
        .filter_map(Value::as_str)
        .filter_map(local)
        .map(str::to_string)
        .collect()
}

fn local(url: &str) -> Option<&str> {
    url.strip_prefix("local://")
        .or_else(|| url.strip_prefix("http://localhost:2900/"))
}

/// The asset files a template names, with their tiles: every file in its
/// directory its name matches, or the one file a template without
/// placeholders names (as every tile).
fn tiles(template: &str, assets: &Path) -> Vec<(Option<[u32; 3]>, PathBuf)> {
    let path = assets.join(template);
    if !template.contains('{') {
        return vec![(None, path)];
    }
    let (Some(dir), Some(name)) = (path.parent(), path.file_name()) else {
        return Vec::new();
    };
    let name = name.to_string_lossy();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<_> = (entries.flatten())
        .filter_map(|e| {
            let file = e.file_name();
            let tile = matches(&name, &file.to_string_lossy())?;
            Some((Some(tile), e.path()))
        })
        .collect();
    out.sort();
    out
}

/// `[z, x, y]` from a file name its template (`{z}-{x}-{y}.png`) matches.
fn matches(template: &str, name: &str) -> Option<[u32; 3]> {
    let mut zxy = [None; 3];
    let (mut t, mut n) = (template, name);
    while !t.is_empty() {
        if let Some(rest) = t.strip_prefix('{') {
            let (key, rest) = rest.split_once('}')?;
            let slot = ["z", "x", "y"].iter().position(|k| *k == key)?;
            let digits = n.bytes().take_while(u8::is_ascii_digit).count();
            zxy[slot] = Some(n[..digits].parse().ok()?);
            (t, n) = (rest, &n[digits..]);
        } else {
            let literal = t.find('{').unwrap_or(t.len());
            n = n.strip_prefix(&t[..literal])?;
            t = &t[literal..];
        }
    }
    n.is_empty().then_some([zxy[0]?, zxy[1]?, zxy[2]?])
}

/// The style's camera, roughly: which DEM tiles it could read.
struct Camera {
    lon: f64,
    lat: f64,
    zoom: f64,
    pitch: f64,
    /// The canvas's longer side in logical pixels.
    size: f64,
}

impl Camera {
    /// Whether a DEM tile of `tile_px` pixels (`None`: every tile) lies
    /// near enough the view, at a zoom it could read: within a few zooms
    /// below the view's (terrain reads coarser DEM tiles, pitched views
    /// coarser still), and within a canvas or two of the centre (more when
    /// pitched).
    fn reads(&self, tile: Option<[u32; 3]>, tile_px: f64) -> bool {
        let Some([z, x, y]) = tile else {
            return true;
        };
        let source_zoom = self.zoom + (512.0 / tile_px).log2();
        let z = f64::from(z);
        if z > source_zoom.ceil() + 1.0 || z < source_zoom.floor() - 4.0 {
            return false;
        }
        // The view's centre and reach in world units [0, 1).
        let cx = (self.lon + 180.0) / 360.0;
        let s = self.lat.to_radians().sin().clamp(-0.9999, 0.9999);
        let cy = 0.5 - ((1.0 + s) / (1.0 - s)).ln() / (4.0 * PI);
        let reach = if self.pitch > 0.0 { 8.0 } else { 1.0 };
        let r = reach * self.size / (512.0 * self.zoom.exp2());
        let n = z.exp2();
        let (x0, y0) = (f64::from(x) / n, f64::from(y) / n);
        let gap = |c: f64, lo: f64| (c - (lo + 0.5 / n)).abs() - 0.5 / n;
        gap(cx, x0) <= r && gap(cy, y0) <= r
    }
}

/// A PNG DEM tile's range under `unpack`, cached by file and vector, with
/// terrarium's black as no data (as `orb_tile::DemGrid` reads it). `None`
/// for other formats.
fn decode(path: &Path, unpack: [f64; 4]) -> Option<Decoded> {
    type Cache = Mutex<HashMap<(PathBuf, [u64; 4]), Option<Decoded>>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let key = (path.to_path_buf(), unpack.map(f64::to_bits));
    let cache = CACHE.get_or_init(Default::default);
    if let Some(&d) = cache.lock().expect("lint cache").get(&key) {
        return d;
    }
    let decoded = (path.extension().is_some_and(|e| e == "png"))
        .then(|| Image::read(path).ok())
        .flatten()
        .map(|image| {
            let [r, g, b, w] = unpack;
            let mut d = Decoded {
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
            };
            for t in image.pixels.as_chunks::<4>().0 {
                if unpack == TERRARIUM && t[..3] == [0, 0, 0] {
                    continue;
                }
                let e = f64::from(t[0]) * r + f64::from(t[1]) * g + f64::from(t[2]) * b - w;
                (d.min, d.max) = (d.min.min(e), d.max.max(e));
            }
            d
        });
    cache.lock().expect("lint cache").insert(key, decoded);
    decoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_match_tile_names() {
        let t = "{z}-{x}-{y}.terrain.png";
        assert_eq!(matches(t, "10-189-401.terrain.png"), Some([10, 189, 401]));
        assert_eq!(matches(t, "10-189-401.png"), None);
        assert_eq!(matches(t, "README.md"), None);
        assert_eq!(matches("{z}/{x}/{y}.png", "3.png"), None);
    }

    #[test]
    fn a_camera_reads_tiles_near_it_at_its_zooms() {
        // The Grand Canyon at z9.64, pitched: GL JS's `terrain/skirts-*`.
        let camera = Camera {
            lon: -113.2339,
            lat: 35.9405,
            zoom: 9.64,
            pitch: 36.0,
            size: 512.0,
        };
        assert!(camera.reads(Some([10, 189, 401]), 256.0));
        assert!(!camera.reads(Some([12, 2449, 1677]), 256.0), "the Dead Sea");
        assert!(!camera.reads(Some([2, 0, 1]), 256.0), "too coarse");
        assert!(camera.reads(None, 256.0));
    }
}
