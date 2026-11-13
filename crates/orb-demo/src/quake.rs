//! The Earthquake scene's ground, as the web demo's
//! (`web/demo/app/quake-worker.js`) makes it: Mapterhorn's terrain tiles
//! decoded to heights, a damped radial wave from an epicentre added, each
//! step handed to the map's host-fed `raster-dem` source. Tiles load on
//! the fetch threads and steps are made on a thread of their own, one at a
//! time: the next once the map shows the last, so the map shows every
//! step at the rate it can and the window's thread only passes them on.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

use orb_api::{Geodetic, TileIndex};
use orb_host::Fetch;
use serde_json::{Value, json};

/// The source the restyled style feeds its terrain and hillshade from.
pub(crate) const SOURCE: &str = "quake";
/// Heights a tile side: Mapterhorn's 512-texel tiles averaged 2×2.
pub(crate) const SIDE: u32 = 256;
const MAPTERHORN: &str = "https://tiles.mapterhorn.com";
/// Metres around the Earth at the equator: Web Mercator's world.
const EARTH: f64 = 40_075_016.686;

/// The wave, as the web demo's `WAVE`: a front at `SPEED` m/s, behind which
/// the ground rings at `HERTZ`, `AMPLITUDE` metres at the epicentre,
/// spreading (as 1/√, halving by `SPREAD` m out) and dying away over
/// `DECAY` s; a new shock every `PERIOD` s.
const SPEED: f64 = 3000.0;
const HERTZ: f64 = 0.5;
const AMPLITUDE: f64 = 900.0;
const SPREAD: f64 = 15_000.0;
const DECAY: f64 = 4.0;
const PERIOD: f64 = 12.0;
/// The wave's table: one entry per `BIN` metres out to `REACH`.
const BIN: f64 = 40.0;
const REACH: f64 = 400_000.0;
const BINS: usize = (REACH / BIN) as usize + 2;

/// The Alps style with its Mapterhorn sources made one the app feeds, as
/// `quakeStyle` in `web/demo/app/quake.js` makes it.
pub(crate) fn style(style: &Value) -> Value {
    let mut style = style.clone();
    let dem: Vec<String> = (style["sources"].as_object().into_iter().flatten())
        .filter(|(_, s)| s["type"] == "raster-dem")
        .map(|(id, _)| id.clone())
        .collect();
    if let Some(sources) = style["sources"].as_object_mut() {
        sources.retain(|id, _| !dem.contains(id));
        sources.insert(
            SOURCE.into(),
            json!({
                "type": "raster-dem",
                "tiles": ["orbis-host:"],
                "encoding": "terrarium",
                "tileSize": 512,
                "maxzoom": 11,
                "attribution": "<a href=\"https://mapterhorn.com/attribution\">© Mapterhorn</a>",
            }),
        );
    }
    for layer in style["layers"].as_array_mut().into_iter().flatten() {
        if layer["source"]
            .as_str()
            .is_some_and(|s| dem.iter().any(|d| d == s))
        {
            layer["source"] = SOURCE.into();
        }
    }
    style["terrain"] = json!({"source": SOURCE, "exaggeration": 1.6});
    style
}

enum Command {
    Tile(TileIndex),
    Loaded(TileIndex, Vec<f32>),
    Epicentre(Geodetic),
    Shown,
    Dropped(Vec<TileIndex>),
}

/// A step's heights, for [`orb_api::Map::set_host_elevations`].
pub(crate) type Step = Vec<(TileIndex, Vec<f32>)>;

/// The quake's thread, which stops when this drops.
pub(crate) struct Quake {
    commands: Sender<Command>,
    steps: Receiver<Step>,
}

impl Quake {
    pub(crate) fn start(epicentre: Geodetic, wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        let (commands, inbox) = channel();
        let (out, steps) = channel();
        let fetch = orb_host_native::NativeFetch::new();
        let loads = commands.clone();
        std::thread::Builder::new()
            .name("quake".into())
            .spawn(move || run(epicentre, &inbox, &out, &*wake, &fetch, &loads))
            .expect("the quake's thread");
        Self { commands, steps }
    }

    /// The map asked for `tile`.
    pub(crate) fn tile(&self, tile: TileIndex) {
        let _ = self.commands.send(Command::Tile(tile));
    }

    pub(crate) fn epicentre(&self, at: Geodetic) {
        let _ = self.commands.send(Command::Epicentre(at));
    }

    /// The map showed the last step.
    pub(crate) fn shown(&self) {
        let _ = self.commands.send(Command::Shown);
    }

    /// Tiles the map no longer holds.
    pub(crate) fn dropped(&self, tiles: Vec<TileIndex>) {
        let _ = self.commands.send(Command::Dropped(tiles));
    }

    /// The next step to hand the map, if one is made.
    pub(crate) fn step(&self) -> Option<Step> {
        self.steps.try_recv().ok()
    }
}

/// A tile: its heights, once loaded, and per texel its distance from the
/// epicentre in bins.
struct Tile {
    base: Option<Vec<f32>>,
    bins: Vec<f32>,
}

/// World x, y in [0, 1) of a point.
fn world(at: Geodetic) -> (f64, f64) {
    let s = at.lat.to_radians().sin();
    (
        at.lon / 360.0 + 0.5,
        0.5 - ((1.0 + s) / (1.0 - s)).ln() / (4.0 * std::f64::consts::PI),
    )
}

fn distances(tile: TileIndex, epicentre: Geodetic) -> Vec<f32> {
    let (ex, ey) = world(epicentre);
    let n = f64::from(1u32 << tile.z);
    let scale = EARTH * epicentre.lat.to_radians().cos() / BIN;
    let side = f64::from(SIDE);
    (0..SIDE * SIDE)
        .map(|k| {
            let x = (f64::from(tile.col) + (f64::from(k % SIDE) + 0.5) / side) / n;
            let y = (f64::from(tile.row) + (f64::from(k / SIDE) + 0.5) / side) / n;
            (((x - ex) * scale).hypot((y - ey) * scale)).min(BINS as f64 - 2.0) as f32
        })
        .collect()
}

/// The wave's swing at each bin out from the epicentre, `t` s after the shock.
fn table(t: f64) -> Vec<f32> {
    (0..BINS)
        .map(|k| {
            let d = k as f64 * BIN;
            let s = t - d / SPEED;
            if s <= 0.0 {
                return 0.0;
            }
            let swing = AMPLITUDE / (1.0 + d / SPREAD).sqrt();
            (swing * (-s / DECAY).exp() * (std::f64::consts::TAU * HERTZ * s).sin()) as f32
        })
        .collect()
}

/// A Mapterhorn tile's terrarium texels as SIDE² heights.
fn decode(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let image = image::load_from_memory(bytes)
        .map_err(|e| e.to_string())?
        .into_rgba8();
    let (width, step) = (image.width(), image.width() / SIDE);
    if image.height() != width || step == 0 {
        return Err(format!("a {width}×{} tile", image.height()));
    }
    let pixels = image.as_raw();
    let metres = |x: u32, y: u32| {
        let k = 4 * (y * width + x) as usize;
        f64::from(pixels[k]) * 256.0 + f64::from(pixels[k + 1]) + f64::from(pixels[k + 2]) / 256.0
            - 32768.0
    };
    Ok((0..SIDE * SIDE)
        .map(|k| {
            let (i, j) = (k % SIDE * step, k / SIDE * step);
            let sum: f64 = (0..step * step)
                .map(|q| metres(i + q % step, j + q / step))
                .sum();
            (sum / f64::from(step * step)) as f32
        })
        .collect())
}

fn run(
    mut epicentre: Geodetic,
    inbox: &Receiver<Command>,
    out: &Sender<Step>,
    wake: &(dyn Fn() + Send + Sync),
    fetch: &Arc<orb_host_native::NativeFetch>,
    loads: &Sender<Command>,
) {
    let mut tiles: HashMap<TileIndex, Tile> = HashMap::new();
    let mut shock = Instant::now();
    let mut in_flight = false;
    loop {
        // Wait for news while a step is out or nothing can be made.
        let ready = !in_flight && tiles.values().any(|t| t.base.is_some());
        let command = match ready {
            true => inbox.try_recv().ok(),
            false => match inbox.recv() {
                Ok(command) => Some(command),
                Err(_) => return,
            },
        };
        match command {
            Some(Command::Tile(tile)) => {
                if tiles.contains_key(&tile) {
                    continue;
                }
                let bins = distances(tile, epicentre);
                tiles.insert(tile, Tile { base: None, bins });
                let url = format!("{MAPTERHORN}/{}/{}/{}.webp", tile.z, tile.col, tile.row);
                let loads = loads.clone();
                fetch.get(
                    orb_api::ResourceKind::Tile,
                    &url,
                    &orb_api::Abort::NEVER,
                    Box::new(move |fetched| {
                        // Where Mapterhorn has no tile, the sea.
                        let base = fetched
                            .map_err(|e| e.to_string())
                            .and_then(|bytes| decode(&bytes))
                            .unwrap_or_else(|_| vec![0.0; (SIDE * SIDE) as usize]);
                        let _ = loads.send(Command::Loaded(tile, base));
                    }),
                );
            }
            Some(Command::Loaded(tile, base)) => {
                if let Some(t) = tiles.get_mut(&tile) {
                    t.base = Some(base);
                }
            }
            Some(Command::Epicentre(at)) => {
                epicentre = at;
                shock = Instant::now();
                for (&tile, t) in &mut tiles {
                    t.bins = distances(tile, epicentre);
                }
            }
            Some(Command::Shown) => in_flight = false,
            Some(Command::Dropped(dropped)) => {
                for tile in dropped {
                    tiles.remove(&tile);
                }
            }
            None => {
                let lut = table(shock.elapsed().as_secs_f64() % PERIOD);
                let step: Step = (tiles.iter())
                    .filter_map(|(&tile, t)| {
                        let base = t.base.as_ref()?;
                        let heights = (base.iter().zip(&t.bins))
                            .map(|(&b, &d)| {
                                let lo = d as usize;
                                let f = d - lo as f32;
                                b + lut[lo] + (lut[lo + 1] - lut[lo]) * f
                            })
                            .collect();
                        Some((tile, heights))
                    })
                    .collect();
                in_flight = true;
                if out.send(step).is_err() {
                    return;
                }
                wake();
            }
        }
    }
}
