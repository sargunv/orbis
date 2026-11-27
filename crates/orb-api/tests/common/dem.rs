//! Terrain for tests to draw on: Terrarium DEM tiles of a function of
//! longitude and latitude, written to a scratch directory.

use super::{Scratch, write_png};

/// Terrarium DEM tiles of `metres(lon, lat)` over `(west, south, east,
/// north)`, zooms 0–`max_zoom`, written to a fresh directory; the source
/// `name`'s JSON, and the directory to keep while it's read.
pub(crate) fn dem(
    name: &str,
    (west, south, east, north): (f64, f64, f64, f64),
    max_zoom: u8,
    metres: impl Fn(f64, f64) -> f64,
) -> (String, Scratch) {
    let dir = Scratch::new(&format!("dem-{name}"));
    const SIZE: u32 = 256;
    for z in 0..=max_zoom {
        let n = f64::from(1u32 << z);
        let col = |lon: f64| ((lon + 180.0) / 360.0 * n).floor() as u32;
        let row = |lat: f64| {
            let lat = f64::to_radians(lat);
            ((1.0 - lat.tan().asinh() / std::f64::consts::PI) / 2.0 * n).floor() as u32
        };
        for x in col(west)..=col(east) {
            for y in row(north)..=row(south) {
                let mut pixels = Vec::with_capacity((SIZE * SIZE * 4) as usize);
                for py in 0..SIZE {
                    for px in 0..SIZE {
                        let fx = (f64::from(x) + (f64::from(px) + 0.5) / f64::from(SIZE)) / n;
                        let fy = (f64::from(y) + (f64::from(py) + 0.5) / f64::from(SIZE)) / n;
                        let lon = fx * 360.0 - 180.0;
                        let lat = (std::f64::consts::PI * (1.0 - 2.0 * fy))
                            .sinh()
                            .atan()
                            .to_degrees();
                        let v = ((metres(lon, lat) + 32768.0) * 256.0).round() as u32;
                        pixels.extend([(v >> 16) as u8, (v >> 8) as u8, v as u8, 255]);
                    }
                }
                let path = dir.join(format!("{z}/{x}/{y}.png"));
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                write_png(&path, SIZE, SIZE, &pixels);
            }
        }
    }
    let source = format!(
        r#""{name}": {{"type": "raster-dem", "encoding": "terrarium", "tileSize": {SIZE},
            "maxzoom": {max_zoom}, "bounds": [{west}, {south}, {east}, {north}],
            "tiles": ["file://{}/{{z}}/{{x}}/{{y}}.png"]}},"#,
        dir.display()
    );
    (source, dir)
}

/// Rolling hills over central Tokyo, zooms 0–11.
pub(crate) fn hills() -> (String, Scratch) {
    dem("hills", (139.2, 35.2, 140.3, 36.2), 11, |lon, lat| {
        600.0
            + 500.0 * (lon * 40.0).sin() * (lat * 55.0).cos()
            + 150.0 * (lon * 170.0 + lat * 90.0).sin()
    })
}

/// Steep ridges over the Valais Alps, 700 to about 4,500 m, zooms 0–11,
/// hillshaded on the terrain under a sky: ridge silhouettes stand against
/// it at a high pitch. Crests are rounded: a crease one DEM pixel wide
/// running across the pixel grid shades as dotted lines.
pub(crate) fn alps() -> (String, Scratch) {
    let (dem, dir) = alps_dem();
    let dem = dem.trim_end_matches(',');
    let style = format!(
        r##"{{"version": 8,
            "sources": {{{dem}}},
            "terrain": {{"source": "alps", "exaggeration": 1.2}},
            "sky": {{"sky-color": "#4f8fd9", "horizon-color": "#d6e8fb", "fog-color": "#e4eef8"}},
            "layers": [
              {{"id": "ground", "type": "background", "paint": {{"background-color": "#d8e0c8"}}}},
              {{"id": "relief", "type": "hillshade", "source": "alps"}}
            ]}}"##
    );
    (style, dir)
}

/// The ridges' DEM source `alps`, and its directory.
pub(crate) fn alps_dem() -> (String, Scratch) {
    dem("alps", (7.4, 45.7, 8.4, 46.3), 11, |lon, lat| {
        let ridge = |v: f64| 1.0 - (v.sin().powi(2) + 0.04).sqrt();
        700.0
            + 2600.0 * ridge(lon * 31.0 + lat * 9.0) * ridge(lat * 23.0 - lon * 5.0)
            + 1200.0 * ridge(lon * 97.0 - lat * 61.0)
    })
}
