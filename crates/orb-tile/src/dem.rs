//! DEM tiles on the CPU: a tile's elevations, decoded once in its job, for
//! the terrain's camera, queries and culling (a CPU DEM, never a GPU
//! readback).

/// Terrarium's unpack offset: its black texel, −32768 m, marks no data.
const TERRARIUM_OFFSET: f64 = 32768.0;

/// A square DEM tile's elevations in metres, without exaggeration, rows
/// top to bottom. Pixel `(i, j)` describes the cell centred at tile
/// coordinate `((i + 0.5) / dim, (j + 0.5) / dim)`, as GL JS's `DEMData`
/// and the GPU's DEM pages place it.
#[derive(Clone, Debug, PartialEq)]
pub struct DemGrid {
    dim: u32,
    elevations: Vec<f32>,
    min: f32,
    max: f32,
}

/// Texels in a tile `dim` texels a side: at least one.
fn area(dim: u32) -> Result<usize, String> {
    let side = usize::try_from(dim).map_err(|e| e.to_string())?;
    (side.checked_mul(side))
        .filter(|&n| n > 0)
        .ok_or_else(|| format!("a DEM tile can't be {dim} texels a side"))
}

impl DemGrid {
    /// Decodes RGBA8 texels with GL JS's unpack vector: metres are
    /// `r·x + g·y + b·z − w` over the 0–255 channels, except that
    /// terrarium's no-data texel, black (−32768 m), reads as sea level, as
    /// the GPU's decode reads it (`dem.wgsl`).
    pub fn from_rgba(dim: u32, texels: &[u8], unpack: [f64; 4]) -> Result<Self, String> {
        if area(dim)?.checked_mul(4) != Some(texels.len()) {
            return Err(format!(
                "a {dim}×{dim} DEM tile has {} bytes, not {}",
                texels.len(),
                4 * u128::from(dim).pow(2)
            ));
        }
        let [r, g, b, offset] = unpack;
        let elevations = texels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|t| {
                if offset == TERRARIUM_OFFSET && t[..3] == [0, 0, 0] {
                    return 0.0;
                }
                (f64::from(t[0]) * r + f64::from(t[1]) * g + f64::from(t[2]) * b - offset) as f32
            })
            .collect();
        Self::from_elevations(dim, elevations)
    }

    /// A tile's elevations in metres, rows top to bottom, as
    /// [`elevations`](Self::elevations) gives them (to cross threads).
    pub fn from_elevations(dim: u32, elevations: Vec<f32>) -> Result<Self, String> {
        if elevations.len() != area(dim)? {
            return Err(format!(
                "a {dim}×{dim} DEM tile has {} elevations",
                elevations.len()
            ));
        }
        let (min, max) = elevations
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &e| {
                (lo.min(e), hi.max(e))
            });
        Ok(Self {
            dim,
            elevations,
            min,
            max,
        })
    }

    pub fn elevations(&self) -> &[f32] {
        &self.elevations
    }

    /// Pixels per side.
    pub fn dim(&self) -> u32 {
        self.dim
    }

    pub fn min(&self) -> f32 {
        self.min
    }

    pub fn max(&self) -> f32 {
        self.max
    }

    /// Pixel `(i, j)`'s elevation, clamped to the tile.
    pub fn texel(&self, i: i64, j: i64) -> f32 {
        let last = i64::from(self.dim) - 1;
        let (i, j) = (i.clamp(0, last), j.clamp(0, last));
        self.elevations[(j * i64::from(self.dim) + i) as usize]
    }

    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.elevations.len() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_terrarium_and_bounds() {
        let terrarium = [256.0, 1.0, 1.0 / 256.0, 32768.0];
        // 32768 m offset: (128, 0, 0) is 0 m, (129, 2, 128) is 258.5 m.
        let texels = [
            128, 0, 0, 255, 129, 2, 128, 255, 127, 0, 0, 255, 128, 10, 0, 255,
        ];
        let grid = DemGrid::from_rgba(2, &texels, terrarium).unwrap();
        assert_eq!(grid.texel(0, 0), 0.0);
        assert_eq!(grid.texel(1, 0), 258.5);
        assert_eq!(grid.texel(0, 1), -256.0);
        assert_eq!(grid.texel(5, 5), 10.0, "clamped to the last pixel");
        assert_eq!((grid.min(), grid.max()), (-256.0, 258.5));
        assert!(DemGrid::from_rgba(3, &texels, terrarium).is_err());
        // Tiles without texels, and sides whose texels overflow, are refused.
        assert!(DemGrid::from_rgba(0, &[], terrarium).is_err());
        assert!(DemGrid::from_rgba(1 << 15, &[], terrarium).is_err());
        assert!(DemGrid::from_elevations(0, vec![]).is_err());
    }

    #[test]
    fn terrarium_no_data_is_sea_level() {
        let terrarium = [256.0, 1.0, 1.0 / 256.0, 32768.0];
        let mapbox = [6553.6, 25.6, 0.1, 10000.0];
        // Voids, whatever their alpha, beside 258.5 m and −256 m.
        let texels = [0, 0, 0, 255, 129, 2, 128, 255, 127, 0, 0, 255, 0, 0, 0, 0];
        let grid = DemGrid::from_rgba(2, &texels, terrarium).unwrap();
        assert_eq!((grid.texel(0, 0), grid.texel(1, 1)), (0.0, 0.0));
        assert_eq!((grid.min(), grid.max()), (-256.0, 258.5));
        // Mapbox's black is −10000 m, within Earth's range: data.
        let grid = DemGrid::from_rgba(2, &texels, mapbox).unwrap();
        assert_eq!(grid.texel(0, 0), -10000.0);
    }
}
