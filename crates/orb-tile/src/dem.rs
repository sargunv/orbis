//! DEM tiles on the CPU: a tile's elevations, decoded once in its job, for
//! the terrain's camera, queries and culling (plan §9.4: a CPU DEM, never
//! a GPU readback).

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

impl DemGrid {
    /// Decodes RGBA8 texels with GL JS's unpack vector: metres are
    /// `r·x + g·y + b·z − w` over the 0–255 channels.
    pub fn from_rgba(dim: u32, texels: &[u8], unpack: [f64; 4]) -> Result<Self, String> {
        if texels.len() != (dim * dim * 4) as usize {
            return Err(format!(
                "a {dim}×{dim} DEM tile has {} bytes, not {}",
                texels.len(),
                dim * dim * 4
            ));
        }
        let [r, g, b, offset] = unpack;
        let elevations = texels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|t| {
                (f64::from(t[0]) * r + f64::from(t[1]) * g + f64::from(t[2]) * b - offset) as f32
            })
            .collect();
        Self::from_elevations(dim, elevations)
    }

    /// A tile's elevations in metres, rows top to bottom, as
    /// [`elevations`](Self::elevations) gives them (to cross threads).
    pub fn from_elevations(dim: u32, elevations: Vec<f32>) -> Result<Self, String> {
        if elevations.len() != (dim * dim) as usize {
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
    }
}
