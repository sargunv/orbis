//! A vector tile through a tile job's stages one call at a time, so a
//! benchmark can time each: decode the bytes, lay the tile out, and pack
//! the result as the web's job workers send it ([`crate::pack`]). A job
//! runs the same code ([`TileWorker::build`]), in one call.

use std::sync::Arc;

use orb_crs::TileIndex;
use orb_style::Encoding;

use crate::glyphs::LocalGlyphs;
use crate::loader::TileLocation;
use crate::tile::TileBuild;
use crate::worker::{Built, TileRequest, TileWorker};

/// One style's tile jobs, stage by stage.
#[derive(Debug)]
pub struct Stages {
    worker: TileWorker,
    encoding: Encoding,
}

impl Stages {
    /// Jobs for every layer and source of a v8 style whose tiles come in
    /// `encoding`. Ideographs draw from `local` glyphs in the
    /// `local_ideographs` family when both are given, as a map's do.
    pub fn new(
        style_json: &str,
        encoding: Encoding,
        local: Option<Arc<dyn LocalGlyphs>>,
        local_ideographs: Option<String>,
    ) -> Result<Self, String> {
        let style = orb_style::v8::parse(style_json).map_err(|d| d.to_string())?;
        let layers = (1..=style.layers.len() as u32).collect();
        let sources = (1..=style.sources.len() as u32).collect();
        let worker = TileWorker::with_style(
            None,
            Arc::new(style),
            layers,
            sources,
            &[],
            0,
            local_ideographs,
            crate::surface::SurfaceTiles::default(),
        );
        if let Some(local) = local {
            worker.set_local_glyphs(local);
        }
        Ok(Self { worker, encoding })
    }

    /// Adds a resource a layout asked for (a glyph range), or its failure.
    pub fn add_resource(&self, url: &str, bytes: Result<&[u8], String>) {
        self.worker.add_resource(url, bytes);
    }

    /// Decodes a tile of source `source` (its key: the style's sources
    /// count from 1) into the worker's cache, as a job's first stage.
    pub fn decode(&self, source: u32, tile: TileIndex, bytes: &[u8]) -> Result<(), String> {
        self.worker
            .frame(bytes, self.encoding, &self.request(source, tile))
            .map(drop)
    }

    /// Lays out a tile [`decode`](Self::decode) cached: the build, or the
    /// URLs its labels need first ([`add_resource`](Self::add_resource)).
    pub fn layout(
        &self,
        source: u32,
        tile: TileIndex,
    ) -> Result<Result<TileBuild, Vec<String>>, String> {
        Ok(
            match self
                .worker
                .build(&[], self.encoding, &self.request(source, tile))?
            {
                Built::Tile(build) => Ok(build),
                Built::Fetch(urls) => Err(urls),
            },
        )
    }

    /// The whole tile at its own zoom, as a map requests it.
    fn request(&self, source: u32, tile: TileIndex) -> TileRequest {
        TileRequest {
            style: 1,
            source,
            tile,
            zoom: f64::from(tile.z),
            bend: None,
            data_zoom: tile.z,
            location: TileLocation::Url(String::new()),
            encoding: self.encoding,
            nodes: None,
            rows_only: false,
            dem_unpack: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{ASSETS, SYMBOL_TILE, SYMBOLS};

    #[test]
    fn stages_lay_out_what_a_job_does() {
        let stages = Stages::new(SYMBOLS, Encoding::Mvt, None, None).unwrap();
        let (z, x, y) = SYMBOL_TILE;
        let tile = TileIndex::new(z, x, y);
        let bytes = std::fs::read(format!("{ASSETS}/tiles/{z}-{x}-{y}.mvt")).unwrap();
        stages.decode(1, tile, &bytes).unwrap();
        let urls = stages.layout(1, tile).unwrap().unwrap_err();
        for url in &urls {
            let path = format!("{ASSETS}/{}", url.strip_prefix("local://").unwrap());
            stages.add_resource(url, Ok(&std::fs::read(path).unwrap()));
        }
        let build = stages.layout(1, tile).unwrap().unwrap();
        let (_, job) = crate::testing::symbol_tile();
        assert_eq!(build.symbols.len(), job.symbols.len());
        assert!(!crate::pack(build).is_empty());
    }
}
