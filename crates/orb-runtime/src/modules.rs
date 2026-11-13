//! Feature modules: code a plain vector map (MVT on Web Mercator) never
//! runs, which a host may load apart from the rest, as the web loads each
//! from a wasm module of its own (`cargo xtask web`). A map runs no code
//! of a module its host hasn't loaded: a style that needs one waits to be
//! installed until it is in (edits go to it meanwhile), a source whose
//! TileJSON names MLT sends no tile requests until its decoders are, and
//! an app's GeoJSON answer for a tile it feeds waits for GeoJSON's code.
//! What runs before that gate (reading and parsing a style, and
//! [`of_style`]) stays in the code every map has, whatever it calls
//! (`cargo xtask web`'s split); editing one, which only the API starts,
//! goes with the API's module. Native hosts have every module from the
//! start.

use orb_style::{Encoding, SourceKind, Style};

/// A set of feature modules.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Modules(u8);

impl Modules {
    pub const NONE: Self = Self(0);
    /// MLT tiles' decoders.
    pub const MLT: Self = Self(1);
    /// `pmtiles://` archives and gzip.
    pub const ARCHIVE: Self = Self(1 << 1);
    /// `mbtiles://` archives: SQLite files.
    pub const SQLITE: Self = Self(1 << 2);
    /// `raster-dem` sources: terrain, hillshade and color-relief.
    pub const DEM: Self = Self(1 << 3);
    /// Projections but Web Mercator, other CRSs and their warps.
    pub const GLOBE: Self = Self(1 << 4);
    /// GeoJSON sources' parsing and tiling.
    pub const GEOJSON: Self = Self(1 << 5);
    /// Queries, edits, snapshots and offline regions: the host loads it
    /// on idle, or before the first call that needs it.
    pub const API: Self = Self(1 << 6);
    pub const ALL: Self = Self((1 << 7) - 1);

    /// Each module and its name, as `cargo xtask web` names its file.
    pub const NAMES: [(Self, &'static str); 7] = [
        (Self::MLT, "mlt"),
        (Self::ARCHIVE, "archive"),
        (Self::SQLITE, "sqlite"),
        (Self::DEM, "dem"),
        (Self::GLOBE, "globe"),
        (Self::GEOJSON, "geojson"),
        (Self::API, "api"),
    ];

    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Those of `self` not in `other`.
    #[must_use]
    pub fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// The modules' names.
    pub fn names(self) -> impl Iterator<Item = &'static str> {
        Self::NAMES
            .into_iter()
            .filter(move |&(m, _)| self.contains(m))
            .map(|(_, name)| name)
    }

    /// The modules as bits, one per module in [`Modules::NAMES`]' order.
    pub fn bits(self) -> u8 {
        self.0
    }

    /// The modules of these bits ([`Modules::bits`]).
    pub fn from_bits(bits: u8) -> Self {
        Self(bits & Self::ALL.0)
    }

    /// The module named `name`.
    pub fn named(name: &str) -> Option<Self> {
        Self::NAMES
            .into_iter()
            .find(|&(_, n)| n == name)
            .map(|(m, _)| m)
    }
}

impl std::ops::BitOr for Modules {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl std::ops::BitOrAssign for Modules {
    fn bitor_assign(&mut self, other: Self) {
        *self = self.union(other);
    }
}

/// What `url`, a source's TileJSON URL, needs: its archive's reader, and
/// for MBTiles gzip too, which is in [`Modules::ARCHIVE`] (as the split's
/// `needs` lists).
fn archive(url: Option<&str>) -> Modules {
    match url {
        Some(url) if url.starts_with("pmtiles://") => Modules::ARCHIVE,
        Some(url) if url.starts_with("mbtiles://") => Modules::SQLITE | Modules::ARCHIVE,
        _ => Modules::NONE,
    }
}

/// The modules `style` needs before it's installed: what its projection,
/// sources and terrain declare. A source whose TileJSON or archive names
/// MLT needs [`Modules::MLT`] too, once that is known.
pub fn of_style(style: &Style) -> Modules {
    let mut needed = Modules::NONE;
    let mercator = orb_crs::TileMatrixSet::web_mercator();
    let projected = style
        .projection
        .is_some_and(|p| !p.same_plane(mercator.projection()));
    if style.globe.is_some() || projected || !style.tile_matrix_sets().is_empty() {
        needed |= Modules::GLOBE;
    }
    if style.terrain.is_some() {
        needed |= Modules::DEM;
    }
    for source in &style.sources {
        needed |= match &source.kind {
            SourceKind::Vector(v) => {
                let mlt = match v.encoding {
                    Some(Encoding::Mlt) => Modules::MLT,
                    _ => Modules::NONE,
                };
                mlt | archive(v.tileset.url.as_deref())
            }
            SourceKind::Raster(r) => archive(r.tileset.url.as_deref()),
            SourceKind::RasterDem(d) => Modules::DEM | archive(d.raster.tileset.url.as_deref()),
            SourceKind::GeoJson(_) => Modules::GEOJSON,
            SourceKind::Image(_) => Modules::NONE,
        };
    }
    needed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn needs(style: &str) -> Vec<&'static str> {
        let style = orb_style::v8::parse_with(style, orb_style::Previews::ALL, &[]).unwrap();
        of_style(&style).names().collect()
    }

    #[test]
    fn a_style_needs_what_it_declares() {
        assert!(needs(r#"{"version":8,"sources":{"v":{"type":"vector","tiles":["https://x/{z}/{x}/{y}.pbf"]}},"layers":[]}"#).is_empty());
        assert_eq!(
            needs(
                r#"{"version":8,"projection":{"type":"globe"},"sources":{
                "a":{"type":"vector","url":"pmtiles://https://x/a.pmtiles"},
                "m":{"type":"vector","url":"mbtiles:///a.mbtiles","encoding":"mlt"},
                "g":{"type":"geojson","data":{"type":"FeatureCollection","features":[]}},
                "d":{"type":"raster-dem","tiles":["https://x/{z}/{x}/{y}.png"]}},"layers":[]}"#
            ),
            ["mlt", "archive", "sqlite", "dem", "globe", "geojson"],
        );
        assert_eq!(Modules::named("dem"), Some(Modules::DEM));
        assert_eq!(Modules::from_bits(Modules::DEM.bits()), Modules::DEM);
        assert_eq!(Modules::from_bits(0xff), Modules::ALL);
        assert!(Modules::ALL.contains(Modules::API));
        assert_eq!(Modules::ALL.without(Modules::API).names().count(), 6);
    }
}
