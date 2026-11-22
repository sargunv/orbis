//! Previews: new style syntax beyond the approved v8 spec (style-spec 26.6.0),
//! which a map opts into by name. With a preview off, its inputs are
//! treated like any unknown or invalid style input: a warning, then the
//! default. A preview may change or disappear at any time.

use std::fmt;

/// The namespace of Orbis's style syntax: every key, property, value and
/// expression the approved spec doesn't define is `dev.sargunv.orbis:` and
/// a name, whether it is our own or from an unapproved proposal, and the
/// preview that gates it has the same name. Once the spec approves a
/// proposal, the spec's name replaces ours.
pub const NAMESPACE: &str = "dev.sargunv.orbis:";

/// A style feature beyond the approved spec.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Preview {
    /// style-spec#1623: the `dev.sargunv.orbis:equirectangular` projection
    /// type, handing over to Mercator between zooms 6 and 7.
    Equirectangular,
    /// The `dev.sargunv.orbis:natural-earth` projection type.
    NaturalEarth,
    /// style-spec#1865: the `dev.sargunv.orbis:equal-earth` projection type.
    EqualEarth,
    /// style-spec#1866: `projection["dev.sargunv.orbis:transition"]`,
    /// `[start, end]` zooms or `false`, replacing the zooms a low-zoom
    /// projection hands over between.
    ProjectionTransition,
    /// Coordinate reference systems as the `projection` type (`EPSG:3413`,
    /// or a parametric kind such as `dev.sargunv.orbis:polar-stereographic`
    /// with PROJ-named numeric parameters beside the `type`).
    Crs,
    /// `dev.sargunv.orbis:tileMatrixSet` on sources: the OGC tile matrix
    /// set a source's tiles are cut by.
    TileMatrixSet,
    /// style-spec#1661: `dev.sargunv.orbis:hillshade-layer-blend`, a
    /// hillshade composited in a blend mode.
    HillshadeLayerBlend,
    /// style-spec#1661: `dev.sargunv.orbis:color-relief-layer-blend`, a
    /// colour relief composited in a blend mode.
    ColorReliefLayerBlend,
    /// Render targets after Wet Dog Weather's sketch for Stamen
    /// Watercolor: the root `dev.sargunv.orbis:targets`, layers drawn into
    /// one (`dev.sargunv.orbis:destination-target`), the
    /// `dev.sargunv.orbis.image-process` layer type, and fills drawn
    /// through one's coverage (`dev.sargunv.orbis:source-target`).
    Targets,
    /// style-spec#161: `["dev.sargunv.orbis:distance-from-center"]` in a
    /// symbol layer's filter, after Mapbox's `distance-from-center`.
    DistanceFromCenter,
}

impl Preview {
    /// Every preview, in the order the docs list them.
    pub const ALL: [Self; 10] = [
        Self::Equirectangular,
        Self::NaturalEarth,
        Self::EqualEarth,
        Self::ProjectionTransition,
        Self::Crs,
        Self::TileMatrixSet,
        Self::HillshadeLayerBlend,
        Self::ColorReliefLayerBlend,
        Self::Targets,
        Self::DistanceFromCenter,
    ];

    /// The name a map enables it by: the name of the key, property, value
    /// or expression it gates (`crs` gates a family of projection types).
    pub const fn name(self) -> &'static str {
        match self {
            Self::Equirectangular => "dev.sargunv.orbis:equirectangular",
            Self::NaturalEarth => "dev.sargunv.orbis:natural-earth",
            Self::EqualEarth => "dev.sargunv.orbis:equal-earth",
            Self::ProjectionTransition => "dev.sargunv.orbis:transition",
            Self::Crs => "dev.sargunv.orbis:crs",
            Self::TileMatrixSet => "dev.sargunv.orbis:tileMatrixSet",
            Self::HillshadeLayerBlend => "dev.sargunv.orbis:hillshade-layer-blend",
            Self::ColorReliefLayerBlend => "dev.sargunv.orbis:color-relief-layer-blend",
            Self::Targets => "dev.sargunv.orbis:targets",
            Self::DistanceFromCenter => "dev.sargunv.orbis:distance-from-center",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == name)
    }

    /// The maplibre-style-spec proposal it follows, by issue number; `None`
    /// for our own designs.
    pub const fn proposal(self) -> Option<u32> {
        Some(match self {
            Self::Equirectangular => 1623,
            Self::EqualEarth => 1865,
            Self::ProjectionTransition => 1866,
            Self::HillshadeLayerBlend | Self::ColorReliefLayerBlend => 1661,
            Self::DistanceFromCenter => 161,
            Self::NaturalEarth | Self::Crs | Self::TileMatrixSet | Self::Targets => return None,
        })
    }

    /// What it enables, and the design it follows.
    pub const fn summary(self) -> &'static str {
        match self {
            Self::Equirectangular => {
                "The `dev.sargunv.orbis:equirectangular` projection type, and zoom interpolations from it to `mercator`, handing over to Mercator between zooms 6 and 7 by default. The proposal's name is still open (`equirectangular`, `eqc`, `geodetic`); Orbis takes `equirectangular`."
            }
            Self::NaturalEarth => {
                "The `dev.sargunv.orbis:natural-earth` projection type, and zoom interpolations from it to `mercator`, handing over to Mercator between zooms 6 and 7 by default. Our own; no proposal covers it yet."
            }
            Self::EqualEarth => {
                "The `dev.sargunv.orbis:equal-earth` projection type, and zoom interpolations from it to `mercator`, handing over to Mercator between zooms 6 and 7 by default."
            }
            Self::ProjectionTransition => {
                "`projection[\"dev.sargunv.orbis:transition\"]`: the `[start, end]` zooms a low-zoom projection hands over to Mercator between, or `false` for none. The interpolation syntax stays supported. The proposal calls it `transition`."
            }
            Self::Crs => {
                "Coordinate reference systems as the projection type: an EPSG code Orbis registers (`EPSG:3413`), `OGC:CRS84`, or a parametric kind, `dev.sargunv.orbis:` and one of `web-mercator`, `geographic`, `transverse-mercator`, `utm`, `polar-stereographic`, `lcc`, `albers` or `laea`, with PROJ-named numeric parameters (`lon_0`, `lat_ts`, …) beside the `type`. The world planes (`dev.sargunv.orbis:equirectangular`, `:equal-earth`, `:natural-earth`) need their own previews, and this one for their parameters. CRS identifiers and PROJ parameters keep their standard names. With `dev.sargunv.orbis:transition` a CRS is a low-zoom plane over Web Mercator. Our own design; no proposal yet."
            }
            Self::TileMatrixSet => {
                "`dev.sargunv.orbis:tileMatrixSet` on a source: the OGC 2D tile matrix set its tiles are cut by, a registered identifier (`WorldCRS84Quad`) or an inline document (version 2.0, or 1.0's names). A GeoJSON source ignores it and tiles on the map's own plane. Our own design; no proposal yet."
            }
            Self::HillshadeLayerBlend => {
                "`dev.sargunv.orbis:hillshade-layer-blend` (default `normal`): how a hillshade layer composites with what is beneath it, in the seven modes of the spec's `fill-layer-blend`, `line-layer-blend` and `raster-layer-blend` (W3C Compositing 1). `multiply` shades the colours beneath by the hillshade's own, so land keeps its saturation in the shadows: give it a white or near-white highlight, which multiplies to no change, since multiply can only darken. A constant, as the spec's are. On terrain, a hillshade the terrain lights (viewport-anchored) blends into the draped layers beneath it in the terrain's own draw, where the layer right beneath it drapes and draws normally; otherwise (map-anchored, which drapes, or over a layer that doesn't drape or composites apart) it is a terrain layer of its own, which blends with the map beneath it in its mode, splitting the map's pass there, and fades into the fog by its coverage rather than taking the fog's colour. The proposal names `*-layer-blend` for every layer type; Orbis takes it for the two image layers the spec left out."
            }
            Self::ColorReliefLayerBlend => {
                "`dev.sargunv.orbis:color-relief-layer-blend` (default `normal`): how a colour relief layer composites with what is beneath it, in the same seven modes, so that an elevation ramp can tint (`color`, `multiply`) or light (`screen`, `overlay`) the map beneath rather than cover it. A constant, as the spec's are. The proposal names `*-layer-blend` for every layer type; Orbis takes it for the two image layers the spec left out."
            }
            Self::Targets => {
                "Render targets, after [Wet Dog Weather's sketch](https://www.wetdogweather.com/2024/06/12/how-to-make-stamen-watercolor-in-maplibre/) for Stamen Watercolor: offscreen images the size of the view, a texel a logical pixel, drawn every frame (the sketch's real-time targets), so they keep a vector map's precision at any zoom. The root `dev.sargunv.orbis:targets` declares them (`[{\"id\": \"paint\"}]`). A target's `precision` is `standard` (the default: 8 bits a channel, as the map's) or `high`: 16-bit floats, kept through every pass of the image-process layers that read or write it, whose blurs then reach four standard deviations rather than three, so that contours far out in a wide blur (a cut at a few thousandths of its coverage) and ramps over smooth fields neither round nor band; layers draw at the map's 8 bits, so a high-precision target they draw into holds what image-process layers draw there at those too, which a warning says. Where the GPU can't render to floats (WebGL 2 without `EXT_color_buffer_float`) targets hold 8 bits, and the map reports the `target-precision` gap. A layer with `dev.sargunv.orbis:destination-target` draws into that target rather than the map, in style order; fill-extrusion and heatmap layers draw apart and can't. A `dev.sargunv.orbis.image-process` layer reads its `dev.sargunv.orbis:source-target` through its `dev.sargunv.orbis:operations`, in order, and draws the result into its destination target, or over the map in its place without one. Operations act on premultiplied colour, coverage being alpha, sizes in pixels; their numbers, pairs and colours may follow the zoom and the camera (`[\"pitch\"]`, `[\"bearing\"]`), as a layer-phase paint property may, evaluated for the whole layer as the camera moves: `select` (`value`, a colour; `tolerance`, 0–255 per channel, default 16) keeps the pixels of that colour; `blur` (`size`, the Gaussian's standard deviation, as CSS's `blur()`, a wide one as smooth as a narrow one); `noise` (`size`; `amount`, 0–1, default 0.25; `speed`, logical pixels a second, default 0) moves coverage by gradient noise. With `dev.sargunv.orbis:anchor` set to `map`, `size` is the lattice spacing in logical pixels at `dev.sargunv.orbis:reference-zoom` (0–24, default 14). Use a constant size to make the same marks grow and shrink with the map. With `viewport`, `size` is in logical pixels from the viewport’s top left and the marks stay in place as the map moves. With no anchor, the noise blends between lattices at successive whole zooms to keep the grain roughly the same size on screen. With a speed, the fields drift as time passes, measured in logical pixels per second at the scale where the lattice is `size` across, and the map keeps drawing while the noise shows; `threshold` (`cutoff`, 0–255, default 128) makes coverage whole or none, antialiased; `invert` swaps coverage for its absence; `sharpen` (`size`; `amount`, default 1) pushes a pixel away from its blur; `high-pass` (`size`) keeps the coverage its blur lacks, the band inside an edge; `convolve` (`kernel`, 9, 25 or 49 weights, row by row); `offset` (`translate`, `[x, y]`, right and down) moves the image; `dilate` and `erode` (`size`) grow and shrink shapes by that much; `combine` (`target`; `mode`: `multiply`, the default, `subtract`, `max`, `min`, `over` or `under`) puts another target's image with it; `opacity` (`value`) scales coverage; `color` (`value`) paints it one colour; `luminance` makes coverage of the colour's light; `ramp` (`stops`, up to eight `[coverage, colour]`, either of which may follow the zoom and the camera, a coverage held at least the one before it) colours it by its coverage; `split` (`red`, `green` and `blue`, each `[x, y]`, default `[0, 0]`; `radial`, default false) moves each colour channel by its own pixels, coverage the most of the three, the moves scaled by the distance from the view's centre (none there, whole at its corners) where `radial`, as a lens's chromatic aberration; `gradient` (`kind`: `linear`, the default, along `angle` degrees clockwise from top to bottom, default 0, over `length` pixels, by default the view corner to corner along the angle, or `radial`, out to `radius` pixels, by default the view's corner farthest from its centre; `center`, `[x, y]` pixels right and down of the view's centre, default `[0, 0]`) makes coverage a gradient from 0 to 1, the colour kept, fixed to the screen (with `combine`, vignettes and tilt-shift; an `angle` of `[\"*\", -1, [\"bearing\"]]` turns it with the map); `bands` (`count`, default 8; `width`, 0–1, default 0.5) makes coverage `count` antialiased stripes, whole where the fraction of coverage times the count is under the width and none elsewhere or where nothing covered (rings over a blur, lines over a gradient). A fill with `dev.sargunv.orbis:source-target` and no source draws its colour and pattern everywhere through the target's coverage, at its `fill-layer-opacity` and in its `fill-layer-blend`; what it would read of features draws its default. The view's edges extend what is at them, so a blur sees no edge there. What is drawn into or through a target doesn't drape on terrain, and noise follows the plane on the globe. Our own design; no proposal yet."
            }
            Self::DistanceFromCenter => {
                "`[\"dev.sargunv.orbis:distance-from-center\"]` in a symbol layer's `filter`: a symbol's distance from the view's centre along the camera's bearing, in logical pixels on the ground at the camera's zoom over the view's height, as Mapbox's `distance-from-center` measures it: 0 at the centre, negative toward the camera, positive away from it, so a filter such as `[\"<\", [\"dev.sargunv.orbis:distance-from-center\"], 1]` drops the far labels of a tilted view. Placement evaluates the filter's conditions that read it (or `[\"pitch\"]`, `[\"bearing\"]`) per symbol at each camera, at the zoom the tile was laid out at; a symbol they reject is not placed, so it neither shows nor collides. Only conditions of a top-level `all` that read none of these select a tile's features. The proposal asks for styling by distance from the camera without a syntax; Orbis takes Mapbox's. Only a symbol layer's filter may read it."
            }
        }
    }

    const fn bit(self) -> u32 {
        1 << self as u32
    }
}

impl fmt::Display for Preview {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The previews a map enables: none by default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, orb_wire::Wire)]
pub struct Previews(u32);

impl Previews {
    pub const NONE: Self = Self(0);

    /// Every preview this build has.
    pub const ALL: Self = {
        let mut bits = 0;
        let mut i = 0;
        while i < Preview::ALL.len() {
            bits |= Preview::ALL[i].bit();
            i += 1;
        }
        Self(bits)
    };

    pub const fn contains(self, preview: Preview) -> bool {
        self.0 & preview.bit() != 0
    }

    #[must_use]
    pub const fn with(self, preview: Preview) -> Self {
        Self(self.0 | preview.bit())
    }

    /// The previews `names` enable (`"all"` for every one), and the names
    /// that match none.
    pub fn from_names<'a>(names: impl IntoIterator<Item = &'a str>) -> (Self, Vec<&'a str>) {
        let mut previews = Self::NONE;
        let mut unknown = Vec::new();
        for name in names {
            match Preview::from_name(name) {
                Some(p) => previews = previews.with(p),
                None if name == "all" => previews = Self::ALL,
                None => unknown.push(name),
            }
        }
        (previews, unknown)
    }

    pub fn iter(self) -> impl Iterator<Item = Preview> {
        Preview::ALL.into_iter().filter(move |p| self.contains(*p))
    }
}

impl FromIterator<Preview> for Previews {
    fn from_iter<I: IntoIterator<Item = Preview>>(iter: I) -> Self {
        iter.into_iter().fold(Self::NONE, Self::with)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for p in Preview::ALL {
            assert_eq!(Preview::from_name(p.name()), Some(p));
            assert!(p.name().starts_with(NAMESPACE), "{p}");
        }
        let (previews, unknown) = Previews::from_names([
            "dev.sargunv.orbis:equal-earth",
            "nope",
            "style-spec#1865",
            "dev.sargunv.orbis:crs",
        ]);
        assert_eq!(
            previews.iter().collect::<Vec<_>>(),
            [Preview::EqualEarth, Preview::Crs]
        );
        assert_eq!(unknown, ["nope", "style-spec#1865"]);
        assert_eq!(Previews::from_names(["all"]).0, Previews::ALL);
        assert_eq!(Previews::ALL.iter().count(), Preview::ALL.len());
    }
}
