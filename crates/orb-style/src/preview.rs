//! Previews: new style syntax beyond the approved v8 spec (style-spec 26.5.1),
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
    /// style-spec#1889: `sky["dev.sargunv.orbis:backdrop-color"]`, a colour
    /// behind the sky.
    BackdropColor,
    /// style-spec#1889: `sky["dev.sargunv.orbis:star-opacity"]`, a fixed
    /// pattern of stars behind the sky.
    StarOpacity,
    /// style-spec#1881: `["dev.sargunv.orbis:meters-per-pixel"]`, the
    /// ground resolution at the camera's target, wherever paint reads the
    /// camera.
    MetersPerPixel,
    /// style-spec#1797: `light["dev.sargunv.orbis:shadow-intensity"]`,
    /// extrusions' shadows on the ground, cast from `light.position`.
    ShadowIntensity,
    /// style-spec#1796: extrusions' walls darkened toward their foot.
    AoWallIntensity,
    /// style-spec#1796: how far up the walls that reaches.
    AoWallRadius,
    /// style-spec#1796: the ground darkened along the walls.
    AoGroundIntensity,
    /// style-spec#1796: how far out from the walls that reaches.
    AoGroundRadius,
    /// style-spec#1047: `dev.sargunv.orbis:fill-rounded-corner-distance`,
    /// fills' corners rounded as extrusions' are.
    FillRoundedCornerDistance,
    /// `metadata["dev.sargunv.orbis:color-space"]`: `"srgb"` or `"linear"`,
    /// the space colours mix in.
    ColorSpace,
    /// `light["dev.sargunv.orbis:ground-lighting"]`: how much `light`
    /// shades the ground, from its day side to its night side.
    GroundLighting,
    /// `light["dev.sargunv.orbis:ambient-color"]`: the colour of the light
    /// the ground keeps where the sun is down.
    AmbientColor,
    /// `light["dev.sargunv.orbis:ambient-intensity"]`: how bright that
    /// light is.
    AmbientIntensity,
    /// `light["dev.sargunv.orbis:twilight"]`: the band of sun elevations
    /// the ground passes from night to day through, in degrees.
    Twilight,
    /// `["dev.sargunv.orbis:daylight"]`: how lit the ground is where a
    /// label stands, which its colours may follow.
    Daylight,
}

impl Preview {
    /// Every preview, in the order the docs list them.
    pub const ALL: [Self; 21] = [
        Self::Equirectangular,
        Self::NaturalEarth,
        Self::EqualEarth,
        Self::ProjectionTransition,
        Self::Crs,
        Self::TileMatrixSet,
        Self::BackdropColor,
        Self::StarOpacity,
        Self::MetersPerPixel,
        Self::ShadowIntensity,
        Self::AoWallIntensity,
        Self::AoWallRadius,
        Self::AoGroundIntensity,
        Self::AoGroundRadius,
        Self::FillRoundedCornerDistance,
        Self::ColorSpace,
        Self::GroundLighting,
        Self::AmbientColor,
        Self::AmbientIntensity,
        Self::Twilight,
        Self::Daylight,
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
            Self::BackdropColor => "dev.sargunv.orbis:backdrop-color",
            Self::StarOpacity => "dev.sargunv.orbis:star-opacity",
            Self::MetersPerPixel => "dev.sargunv.orbis:meters-per-pixel",
            Self::ShadowIntensity => "dev.sargunv.orbis:shadow-intensity",
            Self::AoWallIntensity => {
                "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-intensity"
            }
            Self::AoWallRadius => "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-radius",
            Self::AoGroundIntensity => {
                "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-ground-intensity"
            }
            Self::AoGroundRadius => {
                "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-ground-radius"
            }
            Self::FillRoundedCornerDistance => "dev.sargunv.orbis:fill-rounded-corner-distance",
            Self::ColorSpace => "dev.sargunv.orbis:color-space",
            Self::GroundLighting => "dev.sargunv.orbis:ground-lighting",
            Self::AmbientColor => "dev.sargunv.orbis:ambient-color",
            Self::AmbientIntensity => "dev.sargunv.orbis:ambient-intensity",
            Self::Twilight => "dev.sargunv.orbis:twilight",
            Self::Daylight => "dev.sargunv.orbis:daylight",
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
            Self::BackdropColor | Self::StarOpacity => 1889,
            Self::MetersPerPixel => 1881,
            Self::ShadowIntensity => 1797,
            Self::AoWallIntensity
            | Self::AoWallRadius
            | Self::AoGroundIntensity
            | Self::AoGroundRadius => 1796,
            Self::FillRoundedCornerDistance => 1047,
            Self::NaturalEarth
            | Self::Crs
            | Self::TileMatrixSet
            | Self::ColorSpace
            | Self::GroundLighting
            | Self::AmbientColor
            | Self::AmbientIntensity
            | Self::Twilight
            | Self::Daylight => return None,
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
                "Coordinate reference systems as the projection type: an EPSG code Orbis registers (`EPSG:3413`), `OGC:CRS84`, or a parametric kind, `dev.sargunv.orbis:` and one of `web-mercator`, `geographic`, `equirectangular`, `transverse-mercator`, `utm`, `polar-stereographic`, `lcc`, `albers`, `laea`, `equal-earth` or `natural-earth`, with PROJ-named numeric parameters (`lon_0`, `lat_ts`, …) beside the `type`. CRS identifiers and PROJ parameters keep their standard names. With `dev.sargunv.orbis:transition` a CRS is a low-zoom plane over Web Mercator. Our own design; no proposal yet."
            }
            Self::TileMatrixSet => {
                "`dev.sargunv.orbis:tileMatrixSet` on a source: the OGC 2D tile matrix set its tiles are cut by, a registered identifier (`WorldCRS84Quad`) or an inline document (version 2.0, or 1.0's names). A GeoJSON source ignores it and tiles on the map's own plane. Our own design; no proposal yet."
            }
            Self::BackdropColor => {
                "`sky[\"dev.sargunv.orbis:backdrop-color\"]` (default transparent), the colour the map is drawn on, behind the stars and the sky, which covers it where it's opaque. It takes expressions and transitions like the rest of `sky`. The proposal calls it `sky.backdrop-color`."
            }
            Self::StarOpacity => {
                "`sky[\"dev.sargunv.orbis:star-opacity\"]` (0 to 1, default 0), a fixed procedural pattern of stars on the backdrop that turns with the globe. The draw order is backdrop, stars, then the sky, which covers them where it's opaque and reveals them as it fades with altitude; on a flat map they show above the horizon where the sky is transparent. It takes expressions and transitions like the rest of `sky`. The proposal calls it `sky.star-opacity`."
            }
            Self::MetersPerPixel => {
                "`[\"dev.sargunv.orbis:meters-per-pixel\"]`: ground metres per logical pixel at the camera's target, unpitched, from the projection's scale there (Web Mercator's shrinks toward the poles), so `[\"/\", 10, [\"dev.sargunv.orbis:meters-per-pixel\"]]` is 10 m in pixels. Paint evaluated per frame or per layer reads it, in values the same for every feature, as it reads `[\"pitch\"]`, and so do the overlap, optional and ignore-placement settings at each placement; the rest of layout, laid out once per tile, doesn't. The proposal's name is still open (`[\"meters-per-pixel\"]`, `[\"scale\"]`, `[\"meters-per-px\"]`, `[\"px-per-meter\"]`, `[\"meters-to-px\", n]`, or an engine-provided `global-state` key); Orbis takes the spelled-out form the discussion leans to, after the platform SDKs' meters-per-pixel."
            }
            Self::ShadowIntensity => {
                "`light[\"dev.sargunv.orbis:shadow-intensity\"]` (0 to 1, default 0, off), maplibre/maplibre#555's `light.shadow-intensity` (where the proposal's discussion moved): how much extrusions' shadows darken what lies beneath them, with no colour of their own. They fall away from `light.position` and follow `light.anchor`, as the extrusions' shading does: with the spec's default viewport anchor they turn with the map. Each building's shadow is its walls and roof projected along the light onto the ground it stands on, drawn under the buildings: it darkens the ground and the layers beneath the extrusions, not other buildings (the proposal's later shadow maps would). Overlapping shadows darken once. On terrain each shadow falls where the light meets the terrain, shorter on ground rising away from the light and longer, at most twice as long, on ground falling away, and lies on the surface drawn. It transitions like the rest of `light`. Spec PR #1870's per-layer `fill-extrusion-shadow-opacity` and `-color` are on hold and not read."
            }
            Self::AoWallIntensity => {
                "`dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-intensity` (0 to 1, default 0, off), spec PR #1944's `fill-extrusion-ambient-occlusion-wall-intensity`: walls darken toward the ground they stand on, over the wall radius above it, as MapLibre Native's walls PR (#4909) shades them, from each building's own footprint. On terrain a wall measures its height from the terrain under each point of it. A part standing higher than the wall radius isn't darkened, and neither is anything where buildings stand on or face each other closely. Like the other three, it is zoom-interpolated and transitioned."
            }
            Self::AoWallRadius => {
                "`dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-radius` (metres, default 3), spec PR #1944's `fill-extrusion-ambient-occlusion-wall-radius`: how far up the walls their darkening reaches."
            }
            Self::AoGroundIntensity => {
                "`dev.sargunv.orbis:fill-extrusion-ambient-occlusion-ground-intensity` (0 to 1, default 0, off), spec PR #1944's `fill-extrusion-ambient-occlusion-ground-intensity`: the ground darkens along the walls over the ground radius out from them, rounding off past convex corners; a part standing off the ground darkens it less, and none from the radius up. On terrain the darkening lies on the surface drawn."
            }
            Self::AoGroundRadius => {
                "`dev.sargunv.orbis:fill-extrusion-ambient-occlusion-ground-radius` (metres, default 3), spec PR #1944's `fill-extrusion-ambient-occlusion-ground-radius`: how far out from the walls the ground's darkening reaches."
            }
            Self::FillRoundedCornerDistance => {
                "`dev.sargunv.orbis:fill-rounded-corner-distance` (metres, default 0, off), a layout constant mirroring `fill-extrusion-rounded-corner-distance`: each corner replaced by an arc that far along its edges, at most a fifth of either edge, the same arc as an extrusion's, so a footprint drawn flat matches the building. The outline (`fill-antialias`, `fill-outline-color`) follows the rounded shape, and corners the tile's clipping made stay sharp. The name is Orbis's guess: the proposal has no 2D design yet."
            }
            Self::ColorSpace => {
                "`metadata[\"dev.sargunv.orbis:color-space\"]`, the space colours mix in wherever they meet: `\"srgb\"` (the default) mixes gamma-encoded values, as GL JS and MapLibre Native do; `\"linear\"` mixes linear light: translucent layers, antialiased edges, the terrain's drape, hillshade, fog, the sky's gradients and the globe's scattering, which then adds to what's beneath exactly. Colours in the style keep their meaning: one interpolates as the spec says, then mixes in linear light. In `metadata`, so GL JS, Native and validators ignore it. Our own design; no proposal yet."
            }
            Self::GroundLighting => {
                "`light[\"dev.sargunv.orbis:ground-lighting\"]` (0 to 1, default 0, off): how much `light` shades the ground, so that a globe has a day side and a night side. The ground (every layer beneath the first symbol layer, on the globe, on terrain and on flat maps; labels keep their colours, which may follow it through `[\"dev.sargunv.orbis:daylight\"]`) is multiplied, in linear light, by the sun's `light.color` where the sun is up and by the ambient light (`dev.sargunv.orbis:ambient-color` times `dev.sargunv.orbis:ambient-intensity`) where it's down, passing from one to the other over `dev.sargunv.orbis:twilight`; this value mixes that toward no change. The sun is where `light.position` says, its polar angle from the zenith: with a viewport anchor over the camera's target, its azimuth from the top of the view (turning with the bearing, not the pitch); with a map anchor fixed to the globe, as if over 0°, 0°, so the night side turns with the Earth. Each point takes the globe's normal there, and a flat map its target's; the sky is lit as the target's ground, so a night view's sky is dark too, while the space around a globe seen from orbit stays unlit. As it goes from 0 to 1 the globe's scattering turns from GL JS's sun (the light's position negated, behind the globe) to the ground's, so the haze glows over the day side, and at 0 it is GL JS's. Under a white sun the day side keeps the map's own colours: ground lighting only darkens, and brightening stays `sky.atmosphere-blend`'s, which keeps its spec meaning. `light.intensity` stays the extrusions', and slopes stay the hillshade's. It transitions and takes zoom expressions like the rest of `light`. As a style-spec proposal: `light.ground-lighting` with `light.ambient-color`, `light.ambient-intensity` and `light.twilight`, defined as here, the ambient pair shared with maplibre/maplibre#555's ambient light for extrusions; prior art is Mapbox GL v3's ambient and directional `lights`. Our own design; no proposal yet."
            }
            Self::AmbientColor => {
                "`light[\"dev.sargunv.orbis:ambient-color\"]` (colour, default `white`): the colour of the light the ground keeps where the sun is down, under `dev.sargunv.orbis:ground-lighting`. It mirrors maplibre/maplibre#555's proposed `light.ambient-color` and takes that name if the spec approves it; the extrusions don't read it yet."
            }
            Self::AmbientIntensity => {
                "`light[\"dev.sargunv.orbis:ambient-intensity\"]` (0 to 1, default 0.5): how bright the ambient light is, so that the night side is `dev.sargunv.orbis:ambient-color` times this. It mirrors maplibre/maplibre#555's proposed `light.ambient-intensity` and takes that name if the spec approves it; the extrusions don't read it yet."
            }
            Self::Twilight => {
                "`light[\"dev.sargunv.orbis:twilight\"]` (degrees, 0 to 90, default 12): the band of sun elevations, centred on the horizon, over which ground lit by `dev.sargunv.orbis:ground-lighting` passes smoothly from the ambient light to the sun's; 0 is a sharp terminator. Our own design."
            }
            Self::Daylight => {
                "`[\"dev.sargunv.orbis:daylight\"]`: how lit the ground is where a label stands, from 0 at night to 1 by day, so that labels can read well on both sides of the terminator, as night-mode maps do with light text on dark halos: `[\"interpolate\", [\"linear\"], [\"dev.sargunv.orbis:daylight\"], 0, night, 1, day]`. It is the ground lighting's own day side at the label's anchor (the sun's elevation over the globe's normal there, eased over `dev.sargunv.orbis:twilight`), mixed toward 1 as `dev.sargunv.orbis:ground-lighting` falls, so it is 1 everywhere while the light doesn't shade the ground. `text-color`, `text-halo-color`, `icon-color` and `icon-halo-color` read it, per label every frame in the shaders, as the input of one `step` with one stop or one linear or exponential `interpolate` in RGB with at most two, whose stops may read zoom, data and state like any colour; another shape warns and draws the day side's value. Filters and other properties don't read it. Our own design; no proposal yet."
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
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
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
