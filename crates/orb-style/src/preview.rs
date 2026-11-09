//! Previews: style inputs beyond the approved v8 spec (style-spec 26.5.1),
//! which a map opts into by name. With a preview off, its inputs are
//! treated like any unknown or invalid style input: a warning, then the
//! default. A preview may change or disappear at any time.

use std::fmt;

/// A style feature beyond the approved spec.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Preview {
    /// style-spec#1623: `equirectangular` and `natural-earth` projection
    /// types, handing over to Mercator between zooms 6 and 7.
    Equirectangular,
    /// style-spec#1865: the `equal-earth` projection type.
    EqualEarth,
    /// style-spec#1866: `projection.transition`, `[start, end]` zooms or
    /// `false`, replacing the zooms a low-zoom projection hands over
    /// between.
    ProjectionTransition,
    /// Coordinate reference systems as the `projection` type (`EPSG:3413`,
    /// or a parametric kind with PROJ-named numeric parameters beside the
    /// `type`) and OGC tile matrix sets on sources (`tileMatrixSet`).
    Crs,
    /// Extended expressions: a property reads each input Orbis honours in
    /// its evaluation [`Phase`](crate::Phase) ([`Prop::extended`](crate::Prop)),
    /// whatever the spec's `property-type` says, among them the camera's
    /// `["pitch"]` and `["bearing"]` in paint drawn per frame.
    Expressions,
    /// style-spec#1889: `sky.backdrop-color` and `sky.star-opacity`, a
    /// colour and a fixed pattern of stars behind the sky.
    Stars,
    /// The globe's scattering composited in linear light.
    Atmosphere,
}

impl Preview {
    /// Every preview, in the order the docs list them.
    pub const ALL: [Self; 7] = [
        Self::Equirectangular,
        Self::EqualEarth,
        Self::ProjectionTransition,
        Self::Stars,
        Self::Crs,
        Self::Expressions,
        Self::Atmosphere,
    ];

    /// The name a map enables it by: the proposal's `style-spec#n`, or
    /// `orbis:` and a name for extensions of our own.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Equirectangular => "style-spec#1623",
            Self::EqualEarth => "style-spec#1865",
            Self::ProjectionTransition => "style-spec#1866",
            Self::Stars => "style-spec#1889",
            Self::Crs => "orbis:crs",
            Self::Expressions => "orbis:expressions",
            Self::Atmosphere => "orbis:atmosphere",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == name)
    }

    /// What it enables, and the design it follows.
    pub const fn summary(self) -> &'static str {
        match self {
            Self::Equirectangular => {
                "`equirectangular` and `natural-earth` projection types, handing over to Mercator between zooms 6 and 7. The proposal's name is still open (`equirectangular`, `eqc`, `geodetic`); Orbis keeps `equirectangular`. No proposal covers `natural-earth` yet."
            }
            Self::EqualEarth => {
                "The `equal-earth` projection type, and zoom interpolations from it to `mercator`, handing over to Mercator between zooms 6 and 7 by default."
            }
            Self::ProjectionTransition => {
                "`projection.transition`: the `[start, end]` zooms a low-zoom projection hands over to Mercator between, or `false` for none. The interpolation syntax stays supported."
            }
            Self::Stars => {
                "`sky.backdrop-color` (default transparent), the colour the map is drawn on, and `sky.star-opacity` (0 to 1, default 0), a fixed procedural pattern of stars on it that turns with the globe. Both take expressions like the rest of `sky`. The draw order is backdrop, stars, then the sky, which covers them where it's opaque and reveals them as it fades with altitude; on a flat map they show above the horizon where the sky is transparent."
            }
            Self::Crs => {
                "Coordinate reference systems as the projection type (`EPSG:3413`, or a parametric kind such as `polar-stereographic` with PROJ-named numeric parameters beside the `type`) and OGC 2D tile matrix sets on sources (`tileMatrixSet`: a registered identifier or an inline document). Our own design; no proposal yet."
            }
            Self::Expressions => {
                "Each property reads every input Orbis honours in the phase it is evaluated in, whatever the spec's `property-type` says; the support matrix lists them. Among them are the camera's `[\"pitch\"]` and `[\"bearing\"]` (degrees) in paint drawn every frame, for values the same for every feature. Our own design; no proposal yet."
            }
            Self::Atmosphere => {
                "The globe's scattering (`sky.atmosphere-blend`) composited in linear light rather than added gamma-encoded as GL JS does: a map's sunlit face keeps its colours, tinted, rather than lifted toward white, the haze gathers toward the limb, and over the sky and the backdrop it adds to what they show. No style input; a rendering change of our own."
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
        }
        let (previews, unknown) = Previews::from_names(["style-spec#1865", "nope", "orbis:crs"]);
        assert_eq!(
            previews.iter().collect::<Vec<_>>(),
            [Preview::EqualEarth, Preview::Crs]
        );
        assert_eq!(unknown, ["nope"]);
        assert_eq!(Previews::from_names(["all"]).0, Previews::ALL);
        assert_eq!(Previews::ALL.iter().count(), Preview::ALL.len());
    }
}
