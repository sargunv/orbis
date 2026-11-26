//! What a property value may read and what it does read, in one set: a
//! property's spec says which inputs it allows, a parsed value which it
//! reads, and the value fits the property when the one holds the other.

use crate::ir::Deps;

/// A set of the inputs a value reads beside constants and global state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Inputs(u8);

impl Inputs {
    pub const NONE: Self = Self(0);
    /// `["zoom"]`.
    pub const ZOOM: Self = Self(1);
    /// The camera beyond its zoom: `["pitch"]` and `["bearing"]`, in values
    /// the same for every feature.
    pub const CAMERA: Self = Self(2);
    /// The feature's properties, id and geometry.
    pub const DATA: Self = Self(4);
    /// `["feature-state"]`.
    pub const STATE: Self = Self(8);
    /// `["elevation"]`: a DEM's, or the height on an extrusion.
    pub const ELEVATION: Self = Self(16);
    /// The camera and data in one value: a per-feature value scaled, and
    /// bounded, by values of the camera ([`CameraScale`](crate::CameraScale)).
    /// Implies [`CAMERA`](Self::CAMERA).
    pub const CAMERA_DATA: Self = Self(32 | 2);

    pub const ALL: [(Self, &'static str); 6] = [
        (Self::ZOOM, "zoom"),
        (Self::CAMERA, "camera"),
        (Self::CAMERA_DATA, "camera × data"),
        (Self::DATA, "data"),
        (Self::STATE, "state"),
        (Self::ELEVATION, "elevation"),
    ];

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[must_use]
    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[must_use]
    pub const fn minus(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The names of the inputs it holds.
    pub fn names(self) -> impl Iterator<Item = &'static str> {
        Self::ALL
            .into_iter()
            .filter(move |(i, _)| self.contains(*i))
            .map(|(_, name)| name)
    }

    /// The inputs an expression depending on `deps` reads: the camera
    /// together with the feature counts as [`CAMERA_DATA`](Self::CAMERA_DATA).
    pub(crate) fn of(deps: Deps) -> Self {
        let mut inputs = Self::NONE;
        for (dep, input) in [
            (Deps::ZOOM, Self::ZOOM),
            (Deps::CAMERA, Self::CAMERA),
            (Deps::PROPERTIES, Self::DATA),
            (Deps::FEATURE_STATE, Self::STATE),
            (Deps::ELEVATION, Self::ELEVATION),
        ] {
            if deps.contains(dep) {
                inputs = inputs.union(input);
            }
        }
        if deps.contains(Deps::CAMERA) && deps.contains(Deps::FEATURE) {
            inputs = inputs.union(Self::CAMERA_DATA);
        }
        inputs
    }
}
