//! Render targets (the `dev.sargunv.orbis:targets` preview): offscreen
//! images the size of the view that layers draw into, image-process
//! layers transform, and fills draw through.

use orb_expr::Color;

use crate::Property;
use crate::property::v8_enum;

/// A target the root `dev.sargunv.orbis:targets` declares.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderTarget {
    pub id: String,
    pub precision: Precision,
}

v8_enum!(
    /// What a target's channels hold.
    Precision {
        /// 8 bits each, as the map's own.
        Standard = "standard",
        /// 16-bit floats, kept through the passes of the image processes
        /// that read or write the target: contours of a wide blur at a few
        /// thousandths of its coverage, ramps of smooth fields without
        /// bands.
        High = "high",
    }
);

/// The coordinate system and scale of a noise lattice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoiseAnchor {
    /// Map coordinates, fading between lattices at consecutive camera zooms.
    Adaptive,
    /// Map coordinates, with size in pixels at this style zoom.
    Map { reference_zoom: f64 },
    /// Logical pixels from the viewport's top left.
    Viewport,
}

/// One step of an image-process layer's `dev.sargunv.orbis:operations`.
/// Images are premultiplied; coverage is their alpha. Sizes are logical
/// pixels. Numbers and colours may follow the zoom and the camera
/// (`["pitch"]`, `["bearing"]`), evaluated once a frame for the whole
/// layer, as a layer-phase paint property is.
#[derive(Clone, Debug)]
pub enum Operation {
    /// Keeps the pixels whose colour is within `tolerance` (0–255, per
    /// channel) of `value`.
    Select {
        value: Property<Color>,
        tolerance: Property<f32>,
    },
    /// A Gaussian blur of standard deviation `size`, as CSS's `blur()`.
    Blur {
        size: Property<f32>,
    },
    /// Moves coverage by up to `amount` (0–1) of gradient noise with
    /// features `size` across. With a `speed` (logical pixels a second
    /// at the scale where its features are `size` across) it drifts and changes.
    Noise {
        size: Property<f32>,
        amount: Property<f32>,
        speed: Property<f32>,
        anchor: NoiseAnchor,
    },
    /// Coverage at or above `cutoff` (0–255) becomes whole, the rest none.
    Threshold {
        cutoff: Property<f32>,
    },
    /// Coverage becomes its absence.
    Invert,
    /// An unsharp mask: the pixel pushed `amount` away from its blur of
    /// standard deviation `size`.
    Sharpen {
        size: Property<f32>,
        amount: Property<f32>,
    },
    /// The coverage a pixel has beyond its blur of standard deviation
    /// `size`: the band inside an edge.
    HighPass {
        size: Property<f32>,
    },
    /// A convolution by a square kernel of 3, 5 or 7 weights a side, row by
    /// row.
    Convolve {
        kernel: Vec<f32>,
    },
    /// The image moved by `translate` pixels, right and down.
    Offset {
        translate: Property<[f32; 2]>,
    },
    /// Each pixel the most (`Dilate`) or least (`Erode`) of the square
    /// `size` pixels around it: shapes grown or shrunk by that much.
    Dilate {
        size: Property<f32>,
    },
    Erode {
        size: Property<f32>,
    },
    /// The image with another target's (`target`), by `mode`.
    Combine {
        target: String,
        mode: Combine,
    },
    /// Coverage scaled by `value` (0–1).
    Opacity {
        value: Property<f32>,
    },
    /// Every pixel painted `value`, its coverage kept.
    Color {
        value: Property<Color>,
    },
    /// Coverage from the colour's luminance, in white: what an image
    /// (a raster's, say) has of light.
    Luminance,
    /// Coverage looked up in a colour ramp: `(coverage, colour)` stops in
    /// order, interpolated linearly, up to eight. A stop's coverage
    /// (0–1) and colour may follow the zoom and the camera too; a stop
    /// that falls before the one ahead of it is held at that one's.
    Ramp {
        stops: Vec<(Property<f32>, Property<Color>)>,
    },
    /// Each colour channel moved by its own `[x, y]` pixels, right and
    /// down, coverage the most of the three: chromatic aberration. With
    /// `radial`, the moves scale with the distance from the view's centre,
    /// none there and whole at its corners, as a lens's.
    Split {
        red: Property<[f32; 2]>,
        green: Property<[f32; 2]>,
        blue: Property<[f32; 2]>,
        radial: bool,
    },
    /// Coverage replaced by a gradient across the view, 0 to 1, the colour
    /// kept: fixed to the screen, not the map.
    Gradient {
        kind: Gradient,
    },
    /// Coverage made `count` antialiased stripes: whole where the fraction
    /// of coverage times `count` is under `width` (0–1), none elsewhere
    /// and where nothing covered.
    Bands {
        count: Property<f32>,
        width: Property<f32>,
    },
}

/// The shape of an [`Operation::Gradient`]. Its `center` is logical
/// pixels right and down of the view's centre.
#[derive(Clone, Debug)]
pub enum Gradient {
    /// Along `angle` degrees, clockwise from top to bottom (90 runs right
    /// to left), through `center`: 0 to 1 over `length` logical pixels,
    /// or by default over the view corner to corner along the angle.
    Linear {
        angle: Property<f32>,
        center: Property<[f32; 2]>,
        length: Option<Property<f32>>,
    },
    /// From `center` out: 1 at `radius` logical pixels, or by default at
    /// the view's corner farthest from it.
    Radial {
        center: Property<[f32; 2]>,
        radius: Option<Property<f32>>,
    },
}

v8_enum!(
    /// How [`Operation::Combine`] puts another target's image with the one
    /// processed.
    Combine {
        /// Only where the other covers: a mask.
        Multiply = "multiply",
        /// Only where the other doesn't: a cut-out.
        Subtract = "subtract",
        /// The more covered of the two: a union.
        Max = "max",
        /// The less covered: an intersection.
        Min = "min",
        /// The other drawn over it.
        Over = "over",
        /// The other drawn beneath it.
        Under = "under",
    }
);

/// The render target a layer reads (`dev.sargunv.orbis:source-target`),
/// as only fills and image-process layers do.
#[derive(Clone, Debug)]
pub enum TargetRead {
    /// A fill with no source, drawn everywhere through this target's
    /// coverage.
    Fill(String),
    /// An image-process layer: its source target, put through its
    /// operations.
    Process {
        source: String,
        operations: Operations,
    },
}

impl TargetRead {
    /// The target read.
    pub fn source(&self) -> &str {
        match self {
            Self::Fill(source) | Self::Process { source, .. } => source,
        }
    }

    /// The operations an image-process layer puts its source through;
    /// none for a fill.
    pub fn operations(&self) -> &[Operation] {
        match self {
            Self::Fill(_) => &[],
            Self::Process { operations, .. } => operations,
        }
    }
}

/// An image-process layer's operations, in order, and whether any of their
/// values reads the camera beyond its zoom: they are evaluated again as the
/// camera turns and tilts. Only the parser makes them, so the two agree.
#[derive(Clone, Debug, Default)]
pub struct Operations {
    ops: Vec<Operation>,
    reads_camera: bool,
}

impl Operations {
    pub(crate) fn new(ops: Vec<Operation>, reads_camera: bool) -> Self {
        Self { ops, reads_camera }
    }

    /// Whether any of their values reads the camera beyond its zoom.
    pub fn reads_camera(&self) -> bool {
        self.reads_camera
    }
}

impl std::ops::Deref for Operations {
    type Target = [Operation];

    fn deref(&self) -> &[Operation] {
        &self.ops
    }
}

impl<'a> IntoIterator for &'a Operations {
    type Item = &'a Operation;
    type IntoIter = std::slice::Iter<'a, Operation>;

    fn into_iter(self) -> Self::IntoIter {
        self.ops.iter()
    }
}
