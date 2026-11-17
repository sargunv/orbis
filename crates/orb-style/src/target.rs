//! Render targets (the `dev.sargunv.orbis:targets` preview): offscreen
//! images the size of the view that layers draw into, image-process
//! layers transform, and fills draw through.

use orb_expr::{Color, PropertyExpression};

use crate::Property;

/// A target the root `dev.sargunv.orbis:targets` declares.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderTarget {
    pub id: String,
    pub precision: Precision,
}

/// What a target's channels hold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    /// 8 bits each, as the map's own.
    #[default]
    Standard,
    /// 16-bit floats, kept through the passes of the image processes that
    /// read or write the target: contours of a wide blur at a few
    /// thousandths of its coverage, ramps of smooth fields without bands.
    High,
}

impl Precision {
    pub const NAMES: [&'static str; 2] = ["standard", "high"];

    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "standard" => Self::Standard,
            "high" => Self::High,
            _ => return None,
        })
    }
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
    /// features `size` across, fixed to the map. With a `speed` (logical
    /// pixels a second at the zoom its features are `size` across) it
    /// drifts and changes as time passes: two lattices sliding apart.
    Noise {
        size: Property<f32>,
        amount: Property<f32>,
        speed: Property<f32>,
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

/// How [`Operation::Combine`] puts another target's image with the one
/// processed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Combine {
    /// Only where the other covers: a mask.
    Multiply,
    /// Only where the other doesn't: a cut-out.
    Subtract,
    /// The more covered of the two: a union.
    Max,
    /// The less covered: an intersection.
    Min,
    /// The other drawn over it.
    Over,
    /// The other drawn beneath it.
    Under,
}

impl Combine {
    pub const NAMES: [&'static str; 6] = ["multiply", "subtract", "max", "min", "over", "under"];

    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "multiply" => Self::Multiply,
            "subtract" => Self::Subtract,
            "max" => Self::Max,
            "min" => Self::Min,
            "over" => Self::Over,
            "under" => Self::Under,
            _ => return None,
        })
    }
}

impl Operation {
    /// Whether any of its values reads the camera beyond its zoom: it is
    /// evaluated again as the camera turns and tilts.
    pub fn reads_camera(&self) -> bool {
        self.expressions().any(PropertyExpression::reads_camera)
    }

    /// The expressions among its values.
    fn expressions(&self) -> impl Iterator<Item = &PropertyExpression> {
        let n = Property::<f32>::as_expression;
        let c = Property::<Color>::as_expression;
        let pair = Property::<[f32; 2]>::as_expression;
        let values: Vec<Option<&PropertyExpression>> = match self {
            Self::Select { value, tolerance } => vec![c(value), n(tolerance)],
            Self::Blur { size }
            | Self::HighPass { size }
            | Self::Dilate { size }
            | Self::Erode { size } => vec![n(size)],
            Self::Noise {
                size,
                amount,
                speed,
            } => vec![n(size), n(amount), n(speed)],
            Self::Threshold { cutoff } => vec![n(cutoff)],
            Self::Sharpen { size, amount } => vec![n(size), n(amount)],
            Self::Offset { translate } => vec![pair(translate)],
            Self::Opacity { value } => vec![n(value)],
            Self::Color { value } => vec![c(value)],
            Self::Ramp { stops } => stops.iter().flat_map(|(at, v)| [n(at), c(v)]).collect(),
            Self::Split {
                red, green, blue, ..
            } => vec![pair(red), pair(green), pair(blue)],
            Self::Gradient {
                kind:
                    Gradient::Linear {
                        angle,
                        center,
                        length,
                    },
            } => vec![n(angle), pair(center), length.as_ref().and_then(n)],
            Self::Gradient {
                kind: Gradient::Radial { center, radius },
            } => vec![pair(center), radius.as_ref().and_then(n)],
            Self::Bands { count, width } => vec![n(count), n(width)],
            Self::Invert | Self::Luminance | Self::Convolve { .. } | Self::Combine { .. } => {
                Vec::new()
            }
        };
        values.into_iter().flatten()
    }

    /// The operation names, as `type` gives them.
    pub const NAMES: [&'static str; 19] = [
        "select",
        "blur",
        "noise",
        "threshold",
        "invert",
        "sharpen",
        "high-pass",
        "convolve",
        "offset",
        "dilate",
        "erode",
        "combine",
        "opacity",
        "color",
        "luminance",
        "ramp",
        "split",
        "gradient",
        "bands",
    ];
}
