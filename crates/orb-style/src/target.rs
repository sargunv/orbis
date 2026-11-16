//! Render targets (the `dev.sargunv.orbis:targets` preview): offscreen
//! images the size of the view that layers draw into, image-process
//! layers transform, and fills draw through.

use orb_expr::Color;

use crate::Property;

/// A target the root `dev.sargunv.orbis:targets` declares.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderTarget {
    pub id: String,
}

/// One step of an image-process layer's `dev.sargunv.orbis:operations`.
/// Images are premultiplied; coverage is their alpha. Sizes are logical
/// pixels. Numbers and colours may follow the zoom.
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
    /// features `size` across, fixed to the map.
    Noise {
        size: Property<f32>,
        amount: Property<f32>,
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
    /// order, interpolated linearly, up to eight.
    Ramp {
        stops: Vec<(f32, Color)>,
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
    /// The operation names, as `type` gives them.
    pub const NAMES: [&'static str; 16] = [
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
    ];
}
