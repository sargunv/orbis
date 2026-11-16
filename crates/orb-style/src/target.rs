//! Render targets (the `dev.sargunv.orbis:targets` preview): offscreen
//! images the size of the view that layers draw into, image-process
//! layers transform, and fills draw through.

use orb_expr::Color;

/// A target the root `dev.sargunv.orbis:targets` declares.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderTarget {
    pub id: String,
}

/// One step of an image-process layer's `dev.sargunv.orbis:operations`.
/// Images are premultiplied; coverage is their alpha. Sizes are logical
/// pixels.
#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    /// Keeps the pixels whose colour is within `tolerance` (0–255, per
    /// channel) of `value`.
    Select { value: Color, tolerance: f32 },
    /// A Gaussian blur of standard deviation `size`, as CSS's `blur()`.
    Blur { size: f32 },
    /// Moves coverage by up to `amount` (0–1) of gradient noise with
    /// features `size` across, fixed to the map.
    Noise { size: f32, amount: f32 },
    /// Coverage at or above `cutoff` (0–255) becomes whole, the rest none.
    Threshold { cutoff: f32 },
    /// Coverage becomes its absence.
    Invert,
    /// An unsharp mask: the pixel pushed `amount` away from its blur of
    /// standard deviation `size`.
    Sharpen { size: f32, amount: f32 },
    /// The coverage a pixel has beyond its blur of standard deviation
    /// `size`: the band inside an edge.
    HighPass { size: f32 },
    /// A convolution by a square kernel of 3, 5 or 7 weights a side, row by
    /// row.
    Convolve { kernel: Vec<f32> },
}

impl Operation {
    /// The operation names, as `type` gives them.
    pub const NAMES: [&'static str; 8] = [
        "select",
        "blur",
        "noise",
        "threshold",
        "invert",
        "sharpen",
        "high-pass",
        "convolve",
    ];
}
