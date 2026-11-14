//! Icons: sprite metadata, and GL JS's `shapeIcon`, `fitIconToText` and
//! `applyTextFit` (`shaping.ts`).

use orb_text::{Anchor, Shaping};

/// How an icon's content box follows the text on one axis
/// (`textFitWidth`, `textFitHeight` in sprite metadata).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, orb_wire::Wire)]
pub enum TextFit {
    #[default]
    StretchOrShrink,
    StretchOnly,
    Proportional,
}

/// A sprite image as layout sees it (GL JS `StyleImage` and
/// `ImagePosition`), without its pixels or atlas position.
#[derive(Clone, Debug, PartialEq, orb_wire::Wire)]
pub struct IconImage {
    /// Size in image pixels.
    pub size: [u32; 2],
    pub pixel_ratio: f64,
    pub sdf: bool,
    /// Stretchable ranges in image pixels; `None` when the metadata has
    /// none (an empty list stretches nothing).
    pub stretch_x: Option<Vec<[f64; 2]>>,
    pub stretch_y: Option<Vec<[f64; 2]>>,
    /// `[left, top, right, bottom]` in image pixels.
    pub content: Option<[f64; 4]>,
    pub text_fit_width: Option<TextFit>,
    pub text_fit_height: Option<TextFit>,
}

impl IconImage {
    /// A plain image: no stretch, content or text fit.
    pub fn new(size: [u32; 2], pixel_ratio: f64, sdf: bool) -> Self {
        Self {
            size,
            pixel_ratio,
            sdf,
            stretch_x: None,
            stretch_y: None,
            content: None,
            text_fit_width: None,
            text_fit_height: None,
        }
    }

    /// Size in layout pixels (GL JS `ImagePosition.displaySize`).
    pub fn display_size(&self) -> [f64; 2] {
        [
            f64::from(self.size[0]) / self.pixel_ratio,
            f64::from(self.size[1]) / self.pixel_ratio,
        ]
    }

    /// As text sees it inline.
    pub fn inline(&self) -> orb_text::InlineImage {
        orb_text::InlineImage {
            padded_size: [self.size[0] + 2, self.size[1] + 2],
            pixel_ratio: self.pixel_ratio,
        }
    }

    pub(crate) fn has_text_fit(&self) -> bool {
        self.content.is_some() && (self.text_fit_width.is_some() || self.text_fit_height.is_some())
    }
}

/// An icon's box around the anchor, in layout pixels (GL JS
/// `PositionedIcon`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PositionedIcon<'a> {
    pub(crate) image: &'a IconImage,
    pub(crate) top: f64,
    pub(crate) bottom: f64,
    pub(crate) left: f64,
    pub(crate) right: f64,
    /// Content padding `[left, top, right, bottom]` when fitted to text.
    pub(crate) collision_padding: Option<[f64; 4]>,
}

/// An axis-aligned box `x1, y1, x2, y2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Box2 {
    pub(crate) x1: f64,
    pub(crate) y1: f64,
    pub(crate) x2: f64,
    pub(crate) y2: f64,
}

/// GL JS `shapeIcon`.
pub(crate) fn shape_icon(
    image: &IconImage,
    offset: [f64; 2],
    anchor: Anchor,
) -> PositionedIcon<'_> {
    let (horizontal, vertical) = anchor.alignment();
    let [w, h] = image.display_size();
    let x1 = offset[0] - w * horizontal;
    let y1 = offset[1] - h * vertical;
    PositionedIcon {
        image,
        top: y1,
        bottom: y1 + h,
        left: x1,
        right: x1 + w,
        collision_padding: None,
    }
}

/// `icon-text-fit`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IconTextFit {
    #[default]
    None,
    Width,
    Height,
    Both,
}

/// Stretches or centres an icon on the text (GL JS `fitIconToText`).
/// `padding` is `icon-text-fit-padding`, CSS order.
pub(crate) fn fit_icon_to_text<'a>(
    icon: &PositionedIcon<'a>,
    text: &Shaping,
    fit: IconTextFit,
    padding: [f64; 4],
    offset: [f64; 2],
    font_scale: f64,
) -> PositionedIcon<'a> {
    let image = icon.image;
    let [w, h] = image.display_size();
    let collision_padding = image.content.map(|content| {
        let pixel_ratio = if image.pixel_ratio == 0.0 || image.pixel_ratio.is_nan() {
            1.0
        } else {
            image.pixel_ratio
        };
        [
            content[0] / pixel_ratio,
            content[1] / pixel_ratio,
            w - content[2] / pixel_ratio,
            h - content[3] / pixel_ratio,
        ]
    });

    let text_left = text.left * font_scale;
    let text_right = text.right * font_scale;
    let (left, right) = if matches!(fit, IconTextFit::Width | IconTextFit::Both) {
        (
            offset[0] + text_left - padding[3],
            offset[0] + text_right + padding[1],
        )
    } else {
        let left = offset[0] + (text_left + text_right - w) / 2.0;
        (left, left + w)
    };

    let text_top = text.top * font_scale;
    let text_bottom = text.bottom * font_scale;
    let (top, bottom) = if matches!(fit, IconTextFit::Height | IconTextFit::Both) {
        (
            offset[1] + text_top - padding[0],
            offset[1] + text_bottom + padding[2],
        )
    } else {
        let top = offset[1] + (text_top + text_bottom - h) / 2.0;
        (top, top + h)
    };

    PositionedIcon {
        image,
        top,
        bottom,
        left,
        right,
        collision_padding,
    }
}

/// The box of an icon fitted to text, kept to its content's aspect ratio
/// on proportional axes (GL JS `applyTextFit`). The image has content.
pub(crate) fn apply_text_fit(icon: &PositionedIcon<'_>) -> Box2 {
    let mut left = icon.left;
    let mut top = icon.top;
    let mut width = icon.right - left;
    let mut height = icon.bottom - top;
    let content = icon.image.content.expect("text fit needs content");
    let content_width = content[2] - content[0];
    let content_height = content[3] - content[1];
    let fit_width = icon.image.text_fit_width.unwrap_or_default();
    let fit_height = icon.image.text_fit_height.unwrap_or_default();
    let aspect = content_width / content_height;
    if fit_height == TextFit::Proportional {
        if (fit_width == TextFit::StretchOnly && width / height < aspect)
            || fit_width == TextFit::Proportional
        {
            let new_width = (height * aspect).ceil();
            left *= new_width / width;
            width = new_width;
        }
    } else if fit_width == TextFit::Proportional
        && fit_height == TextFit::StretchOnly
        && aspect != 0.0
        && width / height > aspect
    {
        let new_height = (width / aspect).ceil();
        top *= new_height / height;
        height = new_height;
    }
    Box2 {
        x1: left,
        y1: top,
        x2: left + width,
        y2: top + height,
    }
}
