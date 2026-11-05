//! Glyph and icon quads: GL JS `quads.ts`, line by line.
//!
//! Corners are em-space offsets from the anchor, in layout pixels at
//! [`ONE_EM`]. Texture rectangles are relative to the glyph's or image's
//! padded rectangle, never to an atlas.

use orb_text::{
    DEFAULT_OFFSET, GlyphId, GlyphMap, GlyphSource, ONE_EM, ShapedSection, Shaping, pbf::BORDER,
};

use crate::geometry::{Point, degrees_to_radians, rotation_matrix};
use crate::icon::{Box2, PositionedIcon, apply_text_fit};
use crate::layout::TileImages;

/// GL JS `IMAGE_PADDING`: transparent texels around every image.
const IMAGE_PADDING: f64 = 1.0;
/// GL JS's glyph atlas `padding`: texels around every glyph bitmap.
pub(crate) const GLYPH_PADDING: u32 = 1;

/// What a quad samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Texture {
    Glyph(GlyphId),
    /// A tile-local image index.
    Image(u32),
}

/// A texel rectangle relative to the padded glyph or image rectangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TexRect {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) w: f64,
    pub(crate) h: f64,
}

/// One textured quad of a glyph or icon (GL JS `SymbolQuad`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SymbolQuad {
    pub(crate) tl: Point,
    pub(crate) tr: Point,
    pub(crate) bl: Point,
    pub(crate) br: Point,
    pub(crate) tex: TexRect,
    pub(crate) texture: Texture,
    pub(crate) pixel_offset_tl: Point,
    pub(crate) pixel_offset_br: Point,
    pub(crate) glyph_offset: [f64; 2],
    pub(crate) section: usize,
    pub(crate) sdf: bool,
    pub(crate) min_font_scale: [f64; 2],
}

#[derive(Clone, Copy)]
struct Cut {
    fixed: f64,
    stretch: f64,
}

/// The quads of an icon: one, or nine-patch pieces when it stretches to
/// fit text (GL JS `getIconQuads`). `image` is the icon's tile-local index.
pub(crate) fn icon_quads(
    icon: &PositionedIcon<'_>,
    image: u32,
    rotate: f64,
    sdf: bool,
    has_icon_text_fit: bool,
    out: &mut Vec<SymbolQuad>,
) {
    let meta = icon.image;
    let pixel_ratio = meta.pixel_ratio;
    let image_width = f64::from(meta.size[0]);
    let image_height = f64::from(meta.size[1]);

    let mut bounds = Box2 {
        x1: icon.left,
        y1: icon.top,
        x2: icon.right,
        y2: icon.bottom,
    };

    let whole_x = [[0.0, image_width]];
    let whole_y = [[0.0, image_height]];
    let stretch_x: &[[f64; 2]] = meta.stretch_x.as_deref().unwrap_or(&whole_x);
    let stretch_y: &[[f64; 2]] = meta.stretch_y.as_deref().unwrap_or(&whole_y);

    let sum = |ranges: &[[f64; 2]]| ranges.iter().fold(0.0, |s, r| s + r[1] - r[0]);
    let stretch_width = sum(stretch_x);
    let stretch_height = sum(stretch_y);
    let fixed_width = image_width - stretch_width;
    let fixed_height = image_height - stretch_height;

    let mut stretch_offset_x = 0.0;
    let mut stretch_content_width = stretch_width;
    let mut stretch_offset_y = 0.0;
    let mut stretch_content_height = stretch_height;
    let mut fixed_offset_x = 0.0;
    let mut fixed_content_width = fixed_width;
    let mut fixed_offset_y = 0.0;
    let mut fixed_content_height = fixed_height;

    if let Some(content) = meta.content.filter(|_| has_icon_text_fit) {
        let content_width = content[2] - content[0];
        let content_height = content[3] - content[1];
        if meta.text_fit_width.is_some() || meta.text_fit_height.is_some() {
            bounds = apply_text_fit(icon);
        }
        stretch_offset_x = sum_within_range(stretch_x, 0.0, content[0]);
        stretch_offset_y = sum_within_range(stretch_y, 0.0, content[1]);
        stretch_content_width = sum_within_range(stretch_x, content[0], content[2]);
        stretch_content_height = sum_within_range(stretch_y, content[1], content[3]);
        fixed_offset_x = content[0] - stretch_offset_x;
        fixed_offset_y = content[1] - stretch_offset_y;
        fixed_content_width = content_width - stretch_content_width;
        fixed_content_height = content_height - stretch_content_height;
    }

    let icon_left = bounds.x1;
    let icon_top = bounds.y1;
    let icon_width = bounds.x2 - icon_left;
    let icon_height = bounds.y2 - icon_top;

    let make_box = |left: Cut, top: Cut, right: Cut, bottom: Cut| {
        // A cut's em offset and pixel offset along x or y.
        let x = |c: Cut| {
            let em = em_offset(
                c.stretch - stretch_offset_x,
                stretch_content_width,
                icon_width,
                icon_left,
            );
            let px = px_offset(
                c.fixed - fixed_offset_x,
                fixed_content_width,
                c.stretch,
                stretch_width,
            );
            (em, px)
        };
        let y = |c: Cut| {
            let em = em_offset(
                c.stretch - stretch_offset_y,
                stretch_content_height,
                icon_height,
                icon_top,
            );
            let px = px_offset(
                c.fixed - fixed_offset_y,
                fixed_content_height,
                c.stretch,
                stretch_height,
            );
            (em, px)
        };
        let ((left_em, left_px), (right_em, right_px)) = (x(left), x(right));
        let ((top_em, top_px), (bottom_em, bottom_px)) = (y(top), y(bottom));

        let mut tl = Point::new(left_em, top_em);
        let mut tr = Point::new(right_em, top_em);
        let mut br = Point::new(right_em, bottom_em);
        let mut bl = Point::new(left_em, bottom_em);
        let pixel_offset_tl = Point::new(left_px / pixel_ratio, top_px / pixel_ratio);
        let pixel_offset_br = Point::new(right_px / pixel_ratio, bottom_px / pixel_ratio);

        let angle = degrees_to_radians(rotate);
        if angle != 0.0 && !angle.is_nan() {
            let matrix = rotation_matrix(angle);
            tl = tl.mat_mult(matrix);
            tr = tr.mat_mult(matrix);
            bl = bl.mat_mult(matrix);
            br = br.mat_mult(matrix);
        }

        let x1 = left.stretch + left.fixed;
        let x2 = right.stretch + right.fixed;
        let y1 = top.stretch + top.fixed;
        let y2 = bottom.stretch + bottom.fixed;

        SymbolQuad {
            tl,
            tr,
            bl,
            br,
            tex: TexRect {
                x: IMAGE_PADDING + x1,
                y: IMAGE_PADDING + y1,
                w: x2 - x1,
                h: y2 - y1,
            },
            texture: Texture::Image(image),
            pixel_offset_tl,
            pixel_offset_br,
            glyph_offset: [0.0, 0.0],
            section: 0,
            sdf,
            min_font_scale: [
                fixed_content_width / pixel_ratio / icon_width,
                fixed_content_height / pixel_ratio / icon_height,
            ],
        }
    };

    if !has_icon_text_fit || (meta.stretch_x.is_none() && meta.stretch_y.is_none()) {
        out.push(make_box(
            Cut {
                fixed: 0.0,
                stretch: -1.0,
            },
            Cut {
                fixed: 0.0,
                stretch: -1.0,
            },
            Cut {
                fixed: 0.0,
                stretch: image_width + 1.0,
            },
            Cut {
                fixed: 0.0,
                stretch: image_height + 1.0,
            },
        ));
    } else {
        let x_cuts = stretch_zones_to_cuts(stretch_x, fixed_width, stretch_width);
        let y_cuts = stretch_zones_to_cuts(stretch_y, fixed_height, stretch_height);
        for x in x_cuts.windows(2) {
            for y in y_cuts.windows(2) {
                out.push(make_box(x[0], y[0], x[1], y[1]));
            }
        }
    }
}

fn sum_within_range(ranges: &[[f64; 2]], min: f64, max: f64) -> f64 {
    let clamp = |v: f64| crate::geometry::js_max(min, crate::geometry::js_min(max, v));
    let mut sum = 0.0;
    for range in ranges {
        sum += clamp(range[1]) - clamp(range[0]);
    }
    sum
}

fn stretch_zones_to_cuts(zones: &[[f64; 2]], fixed_size: f64, stretch_size: f64) -> Vec<Cut> {
    let mut cuts = vec![Cut {
        fixed: -IMAGE_PADDING,
        stretch: 0.0,
    }];
    for &[c1, c2] in zones {
        let last = *cuts.last().expect("cuts start with one");
        cuts.push(Cut {
            fixed: c1 - last.stretch,
            stretch: last.stretch,
        });
        cuts.push(Cut {
            fixed: c1 - last.stretch,
            stretch: last.stretch + (c2 - c1),
        });
    }
    cuts.push(Cut {
        fixed: fixed_size + IMAGE_PADDING,
        stretch: stretch_size,
    });
    cuts
}

fn em_offset(stretch_offset: f64, stretch_size: f64, icon_size: f64, icon_offset: f64) -> f64 {
    stretch_offset / stretch_size * icon_size + icon_offset
}

fn px_offset(fixed_offset: f64, fixed_size: f64, stretch_offset: f64, stretch_size: f64) -> f64 {
    fixed_offset - fixed_size * stretch_offset / stretch_size
}

/// Inputs of [`glyph_quads`] shared by every glyph of a label.
#[derive(Clone, Copy)]
pub(crate) struct GlyphQuadContext<'a> {
    pub(crate) glyphs: &'a GlyphMap,
    pub(crate) images: &'a TileImages,
    pub(crate) text_offset: [f64; 2],
    /// `text-rotate`, in degrees.
    pub(crate) rotate: f64,
    pub(crate) along_line: bool,
    pub(crate) allow_vertical_placement: bool,
}

/// The quads of a shaped label (GL JS `getGlyphQuads`). Glyphs without a
/// bitmap, such as spaces, get none.
pub(crate) fn glyph_quads(
    shaping: &Shaping,
    context: GlyphQuadContext<'_>,
    out: &mut Vec<SymbolQuad>,
) {
    let GlyphQuadContext {
        glyphs,
        images,
        text_offset,
        rotate,
        along_line,
        allow_vertical_placement,
    } = context;
    let text_rotate = degrees_to_radians(rotate);

    for line in &shaping.lines {
        for glyph in &line.glyphs {
            let (texture, rect, sdf, pixel_ratio, rect_buffer, texture_scale, image) =
                match glyph.source {
                    GlyphSource::Glyph(id) => {
                        let Some(resolved) = glyphs.glyph(id) else {
                            continue;
                        };
                        let [w, h] = resolved.bitmap;
                        if w == 0 || h == 0 {
                            continue;
                        }
                        let padded = [w + 2 * GLYPH_PADDING, h + 2 * GLYPH_PADDING];
                        let scale = if resolved.double_resolution { 2.0 } else { 1.0 };
                        (
                            Texture::Glyph(id),
                            padded,
                            true,
                            1.0,
                            BORDER + 1.0,
                            scale,
                            false,
                        )
                    }
                    GlyphSource::Image { section } => {
                        let ShapedSection::Image(name) = &shaping.sections[section] else {
                            continue;
                        };
                        let Some((index, image)) = images.get(name) else {
                            continue;
                        };
                        let padded = [image.size[0] + 2, image.size[1] + 2];
                        let pixel_ratio = image.pixel_ratio;
                        (
                            Texture::Image(index),
                            padded,
                            image.sdf,
                            pixel_ratio,
                            IMAGE_PADDING / pixel_ratio,
                            1.0,
                            true,
                        )
                    }
                };

            let mut line_offset = 0.0;
            let rotate_vertical_glyph = (along_line || allow_vertical_placement) && glyph.vertical;
            let half_advance = glyph.metrics.advance * glyph.scale / 2.0;

            if allow_vertical_placement && shaping.verticalizable {
                let scaled_glyph_offset = (glyph.scale - 1.0) * ONE_EM;
                let image_offset = (ONE_EM - glyph.metrics.width * glyph.scale) / 2.0;
                line_offset = line.line_offset / 2.0
                    - if image {
                        -image_offset
                    } else {
                        scaled_glyph_offset
                    };
            }

            let glyph_offset = if along_line {
                [glyph.x + half_advance, glyph.y]
            } else {
                [0.0, 0.0]
            };
            let mut built_in_offset = if along_line {
                [0.0, 0.0]
            } else {
                [
                    glyph.x + half_advance + text_offset[0],
                    glyph.y + text_offset[1] - line_offset,
                ]
            };
            let mut verticalized_label_offset = [0.0, 0.0];
            if rotate_vertical_glyph {
                verticalized_label_offset = built_in_offset;
                built_in_offset = [0.0, 0.0];
            }

            let x1 = (glyph.metrics.left - rect_buffer) * glyph.scale - half_advance
                + built_in_offset[0];
            let y1 = (-glyph.metrics.top - rect_buffer) * glyph.scale + built_in_offset[1];
            let x2 = x1 + f64::from(rect[0]) / texture_scale * glyph.scale / pixel_ratio;
            let y2 = y1 + f64::from(rect[1]) / texture_scale * glyph.scale / pixel_ratio;

            let mut tl = Point::new(x1, y1);
            let mut tr = Point::new(x2, y1);
            let mut bl = Point::new(x1, y2);
            let mut br = Point::new(x2, y2);

            if rotate_vertical_glyph {
                let center = Point::new(-half_advance, half_advance - DEFAULT_OFFSET);
                let vertical_rotation = -std::f64::consts::PI / 2.0;
                let x_half_width_offset_correction = ONE_EM / 2.0 - half_advance;
                let y_image_offset_correction = if image {
                    x_half_width_offset_correction
                } else {
                    0.0
                };
                let half_width_offset_correction = Point::new(
                    5.0 - DEFAULT_OFFSET - x_half_width_offset_correction,
                    -y_image_offset_correction,
                );
                let vertical_offset_correction =
                    Point::new(verticalized_label_offset[0], verticalized_label_offset[1]);
                let correct = |p: Point| {
                    p.rotate_around(vertical_rotation, center)
                        .add(half_width_offset_correction)
                        .add(vertical_offset_correction)
                };
                tl = correct(tl);
                tr = correct(tr);
                bl = correct(bl);
                br = correct(br);
            }

            if text_rotate != 0.0 && !text_rotate.is_nan() {
                let matrix = rotation_matrix(text_rotate);
                tl = tl.mat_mult(matrix);
                tr = tr.mat_mult(matrix);
                bl = bl.mat_mult(matrix);
                br = br.mat_mult(matrix);
            }

            out.push(SymbolQuad {
                tl,
                tr,
                bl,
                br,
                tex: TexRect {
                    x: 0.0,
                    y: 0.0,
                    w: f64::from(rect[0]),
                    h: f64::from(rect[1]),
                },
                texture,
                pixel_offset_tl: Point::default(),
                pixel_offset_br: Point::default(),
                glyph_offset,
                section: glyph.section,
                sdf,
                min_font_scale: [0.0, 0.0],
            });
        }
    }
}
