//! Behaviour on small synthetic layers, beside the GL JS golden layouts.

use orb_text::{
    Anchor, FontStack, Glyph, GlyphKey, GlyphMap, GlyphMetrics, GlyphRequests, Label, Variant,
    WritingMode,
};
use orb_tile::{GeometryType, Layer, LayerBuilder, Value};

use crate::bucket::{
    AnchorOffset, NONE, PlacedSymbol, Quad, SortKeyRange, SymbolBucket, SymbolInstance,
    SymbolWritingMode, instance_flags, justification,
};
use crate::layout::{
    FeatureLayout, LayerLayout, LayoutContext, Placement, SymbolFeature, TextJustify, TileImages,
    TileParams, add_glyph_requests, layout,
};
use crate::{IconImage, IconTextFit};

fn stack() -> FontStack {
    FontStack::new(&["Test"])
}

/// Every single-code-point request resolves: CJK full width, the rest
/// 12 px wide. Vertical alternates are unavailable, as with PBF glyphs.
fn glyphs(layer: &Layer, features: &[SymbolFeature], style: &LayerLayout) -> GlyphMap {
    let mut requests = GlyphRequests::default();
    add_glyph_requests(layer, features, style, &mut requests);
    let mut map = GlyphMap::default();
    for request in requests {
        let mut chars = request.text.chars();
        let (Some(c), None) = (chars.next(), chars.next()) else {
            map.insert(request, None);
            continue;
        };
        if request.variant == Variant::Vertical {
            map.insert(request, None);
            continue;
        }
        let wide = orb_text::unicode::has_upright_vertical_orientation(c);
        let (width, advance) = if wide { (20.0, 24.0) } else { (10.0, 12.0) };
        let glyph = Glyph {
            key: GlyphKey::Pbf {
                stack: request.stack.clone(),
                code_point: c as u32,
            },
            metrics: GlyphMetrics {
                width,
                height: 14.0,
                left: 1.0,
                top: -8.0,
                advance,
            },
            bitmap: [width as u32 + 6, 20],
            double_resolution: false,
        };
        map.insert(request, Some(glyph));
    }
    map
}

fn feature(index: u32, text: &str) -> SymbolFeature {
    SymbolFeature {
        index,
        stable_id: None,
        text: Some(Label::from_text(text)),
        icon: None,
        sort_key: None,
        layout: FeatureLayout {
            text_font: stack(),
            ..FeatureLayout::default()
        },
    }
}

fn points(points: &[[i32; 2]]) -> Layer {
    let mut b = LayerBuilder::new("points", 8192);
    for &p in points {
        b.begin_feature(None, GeometryType::Point);
        b.begin_part();
        b.begin_ring();
        b.push_vertex(p);
    }
    b.finish()
}

fn lines(lines: &[&[[i32; 2]]]) -> Layer {
    let mut b = LayerBuilder::new("lines", 8192);
    for line in lines {
        b.begin_feature(None, GeometryType::LineString);
        b.begin_part();
        b.begin_ring();
        for &p in *line {
            b.push_vertex(p);
        }
    }
    b.finish()
}

fn run_with(
    layer: &Layer,
    features: &[SymbolFeature],
    style: &LayerLayout,
    images: &TileImages,
) -> SymbolBucket {
    let glyphs = glyphs(layer, features, style);
    let tile = TileParams {
        source_layer: layer.name(),
        overscaling: 1.0,
        pixel_ratio: 1.0,
    };
    let context = LayoutContext {
        glyphs: &glyphs,
        images,
        words: None,
    };
    layout(layer, features, style, tile, context)
}

fn run(layer: &Layer, features: &[SymbolFeature], style: &LayerLayout) -> SymbolBucket {
    run_with(layer, features, style, &TileImages::default())
}

fn line_style() -> LayerLayout {
    LayerLayout {
        placement: Placement::Line,
        ..LayerLayout::default()
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn line_labels_repeat_along_the_line_and_share_it() {
    // Continued from the tile edge: the first anchor is half the spacing
    // (16 tile units per pixel × 250 px / 2) along.
    let layer = lines(&[&[[0, 4096], [8192, 4096]]]);
    let bucket = run(&layer, &[feature(0, "Main")], &line_style());

    let anchors: Vec<[i16; 2]> = bucket.instances.iter().map(|i| i.anchor).collect();
    assert_eq!(anchors, [[2000, 4096], [6000, 4096]]);
    // Both ride one copy of the clipped line.
    assert_eq!(bucket.lines, [[0, 4096], [8192, 4096]]);
    for instance in &bucket.instances {
        assert_ne!(
            instance.flags & instance_flags::RUNTIME_COLLISION_CIRCLES,
            0
        );
        assert_eq!(instance.flags & instance_flags::TEXT_BOX, 0);
        assert!(instance.collision_circle_diameter > 0.0);
        let placed = &bucket.text.placed[instance.text[justification::CENTER] as usize];
        assert_eq!((placed.line, placed.segment), ([0, 2], 0));
        // Glyphs advance along the line from left to right of the anchor.
        let quads = placed.quads[0] as usize..(placed.quads[0] + placed.quads[1]) as usize;
        let offsets = &bucket.text.glyph_offsets[quads];
        assert_eq!(offsets.len(), 4);
        assert!(offsets.windows(2).all(|w| w[0] < w[1]));
        assert!(offsets[0] < 0.0 && offsets[3] > 0.0);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn keep_upright_adds_a_vertical_shaping_for_vertical_scripts() {
    let layer = lines(&[&[[0, 4096], [8192, 4096]]]);
    let text = [feature(0, "中央通り")];

    let upright = run(&layer, &text, &line_style());
    let instance = &upright.instances[0];
    let horizontal = &upright.text.placed[instance.text[justification::CENTER] as usize];
    let vertical = &upright.text.placed[instance.text[justification::VERTICAL] as usize];
    assert_eq!(
        SymbolWritingMode::from_u8(horizontal.writing_mode),
        SymbolWritingMode::Horizontal
    );
    assert_eq!(
        SymbolWritingMode::from_u8(vertical.writing_mode),
        SymbolWritingMode::Vertical
    );

    let free = run(
        &layer,
        &text,
        &LayerLayout {
            text_keep_upright: false,
            ..line_style()
        },
    );
    let instance = &free.instances[0];
    assert_eq!(instance.text[justification::VERTICAL], NONE);
    let horizontal = &free.text.placed[instance.text[justification::CENTER] as usize];
    assert_eq!(
        SymbolWritingMode::from_u8(horizontal.writing_mode),
        SymbolWritingMode::HorizontalOnly
    );

    // Latin text never gets a vertical shaping.
    let latin = run(&layer, &[feature(0, "Main")], &line_style());
    assert_eq!(latin.instances[0].text[justification::VERTICAL], NONE);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn variable_anchors_shape_once_per_justification() {
    let layer = points(&[[4096, 4096]]);
    let style = LayerLayout {
        text_variable_anchor: Some(vec![Anchor::Left, Anchor::Right, Anchor::Top]),
        text_radial_offset_set: true,
        ..LayerLayout::default()
    };
    let with = |text: &str| {
        let mut f = feature(0, text);
        f.layout.text_justify = TextJustify::Auto;
        f.layout.text_radial_offset = 1.0;
        f.layout.text_max_width = 3.0;
        f
    };

    // One line: every justification draws the same placed symbol.
    let single = run(&layer, &[with("Cafe")], &style);
    let instance = &single.instances[0];
    assert_eq!(single.text.placed.len(), 1);
    assert_eq!(&instance.text[..3], &[0, 0, 0]);
    let offsets: Vec<AnchorOffset> = single.anchor_offsets.clone();
    assert_eq!(
        offsets,
        [
            AnchorOffset {
                anchor: 2,
                offset: [24.0, 0.0]
            },
            AnchorOffset {
                anchor: 3,
                offset: [-24.0, 0.0]
            },
            AnchorOffset {
                anchor: 4,
                offset: [0.0, 17.0]
            },
        ]
    );
    assert_eq!(instance.anchor_offsets, [0, 3]);
    assert_eq!(offsets[0].text_anchor(), Anchor::Left);

    // Several lines: one shaping per justification, in GL JS's order.
    let multi = run(&layer, &[with("A Long Cafe Name")], &style);
    let instance = &multi.instances[0];
    assert_eq!(multi.text.placed.len(), 3);
    assert_eq!(
        instance.text[..3],
        [1, 2, 0],
        "right, center, left: left was shaped first"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn icon_text_fit_stretches_the_icon_around_the_text() {
    let layer = points(&[[4096, 4096]]);
    let mut images = TileImages::default();
    images.insert("pill", IconImage::new([20, 10], 1.0, false));
    let style = LayerLayout {
        icon_text_fit: IconTextFit::Both,
        icon_text_fit_padding: [1.0, 2.0, 3.0, 4.0],
        text_padding: 0.0,
        ..LayerLayout::default()
    };
    let mut f = feature(0, "Hello");
    f.icon = Some("pill".into());
    f.layout.icon_padding = [0.0; 4];
    // At 24 px text, layout units are pixels.
    f.layout.text_size = crate::FeatureSize::constant(24.0);

    let bucket = run_with(&layer, &[f], &style, &images);
    let instance = &bucket.instances[0];
    let [tx1, ty1, tx2, ty2] = instance.boxes[0].map(i32::from);
    let [ix1, iy1, ix2, iy2] = instance.boxes[2].map(i32::from);
    // 16 tile units per pixel; boxes truncate, so allow one unit.
    let near = |a: i32, b: i32| (a - b).abs() <= 1;
    assert!(near(ix1, tx1 - 4 * 16), "{ix1} {tx1}");
    assert!(near(ix2, tx2 + 2 * 16), "{ix2} {tx2}");
    assert!(near(iy1, ty1 - 16), "{iy1} {ty1}");
    assert!(near(iy2, ty2 + 3 * 16), "{iy2} {ty2}");
    // The text's placed symbol moves the icon with it.
    let text = &bucket.text.placed[instance.text[justification::CENTER] as usize];
    assert_eq!(text.associated_icon, instance.icon[0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn vertical_point_labels_rotate_their_glyph_quads() {
    let layer = points(&[[4096, 4096]]);
    let style = LayerLayout {
        text_writing_mode: vec![WritingMode::Vertical, WritingMode::Horizontal],
        ..LayerLayout::default()
    };
    let bucket = run(&layer, &[feature(0, "東京駅")], &style);
    let instance = &bucket.instances[0];
    assert_ne!(instance.flags & instance_flags::VERTICAL_TEXT_BOX, 0);

    let quads = |placed: &PlacedSymbol| -> Vec<Quad> {
        let start = placed.quads[0] as usize;
        bucket.text.quads[start..start + placed.quads[1] as usize].to_vec()
    };
    let horizontal = &bucket.text.placed[instance.text[justification::CENTER] as usize];
    let vertical = &bucket.text.placed[instance.text[justification::VERTICAL] as usize];
    assert_eq!(
        SymbolWritingMode::from_u8(vertical.writing_mode),
        SymbolWritingMode::Vertical
    );
    for q in quads(horizontal) {
        let [tl, tr, ..] = q.corners;
        assert_eq!(tl[1], tr[1], "a horizontal quad's top edge is level");
    }
    let vertical_quads = quads(vertical);
    assert_eq!(vertical_quads.len(), 3);
    for q in &vertical_quads {
        let [tl, tr, ..] = q.corners;
        assert_eq!(tl[0], tr[0], "an upright glyph's quad is turned a quarter");
        assert!(tr[1] < tl[1]);
    }
    // Glyphs still advance along x; the draw turns the whole label a
    // quarter turn, so the upright glyphs read top to bottom.
    let lefts: Vec<i16> = vertical_quads.iter().map(|q| q.corners[0][0]).collect();
    assert!(lefts.windows(2).all(|w| w[0] < w[1]), "{lefts:?}");
    // Its collision box stands where the horizontal one lies.
    let [hx1, hy1, hx2, hy2] = instance.boxes[0];
    let [vx1, vy1, vx2, vy2] = instance.boxes[1];
    assert!(hx2 - hx1 > hy2 - hy1);
    assert!(vy2 - vy1 > vx2 - vx1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn sort_keys_order_instances_stably() {
    let layer = points(&[[100, 100], [200, 200], [300, 300], [400, 400]]);
    let features: Vec<SymbolFeature> = [3.0, 1.0, 2.0, 1.0]
        .into_iter()
        .enumerate()
        .map(|(i, key)| SymbolFeature {
            sort_key: Some(key),
            ..feature(i as u32, "A")
        })
        .collect();
    let bucket = run(&layer, &features, &LayerLayout::default());
    let order: Vec<u32> = bucket.instances.iter().map(|i| i.feature).collect();
    assert_eq!(order, [1, 3, 2, 0]);
    let range = |sort_key, start, end| SortKeyRange {
        sort_key,
        start,
        end,
    };
    assert_eq!(
        bucket.sort_key_ranges,
        [range(1.0, 0, 2), range(2.0, 2, 3), range(3.0, 3, 4)]
    );

    // Without keys, the layer's order stands.
    let unsorted: Vec<SymbolFeature> = (0..4).map(|i| feature(i, "A")).collect();
    let bucket = run(&layer, &unsorted, &LayerLayout::default());
    let order: Vec<u32> = bucket.instances.iter().map(|i| i.feature).collect();
    assert_eq!(order, [0, 1, 2, 3]);
    assert!(bucket.sort_key_ranges.is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn stable_keys_name_layer_feature_and_text() {
    let key = |layer: &str, id: Option<Value>, text: &str| {
        let points = {
            let mut b = LayerBuilder::new(layer, 8192);
            b.begin_feature(None, GeometryType::Point);
            b.begin_part();
            b.begin_ring();
            b.push_vertex([10, 10]);
            b.finish()
        };
        let f = SymbolFeature {
            stable_id: id,
            ..feature(0, text)
        };
        run(&points, &[f], &LayerLayout::default()).instances[0].key
    };
    let base = key("place", None, "Paris");
    assert_eq!(base, key("place", None, "Paris"), "deterministic");
    assert_ne!(base, key("poi", None, "Paris"), "source layer");
    assert_ne!(base, key("place", None, "Lyon"), "text");
    let seven = key("place", Some(Value::I64(7)), "Paris");
    assert_ne!(base, seven, "id");
    assert_eq!(seven, key("place", Some(Value::U64(7)), "Paris"));
    assert_eq!(seven, key("place", Some(Value::F64(7.0)), "Paris"));
    assert_ne!(seven, key("place", Some(Value::Str("7".into())), "Paris"));
    assert_ne!(seven, key("place", Some(Value::F64(7.5)), "Paris"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn tables_are_flat_rows() {
    use std::mem::size_of;
    assert_eq!(size_of::<SymbolInstance>(), 96);
    assert_eq!(size_of::<PlacedSymbol>(), 44);
    assert_eq!(size_of::<Quad>(), 48);
    assert_eq!(size_of::<AnchorOffset>(), 12);
    assert_eq!(size_of::<SortKeyRange>(), 16);

    let layer = points(&[[4096, 4096]]);
    let bucket = run(&layer, &[feature(0, "Hi")], &LayerLayout::default());
    let bytes: &[u8] = bytemuck::cast_slice(&bucket.text.quads);
    assert_eq!(bytes.len(), 2 * 48);
    let back: &[Quad] = bytemuck::cast_slice(bytes);
    assert_eq!(back, &bucket.text.quads[..]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn points_outside_the_tile_get_no_symbols_or_glyphs() {
    let layer = points(&[[-10, 100], [100, 8192]]);
    let features = [feature(0, "Out"), feature(1, "Edge")];
    let style = LayerLayout::default();
    let mut requests = GlyphRequests::default();
    add_glyph_requests(&layer, &features, &style, &mut requests);
    assert!(requests.is_empty());
    assert!(run(&layer, &features, &style).is_empty());
}
