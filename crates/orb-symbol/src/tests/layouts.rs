//! Whole layouts against GL JS's `performSymbolLayout` on a central Tokyo
//! test tile (`layout-*.json`): the same evaluated features in, every
//! instance, placed symbol, vertex, box and offset compared.

use std::collections::{HashMap, HashSet};

use orb_text::{
    Content, FontStack, Glyph, GlyphKey, GlyphMap, GlyphMetrics, GlyphRequests, Label, Section,
    Variant, VerticalAlign, WritingMode,
};
use orb_tile::{Ancestor, TileFrame, Value as TileValue};
use serde_json::Value;

use super::{anchor, assert_same, image, num, nums};
use crate::bucket::{NONE, SizeKind, SymbolBucket, SymbolStream, instance_flags, quad_flags};
use crate::layout::{
    Alignment, FeatureLayout, FeatureSize, LayerLayout, LayoutContext, Placement, SymbolFeature,
    TextJustify, TileImages, TileParams, add_glyph_requests, layout, retain_line_labels_in_tile,
};
use crate::{IconTextFit, quads::GLYPH_PADDING};

macro_rules! golden {
    ($($name:literal),* $(,)?) => {
        &[$(($name, include_str!(concat!("../../tests/data/layout-", $name, ".json")))),*]
    };
}

const CASES: &[(&str, &str)] = golden![
    "point-labels",
    "line-labels",
    "line-center",
    "line-overscaled",
    "line-icons",
    "variable-anchors",
    "variable-anchor-offset",
    "variable-anchor-text-offset",
    "icon-text-fit",
    "icon-text-fit-proportional",
    "vertical-point",
    "vertical-line",
    "sort-key-sizes",
    "polygons",
    "inline-images",
    "promote-id",
    "poi-cyrillic",
    "poi-variable-many",
    "line-cyrillic",
    "line-repeat",
    "polygon-outlines",
    "line-points",
];

fn tile(name: &str) -> TileFrame {
    let bytes: &[u8] = match name {
        "14-14552-6450.mvt" => include_bytes!("../../tests/data/tiles/14-14552-6450.mvt"),
        other => panic!("tile {other} is not included"),
    };
    orb_tile::mvt::decode(bytes).expect("test tile decodes")
}

fn size_kind(v: &Value) -> SizeKind {
    match v["kind"].as_str().unwrap() {
        "constant" => SizeKind::Constant,
        "source" => SizeKind::Source,
        "camera" => SizeKind::Camera,
        "composite" => SizeKind::Composite {
            zooms: [num(&v["minZoom"]), num(&v["maxZoom"])],
        },
        other => panic!("size kind {other}"),
    }
}

fn alignment(v: &Value) -> Alignment {
    match v.as_str().unwrap() {
        "map" => Alignment::Map,
        "viewport" => Alignment::Viewport,
        "viewport-glyph" => Alignment::ViewportGlyph,
        _ => Alignment::Auto,
    }
}

fn layer_layout(v: &Value) -> LayerLayout {
    LayerLayout {
        placement: match v["placement"].as_str().unwrap() {
            "line" => Placement::Line,
            "line-center" => Placement::LineCenter,
            _ => Placement::Point,
        },
        spacing: num(&v["spacing"]),
        text_rotation_alignment: alignment(&v["textRotationAlignment"]),
        icon_rotation_alignment: alignment(&v["iconRotationAlignment"]),
        text_keep_upright: v["keepUpright"].as_bool().unwrap(),
        text_max_angle: num(&v["maxAngle"]),
        text_line_height: num(&v["lineHeight"]),
        text_padding: num(&v["textPadding"]),
        text_writing_mode: v["writingModes"]
            .as_array()
            .map(|modes| {
                modes
                    .iter()
                    .map(|m| match m.as_str().unwrap() {
                        "vertical" => WritingMode::Vertical,
                        _ => WritingMode::Horizontal,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        text_variable_anchor: v["variableAnchor"]
            .as_array()
            .map(|a| a.iter().map(|a| anchor(a.as_str().unwrap())).collect()),
        text_radial_offset_set: v["radialOffsetSet"].as_bool().unwrap(),
        icon_text_fit: match v["iconTextFit"].as_str().unwrap() {
            "width" => IconTextFit::Width,
            "height" => IconTextFit::Height,
            "both" => IconTextFit::Both,
            _ => IconTextFit::None,
        },
        icon_text_fit_padding: nums(&v["iconTextFitPadding"]),
        icon_rotate_constant: (!v["iconRotateConstant"].is_null())
            .then(|| num(&v["iconRotateConstant"])),
        text_size: size_kind(&v["textSize"]),
        icon_size: size_kind(&v["iconSize"]),
    }
}

fn feature_size(v: &Value) -> FeatureSize {
    FeatureSize {
        tile: num(&v["tile"]),
        layout: num(&v["layout"]),
        max: num(&v["max"]),
        stops: nums(&v["stops"]),
    }
}

fn label(v: &Value) -> Label {
    let sections = v
        .as_array()
        .unwrap()
        .iter()
        .map(|s| Section {
            content: match s["image"].as_str() {
                Some(name) => Content::Image(name.into()),
                None => Content::Text(s["text"].as_str().unwrap().into()),
            },
            scale: (!s["scale"].is_null()).then(|| num(&s["scale"])),
            font_stack: s["fontStack"].as_str().map(FontStack::from_joined),
            vertical_align: match s["verticalAlign"].as_str() {
                Some("center") => VerticalAlign::Center,
                Some("top") => VerticalAlign::Top,
                _ => VerticalAlign::Bottom,
            },
        })
        .collect();
    Label { sections }
}

fn feature(v: &Value) -> SymbolFeature {
    let strings = |v: &Value| -> Vec<String> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_owned())
            .collect()
    };
    let anchor_offsets = v["textVariableAnchorOffset"].as_array().map(|values| {
        values
            .chunks(2)
            .map(|pair| (anchor(pair[0].as_str().unwrap()), nums(&pair[1])))
            .collect()
    });
    SymbolFeature {
        index: v["index"].as_u64().unwrap() as u32,
        stable_id: match &v["id"] {
            Value::Null => None,
            Value::String(s) => Some(TileValue::Str(s.clone())),
            other => Some(TileValue::F64(num(other))),
        },
        text: (!v["text"].is_null()).then(|| label(&v["text"])),
        icon: v["icon"].as_str().map(Into::into),
        sort_key: (!v["sortKey"].is_null()).then(|| num(&v["sortKey"])),
        layout: FeatureLayout {
            text_font: FontStack::new(&strings(&v["textFont"])),
            text_size: feature_size(&v["textSize"]),
            icon_size: feature_size(&v["iconSize"]),
            text_letter_spacing: num(&v["textLetterSpacing"]),
            text_anchor: anchor(v["textAnchor"].as_str().unwrap()),
            text_justify: match v["textJustify"].as_str().unwrap() {
                "auto" => TextJustify::Auto,
                "left" => TextJustify::Left,
                "right" => TextJustify::Right,
                _ => TextJustify::Center,
            },
            text_max_width: num(&v["textMaxWidth"]),
            text_offset: nums(&v["textOffset"]),
            text_radial_offset: num(&v["textRadialOffset"]),
            text_variable_anchor_offset: anchor_offsets,
            text_rotate: num(&v["textRotate"]),
            icon_offset: nums(&v["iconOffset"]),
            icon_anchor: anchor(v["iconAnchor"].as_str().unwrap()),
            icon_rotate: num(&v["iconRotate"]),
            icon_padding: nums(&v["iconPadding"]),
            height_offset: num(&v["heightOffset"]),
        },
    }
}

/// GL JS's glyph table of the case: what each request resolved to, and
/// where its atlas put it.
struct Glyphs {
    resolved: HashMap<RequestKey, (Option<Glyph>, Option<[f64; 4]>)>,
}

/// A request as GL JS keys glyphs: stack, variant, cluster.
type RequestKey = (String, Variant, String);

impl Glyphs {
    fn new(v: &Value) -> Self {
        let mut resolved = HashMap::new();
        for g in v.as_array().unwrap() {
            let stack = g["stack"].as_str().unwrap().to_owned();
            let variant = match g["variant"].as_str().unwrap() {
                "vertical" => Variant::Vertical,
                _ => Variant::Default,
            };
            let grapheme = g["grapheme"].as_str().unwrap().to_owned();
            let glyph = (!g["metrics"].is_null()).then(|| {
                let m = &g["metrics"];
                let bitmap: [f64; 2] = nums(&g["bitmap"]);
                Glyph {
                    key: GlyphKey::Pbf {
                        stack: FontStack::from_joined(&stack),
                        code_point: grapheme.chars().next().unwrap() as u32,
                    },
                    metrics: GlyphMetrics {
                        width: num(&m["width"]),
                        height: num(&m["height"]),
                        left: num(&m["left"]),
                        top: num(&m["top"]),
                        advance: num(&m["advance"]),
                    },
                    bitmap: bitmap.map(|v| v as u32),
                    double_resolution: false,
                }
            });
            let rect = (!g["rect"].is_null()).then(|| nums(&g["rect"]));
            resolved.insert((stack, variant, grapheme), (glyph, rect));
        }
        Self { resolved }
    }

    fn map(&self, requests: GlyphRequests) -> GlyphMap {
        let mut map = GlyphMap::default();
        let mut asked = HashSet::new();
        for request in requests {
            let key = (
                request.stack.as_str().to_owned(),
                request.variant,
                request.text.to_string(),
            );
            let (glyph, _) = self
                .resolved
                .get(&key)
                .unwrap_or_else(|| panic!("GL JS did not request {key:?}"));
            asked.insert(key);
            map.insert(request, glyph.clone());
        }
        let missing: Vec<_> = self
            .resolved
            .keys()
            .filter(|k| !asked.contains(*k))
            .collect();
        assert!(missing.is_empty(), "orb-symbol did not request {missing:?}");
        map
    }

    fn rect(&self, map: &GlyphMap, id: u32) -> [f64; 4] {
        let request = map.request(orb_text::GlyphId(id));
        let key = (
            request.stack.as_str().to_owned(),
            request.variant,
            request.text.to_string(),
        );
        self.resolved[&key].1.expect("a drawn glyph has a rect")
    }
}

fn index_or_none(v: &Value) -> u32 {
    match num(v) {
        i if i < 0.0 => NONE,
        i => i as u32,
    }
}

struct Case<'a> {
    name: &'a str,
    golden: Value,
    glyphs: Glyphs,
    glyph_map: GlyphMap,
    images: TileImages,
    bucket: SymbolBucket,
}

fn run<'a>(name: &'a str, json: &str) -> Case<'a> {
    run_in(name, json, None)
}

/// [`run`] for the slice at `ancestor` of the case's tile. A slice lays
/// out line labels only of the texts near it, as the runtime does, and
/// that drops no symbol of the slice.
fn run_in<'a>(name: &'a str, json: &str, ancestor: Option<Ancestor>) -> Case<'a> {
    let golden: Value = serde_json::from_str(json).expect("golden parses");
    let spec = &golden["case"];
    let frame = tile(spec["tile"].as_str().unwrap());
    let layer = frame
        .layer(spec["sourceLayer"].as_str().unwrap())
        .expect("the case's source layer");
    let style = layer_layout(&golden["layer"]);
    let features: Vec<SymbolFeature> = golden["features"]
        .as_array()
        .unwrap()
        .iter()
        .map(feature)
        .collect();

    let mut requests = GlyphRequests::default();
    add_glyph_requests(layer, &features, &style, &mut requests);
    let glyphs = Glyphs::new(&golden["glyphs"]);
    let glyph_map = glyphs.map(requests);

    let mut images = TileImages::default();
    if let Some(specs) = spec["images"].as_object() {
        for (name, v) in specs {
            images.insert(name, image(v));
        }
    }
    let tile = TileParams {
        source_layer: spec["sourceLayer"].as_str().unwrap(),
        overscaling: spec.get("overscaling").map_or(1.0, num)
            / ancestor.map_or(1.0, |a| f64::from(a.scale)),
        pixel_ratio: spec.get("pixelRatio").map_or(1.0, num),
        ancestor,
    };
    let context = LayoutContext {
        glyphs: &glyph_map,
        images: &images,
        words: None,
    };
    let mut bucket = layout(layer, &features, &style, tile, context);
    if let Some(ancestor) = ancestor.filter(|_| style.placement == Placement::Line) {
        let mut near = features.clone();
        retain_line_labels_in_tile(layer, &mut near, ancestor);
        let culled = layout(layer, &near, &style, tile, context);
        assert_eq!(culled, bucket, "{name} at {ancestor:?}");
        bucket = culled;
    }
    Case {
        name,
        golden,
        glyphs,
        glyph_map,
        images,
        bucket,
    }
}

impl Case<'_> {
    fn check(&self) {
        let out = &self.golden["out"];
        let name = self.name;
        let b = &self.bucket;

        let instances = out["instances"].as_array().unwrap();
        assert_eq!(b.instances.len(), instances.len(), "{name}: instance count");
        let mut keys: HashMap<u64, u64> = HashMap::new();
        for (i, (got, want)) in b.instances.iter().zip(instances).enumerate() {
            let what = format!("{name} instance {i}");
            assert_eq!(
                got.anchor,
                [num(&want["anchorX"]) as i16, num(&want["anchorY"]) as i16],
                "{what} anchor"
            );
            assert_eq!(
                got.feature as f64,
                num(&want["featureIndex"]),
                "{what} feature"
            );
            assert_same(
                f64::from(got.text_box_scale),
                num(&want["textBoxScale"]),
                &format!("{what} textBoxScale"),
            );
            assert_same(
                f64::from(got.height_offset),
                num(&want["heightOffset"]),
                &format!("{what} heightOffset"),
            );
            let circles = num(&want["useRuntimeCollisionCircles"]) != 0.0;
            assert_eq!(
                got.flags & instance_flags::RUNTIME_COLLISION_CIRCLES != 0,
                circles,
                "{what} circles"
            );
            assert_same(
                f64::from(got.collision_circle_diameter),
                num(&want["collisionCircleDiameter"]),
                &format!("{what} collisionCircleDiameter"),
            );
            let text = [
                "rightJustifiedTextSymbolIndex",
                "centerJustifiedTextSymbolIndex",
                "leftJustifiedTextSymbolIndex",
                "verticalPlacedTextSymbolIndex",
            ]
            .map(|f| index_or_none(&want[f]));
            assert_eq!(got.text, text, "{what} text symbols");
            let icon = ["placedIconSymbolIndex", "verticalPlacedIconSymbolIndex"]
                .map(|f| index_or_none(&want[f]));
            assert_eq!(got.icon, icon, "{what} icon symbols");

            let boxes = ["textBox", "verticalTextBox", "iconBox", "verticalIconBox"];
            let flags = [
                instance_flags::TEXT_BOX,
                instance_flags::VERTICAL_TEXT_BOX,
                instance_flags::ICON_BOX,
                instance_flags::VERTICAL_ICON_BOX,
            ];
            for k in 0..4 {
                let want_box = &want[boxes[k]];
                assert_eq!(
                    got.flags & flags[k] != 0,
                    !want_box.is_null(),
                    "{what} has {}",
                    boxes[k]
                );
                if let Some(wb) = want_box.as_array() {
                    let wb: Vec<i16> = wb.iter().map(|v| num(v) as i16).collect();
                    assert_eq!(&got.boxes[k][..], &wb[..4], "{what} {}", boxes[k]);
                    assert_eq!(got.anchor, [wb[4], wb[5]], "{what} {} anchor", boxes[k]);
                }
            }

            let [start, end] = [
                num(&want["textAnchorOffsetStartIndex"]) as usize,
                num(&want["textAnchorOffsetEndIndex"]) as usize,
            ];
            let got_offsets =
                &b.anchor_offsets[got.anchor_offsets[0] as usize..got.anchor_offsets[1] as usize];
            let want_offsets = &out["anchorOffsets"].as_array().unwrap()[start..end];
            assert_eq!(
                got_offsets.len(),
                want_offsets.len(),
                "{what} anchor offsets"
            );
            for (g, w) in got_offsets.iter().zip(want_offsets) {
                let [a, x, y]: [f64; 3] = nums(w);
                assert_eq!(f64::from(g.anchor), a, "{what} anchor offset");
                assert_same(
                    f64::from(g.offset[0]),
                    x,
                    &format!("{what} anchor offset x"),
                );
                assert_same(
                    f64::from(g.offset[1]),
                    y,
                    &format!("{what} anchor offset y"),
                );
            }

            // Equal Orbis keys imply equal GL JS keys; GL JS's 16 bits may
            // also collide where Orbis's don't.
            let gl_key = want["key"].as_u64().unwrap();
            if let Some(&previous) = keys.get(&got.key) {
                assert_eq!(previous, gl_key, "{what} key");
            }
            keys.insert(got.key, gl_key);
        }

        let glyph_offsets: Vec<f64> = out["glyphOffsets"]
            .as_array()
            .unwrap()
            .iter()
            .map(num)
            .collect();
        let gl_lines: Vec<[i16; 2]> = out["lines"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| nums::<2>(p).map(|v| v as i16))
            .collect();
        self.check_stream("text", &b.text, &out["text"], &glyph_offsets, &gl_lines);
        self.check_stream("icon", &b.icon, &out["icon"], &glyph_offsets, &gl_lines);

        let ranges = out["sortKeyRanges"].as_array().unwrap();
        assert_eq!(
            b.sort_key_ranges.len(),
            ranges.len(),
            "{name}: sort key ranges"
        );
        for (g, w) in b.sort_key_ranges.iter().zip(ranges) {
            assert_same(g.sort_key, num(&w["sortKey"]), &format!("{name} sort key"));
            assert_eq!(
                g.start as f64,
                num(&w["symbolInstanceStart"]),
                "{name} sort range start"
            );
            assert_eq!(
                g.end as f64,
                num(&w["symbolInstanceEnd"]),
                "{name} sort range end"
            );
        }
        assert_eq!(b.sdf_icons, out["sdfIcons"].as_bool(), "{name}: sdfIcons");
        assert_eq!(
            b.icons_in_text,
            out["iconsInText"].as_bool().unwrap(),
            "{name}: iconsInText"
        );
        assert_eq!(
            b.icons_need_linear,
            out["iconsNeedLinear"].as_bool().unwrap(),
            "{name}: iconsNeedLinear"
        );
        assert_same(
            b.max_height_offset,
            num(&out["maxHeightOffset"]),
            &format!("{name}: maxHeightOffset"),
        );
    }

    fn check_stream(
        &self,
        kind: &str,
        stream: &SymbolStream,
        want: &Value,
        glyph_offsets: &[f64],
        gl_lines: &[[i16; 2]],
    ) {
        let name = self.name;
        let placed = want["placed"].as_array().unwrap();
        let vertices = want["vertices"].as_array().unwrap();
        assert_eq!(
            stream.placed.len(),
            placed.len(),
            "{name}: {kind} placed count"
        );
        assert_eq!(
            stream.quads.len() * 4,
            vertices.len(),
            "{name}: {kind} vertex count"
        );
        assert_eq!(stream.glyph_offsets.len(), stream.quads.len());

        for (i, (got, w)) in stream.placed.iter().zip(placed).enumerate() {
            let what = format!("{name} {kind} placed {i}");
            assert_eq!(
                got.anchor,
                [num(&w["anchorX"]) as i16, num(&w["anchorY"]) as i16],
                "{what} anchor"
            );
            assert_eq!(f64::from(got.segment), num(&w["segment"]), "{what} segment");
            assert_eq!(
                f64::from(got.writing_mode),
                num(&w["writingMode"]),
                "{what} writing mode"
            );
            assert_eq!(
                got.associated_icon,
                index_or_none(&w["associatedIconIndex"]),
                "{what} associated icon"
            );
            assert_same(
                f64::from(got.line_offset[0]),
                num(&w["lineOffsetX"]),
                &format!("{what} lineOffsetX"),
            );
            assert_same(
                f64::from(got.line_offset[1]),
                num(&w["lineOffsetY"]),
                &format!("{what} lineOffsetY"),
            );
            let count = num(&w["numGlyphs"]) as usize;
            assert_eq!(got.quads[1] as usize, count, "{what} quad count");
            let start = num(&w["glyphStartIndex"]) as usize;
            let quads = got.quads[0] as usize..(got.quads[0] + got.quads[1]) as usize;
            for (k, (g, w)) in stream.glyph_offsets[quads.clone()]
                .iter()
                .zip(&glyph_offsets[start..start + count])
                .enumerate()
            {
                assert_same(
                    f64::from(*g),
                    f64::from(*w as f32),
                    &format!("{what} glyph offset {k}"),
                );
            }
            let line_start = num(&w["lineStartIndex"]) as usize;
            let line_len = num(&w["lineLength"]) as usize;
            let got_line =
                &self.bucket.lines[got.line[0] as usize..(got.line[0] + got.line[1]) as usize];
            assert_eq!(
                got_line,
                &gl_lines[line_start..line_start + line_len],
                "{what} line"
            );
            for q in quads {
                assert_eq!(stream.quads[q].symbol as usize, i, "{what} quad owner");
            }
        }

        for (k, quad) in stream.quads.iter().enumerate() {
            let what = format!("{name} {kind} quad {k}");
            let symbol = &stream.placed[quad.symbol as usize];
            let instance = &self.bucket.instances[symbol.instance as usize];
            let origin = if quad.flags & quad_flags::IMAGE != 0 {
                let image = self.images.names().nth(quad.texture as usize).unwrap();
                let rect: [f64; 2] = nums(&self.golden["imageRects"][image]);
                [rect[0], rect[1]]
            } else {
                let rect = self.glyphs.rect(&self.glyph_map, quad.texture);
                assert_eq!(
                    [
                        f64::from(quad.tex[2] - quad.tex[0]),
                        f64::from(quad.tex[3] - quad.tex[1])
                    ],
                    [rect[2], rect[3]],
                    "{what} glyph rect"
                );
                let glyph = self
                    .glyph_map
                    .glyph(orb_text::GlyphId(quad.texture))
                    .unwrap();
                assert_eq!(
                    [rect[2], rect[3]],
                    glyph.bitmap.map(|v| f64::from(v + 2 * GLYPH_PADDING)),
                    "{what} padding"
                );
                [rect[0], rect[1]]
            };
            let tex = [
                [quad.tex[0], quad.tex[1]],
                [quad.tex[2], quad.tex[1]],
                [quad.tex[0], quad.tex[3]],
                [quad.tex[2], quad.tex[3]],
            ];
            let po = quad.pixel_offsets;
            let pixel = [
                [po[0], po[1]],
                [po[2], po[1]],
                [po[0], po[3]],
                [po[2], po[3]],
            ];
            for c in 0..4 {
                let v: Vec<f64> = vertices[k * 4 + c]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(num)
                    .collect();
                let what = format!("{what} corner {c}");
                assert_eq!(symbol.anchor, [v[0] as i16, v[1] as i16], "{what} anchor");
                assert_eq!(quad.corners[c], [v[2] as i16, v[3] as i16], "{what} offset");
                assert_eq!(
                    [
                        f64::from(tex[c][0]) + origin[0],
                        f64::from(tex[c][1]) + origin[1]
                    ],
                    [v[4], v[5]],
                    "{what} tex"
                );
                let size_x = v[6] as u16;
                assert_eq!(symbol.size, [size_x >> 1, v[7] as u16], "{what} size");
                assert_eq!(
                    quad.flags & quad_flags::SDF != 0,
                    size_x & 1 == 1,
                    "{what} sdf"
                );
                assert_eq!(pixel[c], [v[8] as i16, v[9] as i16], "{what} pixel offset");
                assert_eq!(
                    quad.min_font_scale,
                    [v[10] as i16, v[11] as i16],
                    "{what} min font scale"
                );
                assert_same(
                    f64::from(instance.height_offset),
                    v[12],
                    &format!("{what} height"),
                );
            }
        }
    }
}

macro_rules! golden_tests {
    ($($test:ident => $name:literal),* $(,)?) => {$(
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
        #[cfg_attr(not(target_arch = "wasm32"), test)]
        fn $test() {
            let (name, json) = CASES.iter().find(|(n, _)| *n == $name).unwrap();
            run(name, json).check();
        }
    )*};
}

golden_tests! {
    point_labels_match_gl_js => "point-labels",
    line_labels_match_gl_js => "line-labels",
    line_center_labels_match_gl_js => "line-center",
    overscaled_line_labels_match_gl_js => "line-overscaled",
    line_icons_match_gl_js => "line-icons",
    variable_anchors_match_gl_js => "variable-anchors",
    variable_anchor_offsets_match_gl_js => "variable-anchor-offset",
    variable_anchors_with_text_offset_match_gl_js => "variable-anchor-text-offset",
    icon_text_fit_matches_gl_js => "icon-text-fit",
    proportional_icon_text_fit_matches_gl_js => "icon-text-fit-proportional",
    vertical_point_labels_match_gl_js => "vertical-point",
    vertical_line_labels_match_gl_js => "vertical-line",
    sort_keys_and_sizes_match_gl_js => "sort-key-sizes",
    polygon_labels_match_gl_js => "polygons",
    inline_images_match_gl_js => "inline-images",
    promoted_ids_match_gl_js => "promote-id",
    cyrillic_point_labels_match_gl_js => "poi-cyrillic",
    many_variable_anchors_match_gl_js => "poi-variable-many",
    viewport_glyph_line_labels_match_gl_js => "line-cyrillic",
    repeated_line_labels_match_gl_js => "line-repeat",
    polygon_outline_labels_match_gl_js => "polygon-outlines",
    point_labels_of_lines_match_gl_js => "line-points",
}

/// Every case lays something out, so none passes vacuously.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn every_case_places_symbols() {
    for (name, json) in CASES {
        let case = run(name, json);
        assert!(!case.bucket.is_empty(), "{name} is empty");
    }
}

/// An overzoom slice laid out in its ancestor keeps the symbols GL JS lays
/// out in the whole ancestor whose anchors fall in the slice: over a
/// case's tile cut in four and in sixteen, every instance is laid out once.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn slices_keep_their_ancestors_symbols() {
    for (name, json) in CASES {
        let golden: Value = serde_json::from_str(json).unwrap();
        let mut want: Vec<(i64, i64, u32)> = golden["out"]["instances"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| {
                let at = |k| num(&i[k]) as i64;
                (at("anchorX"), at("anchorY"), num(&i["featureIndex"]) as u32)
            })
            .collect();
        want.sort_unstable();
        for scale in [2u32, 4] {
            let mut got = Vec::new();
            for cell in (0..scale * scale).map(|c| [c % scale, c / scale]) {
                let case = run_in(name, json, Some(Ancestor { scale, cell }));
                for instance in &case.bucket.instances {
                    // Anchors are kept to a unit, a finer one in the
                    // slice where they aren't rounded (polygons' poles).
                    let ancestor = |i: usize| {
                        let v = i64::from(instance.anchor[i]) + i64::from(cell[i]) * 8192;
                        v.div_euclid(i64::from(scale))
                    };
                    got.push((ancestor(0), ancestor(1), instance.feature));
                }
            }
            got.sort_unstable();
            assert_eq!(got, want, "{name} in slices of {scale}");
        }
    }
}
