use orb_tile::{GeometryType, LayerBuilder};

use crate::v8::*;

fn style(layers: &str) -> Result<Style, Diagnostics> {
    parse(&format!(
        r#"{{
            "version": 8,
            "sources": {{
                "v": {{"type": "vector", "tiles": ["https://example.com/{{z}}/{{x}}/{{y}}.pbf"], "maxzoom": 14, "scheme": "tms"}},
                "r": {{"type": "raster", "tiles": ["https://example.com/{{z}}.png"]}}
            }},
            "layers": {layers}
        }}"#
    ))
}

fn errors(layers: &str) -> Vec<String> {
    style(layers)
        .unwrap_err()
        .errors
        .iter()
        .map(|e| e.to_string())
        .collect()
}

fn line(paint: &str, layout: &str) -> Layer {
    let style = style(&format!(
        r#"[{{"id": "l", "type": "line", "source": "v", "source-layer": "roads", "paint": {paint}, "layout": {layout}}}]"#
    ))
    .unwrap_or_else(|d| panic!("{d}"));
    (*style.layers[0]).clone()
}

#[test]
fn defaults_follow_the_spec() {
    let l = line("{}", "{}");
    assert_eq!(l.get(LINE_WIDTH).constant(), Some(1.0));
    assert_eq!(l.get(LINE_OPACITY).constant(), Some(1.0));
    assert_eq!(l.get(LINE_COLOR).constant(), Some(Color::BLACK));
    assert_eq!(l.get(LINE_GAP_WIDTH).constant(), Some(0.0));
    assert_eq!(
        l.get(LINE_TRANSLATE_ANCHOR).constant(),
        Some(TranslateAnchor::Map)
    );
    assert_eq!(l.get(LINE_DASHARRAY).constant(), Some(None));
    assert_eq!(l.get(LINE_ROUND_LIMIT).constant(), Some(1.05));
    assert_eq!(l.get(LINE_SORT_KEY).constant(), Some(None));
    assert!(l.ramp(LINE_GRADIENT).is_none());
}

#[test]
fn sources_are_typed() {
    let s = style("[]").unwrap();
    let SourceKind::Vector(v) = &s.source("v").unwrap().kind else {
        panic!()
    };
    assert_eq!(v.tileset.maxzoom, Some(14.0));
    assert_eq!(v.tileset.minzoom, None);
    assert_eq!(v.tileset.scheme, Some(Scheme::Tms));
    assert_eq!(v.tileset.tiles.as_ref().unwrap().len(), 1);
    assert!(s.warnings.is_empty(), "{:?}", s.warnings);
    let SourceKind::Raster(r) = &s.source("r").unwrap().kind else {
        panic!()
    };
    assert_eq!(r.tileset.tiles.as_ref().unwrap().len(), 1);
    assert_eq!(r.tile_size, None);
}

#[test]
fn raster_dem_sources_and_image_layers() {
    let s = parse(
        r##"{"version": 8, "sources": {
            "dem": {"type": "raster-dem", "tiles": ["t/{z}/{x}/{y}.png"], "tileSize": 256,
                    "encoding": "custom", "redFactor": 0, "baseShift": 3.5},
            "r": {"type": "raster", "url": "https://example.com/tiles.json"}
        }, "layers": [
            {"id": "r", "type": "raster", "source": "r", "paint": {"raster-opacity": 0.5, "resampling": "nearest"}},
            {"id": "h", "type": "hillshade", "source": "dem", "paint": {
                "hillshade-highlight-color": ["#FF4000", "#FFFF00"],
                "hillshade-illumination-direction": [270, 315, 0],
                "hillshade-method": "multidirectional"}},
            {"id": "c", "type": "color-relief", "source": "dem", "paint": {
                "color-relief-color": ["interpolate", ["linear"], ["elevation"], 0, "black", 100, "white"]}}
        ]}"##,
    )
    .unwrap();
    assert!(s.warnings.is_empty(), "{:?}", s.warnings);
    let SourceKind::RasterDem(dem) = &s.source("dem").unwrap().kind else {
        panic!()
    };
    assert_eq!(dem.encoding, Some(crate::v8::DemEncoding::Custom));
    assert_eq!(dem.factors, [0.0, 1.0, 1.0, 3.5]);
    assert_eq!(dem.raster.tile_size, Some(256.0));
    let h = &s.layers[1];
    assert_eq!(h.list(HILLSHADE_HIGHLIGHT_COLOR).len(), 2);
    assert_eq!(h.list(HILLSHADE_ILLUMINATION_DIRECTION).len(), 3);
    assert_eq!(h.list(HILLSHADE_SHADOW_COLOR).len(), 1);
    assert_eq!(
        h.list(HILLSHADE_ILLUMINATION_DIRECTION)[2].constant(),
        Some(0.0)
    );
    let c = &s.layers[2];
    let ramp = c.ramp(COLOR_RELIEF_COLOR).as_ref().unwrap().expression();
    assert_eq!(ramp.interpolate_labels(), Some(&[0.0, 100.0][..]));
}

#[test]
fn image_layers_need_matching_sources() {
    let errors = errors(
        r#"[{"id": "h", "type": "hillshade", "source": "r"},
            {"id": "f", "type": "fill", "source": "r"}]"#,
    );
    assert!(
        errors[0].contains("requires a raster-dem source"),
        "{errors:?}"
    );
    assert!(errors[1].contains("requires a vector source"), "{errors:?}");
}

fn tile() -> orb_tile::Layer {
    let mut b = LayerBuilder::new("roads", 4096);
    for (class, width) in [("primary", 4.0), ("minor", 1.0)] {
        b.begin_feature(None, GeometryType::LineString);
        b.begin_part();
        b.begin_ring();
        b.push_vertex([0, 0]);
        b.set_property("class", orb_tile::Value::Str(class.into()));
        b.set_property("width", orb_tile::Value::F64(width));
    }
    b.finish()
}

#[test]
fn source_and_composite_values_evaluate_per_feature() {
    let l = line(
        r#"{
            "line-color": ["match", ["get", "class"], "primary", "red", "blue"],
            "line-width": ["interpolate", ["linear"], ["zoom"], 10, ["get", "width"], 20, ["*", 4, ["get", "width"]]],
            "line-opacity": ["get", "missing"]
        }"#,
        r#"{"line-cap": ["match", ["get", "class"], "primary", "round", "bogus"]}"#,
    );
    let tile = tile();
    assert_eq!(l.get(LINE_COLOR).kind(), Kind::Source);
    let color = l.get(LINE_COLOR).bind(&tile);
    assert_eq!(color.evaluate(0.0, 0), Color::parse("red").unwrap());
    assert_eq!(color.evaluate(0.0, 1), Color::parse("blue").unwrap());

    assert_eq!(l.get(LINE_WIDTH).kind(), Kind::Composite);
    let width = l.get(LINE_WIDTH).bind(&tile);
    // The tile job evaluates composite values at its zoom and the next.
    assert_eq!(
        (width.evaluate(10.0, 0), width.evaluate(20.0, 0)),
        (4.0, 16.0)
    );
    assert_eq!(width.evaluate(15.0, 1), 2.5);
    assert_eq!(
        l.get(LINE_WIDTH)
            .as_expression()
            .unwrap()
            .interpolation_factor(15.0, 10.0, 20.0),
        0.5
    );

    // A missing property evaluates to the default, as GL JS falls back.
    assert_eq!(l.get(LINE_OPACITY).bind(&tile).evaluate(0.0, 0), 1.0);
    // So does a value outside an enum.
    let cap = l.get(LINE_CAP).bind(&tile);
    assert_eq!(
        (cap.evaluate(0.0, 0), cap.evaluate(0.0, 1)),
        (LineCap::Round, LineCap::Butt)
    );
}

#[test]
fn legacy_functions_and_filters() {
    let l = line(
        r#"{"line-width": {"base": 1.5, "stops": [[10, 2], [20, 8]]}, "line-color": {"property": "class", "type": "categorical", "stops": [["primary", "red"]], "default": "blue"}}"#,
        "{}",
    );
    assert_eq!(l.get(LINE_WIDTH).kind(), Kind::Camera);
    assert_eq!(l.get(LINE_WIDTH).evaluate(10.0), 2.0);
    assert_eq!(l.get(LINE_COLOR).kind(), Kind::Source);
    let tile = tile();
    assert_eq!(
        l.get(LINE_COLOR).bind(&tile).evaluate(0.0, 1),
        Color::parse("blue").unwrap()
    );

    let s = style(
        r#"[{"id": "a", "type": "line", "source": "v", "source-layer": "roads", "filter": ["all", ["==", "class", "minor"], [">=", "width", 1]]}]"#,
    )
    .unwrap();
    let line = &s.layers[0];
    assert_eq!(line.filter.as_ref().unwrap().select(&tile, 0.0, None), [1]);
}

#[test]
fn errors_carry_json_paths() {
    let e =
        errors(r#"[{"id": "bg", "type": "background", "paint": {"background-color": "nope"}}]"#);
    assert_eq!(
        e,
        ["layers[0].paint.background-color: Could not parse color from value 'nope'"]
    );

    let e = errors(
        r#"[{"id": "l", "type": "line", "source": "v", "source-layer": "x", "paint": {"line-width": ["+", 1, "two"]}}]"#,
    );
    assert_eq!(
        e,
        ["layers[0].paint.line-width[2]: Expected number but found string instead."]
    );

    let e = errors(
        r#"[{"id": "l", "type": "line", "source": "v", "source-layer": "x", "paint": {"line-translate": ["get", "t"]}}]"#,
    );
    assert_eq!(
        e,
        ["layers[0].paint.line-translate: data expressions not supported"]
    );

    let e = errors(
        r#"[{"id": "l", "type": "line", "source": "v", "source-layer": "x", "paint": {"line-opacity": 2, "line-width": -1}, "layout": {"line-cap": "flat"}}]"#,
    );
    assert_eq!(
        e,
        [
            "layers[0].layout.line-cap: expected one of [butt, round, square], \"flat\" found",
            "layers[0].paint.line-opacity: 2 is greater than the maximum value 1",
            "layers[0].paint.line-width: -1 is less than the minimum value 0",
        ]
    );

    let e = errors(
        r#"[{"id": "a", "type": "fill", "source": "nowhere", "source-layer": "x"},
            {"id": "a", "type": "line", "source": "v"},
            {"id": "c", "type": "fill", "source": "r"},
            {"id": "d", "type": "bogus"},
            {"id": "e", "type": "line", "source": "v", "source-layer": "x", "filter": ["==", "$type", "Circle"]}]"#,
    );
    assert_eq!(
        e,
        [
            "layers[0].source: source \"nowhere\" not found",
            "layers[1].id: duplicate layer id \"a\"",
            "layers[1]: layer \"a\" must specify a \"source-layer\"",
            "layers[2].source: layer \"c\" requires a vector source",
            "layers[3].type: expected one of [fill, line, symbol, circle, heatmap, fill-extrusion, raster, hillshade, color-relief, background], \"bogus\" found",
            "layers[4].filter[2]: expected one of [Point, LineString, Polygon], \"Circle\" found",
        ]
    );

    let e = parse(r#"{"version": 7, "sources": {}, "layers": []}"#).unwrap_err();
    assert_eq!(
        e.errors[0].to_string(),
        "version: expected one of [8], 7 found"
    );
    let e = parse(r#"{"version": 8, "sources": {}, "layers": [}"#).unwrap_err();
    assert_eq!(e.errors[0].span.start, 41);
}

#[test]
fn warnings_for_what_is_not_supported() {
    let s = style(
        r#"[{"id": "s", "type": "symbol", "source": "v", "source-layer": "x", "layout": {"text-field": "x", "text-wrap": 1}},
            {"id": "f", "type": "fill", "source": "v", "source-layer": "x", "minzoom": 3, "maxzoom": 0,
             "paint": {"fill-color-transition": {"duration": 0}, "fill-glow": 1,"fill-pattern": "p"}}]"#,
    )
    .unwrap();
    let warnings: Vec<String> = s.warnings.iter().map(|w| w.to_string()).collect();
    assert_eq!(
        warnings,
        [
            "layers[0].layout.text-wrap: unknown property \"text-wrap\"",
            "layers[1].paint.fill-glow: unknown property \"fill-glow\"",
        ]
    );
    // GL JS reads a `maxzoom` of 0 as unset.
    assert!(s.layers[1].is_visible_at(23.0));
    assert!(!s.layers[1].is_visible_at(2.0));
}

#[test]
fn default_layers_draw_at_negative_zooms() {
    let s = style(
        r#"[{"id": "f", "type": "fill", "source": "v", "source-layer": "x"},
            {"id": "g", "type": "fill", "source": "v", "source-layer": "x", "minzoom": 2}]"#,
    )
    .unwrap();
    // GL JS reads a `minzoom` of 0 as unset, so maps can go below zoom 0.
    assert!(s.layers[0].is_visible_at(-0.001));
    assert!(s.layers[0].is_visible_at(-2.0));
    assert!(!s.layers[1].is_visible_at(1.99));
    assert!(s.layers[1].is_visible_at(2.0));
}

#[test]
fn extrusion_paint_and_light_are_typed() {
    let s = parse(r##"{"version":8,"light":{"anchor":"map","position":[1.5,120,50],"color":"#abcd12","intensity":0.7},
      "sources":{"s":{"type":"geojson","data":{"type":"FeatureCollection","features":[]}}},
      "layers":[{"id":"e","type":"fill-extrusion","source":"s","paint":{"fill-extrusion-height":["get","height"],"fill-extrusion-base":12,
      "fill-extrusion-opacity":0.5,"fill-extrusion-vertical-gradient":false}}]}"##).unwrap();
    assert!(s.warnings.is_empty(), "{:?}", s.warnings);
    assert_eq!(s.light.position.evaluate(0.0), [1.5, 120.0, 50.0]);
    assert_eq!(s.light.anchor.evaluate(0.0), TranslateAnchor::Map);
    assert_eq!(s.light.intensity.evaluate(0.0), 0.7);
    let e = &s.layers[0];
    assert_eq!(e.get(EXTRUSION_HEIGHT).kind(), Kind::Source);
    assert_eq!(e.get(EXTRUSION_BASE).evaluate(0.0), 12.0);
    assert_eq!(e.get(EXTRUSION_OPACITY).evaluate(0.0), 0.5);
    assert!(!e.get(EXTRUSION_VERTICAL_GRADIENT).evaluate(0.0));
}

#[test]
fn transitions_parse_and_merge_over_the_style() {
    let style = parse(
        r#"{"version": 8, "transition": {"duration": 500}, "sources": {}, "layers": [
            {"id": "b", "type": "background", "paint": {"background-color": "red",
                "background-color-transition": {"delay": 100}}}]}"#,
    )
    .unwrap();
    assert!(style.warnings.is_empty(), "{:?}", style.warnings);
    assert_eq!(
        style.transition,
        crate::v8::Transition {
            duration: 500.0,
            delay: 0.0
        }
    );
    let (name, t) = &style.layers[0].transitions[0];
    assert_eq!(name, "background-color");
    assert_eq!(
        t.over(style.transition),
        crate::v8::Transition {
            duration: 500.0,
            delay: 100.0
        }
    );
    let default = parse(r#"{"version": 8, "sources": {}, "layers": []}"#).unwrap();
    assert_eq!(default.transition, crate::v8::Transition::default());
}

#[test]
fn global_state_resolves_at_parse() {
    let text = r#"{"version": 8, "state": {"c": {"default": "red"}, "w": {"default": 2}},
        "sources": {"v": {"type": "vector", "tiles": ["https://example.com/{z}/{x}/{y}.pbf"],
                          "promoteId": {"roads": "osm_id"}}},
        "layers": [{"id": "l", "type": "line", "source": "v", "source-layer": "roads",
            "filter": ["==", ["get", "class"], ["global-state", "cls"]],
            "paint": {"line-color": ["global-state", "c"],
                      "line-width": ["coalesce", ["feature-state", "w"], ["global-state", "w"]]}}]}"#;
    let line = |s: &Style| s.layers[0].clone();
    let s = parse(text).unwrap();
    assert!(s.warnings.is_empty(), "{:?}", s.warnings);
    let l = line(&s);
    // Only global state: folded to a constant.
    assert_eq!(
        l.get(LINE_COLOR).constant(),
        Some(Color::parse("red").unwrap())
    );
    // State-only: a state lookup, with the global default as the fallback.
    assert!(l.get(LINE_WIDTH).is_state_lookup());
    assert_eq!(l.get(LINE_WIDTH).evaluate_state(0.0, &[], None), 2.0);
    let state = [("w".to_string(), Value::Number(5.0))];
    assert_eq!(l.get(LINE_WIDTH).evaluate_state(0.0, &state, None), 5.0);
    let s = parse_with(
        text,
        Previews::NONE,
        &[("c".into(), Value::String("blue".into()))],
    )
    .unwrap();
    assert_eq!(
        line(&s).get(LINE_COLOR).constant(),
        Some(Color::parse("blue").unwrap())
    );
    let SourceKind::Vector(v) = &s.sources[0].kind else {
        unreachable!()
    };
    assert_eq!(
        v.promote_id.as_ref().and_then(|p| p.key("roads")),
        Some("osm_id")
    );
    assert_eq!(v.promote_id.as_ref().and_then(|p| p.key("water")), None);
}

fn symbol(layout: &str, paint: &str) -> Layer {
    let style = style(&format!(
        r#"[{{"id": "s", "type": "symbol", "source": "v", "source-layer": "poi", "layout": {layout}, "paint": {paint}}}]"#
    ))
    .unwrap_or_else(|d| panic!("{d}"));
    (*style.layers[0]).clone()
}

#[test]
fn image_expressions_resolve_with_the_styles_images() {
    let plain = symbol(r#"{"icon-image": "marker"}"#, "{}");
    assert_eq!(
        plain.get(ICON_IMAGE).constant(),
        Some(Some("marker".into()))
    );
    // GL JS evaluates constant image expressions with the available images.
    let s = symbol(
        r#"{"icon-image": ["coalesce", ["image", "a"], ["image", "b"], ["image", "c"]]}"#,
        "{}",
    );
    let mut b = LayerBuilder::new("poi", 4096);
    b.begin_feature(None, GeometryType::Point);
    b.begin_part();
    b.begin_ring();
    b.push_vertex([0, 0]);
    let tile = b.finish();
    let icon = s.get(ICON_IMAGE);
    let images = ["c".to_string(), "b".to_string()];
    let with = |images: &[String]| icon.bind_with_images(&tile, images).evaluate(0.0, 0);
    assert_eq!(with(&images).as_deref(), Some("b"));
    // None available: the first requested.
    assert_eq!(with(&[]).as_deref(), Some("a"));
}

#[test]
fn symbol_layers_are_typed() {
    let s = symbol("{}", "{}");
    let l = &s;
    assert_eq!(
        l.get(SYMBOL_PLACEMENT).constant(),
        Some(SymbolPlacement::Point)
    );
    assert_eq!(l.get(TEXT_SIZE).constant(), Some(16.0));
    assert_eq!(
        l.get(TEXT_FONT).constant().unwrap(),
        ["Open Sans Regular", "Arial Unicode MS Regular"]
    );
    assert_eq!(l.get(ICON_PADDING).constant(), Some(Padding([2.0; 4])));
    assert_eq!(l.get(TEXT_OVERLAP).constant(), Some(None));
    assert_eq!(l.get(TEXT_VARIABLE_ANCHOR).constant(), Some(None));
    assert_eq!(l.get(TEXT_FIELD).constant().unwrap()[0].text, "");
    assert_eq!(
        s.get(TEXT_HALO_COLOR).constant().unwrap().premultiplied(),
        [0.0; 4]
    );

    let s = symbol(
        r#"{
            "text-field": "{name} ({ref})",
            "icon-image": "{maki}-15",
            "text-line-height": 1.25,
            "icon-padding": [1, 2],
            "text-writing-mode": ["vertical", "horizontal"],
            "text-variable-anchor": ["top", "bogus"],
            "text-variable-anchor-offset": ["top", [0, 1], "left", [2, 0]],
            "symbol-placement": "line-center"
        }"#,
        r#"{"text-color": ["get", "color"]}"#,
    );
    let l = &s;
    // Plain strings expand tokens (GL JS `resolveTokens`).
    assert_eq!(l.get(TEXT_FIELD).kind(), Kind::Source);
    let mut b = LayerBuilder::new("poi", 4096);
    b.begin_feature(None, GeometryType::Point);
    b.begin_part();
    b.begin_ring();
    b.push_vertex([0, 0]);
    b.set_property("name", orb_tile::Value::Str("Café".into()));
    b.set_property("maki", orb_tile::Value::Str("bar".into()));
    let tile = b.finish();
    let text = l.get(TEXT_FIELD).bind(&tile).evaluate(0.0, 0);
    assert_eq!(
        text.iter().map(|s| s.text.as_str()).collect::<String>(),
        "Café ()"
    );
    assert_eq!(
        l.get(ICON_IMAGE).bind(&tile).evaluate(0.0, 0).as_deref(),
        Some("bar-15")
    );
    // Numbers are f64, as GL JS lays out with doubles.
    assert_eq!(l.get(TEXT_LINE_HEIGHT).constant(), Some(1.25));
    assert_eq!(
        l.get(ICON_PADDING).constant(),
        Some(Padding([1.0, 2.0, 1.0, 2.0]))
    );
    assert_eq!(
        l.get(TEXT_WRITING_MODE).constant(),
        Some(Some(vec![WritingMode::Vertical, WritingMode::Horizontal]))
    );
    // An item outside the enum falls back to the default.
    assert_eq!(l.get(TEXT_VARIABLE_ANCHOR).constant(), Some(None));
    assert_eq!(
        l.get(TEXT_VARIABLE_ANCHOR_OFFSET).constant(),
        Some(Some(vec![
            (Anchor::Top, [0.0, 1.0]),
            (Anchor::Left, [2.0, 0.0])
        ]))
    );
    assert_eq!(
        l.get(SYMBOL_PLACEMENT).constant(),
        Some(SymbolPlacement::LineCenter)
    );
    assert_eq!(s.get(TEXT_COLOR).kind(), Kind::Source);
    assert_eq!(s.get(ICON_COLOR).kind(), Kind::Constant);
}

#[test]
fn glyphs_urls_need_their_tokens() {
    let glyphs = |url: &str| {
        parse(&format!(
            r#"{{"version": 8, "glyphs": "{url}", "sources": {{}}, "layers": []}}"#
        ))
    };
    assert_eq!(
        glyphs("https://example.com/{fontstack}/{range}.pbf")
            .unwrap()
            .glyphs
            .as_deref(),
        Some("https://example.com/{fontstack}/{range}.pbf")
    );
    let errors = glyphs("https://example.com/{range}.pbf").unwrap_err();
    assert_eq!(
        errors.to_string(),
        "glyphs: \"glyphs\" url must include a \"{fontstack}\" token"
    );
}

#[test]
fn projections_parse_to_globes() {
    let globe = |projection: &str| {
        let style = parse_with(
            &format!(
                r#"{{"version": 8, "projection": {projection}, "sources": {{}}, "layers": []}}"#
            ),
            Previews::ALL,
            &[],
        )
        .unwrap();
        (style.globe, style.warnings.len())
    };
    assert_eq!(globe(r#"{"type": "mercator"}"#), (None, 0));
    assert_eq!(globe(r#"{"type": "globe"}"#), (Some(orb_crs::Globe::V8), 0));
    assert_eq!(
        globe(r#"{"type": "vertical-perspective"}"#),
        (Some(orb_crs::Globe::SPHERE), 0)
    );
    assert_eq!(
        globe(
            r#"{"type": ["interpolate", ["linear"], ["zoom"], 5, "vertical-perspective", 7, "mercator"]}"#
        ),
        (
            Some(orb_crs::Globe {
                handover: Some([5.0, 7.0]),
                plane: None,
            }),
            0
        )
    );
    // style-spec#1865 and #1866: low-zoom planes, and transitions.
    let equal_earth = orb_crs::Projection::from_descriptor("equal-earth").ok();
    assert_eq!(
        globe(r#"{"type": "equal-earth"}"#),
        (
            Some(orb_crs::Globe {
                handover: Some([6.0, 7.0]),
                plane: equal_earth,
            }),
            0
        )
    );
    assert_eq!(
        globe(r#"{"type": "equal-earth", "transition": false}"#),
        (
            Some(orb_crs::Globe {
                handover: None,
                plane: equal_earth,
            }),
            0
        )
    );
    assert_eq!(
        globe(r#"{"type": "globe", "transition": [4, 5]}"#),
        (
            Some(orb_crs::Globe {
                handover: Some([4.0, 5.0]),
                plane: None,
            }),
            0
        )
    );
    assert_eq!(
        globe(r#"{"type": "globe", "transition": false}"#),
        (Some(orb_crs::Globe::SPHERE), 0)
    );
    assert_eq!(
        globe(r#"{"type": "globe", "transition": [5, 4]}"#),
        (Some(orb_crs::Globe::V8), 1)
    );
    assert_eq!(globe(r#"{"type": "lcc"}"#), (None, 1));
}

#[test]
fn projections_read_the_global_state() {
    let style = parse(
        r#"{"version": 8, "state": {"kind": {"default": "vertical-perspective"}},
            "projection": {"type": ["global-state", "kind"]}, "sources": {}, "layers": []}"#,
    )
    .unwrap();
    assert_eq!(style.globe, Some(orb_crs::Globe::SPHERE));
    let state = vec![("kind".to_string(), Value::String("mercator".into()))];
    let json = orb_expr::json::parse(r#"{"type": ["global-state", "kind"]}"#).unwrap();
    assert_eq!(parse_projection(&json, &state, Previews::NONE).0, None);
}

/// style-spec's `layer-blend` validation test: every mode of both
/// properties parses, with no warnings.
#[test]
fn layer_blends_parse() {
    let modes = [
        ("normal", LayerBlend::Normal),
        ("multiply", LayerBlend::Multiply),
        ("color", LayerBlend::Color),
        ("luminosity", LayerBlend::Luminosity),
        ("plus-lighter", LayerBlend::PlusLighter),
        ("screen", LayerBlend::Screen),
        ("overlay", LayerBlend::Overlay),
    ];
    let layers: Vec<String> = ["fill", "line"]
        .iter()
        .flat_map(|ty| {
            modes.iter().map(move |(mode, _)| {
                format!(
                    r#"{{"id": "{ty}-{mode}", "type": "{ty}", "source": "geojson",
                        "paint": {{"{ty}-layer-blend": "{mode}"}}}}"#
                )
            })
        })
        .collect();
    let style = parse(&format!(
        r#"{{"version": 8, "sources": {{"geojson": {{"type": "geojson",
            "data": {{"type": "FeatureCollection", "features": []}}}}}},
            "layers": [{}]}}"#,
        layers.join(",")
    ))
    .unwrap();
    assert!(style.warnings.is_empty(), "{:?}", style.warnings);
    let blends: Vec<LayerBlend> = style
        .layers
        .iter()
        .map(|l| match l.ty {
            LayerType::Fill => l.get(FILL_LAYER_BLEND).evaluate(0.0),
            _ => l.get(LINE_LAYER_BLEND).evaluate(0.0),
        })
        .collect();
    let expected: Vec<LayerBlend> = modes.iter().chain(&modes).map(|&(_, b)| b).collect();
    assert_eq!(blends, expected);
    // `erase` was dropped from the spec: not a mode.
    let error = parse(
        r#"{"version": 8, "sources": {"s": {"type": "geojson",
            "data": {"type": "FeatureCollection", "features": []}}},
            "layers": [{"id": "f", "type": "fill", "source": "s",
            "paint": {"fill-layer-blend": "erase"}}]}"#,
    )
    .unwrap_err();
    assert_eq!(error.errors[0].path, "layers[0].paint.fill-layer-blend");
}

#[test]
fn parses_projection_and_source_matrix_sets() {
    let style = crate::v8::parse_with(
        r#"{
            "version": 8,
            "projection": {"type": "EPSG:3413"},
            "sources": {
                "linz": {"type": "raster", "tiles": ["a/{z}/{x}/{y}.png"],
                         "tileMatrixSet": "NZTM2000Quad"},
                "gibs": {"type": "raster", "tiles": ["b/{z}/{x}/{y}.png"], "tileMatrixSet": {
                    "id": "GIBS3413", "crs": "http://www.opengis.net/def/crs/EPSG/0/3413",
                    "tileMatrices": [
                        {"id": "0", "scaleDenominator": 29257142.85714286, "pointOfOrigin": [-4194304, 4194304],
                         "tileWidth": 512, "tileHeight": 512, "matrixWidth": 2, "matrixHeight": 2},
                        {"id": "1", "scaleDenominator": 14628571.42857143, "pointOfOrigin": [-4194304, 4194304],
                         "tileWidth": 512, "tileHeight": 512, "matrixWidth": 4, "matrixHeight": 4}
                    ]}}
            },
            "layers": []
        }"#,
        Previews::ALL,
        &[],
    )
    .unwrap();
    assert!(style.warnings.is_empty(), "{:?}", style.warnings);
    let projection = style.projection.unwrap();
    assert_eq!(
        projection.descriptor(),
        orb_crs::Descriptor::named("EPSG:3413")
    );
    let sets = style.tile_matrix_sets();
    assert_eq!(sets.len(), 2);
    let gibs = &sets[1].1;
    assert!(gibs.projection().same_plane(&projection));
    assert_eq!(gibs.matrix_size(0), (2, 2));
    // 8,192 m pixels, 512 per tile.
    assert!((gibs.tile_span(0) - 4_194_304.0).abs() < 1e-3);
    assert_eq!(style.globe, None, "national planes stay fixed");
}

#[test]
fn world_planes_adapt_to_mercator() {
    let style = |projection: &str| {
        crate::v8::parse_with(
            &format!(
                r#"{{"version": 8, "projection": {projection}, "sources": {{}}, "layers": []}}"#
            ),
            Previews::ALL,
            &[],
        )
        .unwrap()
    };
    let equal_earth = orb_crs::Projection::from_descriptor("equal-earth").unwrap();
    let adaptive = style(r#"{"type": "equal-earth"}"#);
    assert_eq!(adaptive.projection, None);
    assert_eq!(
        adaptive.globe,
        Some(orb_crs::Globe {
            handover: Some([6.0, 7.0]),
            plane: Some(equal_earth),
        })
    );
    let utm = style(r#"{"type": "EPSG:32633", "transition": [8, 9]}"#);
    assert_eq!(utm.projection, None);
    assert_eq!(utm.globe.unwrap().handover, Some([8.0, 9.0]));
    let polar =
        style(r#"{"type": "polar-stereographic", "lat_0": 90, "lat_ts": 70, "lon_0": -45}"#);
    assert!(polar.projection.is_some() && polar.warnings.is_empty());
    // Unknown planes and invalid transitions warn and draw Web Mercator.
    for bad in [
        r#"{"type": "EPSG:31467"}"#,
        r#"{"type": "equal-earth", "lon_0": "east"}"#,
    ] {
        let style = style(bad);
        assert_eq!(style.warnings.len(), 1, "{bad}: {:?}", style.warnings);
    }
    let reversed = style(r#"{"type": "equal-earth", "transition": [9, 8]}"#);
    assert_eq!(reversed.warnings.len(), 1);
    assert_eq!(reversed.globe.unwrap().handover, Some([6.0, 7.0]));
    let bad_tms = crate::v8::parse_with(
        r#"{"version": 8, "sources": {"a": {"type": "raster", "tiles": ["a"],
            "tileMatrixSet": "GNOSISGlobalGrid"}}, "layers": []}"#,
        Previews::ALL,
        &[],
    )
    .unwrap_err();
    let paths: Vec<_> = bad_tms.errors.iter().map(|e| e.path.as_str()).collect();
    assert_eq!(paths, ["sources.a.tileMatrixSet"]);
}

/// With their previews off, inputs beyond the approved spec warn and fall
/// back to the default: Web Mercator, and sources in `WebMercatorQuad`.
#[test]
fn previews_gate_projections_and_tile_matrix_sets() {
    let style = |previews: Previews, projection: &str| {
        crate::v8::parse_with(
            &format!(
                r#"{{"version": 8, "projection": {projection}, "sources": {{"a": {{"type": "raster",
                    "tiles": ["a"], "tileMatrixSet": "UPSArcticWGS84Quad"}}}}, "layers": []}}"#
            ),
            previews,
            &[],
        )
        .unwrap()
    };
    let paths = |style: &Style| -> Vec<String> {
        let mut paths: Vec<String> = style.warnings.iter().map(|w| w.path.clone()).collect();
        paths.sort();
        paths
    };
    for (projection, preview) in [
        (r#"{"type": "equal-earth"}"#, Preview::EqualEarth),
        (r#"{"type": "natural-earth"}"#, Preview::Equirectangular),
        (r#"{"type": "equirectangular"}"#, Preview::Equirectangular),
        (
            r#"{"type": ["interpolate", ["linear"], ["zoom"], 5, "equal-earth", 7, "mercator"]}"#,
            Preview::EqualEarth,
        ),
        (r#"{"type": "EPSG:3413"}"#, Preview::Crs),
    ] {
        let off = style(Previews::NONE, projection);
        assert_eq!((off.projection, off.globe), (None, None), "{projection}");
        assert_eq!(
            paths(&off),
            ["projection.type", "sources.a.tileMatrixSet"],
            "{projection}"
        );
        let warning = off.warnings.iter().find(|w| w.path == "projection.type");
        assert!(warning.unwrap().message.contains(preview.name()));
        assert!(off.tile_matrix_sets().is_empty());
        let on = style(Previews::NONE.with(preview).with(Preview::Crs), projection);
        assert!(on.warnings.is_empty(), "{projection}: {:?}", on.warnings);
        assert!(
            on.projection.is_some() || on.globe.is_some(),
            "{projection}"
        );
    }
    // A spec projection keeps working; the transition and parameters are
    // ignored without their previews.
    let off = style(
        Previews::NONE.with(Preview::Crs),
        r#"{"type": "globe", "transition": false}"#,
    );
    assert_eq!(off.globe, Some(orb_crs::Globe::V8));
    assert_eq!(paths(&off), ["projection.transition"]);
    let off = style(
        Previews::NONE.with(Preview::EqualEarth),
        r#"{"type": "equal-earth", "lon_0": 150}"#,
    );
    assert_eq!(paths(&off), ["projection.lon_0", "sources.a.tileMatrixSet"]);
}

/// With `orbis:expressions`, paint Orbis evaluates per frame reads the
/// camera; without it, such a value warns and is the default. Inputs no
/// phase honours stay errors, as do the spec's own limits.
#[test]
fn extended_inputs_follow_the_preview() {
    let layer = |paint: &str, layout: &str| {
        format!(
            r#"{{"version": 8, "sources": {{"s": {{"type": "geojson",
                "data": {{"type": "FeatureCollection", "features": []}}}}}},
                "layers": [{{"id": "f", "type": "fill", "source": "s",
                "paint": {{{paint}}}, "layout": {{{layout}}}}}]}}"#
        )
    };
    let pitch = r#""fill-opacity": ["interpolate", ["linear"], ["pitch"], 0, 1, 60, 0.5]"#;
    let on = parse_with(&layer(pitch, ""), Previews::ALL, &[]).unwrap();
    assert!(on.warnings.is_empty(), "{:?}", on.warnings);
    let opacity = on.layers[0].get(FILL_OPACITY);
    assert_eq!(opacity.kind(), Kind::Camera);
    let folded = opacity
        .as_expression()
        .unwrap()
        .at_camera(30.0, 0.0)
        .unwrap();
    assert_eq!(folded.kind(), Kind::Constant);
    assert_eq!(
        folded.expression().evaluate(0.0).unwrap(),
        Value::Number(0.75)
    );

    let off = parse(&layer(pitch, "")).unwrap();
    assert_eq!(off.warnings.len(), 1, "{:?}", off.warnings);
    assert_eq!(off.warnings[0].path, "layers[0].paint.fill-opacity");
    assert!(off.warnings[0].message.contains("orbis:expressions"));
    assert!(!off.layers[0].get(FILL_OPACITY).is_set());

    for (paint, layout) in [
        // Not honoured: the camera where tiles are laid out, or with data.
        (
            r#""fill-translate": ["literal", [0, 0]], "fill-pattern": ["case", [">", ["pitch"], 1], "a", "b"]"#,
            "",
        ),
        (r#""fill-opacity": ["*", ["pitch"], ["get", "x"]]"#, ""),
        ("", r#""fill-sort-key": ["pitch"]"#),
        // The spec's validator: no feature state in layout.
        ("", r#""fill-sort-key": ["feature-state", "x"]"#),
    ] {
        for previews in [Previews::NONE, Previews::ALL] {
            assert!(
                parse_with(&layer(paint, layout), previews, &[]).is_err(),
                "{paint} {layout}"
            );
        }
    }
    // Patterns are laid out per tile: feature state parses, as GL JS has
    // it, but warns.
    let state = parse(&layer(
        r#""fill-pattern": ["coalesce", ["feature-state", "p"], "a"]"#,
        "",
    ))
    .unwrap();
    assert_eq!(state.warnings[0].path, "layers[0].paint.fill-pattern");
    // Filters run per tile.
    let filter = r#"{"version": 8, "sources": {"s": {"type": "geojson",
        "data": {"type": "FeatureCollection", "features": []}}},
        "layers": [{"id": "f", "type": "fill", "source": "s", "filter": [">", ["pitch"], 10]}]}"#;
    assert!(parse_with(filter, Previews::ALL, &[]).is_err());
}
