use orb_tile::{GeometryType, LayerBuilder};

use crate::*;

fn style(layers: &str) -> Result<Scene, Diagnostics> {
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
    style.layers.into_iter().next().unwrap()
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
    assert_eq!(dem.encoding, Some(crate::DemEncoding::Custom));
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
    assert_eq!(line.filter.as_ref().unwrap().select(&tile, 0.0), [1]);
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
             "paint": {"fill-color-transition": {"duration": 0}, "fill-emissive": 1, "fill-pattern": "p"}}]"#,
    )
    .unwrap();
    let warnings: Vec<String> = s.warnings.iter().map(|w| w.to_string()).collect();
    assert_eq!(
        warnings,
        [
            "layers[0].layout.text-wrap: unknown property \"text-wrap\"",
            "layers[1].paint.fill-emissive: unknown property \"fill-emissive\"",
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
    // v8's light lowers to ambient `1 − i` and directional `2i`.
    let directional = s.lights.directional.as_ref().unwrap();
    let crate::Direction::Position(position) = &directional.direction else {
        panic!("a positioned light")
    };
    assert_eq!(position.evaluate(0.0), [1.5, 120.0, 50.0]);
    assert_eq!(directional.anchor.evaluate(0.0), TranslateAnchor::Map);
    assert!((directional.level.evaluate(0.0) - 1.4).abs() < 1e-6);
    let ambient = s.lights.ambient.as_ref().unwrap();
    assert!((ambient.level.evaluate(0.0) - 0.3).abs() < 1e-6);
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
        crate::Transition {
            duration: 500.0,
            delay: 0.0
        }
    );
    let (name, t) = &style.layers[0].transitions[0];
    assert_eq!(name, "background-color");
    assert_eq!(
        t.over(style.transition),
        crate::Transition {
            duration: 500.0,
            delay: 100.0
        }
    );
    let default = parse(r#"{"version": 8, "sources": {}, "layers": []}"#).unwrap();
    assert_eq!(default.transition, crate::Transition::default());
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
    let line = |s: &Scene| s.layers[0].clone();
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
    let s = parse_with_state(text, &[("c".into(), Value::String("blue".into()))]).unwrap();
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
    style.layers.into_iter().next().unwrap()
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
        let style = parse(&format!(
            r#"{{"version": 8, "projection": {projection}, "sources": {{}}, "layers": []}}"#
        ))
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
    assert_eq!(parse_projection(&json, &state).0, None);
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
