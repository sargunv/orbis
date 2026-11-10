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
        r#"[{"id": "l", "type": "line", "source": "v", "source-layer": "x", "paint": {"line-translate-anchor": ["get", "t"]}}]"#,
    );
    assert_eq!(
        e,
        ["layers[0].paint.line-translate-anchor: data expressions not supported"]
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
    let with = |images: &[String]| icon.bind(&tile).with_images(images).evaluate(0.0, 0);
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
        globe(r#"{"type": "dev.sargunv.orbis:equal-earth"}"#),
        (
            Some(orb_crs::Globe {
                handover: Some([6.0, 7.0]),
                plane: equal_earth,
            }),
            0
        )
    );
    assert_eq!(
        globe(
            r#"{"type": "dev.sargunv.orbis:equal-earth", "dev.sargunv.orbis:transition": false}"#
        ),
        (
            Some(orb_crs::Globe {
                handover: None,
                plane: equal_earth,
            }),
            0
        )
    );
    assert_eq!(
        globe(r#"{"type": "globe", "dev.sargunv.orbis:transition": [4, 5]}"#),
        (
            Some(orb_crs::Globe {
                handover: Some([4.0, 5.0]),
                plane: None,
            }),
            0
        )
    );
    assert_eq!(
        globe(r#"{"type": "globe", "dev.sargunv.orbis:transition": false}"#),
        (Some(orb_crs::Globe::SPHERE), 0)
    );
    assert_eq!(
        globe(r#"{"type": "globe", "dev.sargunv.orbis:transition": [5, 4]}"#),
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
                         "dev.sargunv.orbis:tileMatrixSet": "NZTM2000Quad"},
                "gibs": {"type": "raster", "tiles": ["b/{z}/{x}/{y}.png"], "dev.sargunv.orbis:tileMatrixSet": {
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
    let adaptive = style(r#"{"type": "dev.sargunv.orbis:equal-earth"}"#);
    assert_eq!(adaptive.projection, None);
    assert_eq!(
        adaptive.globe,
        Some(orb_crs::Globe {
            handover: Some([6.0, 7.0]),
            plane: Some(equal_earth),
        })
    );
    let utm = style(r#"{"type": "EPSG:32633", "dev.sargunv.orbis:transition": [8, 9]}"#);
    assert_eq!(utm.projection, None);
    assert_eq!(utm.globe.unwrap().handover, Some([8.0, 9.0]));
    let polar = style(
        r#"{"type": "dev.sargunv.orbis:polar-stereographic", "lat_0": 90, "lat_ts": 70, "lon_0": -45}"#,
    );
    assert!(polar.projection.is_some() && polar.warnings.is_empty());
    // Unknown planes and invalid transitions warn and draw Web Mercator,
    // as do Orbis's kinds outside its namespace.
    for bad in [
        r#"{"type": "EPSG:31467"}"#,
        r#"{"type": "equal-earth"}"#,
        r#"{"type": "lcc", "lat_1": 30}"#,
        r#"{"type": "dev.sargunv.orbis:globe"}"#,
        r#"{"type": "dev.sargunv.orbis:equal-earth", "lon_0": "east"}"#,
    ] {
        let style = style(bad);
        assert_eq!(style.warnings.len(), 1, "{bad}: {:?}", style.warnings);
    }
    let reversed = style(
        r#"{"type": "dev.sargunv.orbis:equal-earth", "dev.sargunv.orbis:transition": [9, 8]}"#,
    );
    assert_eq!(reversed.warnings.len(), 1);
    assert_eq!(reversed.globe.unwrap().handover, Some([6.0, 7.0]));
    let bad_tms = crate::v8::parse_with(
        r#"{"version": 8, "sources": {"a": {"type": "raster", "tiles": ["a"],
            "dev.sargunv.orbis:tileMatrixSet": "GNOSISGlobalGrid"}}, "layers": []}"#,
        Previews::ALL,
        &[],
    )
    .unwrap_err();
    let paths: Vec<_> = bad_tms.errors.iter().map(|e| e.path.as_str()).collect();
    assert_eq!(paths, ["sources.a.dev.sargunv.orbis:tileMatrixSet"]);
    // GeoJSON tiles on the plane, whatever set it names: drawn through
    // that set's warp, it would land in the wrong place.
    let geojson = crate::v8::parse_with(
        r#"{"version": 8, "sources": {"g": {"type": "geojson",
            "data": {"type": "FeatureCollection", "features": []},
            "dev.sargunv.orbis:tileMatrixSet": "WorldCRS84Quad"}}, "layers": []}"#,
        Previews::ALL,
        &[],
    )
    .unwrap();
    assert!(geojson.tile_matrix_sets().is_empty());
    let paths: Vec<_> = geojson.warnings.iter().map(|w| w.path.as_str()).collect();
    assert_eq!(paths, ["sources.g.dev.sargunv.orbis:tileMatrixSet"]);
}

/// With their previews off, inputs beyond the approved spec warn and fall
/// back to the default: Web Mercator, and sources in `WebMercatorQuad`.
#[test]
fn previews_gate_projections_and_tile_matrix_sets() {
    let style = |previews: Previews, projection: &str| {
        crate::v8::parse_with(
            &format!(
                r#"{{"version": 8, "projection": {projection}, "sources": {{"a": {{"type": "raster",
                    "tiles": ["a"], "dev.sargunv.orbis:tileMatrixSet": "UPSArcticWGS84Quad"}}}}, "layers": []}}"#
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
        (
            r#"{"type": "dev.sargunv.orbis:equal-earth"}"#,
            Preview::EqualEarth,
        ),
        (
            r#"{"type": "dev.sargunv.orbis:natural-earth"}"#,
            Preview::NaturalEarth,
        ),
        (
            r#"{"type": "dev.sargunv.orbis:equirectangular"}"#,
            Preview::Equirectangular,
        ),
        (
            r#"{"type": ["interpolate", ["linear"], ["zoom"], 5, "dev.sargunv.orbis:equal-earth", 7, "mercator"]}"#,
            Preview::EqualEarth,
        ),
        (r#"{"type": "EPSG:3413"}"#, Preview::Crs),
    ] {
        let off = style(Previews::NONE, projection);
        assert_eq!((off.projection, off.globe), (None, None), "{projection}");
        assert_eq!(
            paths(&off),
            [
                "projection.type",
                "sources.a.dev.sargunv.orbis:tileMatrixSet"
            ],
            "{projection}"
        );
        let warning = off.warnings.iter().find(|w| w.path == "projection.type");
        assert!(warning.unwrap().message.contains(preview.name()));
        assert!(off.tile_matrix_sets().is_empty());
        let on = style(
            Previews::NONE.with(preview).with(Preview::TileMatrixSet),
            projection,
        );
        assert!(on.warnings.is_empty(), "{projection}: {:?}", on.warnings);
        assert!(
            on.projection.is_some() || on.globe.is_some(),
            "{projection}"
        );
    }
    // A spec projection keeps working; the transition and parameters are
    // ignored without their previews.
    let off = style(
        Previews::NONE.with(Preview::TileMatrixSet),
        r#"{"type": "globe", "dev.sargunv.orbis:transition": false}"#,
    );
    assert_eq!(off.globe, Some(orb_crs::Globe::V8));
    assert_eq!(paths(&off), ["projection.dev.sargunv.orbis:transition"]);
    let off = style(
        Previews::NONE.with(Preview::EqualEarth),
        r#"{"type": "dev.sargunv.orbis:equal-earth", "lon_0": 150}"#,
    );
    assert_eq!(
        paths(&off),
        [
            "projection.lon_0",
            "sources.a.dev.sargunv.orbis:tileMatrixSet"
        ]
    );
}

/// Paint Orbis evaluates per frame reads the camera, previews or not,
/// with one warning per property that the style isn't portable. Inputs no
/// phase honours stay errors, as do the spec's own limits.
#[test]
fn extended_inputs_are_drawn_and_warn_once() {
    let layer = |paint: &str, layout: &str| {
        format!(
            r#"{{"version": 8, "sources": {{"s": {{"type": "geojson",
                "data": {{"type": "FeatureCollection", "features": []}}}}}},
                "layers": [{{"id": "f", "type": "fill", "source": "s",
                "paint": {{{paint}}}, "layout": {{{layout}}}}},
                {{"id": "g", "type": "fill", "source": "s",
                "paint": {{{paint}}}, "layout": {{{layout}}}}}]}}"#
        )
    };
    let pitch = r#""fill-opacity": ["interpolate", ["linear"], ["pitch"], 0, 1, 60, 0.5]"#;
    for previews in [Previews::NONE, Previews::ALL] {
        let style = parse_with(&layer(pitch, ""), previews, &[]).unwrap();
        assert_eq!(style.warnings.len(), 1, "{:?}", style.warnings);
        assert_eq!(style.warnings[0].path, "layers[0].paint.fill-opacity");
        assert!(style.warnings[0].message.contains("reads camera"));
        let opacity = style.layers[1].get(FILL_OPACITY);
        assert_eq!(opacity.kind(), Kind::Camera);
        let folded = opacity
            .as_expression()
            .unwrap()
            .at_camera(orb_expr::CameraInputs {
                pitch: 30.0,
                bearing: 0.0,
                meters_per_pixel: 1.0,
            })
            .unwrap();
        assert_eq!(folded.kind(), Kind::Constant);
        assert_eq!(
            folded.expression().evaluate(0.0).unwrap(),
            Value::Number(0.75)
        );
    }
    // Another property warns once of its own.
    let color = r#""fill-color": ["case", [">", ["bearing"], 90], "red", "blue"]"#;
    let style = parse(&layer(&format!("{pitch}, {color}"), "")).unwrap();
    let paths: Vec<&str> = style.warnings.iter().map(|w| w.path.as_str()).collect();
    assert_eq!(
        paths,
        ["layers[0].paint.fill-opacity", "layers[0].paint.fill-color"]
    );

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

#[test]
fn layer_refs_take_their_parents_type_source_filter_and_layout() {
    let style = parse(
        r#"{"version":8,"sources":{"s":{"type":"geojson","data":{"type":"FeatureCollection","features":[]}}},
        "layers":[
            {"id":"a","type":"line","source":"s","filter":["==","k",1],"layout":{"line-cap":"round"},"paint":{"line-color":"red"}},
            {"id":"b","ref":"a","type":"fill","paint":{"line-color":"blue"}}
        ]}"#,
    )
    .expect("parses");
    assert!(style.warnings.is_empty(), "{:?}", style.warnings);
    assert_eq!(style.layers.len(), 2);
    assert_eq!(style.layers[1].id, "b");
    assert_eq!(style.layers[1].ty, style.layers[0].ty);
    assert_eq!(style.layers[1].source, style.layers[0].source);
    let doc = StyleDoc::parse(
        r#"{"version":8,"sources":{},"layers":[{"id":"a","type":"background"},{"id":"b","ref":"a"}]}"#,
    )
    .expect("parses");
    assert!(!doc.to_json().contains("\"ref\""));
}

#[test]
fn layers_that_lay_out_alike_share_a_group() {
    let style = style(
        r#"[
            {"id": "a", "type": "symbol", "source": "v", "source-layer": "poi",
             "layout": {"text-field": "{name}", "text-size": 12}, "paint": {"text-color": "red"}},
            {"id": "b", "type": "symbol", "source": "v", "source-layer": "poi",
             "layout": {"text-size": 12, "text-field": "{name}"}, "paint": {"text-color": "blue"}},
            {"id": "c", "type": "symbol", "source": "v", "source-layer": "poi",
             "layout": {"text-field": "{name}", "text-size": 13}},
            {"id": "d", "type": "symbol", "source": "v", "source-layer": "poi", "minzoom": 3,
             "layout": {"text-field": "{name}", "text-size": 12}}
        ]"#,
    )
    .unwrap_or_else(|d| panic!("{d}"));
    let group = |i: usize| {
        style.layers[i]
            .layout_group
            .expect("parsed layers have one")
    };
    assert_eq!(group(0), group(1), "paint and key order aside");
    assert_ne!(group(0), group(2), "another layout");
    assert_ne!(group(0), group(3), "another zoom range");
}

/// style-spec#1881's `["dev.sargunv.orbis:meters-per-pixel"]` is new syntax: with its preview
/// a camera value of paint, without it a warning and the default. Layout
/// can't read it.
#[test]
fn meters_per_pixel_follows_its_preview() {
    let layer = |paint: &str, layout: &str| {
        format!(
            r#"{{"version": 8, "sources": {{"s": {{"type": "geojson",
                "data": {{"type": "FeatureCollection", "features": []}}}}}},
                "layers": [{{"id": "l", "type": "line", "source": "s",
                "paint": {{{paint}}}, "layout": {{{layout}}}}}]}}"#
        )
    };
    let width = r#""line-width": ["/", 10, ["dev.sargunv.orbis:meters-per-pixel"]]"#;
    let on = parse_with(&layer(width, ""), Previews::ALL, &[]).unwrap();
    assert!(on.warnings.is_empty(), "{:?}", on.warnings);
    let w = on.layers[0].get(LINE_WIDTH);
    assert_eq!(w.kind(), Kind::Camera);
    let camera = orb_expr::CameraInputs {
        pitch: 0.0,
        bearing: 0.0,
        meters_per_pixel: 2.5,
    };
    assert_eq!(w.evaluate_camera(0.0, camera), 4.0);
    let off = parse(&layer(width, "")).unwrap();
    assert_eq!(off.warnings.len(), 1, "{:?}", off.warnings);
    assert!(
        off.warnings[0]
            .message
            .contains(Preview::MetersPerPixel.name())
    );
    // The proposal's own spelling is an unknown expression.
    let bare = r#""line-width": ["/", 10, ["meters-per-pixel"]]"#;
    assert!(parse_with(&layer(bare, ""), Previews::ALL, &[]).is_err());
    assert!(!off.layers[0].get(LINE_WIDTH).is_set());
    let layout = r#""line-miter-limit": ["dev.sargunv.orbis:meters-per-pixel"]"#;
    assert!(parse_with(&layer("", layout), Previews::ALL, &[]).is_err());
}

/// `["dev.sargunv.orbis:daylight"]` behind its preview: a label colour
/// over it with the preview, without it a warning and the default. Other
/// properties can't read it, even with the preview.
#[test]
fn daylight_follows_its_preview() {
    let layer = |ty: &str, paint: &str| {
        format!(
            r#"{{"version": 8, "sources": {{"s": {{"type": "geojson",
                "data": {{"type": "FeatureCollection", "features": []}}}}}},
                "layers": [{{"id": "l", "type": "{ty}", "source": "s",
                "paint": {{{paint}}}}}]}}"#
        )
    };
    let curve =
        r##"["interpolate", ["linear"], ["dev.sargunv.orbis:daylight"], 0, "#fff", 1, "#000"]"##;
    let halo = layer("symbol", &format!(r#""text-halo-color": {curve}"#));
    let on = parse_with(&halo, Previews::NONE.with(Preview::Daylight), &[]).unwrap();
    assert!(on.warnings.is_empty(), "{:?}", on.warnings);
    let color = on.layers[0].get(TEXT_HALO_COLOR);
    assert!(color.as_expression().is_some_and(|e| e.reads_daylight()));
    let off = parse(&halo).unwrap();
    assert_eq!(off.warnings.len(), 1, "{:?}", off.warnings);
    assert!(
        off.warnings[0]
            .message
            .contains("dev.sargunv.orbis:daylight")
    );
    assert!(!off.layers[0].get(TEXT_HALO_COLOR).is_set());
    let line = layer("line", &format!(r#""line-color": {curve}"#));
    assert!(parse_with(&line, Previews::ALL, &[]).is_err());
}

/// `metadata["dev.sargunv.orbis:color-space"]` behind its preview: off, the
/// key warns and colours mix as sRGB; an unknown value warns too. Other
/// metadata is anyone's.
#[test]
fn color_space_reads_metadata_with_its_preview() {
    let style = |metadata: &str| {
        format!(r#"{{"version": 8, "metadata": {metadata}, "sources": {{}}, "layers": []}}"#)
    };
    let on = Previews::NONE.with(Preview::ColorSpace);
    let linear = style(r#"{"dev.sargunv.orbis:color-space": "linear", "other": 1}"#);
    let parsed = parse_with(&linear, on, &[]).unwrap();
    assert_eq!(parsed.color_space, ColorSpace::Linear);
    assert!(parsed.warnings.is_empty(), "{:?}", parsed.warnings);
    let off = parse(&linear).unwrap();
    assert_eq!(off.color_space, ColorSpace::Srgb);
    assert_eq!(off.warnings.len(), 1, "{:?}", off.warnings);
    assert!(off.warnings[0].message.contains("preview"));
    let unknown = parse_with(
        &style(r#"{"dev.sargunv.orbis:color-space": "p3"}"#),
        on,
        &[],
    )
    .unwrap();
    assert_eq!(unknown.color_space, ColorSpace::Srgb);
    assert_eq!(
        unknown.warnings[0].path,
        "metadata.dev.sargunv.orbis:color-space"
    );
    for metadata in [
        r#"{"dev.sargunv.orbis:color-space": "srgb"}"#,
        r#"{"other": true}"#,
        "7",
    ] {
        let parsed = parse_with(&style(metadata), on, &[]).unwrap();
        assert_eq!(parsed.color_space, ColorSpace::Srgb, "{metadata}");
        assert!(
            parsed.warnings.is_empty(),
            "{metadata}: {:?}",
            parsed.warnings
        );
    }
}

/// The 3D proposals' syntax follows each one's preview: style-spec#1797's
/// `light["dev.sargunv.orbis:shadow-intensity"]`, #1796's ambient
/// occlusion and #1047's rounded fills. Off, each warns once (a transition
/// too) and stays at its default; the proposals' own spellings are unknown.
#[test]
fn shadows_occlusion_and_rounded_fills_follow_their_previews() {
    let doc = r#"{"version": 8,
        "light": {"dev.sargunv.orbis:shadow-intensity": 0.4, "dev.sargunv.orbis:shadow-intensity-transition": {"duration": 0}},
        "sources": {"s": {"type": "geojson", "data": {"type": "FeatureCollection", "features": []}}},
        "layers": [
            {"id": "f", "type": "fill", "source": "s",
             "layout": {"dev.sargunv.orbis:fill-rounded-corner-distance": 2}},
            {"id": "b", "type": "fill-extrusion", "source": "s",
             "paint": {"dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-intensity": 0.5,
                       "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-ground-radius":
                           ["interpolate", ["linear"], ["zoom"], 15, 2, 18, 6]}}]}"#;
    let on = parse_with(doc, Previews::ALL, &[]).unwrap();
    assert!(on.warnings.is_empty(), "{:?}", on.warnings);
    assert_eq!(on.light.shadow_intensity.evaluate(0.0), 0.4);
    assert!(
        (on.light.transitions.iter()).any(|(name, _)| name == "dev.sargunv.orbis:shadow-intensity")
    );
    assert_eq!(on.layers[0].get(FILL_ROUNDED_CORNER).evaluate(0.0), 2.0);
    let building = &on.layers[1];
    assert_eq!(building.get(EXTRUSION_AO_WALL_INTENSITY).evaluate(0.0), 0.5);
    assert_eq!(building.get(EXTRUSION_AO_GROUND_RADIUS).evaluate(16.5), 4.0);
    assert_eq!(building.get(EXTRUSION_AO_WALL_RADIUS).evaluate(0.0), 3.0);

    let off = parse(doc).unwrap();
    let mut warned: Vec<(&str, &str)> = (off.warnings.iter())
        .map(|w| (w.path.as_str(), w.message.as_str()))
        .collect();
    warned.sort_unstable();
    let expected = [
        (
            "layers[0].layout.dev.sargunv.orbis:fill-rounded-corner-distance",
            Preview::FillRoundedCornerDistance,
        ),
        (
            "layers[1].paint.dev.sargunv.orbis:fill-extrusion-ambient-occlusion-ground-radius",
            Preview::AoGroundRadius,
        ),
        (
            "layers[1].paint.dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-intensity",
            Preview::AoWallIntensity,
        ),
        (
            "light.dev.sargunv.orbis:shadow-intensity",
            Preview::ShadowIntensity,
        ),
        (
            "light.dev.sargunv.orbis:shadow-intensity-transition",
            Preview::ShadowIntensity,
        ),
    ];
    assert_eq!(warned.len(), expected.len(), "{warned:?}");
    for ((path, message), (want, preview)) in warned.into_iter().zip(expected) {
        assert_eq!(path, want);
        assert!(message.contains(preview.name()), "{path}: {message}");
    }
    assert_eq!(off.light.shadow_intensity.evaluate(0.0), 0.0);
    assert!(off.light.transitions.is_empty());
    assert_eq!(off.layers[0].get(FILL_ROUNDED_CORNER).evaluate(0.0), 0.0);
    assert!(!off.layers[1].get(EXTRUSION_AO_WALL_INTENSITY).is_set());
    assert!(!off.layers[1].get(EXTRUSION_AO_GROUND_RADIUS).is_set());
    let bare = parse_with(&doc.replace("dev.sargunv.orbis:", ""), Previews::ALL, &[]).unwrap();
    assert_eq!(bare.warnings.len(), 5, "{:?}", bare.warnings);
    assert!((bare.warnings.iter()).all(|w| w.message.starts_with("unknown property")));
    assert_eq!(bare.light.shadow_intensity.evaluate(0.0), 0.0);

    // Drawn per frame, ambient occlusion reads the camera as the rest of
    // the extrusion's paint does, warning that it isn't portable.
    let pitch = doc.replace(
        "\"dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-intensity\": 0.5",
        r#""dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-intensity": ["interpolate", ["linear"], ["pitch"], 0, 0, 60, 0.5]"#,
    );
    let camera = parse_with(&pitch, Previews::ALL, &[]).unwrap();
    let paths: Vec<&str> = camera.warnings.iter().map(|w| w.path.as_str()).collect();
    assert_eq!(
        paths,
        ["layers[1].paint.dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-intensity"]
    );
    let intensity = camera.layers[1].get(EXTRUSION_AO_WALL_INTENSITY);
    assert_eq!(intensity.kind(), Kind::Camera);
}
