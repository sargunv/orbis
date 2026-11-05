use orb_tile::{GeometryType, LayerBuilder};

use crate::*;

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

fn line(paint: &str, layout: &str) -> Line {
    let style = style(&format!(
        r#"[{{"id": "l", "type": "line", "source": "v", "source-layer": "roads", "paint": {paint}, "layout": {layout}}}]"#
    ))
    .unwrap_or_else(|d| panic!("{d}"));
    match style.layers.into_iter().next().unwrap().kind {
        LayerKind::Line(line) => *line,
        _ => panic!("a line layer"),
    }
}

#[test]
fn defaults_follow_the_spec() {
    let l = line("{}", "{}");
    assert_eq!(l.paint.width.as_constant(), Some(&1.0));
    assert_eq!(l.paint.opacity.as_constant(), Some(&1.0));
    assert_eq!(l.paint.color.as_constant(), Some(&Color::BLACK));
    assert_eq!(l.paint.gap_width.as_constant(), Some(&0.0));
    assert_eq!(
        l.paint.translate_anchor.as_constant(),
        Some(&TranslateAnchor::Map)
    );
    assert_eq!(l.paint.dasharray.as_constant(), Some(&None));
    assert_eq!(l.layout.round_limit.as_constant(), Some(&1.05));
    assert_eq!(l.layout.sort_key.as_constant(), Some(&None));
    assert!(l.paint.gradient.is_none());
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
    let LayerKind::Hillshade(h) = &s.layers[1].kind else {
        panic!()
    };
    assert_eq!(h.paint.highlight_color.len(), 2);
    assert_eq!(h.paint.illumination_direction.len(), 3);
    assert_eq!(h.paint.shadow_color.len(), 1);
    assert_eq!(h.paint.illumination_direction[2].as_constant(), Some(&0.0));
    let LayerKind::ColorRelief(c) = &s.layers[2].kind else {
        panic!()
    };
    let ramp = c.paint.color.as_ref().unwrap().expression();
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
    assert_eq!(l.paint.color.kind(), Kind::Source);
    let color = l.paint.color.bind(&tile);
    assert_eq!(color.evaluate(0.0, 0), Color::parse("red").unwrap());
    assert_eq!(color.evaluate(0.0, 1), Color::parse("blue").unwrap());

    assert_eq!(l.paint.width.kind(), Kind::Composite);
    let width = l.paint.width.bind(&tile);
    // The tile job evaluates composite values at its zoom and the next.
    assert_eq!(
        (width.evaluate(10.0, 0), width.evaluate(20.0, 0)),
        (4.0, 16.0)
    );
    assert_eq!(width.evaluate(15.0, 1), 2.5);
    assert_eq!(
        l.paint
            .width
            .as_expression()
            .unwrap()
            .interpolation_factor(15.0, 10.0, 20.0),
        0.5
    );

    // A missing property evaluates to the default, as GL JS falls back.
    assert_eq!(l.paint.opacity.bind(&tile).evaluate(0.0, 0), 1.0);
    // So does a value outside an enum.
    let cap = l.layout.cap.bind(&tile);
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
    assert_eq!(l.paint.width.kind(), Kind::Camera);
    assert_eq!(l.paint.width.evaluate(10.0), 2.0);
    assert_eq!(l.paint.color.kind(), Kind::Source);
    let tile = tile();
    assert_eq!(
        l.paint.color.bind(&tile).evaluate(0.0, 1),
        Color::parse("blue").unwrap()
    );

    let s = style(
        r#"[{"id": "a", "type": "line", "source": "v", "source-layer": "roads", "filter": ["all", ["==", "class", "minor"], [">=", "width", 1]]}]"#,
    )
    .unwrap();
    let LayerKind::Line(line) = &s.layers[0].kind else {
        panic!()
    };
    assert_eq!(
        line.selection.filter.as_ref().unwrap().select(&tile, 0.0),
        [1]
    );
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
            "layers[1].paint.fill-color-transition: transitions are not supported yet and are ignored",
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
    let LayerKind::FillExtrusion(e) = &s.layers[0].kind else {
        panic!()
    };
    assert_eq!(e.paint.height.kind(), Kind::Source);
    assert_eq!(e.paint.base.evaluate(0.0), 12.0);
    assert_eq!(e.paint.opacity.evaluate(0.0), 0.5);
    assert!(!e.paint.vertical_gradient.evaluate(0.0));
}

#[test]
fn zero_duration_transitions_are_honoured() {
    let text =
        |t: &str| format!(r#"{{"version": 8, "transition": {t}, "sources": {{}}, "layers": []}}"#);
    assert!(
        parse(&text(r#"{"duration": 0}"#))
            .unwrap()
            .warnings
            .is_empty()
    );
    assert_eq!(
        parse(&text(r#"{"duration": 300}"#)).unwrap().warnings.len(),
        1
    );
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
    let line = |s: &Style| match &s.layers[0].kind {
        LayerKind::Line(l) => (**l).clone(),
        _ => unreachable!(),
    };
    let s = parse(text).unwrap();
    assert!(s.warnings.is_empty(), "{:?}", s.warnings);
    let l = line(&s);
    // Only global state: folded to a constant.
    assert_eq!(
        l.paint.color.as_constant(),
        Some(&Color::parse("red").unwrap())
    );
    // State-only: a state lookup, with the global default as the fallback.
    assert!(l.paint.width.is_state_lookup());
    assert_eq!(l.paint.width.evaluate_state(0.0, &[]), 2.0);
    let state = [("w".to_string(), Value::Number(5.0))];
    assert_eq!(l.paint.width.evaluate_state(0.0, &state), 5.0);
    let s = parse_with_state(text, &[("c".into(), Value::String("blue".into()))]).unwrap();
    assert_eq!(
        line(&s).paint.color.as_constant(),
        Some(&Color::parse("blue").unwrap())
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

fn symbol(layout: &str, paint: &str) -> Symbol {
    let style = style(&format!(
        r#"[{{"id": "s", "type": "symbol", "source": "v", "source-layer": "poi", "layout": {layout}, "paint": {paint}}}]"#
    ))
    .unwrap_or_else(|d| panic!("{d}"));
    match style.layers.into_iter().next().unwrap().kind {
        LayerKind::Symbol(symbol) => *symbol,
        _ => panic!("a symbol layer"),
    }
}

#[test]
fn symbol_layers_are_typed() {
    let s = symbol("{}", "{}");
    let l = &s.layout;
    assert_eq!(l.placement.as_constant(), Some(&SymbolPlacement::Point));
    assert_eq!(l.text_size.as_constant(), Some(&16.0));
    assert_eq!(
        l.text_font.as_constant().unwrap(),
        &["Open Sans Regular", "Arial Unicode MS Regular"]
    );
    assert_eq!(l.icon_padding.as_constant(), Some(&Padding([2.0; 4])));
    assert_eq!(l.text_overlap.as_constant(), Some(&None));
    assert_eq!(l.text_variable_anchor.as_constant(), Some(&None));
    assert_eq!(l.text_field.as_constant().unwrap()[0].text, "");
    assert_eq!(
        s.paint
            .text
            .halo_color
            .as_constant()
            .unwrap()
            .premultiplied(),
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
    let l = &s.layout;
    // Plain strings expand tokens (GL JS `resolveTokens`).
    assert_eq!(l.text_field.kind(), Kind::Source);
    let mut b = LayerBuilder::new("poi", 4096);
    b.begin_feature(None, GeometryType::Point);
    b.begin_part();
    b.begin_ring();
    b.push_vertex([0, 0]);
    b.set_property("name", orb_tile::Value::Str("Café".into()));
    b.set_property("maki", orb_tile::Value::Str("bar".into()));
    let tile = b.finish();
    let text = l.text_field.bind(&tile).evaluate(0.0, 0);
    assert_eq!(
        text.iter().map(|s| s.text.as_str()).collect::<String>(),
        "Café ()"
    );
    assert_eq!(
        l.icon_image.bind(&tile).evaluate(0.0, 0).as_deref(),
        Some("bar-15")
    );
    // Numbers are f64, as GL JS lays out with doubles.
    assert_eq!(l.text_line_height.as_constant(), Some(&1.25));
    assert_eq!(
        l.icon_padding.as_constant(),
        Some(&Padding([1.0, 2.0, 1.0, 2.0]))
    );
    assert_eq!(
        l.text_writing_mode.as_constant(),
        Some(&Some(vec![WritingMode::Vertical, WritingMode::Horizontal]))
    );
    // An item outside the enum falls back to the default.
    assert_eq!(l.text_variable_anchor.as_constant(), Some(&None));
    assert_eq!(
        l.text_variable_anchor_offset.as_constant(),
        Some(&Some(vec![
            (Anchor::Top, [0.0, 1.0]),
            (Anchor::Left, [2.0, 0.0])
        ]))
    );
    assert_eq!(
        l.placement.as_constant(),
        Some(&SymbolPlacement::LineCenter)
    );
    assert_eq!(s.paint.text.color.kind(), Kind::Source);
    assert_eq!(s.paint.icon.color.kind(), Kind::Constant);
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
        (Some(orb_crs::Globe { handover: None }), 0)
    );
    assert_eq!(
        globe(
            r#"{"type": ["interpolate", ["linear"], ["zoom"], 5, "vertical-perspective", 7, "mercator"]}"#
        ),
        (
            Some(orb_crs::Globe {
                handover: Some([5.0, 7.0])
            }),
            0
        )
    );
    assert_eq!(globe(r#"{"type": "equal-earth"}"#), (None, 1));
}
