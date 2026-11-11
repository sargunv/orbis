//! Tests of the tile-facing expression API.
use crate::json::{self, Json};
use crate::property::{PropertyExpression, PropertySpec, PropertyType};
use crate::value::{Type, Value};
use crate::{CanonicalTile, Color, Expression, Filter, Kind};
use orb_tile::{GeometryType, LayerBuilder};
use std::borrow::Cow;
fn parse(text: &str) -> Json {
    json::parse(text).unwrap()
}

fn spec(ty: PropertyType, zoom: bool, feature: bool, interpolated: bool) -> PropertySpec {
    PropertySpec {
        ty,
        default: None,
        zoom,
        feature,
        state: feature,
        camera: false,
        camera_data: false,
        elevation: false,
        daylight: false,
        interpolated,
        tokens: false,
    }
}

fn roads() -> orb_tile::Layer {
    let mut b = LayerBuilder::new("roads", 4096);
    for (id, class, width, ty) in [
        (
            Some(1),
            Some("primary"),
            Some(4.0),
            GeometryType::LineString,
        ),
        (Some(2), Some("minor"), None, GeometryType::LineString),
        (None, None, Some(2.0), GeometryType::Polygon),
        (Some(4), Some("primary"), Some(6.0), GeometryType::Point),
    ] {
        b.begin_feature(id, ty);
        b.begin_part();
        b.begin_ring();
        b.push_vertex([0, 0]);
        b.push_vertex([1, 1]);
        if let Some(class) = class {
            b.set_property("class", orb_tile::Value::Str(class.into()));
        }
        if let Some(width) = width {
            b.set_property("width", orb_tile::Value::F64(width));
        }
        b.set_property("rank", orb_tile::Value::I64(id.unwrap_or(0) as i64));
    }
    b.finish()
}

fn select(filter: &str, zoom: f64) -> Vec<u32> {
    let mut warnings = Vec::new();
    let filter = Filter::parse(&parse(filter), &mut warnings).unwrap();
    filter.select(&roads(), zoom, None)
}

#[test]
fn within_filters_read_each_feature_where_its_tile_is() {
    // Points in the west and east halves of the zoom-1 tile 0/0, the
    // north-west quarter of the world.
    let mut b = LayerBuilder::new("points", 4096);
    for x in [1024, 3072] {
        b.begin_feature(None, GeometryType::Point);
        b.begin_part();
        b.begin_ring();
        b.push_vertex([x, 2048]);
    }
    let points = b.finish();
    let mut warnings = Vec::new();
    let west = Filter::parse(
        &parse(
            r#"["within", {"type": "Polygon", "coordinates":
                [[[-180, 0], [-90, 0], [-90, 85], [-180, 85], [-180, 0]]]}]"#,
        ),
        &mut warnings,
    )
    .unwrap();
    assert!(west.expression().reads_geometry());
    let tile = CanonicalTile { z: 1, x: 0, y: 0 };
    assert_eq!(west.select(&points, 1.0, Some(tile)), [0]);
    // Nowhere to find the geometry: nothing is within.
    assert_eq!(west.select(&points, 1.0, None), Vec::<u32>::new());
}

#[test]
fn filters_select_tile_rows() {
    assert_eq!(
        select(r#"["==", ["get", "class"], "primary"]"#, 10.0),
        [0, 3]
    );
    assert_eq!(select(r#"[">", ["get", "width"], 3]"#, 10.0), [0, 3]);
    assert_eq!(
        select(r#"["==", ["geometry-type"], "LineString"]"#, 10.0),
        [0, 1]
    );
    assert_eq!(select(r#"["==", ["id"], 2]"#, 10.0), [1]);
    assert_eq!(select(r#"[">=", ["zoom"], 12]"#, 10.0), Vec::<u32>::new());
    assert_eq!(select("true", 10.0), [0, 1, 2, 3]);
    assert_eq!(
        select(
            r#"["match", ["get", "class"], ["minor", "primary"], true, false]"#,
            1.0
        ),
        [0, 1, 3]
    );
}

#[test]
fn legacy_filters_keep_legacy_semantics() {
    assert_eq!(select(r#"["==", "class", "primary"]"#, 0.0), [0, 3]);
    assert_eq!(select(r#"["!=", "class", "primary"]"#, 0.0), [1, 2]);
    // A missing or mistyped property is false, not an error that sinks the
    // whole `any`.
    assert_eq!(
        select(
            r#"["any", [">", "width", 3], ["==", "class", "minor"]]"#,
            0.0
        ),
        [0, 1, 3]
    );
    assert_eq!(select(r#"["<", "class", 5]"#, 0.0), Vec::<u32>::new());
    assert_eq!(
        select(r#"["in", "class", "minor", "primary"]"#, 0.0),
        [0, 1, 3]
    );
    assert_eq!(select(r#"["!in", "class", "minor", "primary"]"#, 0.0), [2]);
    assert_eq!(select(r#"["has", "width"]"#, 0.0), [0, 2, 3]);
    assert_eq!(select(r#"["!has", "width"]"#, 0.0), [1]);
    assert_eq!(select(r#"["==", "$type", "Polygon"]"#, 0.0), [2]);
    assert_eq!(
        select(r#"["in", "$type", "Point", "Polygon"]"#, 0.0),
        [2, 3]
    );
    assert_eq!(select(r#"["==", "$id", 4]"#, 0.0), [3]);
    assert_eq!(select(r#"["has", "$id"]"#, 0.0), [0, 1, 3]);
    assert_eq!(
        select(
            r#"["none", ["==", "class", "minor"], ["has", "width"]]"#,
            0.0
        ),
        Vec::<u32>::new()
    );
    assert_eq!(select(r#"["all"]"#, 0.0), [0, 1, 2, 3]);
    assert_eq!(select(r#"["any"]"#, 0.0), Vec::<u32>::new());
    assert_eq!(select(r#"["in", "class"]"#, 0.0), Vec::<u32>::new());
    // Integers in tiles compare as numbers.
    assert_eq!(select(r#"["<=", "rank", 2]"#, 0.0), [0, 1, 2]);
}

#[test]
fn legacy_filter_errors_and_mixed_warnings() {
    let mut warnings = Vec::new();
    let err = Filter::parse(&parse(r#"["<", "$type", "Point"]"#), &mut warnings).unwrap_err();
    assert_eq!(
        err[0].message,
        "\"$type\" cannot be use with operator \"<\""
    );
    let err = Filter::parse(&parse(r#"["==", "$type", "Circle"]"#), &mut warnings).unwrap_err();
    assert_eq!(err[0].path, "[2]");
    let err = Filter::parse(&parse(r#"["==", "x", [1]]"#), &mut warnings);
    assert!(
        err.is_err(),
        "an array compared to a string is an expression type error"
    );

    let mixed = parse(r#"["all", ["==", ["get", "a"], 1], ["!in", "b", 1, 2]]"#);
    Filter::parse(&mixed, &mut warnings).unwrap_err();
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].path, "[2]");
    assert_eq!(
        warnings[0].message,
        r#"Mixing deprecated filter syntax with expression syntax is not supported. Replace ["!in","b",1,2] with ["!",["in",["get","b"],["literal",[1,2]]]]."#
    );
}

#[test]
fn bound_expressions_read_columns() {
    let layer = roads();
    let expr = Expression::parse(
        &parse(
            r#"["concat", ["get", "class"], "/", ["to-string", ["id"]], "/", ["geometry-type"]]"#,
        ),
        None,
    )
    .unwrap();
    let bound = expr.bind(&layer);
    let rows: Vec<String> = (0..layer.len())
        .map(|r| bound.evaluate(0.0, r).unwrap().to_string())
        .collect();
    assert_eq!(
        rows,
        [
            "primary/1/LineString",
            "minor/2/LineString",
            "//Polygon",
            "primary/4/Point"
        ]
    );

    // A string property is borrowed from the column, not copied.
    let get = Expression::parse(&parse(r#"["get", "class"]"#), None).unwrap();
    let bound = get.bind(&layer);
    assert!(matches!(
        bound.evaluate(0.0, 0),
        Ok(Value::String(Cow::Borrowed("primary")))
    ));
    assert_eq!(bound.evaluate(0.0, 2), Ok(Value::Null));

    let props = Expression::parse(&parse(r#"["properties"]"#), None).unwrap();
    let Ok(Value::Object(members)) = props.bind(&layer).evaluate(0.0, 1) else {
        panic!()
    };
    assert_eq!(
        members.len(),
        2,
        "absent columns are not properties: {members:?}"
    );
}

#[test]
fn classification_and_zoom_curves() {
    let number = spec(PropertyType::Number, true, true, true);
    let parse_kind =
        |text: &str, spec: &PropertySpec| PropertyExpression::parse(&parse(text), spec);

    assert_eq!(parse_kind("2", &number).unwrap().kind(), Kind::Constant);
    assert_eq!(
        parse_kind(r#"["+", 1, 2]"#, &number).unwrap().kind(),
        Kind::Constant
    );
    assert_eq!(
        parse_kind(r#"["get", "w"]"#, &number).unwrap().kind(),
        Kind::Source
    );
    let camera = parse_kind(
        r#"["interpolate", ["exponential", 2], ["zoom"], 10, 1, 12, 5]"#,
        &number,
    )
    .unwrap();
    assert_eq!(camera.kind(), Kind::Camera);
    assert_eq!(camera.zoom_curve().unwrap().stops, [10.0, 12.0]);
    assert!((camera.interpolation_factor(11.0, 10.0, 12.0) - 1.0 / 3.0).abs() < 1e-12);
    assert_eq!(
        camera.expression().evaluate(11.0),
        Ok(Value::Number(1.0 + 4.0 / 3.0))
    );
    let composite = parse_kind(
        r#"["step", ["zoom"], ["get", "w"], 14, ["*", 2, ["get", "w"]]]"#,
        &number,
    )
    .unwrap();
    assert_eq!(composite.kind(), Kind::Composite);
    assert_eq!(composite.interpolation_factor(14.5, 14.0, 15.0), 0.0);

    let constant_only = spec(PropertyType::Number, true, false, true);
    assert_eq!(
        parse_kind(r#"["get", "w"]"#, &constant_only).unwrap_err()[0].message,
        "data expressions not supported"
    );
    let no_zoom = spec(PropertyType::Number, false, true, true);
    assert_eq!(
        parse_kind(r#"["step", ["zoom"], 1, 5, 2]"#, &no_zoom).unwrap_err()[0].message,
        "zoom expressions not supported"
    );
    let stepped = spec(PropertyType::Number, true, true, false);
    assert_eq!(
        parse_kind(
            r#"["interpolate", ["linear"], ["zoom"], 0, 1, 5, 2]"#,
            &stepped
        )
        .unwrap_err()[0]
            .message,
        "\"interpolate\" expressions cannot be used with this property"
    );
}

/// GL JS's `findZoomCurve` sees no child in a `var`: a zoom curve reached
/// only through one, or a curve whose input is a variable bound to
/// `["zoom"]`, is no top-level curve, as in GL JS and Native.
#[test]
fn zoom_curves_behind_variables_are_not_top_level() {
    let number = spec(PropertyType::Number, true, true, true);
    for text in [
        r#"["let", "a", ["interpolate", ["linear"], ["zoom"], 0, 0, 10, 1], ["var", "a"]]"#,
        r#"["let", "z", ["zoom"], ["interpolate", ["linear"], ["var", "z"], 0, 0, 10, 1]]"#,
    ] {
        let errors = PropertyExpression::parse(&parse(text), &number).unwrap_err();
        assert_eq!(
            errors[0].message,
            "\"zoom\" expression may only be used as input to a top-level \"step\" or \"interpolate\" expression.",
            "{text}"
        );
    }
}

#[test]
fn constants_and_legacy_functions() {
    let color = spec(PropertyType::Color, true, true, true);
    let c = PropertyExpression::parse(&parse(r##""#f00""##), &color).unwrap();
    assert_eq!(c.kind(), Kind::Constant);
    assert_eq!(
        c.expression().constant(),
        Some(&Value::Color(Color::parse("red").unwrap()))
    );
    let err = PropertyExpression::parse(&parse(r#""nope""#), &color).unwrap_err();
    assert_eq!(err[0].message, "Could not parse color from value 'nope'");

    let translate = spec(
        PropertyType::Array(Type::Number, Some(2)),
        true,
        false,
        true,
    );
    let t = PropertyExpression::parse(&parse("[1, 2]"), &translate).unwrap();
    assert_eq!(t.expression().ty().to_string(), "array<number, 2>");
    assert!(PropertyExpression::parse(&parse("[1, 2, 3]"), &translate).is_err());

    let width = spec(PropertyType::Number, true, true, true);
    let f = PropertyExpression::parse(
        &parse(r#"{"base": 1.4, "stops": [[10, 1], [20, 9]]}"#),
        &width,
    )
    .unwrap();
    assert_eq!(f.kind(), Kind::Camera);
    assert_eq!(
        f.zoom_curve().unwrap().interpolation,
        Some(crate::Interpolation::Exponential(1.4))
    );
    let single =
        PropertyExpression::parse(&parse(r#"{"type": "interval", "stops": [[5, 3]]}"#), &width)
            .unwrap();
    assert_eq!(single.expression().evaluate(9.0), Ok(Value::Number(3.0)));
    let err = PropertyExpression::parse(&parse(r#"{"type": "bogus", "stops": [[5, 3]]}"#), &width)
        .unwrap_err();
    assert_eq!(err[0].path, ".type");
}

#[test]
fn all_v8_operators_are_expressions() {
    assert!(crate::parse::is_expression(&parse(
        r#"["format", "a", {}]"#
    )));
    assert!(!crate::parse::is_expression(&parse(
        r#"["Open Sans Regular"]"#
    )));
    assert!(Expression::parse(&parse(r#"["format", "a", {}]"#), None).is_ok());
}

#[test]
fn diagnostics_carry_spans() {
    let text = r#"["+", 1, "two"]"#;
    let err = Expression::parse(&parse(text), None).unwrap_err();
    assert_eq!(err[0].path, "[2]");
    assert_eq!(
        &text[err[0].span.start as usize..err[0].span.end as usize],
        "\"two\""
    );
}

#[test]
fn deepest_document_fits_a_small_stack() {
    // The wasm default stack is 1 MiB. Parse and evaluate the deepest
    // expression the JSON parser accepts on one that size, with a feature
    // read at the bottom so no level folds to a constant.
    let adds = json::MAX_DEPTH - 3;
    let text = format!(
        r#"["-", {}["get", "rank"]{}]"#,
        r#"["+", 1, "#.repeat(adds),
        "]".repeat(adds)
    );
    let result = std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || {
            let expr = Expression::parse(&parse(&text), None).unwrap();
            let layer = roads();
            expr.bind(&layer).evaluate(0.0, 0).map(Value::into_owned)
        })
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(result, Ok(Value::Number(-(adds as f64) - 1.0)));
}

#[test]
fn bound_evaluation_provides_all_state_and_global_inputs() {
    use crate::{EvaluationContext, ir::Deps};
    let expr = Expression::parse(&parse(r#"["+", ["get", "rank"], ["number", ["feature-state", "weight"]], ["number", ["global-state", "offset"]], ["line-progress"], ["heatmap-density"], ["elevation"], ["number", ["accumulated"]]]"#),None).unwrap();
    let layer = roads();
    let bound = expr.bind(&layer);
    let feature_state = vec![("weight".into(), Value::Number(2.0))];
    let global_state = vec![("offset".into(), Value::Number(3.0))];
    let context = EvaluationContext {
        line_progress: 4.0,
        heatmap_density: 5.0,
        elevation: 6.0,
        accumulated: Value::Number(7.0),
        feature_state: &feature_state,
        global_state: &global_state,
        ..EvaluationContext::default()
    };
    assert_eq!(bound.evaluate_with(&context, 0), Ok(Value::Number(28.0)));
    assert_eq!(bound.evaluate_with(&context, 1), Ok(Value::Number(29.0)));
    for dependency in [
        Deps::FEATURE_STATE,
        Deps::GLOBAL_STATE,
        Deps::LINE_PROGRESS,
        Deps::HEATMAP,
        Deps::ELEVATION,
        Deps::ACCUMULATED,
    ] {
        assert!(expr.deps().contains(dependency));
    }
    assert!(!expr.is_feature_constant());
    assert!(expr.constant().is_none());
}

#[test]
fn bound_geometry_uses_layer_extent_and_tile_location() {
    use crate::{CanonicalTile, EvaluationContext, FeatureGeometry};
    let tile = CanonicalTile { z: 0, x: 0, y: 0 };
    let mut builder = LayerBuilder::new("points", 4096);
    builder.begin_feature(Some(1), GeometryType::Point);
    builder.begin_part();
    builder.begin_ring();
    builder.push_vertex([2048, 2048]);
    let layer = builder.finish();
    let geometry = FeatureGeometry::from_layer(&layer, 0, tile);
    let expr=Expression::parse(&parse(r#"["within", {"type":"Polygon","coordinates":[[[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]]]}]"#),None).unwrap();
    let context = EvaluationContext {
        geometry: Some(&geometry),
        ..EvaluationContext::default()
    };
    assert_eq!(
        expr.bind(&layer).evaluate_with(&context, 0),
        Ok(Value::Bool(true))
    );
    assert_eq!(expr.bind(&layer).evaluate(0.0, 0), Ok(Value::Bool(false)));
    // Bound to its tile, a row finds its geometry as paint and layout
    // evaluate it.
    assert_eq!(
        expr.bind(&layer).with_tile(Some(tile)).evaluate(0.0, 0),
        Ok(Value::Bool(true))
    );
}

#[test]
fn number_format_rejects_out_of_range_digits() {
    assert!(
        Expression::parse(
            &parse(r#"["number-format",1,{"min-fraction-digits":-1}]"#),
            None
        )
        .is_err()
    );
}

/// A failing argument is parsed once however deep it is nested: when
/// `all` had a two-argument overload beside its variadic one, each level
/// parsed its failing child three times.
#[test]
fn failing_nested_expressions_parse_in_linear_time() {
    for template in [
        r#"["all", {}, true]"#,
        r#"["any", {}, false]"#,
        r#"["!", {}]"#,
        r#"["-", {}, 1]"#,
        r#"["+", {}, 1]"#,
        r#"["==", {}, true]"#,
        r#"["case", {}, true, false]"#,
        r#"["coalesce", {}, true]"#,
        r#"["to-boolean", {}]"#,
        r#"["let", "a", 1, {}]"#,
        r#"["match", {}, 1, true, false]"#,
        r#"["step", {}, true, 1, false]"#,
        r#"["at", 0, ["array", {}]]"#,
    ] {
        for leaf in [r#"["!", "x"]"#, r#"["get", 1]"#] {
            let text = (0..30).fold(leaf.to_string(), |inner, _| template.replace("{}", &inner));
            let started = std::time::Instant::now();
            assert!(
                Expression::parse(&parse(&text), None).is_err(),
                "{template}"
            );
            let elapsed = started.elapsed();
            assert!(
                elapsed.as_millis() < 50,
                "{template} over {leaf}: {elapsed:?}"
            );
        }
    }
    // A failing argument of a two-argument `all` is named, as Native names it.
    let errors = Expression::parse(&parse(r#"["all", 1, true]"#), None).unwrap_err();
    assert_eq!(
        errors[0].to_string(),
        "[1]: Expected boolean but found number instead."
    );
}
