//! OpenFreeMap Liberty (`tests/data/liberty.json`, a copy of
//! https://tiles.openfreemap.org/styles/liberty).

use orb_expr::Interpolation;
use orb_tile::{GeometryType, LayerBuilder};
use orb_v8::*;

fn liberty() -> Style {
    orb_v8::parse(include_str!("data/liberty.json")).unwrap_or_else(|d| panic!("{d}"))
}

fn layer<'a>(style: &'a Style, id: &str) -> &'a orb_v8::Layer {
    style.layers.iter().find(|l| l.id == id).expect(id)
}

fn constant<T: PropertyValue + Clone>(p: PropertyRef<'_, T>) -> T {
    assert_eq!(p.kind(), Kind::Constant);
    p.constant().unwrap()
}

fn color(css: &str) -> Color {
    Color::parse(css).unwrap()
}

#[test]
fn parses_with_the_expected_warnings() {
    let style = liberty();
    assert_eq!(style.layers.len(), 111);

    // Liberty parses cleanly: every layer type and root key is modelled.
    let warnings: Vec<String> = style.warnings.iter().map(ToString::to_string).collect();
    assert_eq!(warnings, Vec::<String>::new());

    let kinds = |ty: LayerType| style.layers.iter().filter(|l| l.ty == ty).count();
    assert_eq!(kinds(LayerType::Background), 1);
    assert_eq!(kinds(LayerType::Fill), 16);
    assert_eq!(kinds(LayerType::Line), 67);
    assert_eq!(kinds(LayerType::Symbol), 25);
}

#[test]
fn sources() {
    let style = liberty();
    let SourceKind::Vector(omt) = &style.source("openmaptiles").unwrap().kind else {
        panic!("openmaptiles is a vector source");
    };
    assert_eq!(
        omt.tileset.url.as_deref(),
        Some("https://tiles.openfreemap.org/planet")
    );
    assert_eq!(omt.tileset.tiles, None);
    assert!(matches!(
        &style.source("ne2_shaded").unwrap().kind,
        SourceKind::Raster(r) if r.tile_size == Some(256.0)
    ));
}

#[test]
fn constant_properties() {
    let style = liberty();
    let bg = &layer(&style, "background");
    assert_eq!(constant(bg.get(BACKGROUND_COLOR)), color("#f8f4f0"));
    assert_eq!(constant(bg.get(BACKGROUND_OPACITY)), 1.0);

    let park = &layer(&style, "park");
    assert_eq!(park.source.as_deref().unwrap(), "openmaptiles");
    assert_eq!(park.source_layer.as_deref(), Some("park"));
    assert_eq!(constant(park.get(FILL_OPACITY)), 0.7);
    assert_eq!(
        constant(park.get(FILL_OUTLINE_COLOR)),
        Some(color("rgba(95, 208, 100, 1)"))
    );
    assert!(constant(park.get(FILL_ANTIALIAS)));

    let wood = &layer(&style, "landcover_wood");
    assert!(!constant(wood.get(FILL_ANTIALIAS)));
    assert_eq!(
        constant(wood.get(FILL_COLOR)),
        color("hsla(98,61%,72%,0.7)")
    );

    let wetland = &layer(&style, "landcover_wetland");
    assert_eq!(
        constant(wetland.get(FILL_PATTERN)).as_deref(),
        Some("wetland_bg_11")
    );

    let outline = &layer(&style, "park_outline");
    assert_eq!(constant(outline.get(LINE_DASHARRAY)), Some(vec![1.0, 1.5]));
    // Unset properties take the spec defaults.
    assert_eq!(constant(outline.get(LINE_WIDTH)), 1.0);
    assert_eq!(constant(outline.get(LINE_CAP)), LineCap::Butt);
    assert_eq!(constant(outline.get(LINE_JOIN)), LineJoin::Miter);
    assert_eq!(constant(outline.get(LINE_MITER_LIMIT)), 2.0);
    assert_eq!(constant(outline.get(LINE_TRANSLATE)), [0.0, 0.0]);
}

#[test]
fn camera_properties() {
    let style = liberty();
    let residential = layer(&style, "landuse_residential");
    assert_eq!(residential.maxzoom, 12.0);
    assert!(residential.is_visible_at(11.9) && !residential.is_visible_at(12.0));
    let fill = &residential;
    assert_eq!(fill.get(FILL_COLOR).kind(), Kind::Camera);
    let curve = fill
        .get(FILL_COLOR)
        .as_expression()
        .unwrap()
        .zoom_curve()
        .unwrap();
    assert_eq!(curve.stops, [9.0, 12.0]);
    assert_eq!(curve.interpolation, Some(Interpolation::Linear));
    assert_eq!(
        fill.get(FILL_COLOR).evaluate(9.0),
        color("hsla(0,3%,85%,0.84)")
    );
    assert_eq!(
        fill.get(FILL_COLOR).evaluate(13.0),
        color("hsla(35,57%,88%,0.49)")
    );

    let river = &layer(&style, "waterway_river");
    assert_eq!(constant(river.get(LINE_CAP)), LineCap::Round);
    let width = &river.get(LINE_WIDTH);
    assert_eq!(width.kind(), Kind::Camera);
    let expression = width.as_expression().unwrap();
    assert_eq!(
        expression.zoom_curve().unwrap().interpolation,
        Some(Interpolation::Exponential(1.2))
    );
    assert_eq!(width.evaluate(11.0), 0.5);
    assert_eq!(width.evaluate(20.0), 6.0);
    let t = expression.interpolation_factor(15.0, 11.0, 20.0);
    let expected = 0.5 + (6.0 - 0.5) * t;
    assert!((f64::from(width.evaluate(15.0)) - expected).abs() < 1e-5);

    let motorway = layer(&style, "road_motorway");
    assert_eq!(motorway.minzoom, 5.0);
    let line = &motorway;
    assert_eq!(line.get(LINE_COLOR).kind(), Kind::Camera);
    assert_eq!(line.get(LINE_COLOR).evaluate(6.0), color("#fc8"));

    let building = &layer(&style, "building");
    assert_eq!(building.get(FILL_OUTLINE_COLOR).kind(), Kind::Camera);
    assert_eq!(
        building.get(FILL_OUTLINE_COLOR).evaluate(14.0),
        Some(color("hsl(35,6%,79%)"))
    );
}

#[test]
fn filters_select_features() {
    let style = liberty();
    let mut b = LayerBuilder::new("transportation", 4096);
    for (ty, class, brunnel) in [
        (GeometryType::LineString, "minor", None),
        (GeometryType::LineString, "minor", Some("bridge")),
        (GeometryType::Polygon, "minor", None),
        (GeometryType::LineString, "primary", None),
    ] {
        b.begin_feature(None, ty);
        b.begin_part();
        b.begin_ring();
        b.push_vertex([0, 0]);
        b.set_property("class", orb_tile::Value::Str(class.into()));
        if let Some(brunnel) = brunnel {
            b.set_property("brunnel", orb_tile::Value::Str(brunnel.into()));
        }
    }
    let tile = b.finish();
    let select = |id: &str| {
        let layer = layer(&style, id);
        layer.filter.as_ref().unwrap().select(&tile, 14.0)
    };
    assert_eq!(select("road_minor"), [0]);
    assert_eq!(select("road_trunk_primary"), [3]);
    assert_eq!(select("bridge_street"), [1]);
}
