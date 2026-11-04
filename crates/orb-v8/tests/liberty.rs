//! OpenFreeMap Liberty (`tests/data/liberty.json`, a copy of
//! https://tiles.openfreemap.org/styles/liberty).

use orb_expr::Interpolation;
use orb_tile::{GeometryType, LayerBuilder};
use orb_v8::{
    Color, Kind, LayerKind, LineCap, LineJoin, Property, PropertyValue, SourceKind, Style,
};

fn liberty() -> Style {
    orb_v8::parse(include_str!("data/liberty.json")).unwrap_or_else(|d| panic!("{d}"))
}

fn layer<'a>(style: &'a Style, id: &str) -> &'a orb_v8::Layer {
    style.layers.iter().find(|l| l.id == id).expect(id)
}

fn constant<T: PropertyValue + Clone>(p: &Property<T>) -> T {
    assert_eq!(p.kind(), Kind::Constant);
    p.as_constant().cloned().unwrap()
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

    let kinds =
        |pred: fn(&LayerKind) -> bool| style.layers.iter().filter(|l| pred(&l.kind)).count();
    assert_eq!(kinds(|k| matches!(k, LayerKind::Background(_))), 1);
    assert_eq!(kinds(|k| matches!(k, LayerKind::Fill(_))), 16);
    assert_eq!(kinds(|k| matches!(k, LayerKind::Line(_))), 67);
    assert_eq!(kinds(|k| matches!(k, LayerKind::Symbol(_))), 25);
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
    let LayerKind::Background(bg) = &layer(&style, "background").kind else {
        panic!()
    };
    assert_eq!(constant(&bg.paint.color), color("#f8f4f0"));
    assert_eq!(constant(&bg.paint.opacity), 1.0);

    let LayerKind::Fill(park) = &layer(&style, "park").kind else {
        panic!()
    };
    assert_eq!(park.selection.source, "openmaptiles");
    assert_eq!(park.selection.source_layer.as_deref(), Some("park"));
    assert_eq!(constant(&park.paint.opacity), 0.7);
    assert_eq!(
        constant(&park.paint.outline_color),
        Some(color("rgba(95, 208, 100, 1)"))
    );
    assert!(constant(&park.paint.antialias));

    let LayerKind::Fill(wood) = &layer(&style, "landcover_wood").kind else {
        panic!()
    };
    assert!(!constant(&wood.paint.antialias));
    assert_eq!(constant(&wood.paint.color), color("hsla(98,61%,72%,0.7)"));

    let LayerKind::Fill(wetland) = &layer(&style, "landcover_wetland").kind else {
        panic!()
    };
    assert_eq!(
        constant(&wetland.paint.pattern).as_deref(),
        Some("wetland_bg_11")
    );

    let LayerKind::Line(outline) = &layer(&style, "park_outline").kind else {
        panic!()
    };
    assert_eq!(constant(&outline.paint.dasharray), Some(vec![1.0, 1.5]));
    // Unset properties take the spec defaults.
    assert_eq!(constant(&outline.paint.width), 1.0);
    assert_eq!(constant(&outline.layout.cap), LineCap::Butt);
    assert_eq!(constant(&outline.layout.join), LineJoin::Miter);
    assert_eq!(constant(&outline.layout.miter_limit), 2.0);
    assert_eq!(constant(&outline.paint.translate), [0.0, 0.0]);
}

#[test]
fn camera_properties() {
    let style = liberty();
    let residential = layer(&style, "landuse_residential");
    assert_eq!(residential.maxzoom, 12.0);
    assert!(residential.is_visible_at(11.9) && !residential.is_visible_at(12.0));
    let LayerKind::Fill(fill) = &residential.kind else {
        panic!()
    };
    assert_eq!(fill.paint.color.kind(), Kind::Camera);
    let curve = fill
        .paint
        .color
        .as_expression()
        .unwrap()
        .zoom_curve()
        .unwrap();
    assert_eq!(curve.stops, [9.0, 12.0]);
    assert_eq!(curve.interpolation, Some(Interpolation::Linear));
    assert_eq!(fill.paint.color.evaluate(9.0), color("hsla(0,3%,85%,0.84)"));
    assert_eq!(
        fill.paint.color.evaluate(13.0),
        color("hsla(35,57%,88%,0.49)")
    );

    let LayerKind::Line(river) = &layer(&style, "waterway_river").kind else {
        panic!()
    };
    assert_eq!(constant(&river.layout.cap), LineCap::Round);
    let width = &river.paint.width;
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
    let LayerKind::Line(line) = &motorway.kind else {
        panic!()
    };
    assert_eq!(line.paint.color.kind(), Kind::Camera);
    assert_eq!(line.paint.color.evaluate(6.0), color("#fc8"));

    let LayerKind::Fill(building) = &layer(&style, "building").kind else {
        panic!()
    };
    assert_eq!(building.paint.outline_color.kind(), Kind::Camera);
    assert_eq!(
        building.paint.outline_color.evaluate(14.0),
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
        let selection = match &layer(&style, id).kind {
            LayerKind::Line(line) => &line.selection,
            LayerKind::Fill(fill) => &fill.selection,
            _ => panic!("{id} draws features"),
        };
        selection.filter.as_ref().unwrap().select(&tile, 14.0)
    };
    assert_eq!(select("road_minor"), [0]);
    assert_eq!(select("road_trunk_primary"), [3]);
    assert_eq!(select("bridge_street"), [1]);
}
