use crate::v8::*;

const STYLE: &str = r#"{
    "version": 8,
    "state": {"accent": {"default": "red"}},
    "sources": {
        "v": {"type": "vector", "tiles": ["https://example.com/{z}/{x}/{y}.pbf"]},
        "g": {"type": "geojson", "data": {"type": "FeatureCollection", "features": []}}
    },
    "layers": [
        {"id": "bg", "type": "background", "paint": {"background-color": "white"}},
        {"id": "water", "type": "fill", "source": "v", "source-layer": "water",
         "paint": {"fill-color": "blue"}},
        {"id": "roads", "type": "line", "source": "v", "source-layer": "roads",
         "paint": {"line-color": ["global-state", "accent"]}},
        {"id": "points", "type": "circle", "source": "g"}
    ]
}"#;

fn doc() -> StyleDoc {
    StyleDoc::parse(STYLE).unwrap()
}

fn commands(doc: &mut StyleDoc, text: &str) -> Patch {
    doc.apply(&Transaction::from_commands(text).unwrap())
        .unwrap()
}

fn layer_keys(doc: &StyleDoc) -> Vec<LayerKey> {
    doc.style().layers.iter().map(|l| l.key).collect()
}

fn source_keys(doc: &StyleDoc) -> Vec<SourceKey> {
    doc.style().sources.iter().map(|s| s.key).collect()
}

fn classes(patch: &Patch) -> Vec<Class> {
    patch.changes.iter().map(Change::class).collect()
}

#[test]
fn paint_edits_classify_by_binding() {
    let mut doc = doc();
    let water = layer_keys(&doc)[1];
    let patch = commands(
        &mut doc,
        r#"[["setPaintProperty", "water", "fill-color", "green"]]"#,
    );
    assert_eq!(patch.version, 2);
    assert_eq!(
        patch.changes,
        [Change::Paint {
            layer: water,
            id: FILL_COLOR.id(),
            from: Binding::Constant,
            to: Binding::Constant
        }]
    );
    assert_eq!(classes(&patch), [Class::Block]);
    // A flip to data-driven re-evaluates rows; back is a uniform again.
    let flip = r#"[["setPaintProperty", "water", "fill-color", ["get", "c"]]]"#;
    assert_eq!(classes(&commands(&mut doc, flip)), [Class::Column]);
    let back = r#"[["setPaintProperty", "water", "fill-color", null]]"#;
    let patch = commands(&mut doc, back);
    assert_eq!(classes(&patch), [Class::Block]);
    assert!(matches!(
        patch.changes[0],
        Change::Paint {
            from: Binding::Source,
            to: Binding::Constant,
            ..
        }
    ));
    let state = r#"[["setPaintProperty", "water", "fill-opacity",
        ["case", ["boolean", ["feature-state", "hover"], false], 1, 0.5]]]"#;
    assert_eq!(classes(&commands(&mut doc, state)), [Class::State]);
    // Unchanged layers keep their parsed IR.
    assert_eq!(doc.style().layers.len(), 4);
    assert_eq!(doc.version(), 5);
}

#[test]
fn mixed_paint_binds_as_rows() {
    let mut doc = doc();
    let patch = commands(
        &mut doc,
        r#"[["setPaintProperty", "water", "fill-color",
            ["case", ["boolean", ["feature-state", "hover"], false], "red", ["get", "c"]]]]"#,
    );
    assert!(matches!(
        patch.changes[0],
        Change::Paint {
            to: Binding::Source,
            ..
        }
    ));
    assert_eq!(classes(&patch), [Class::Column]);
}

#[test]
fn layout_filter_and_layer_edits() {
    let mut doc = doc();
    let edits = r#"[
        ["setLayoutProperty", "roads", "line-cap", "round"],
        ["setLayoutProperty", "roads", "visibility", "none"],
        ["setFilter", "water", ["==", ["get", "kind"], "lake"]],
        ["setLayerZoomRange", "points", 2, 10]
    ]"#;
    let patch = commands(&mut doc, edits);
    // In layer order: the water filter, the road cap and visibility, the
    // point zoom range.
    assert_eq!(
        classes(&patch),
        [
            Class::Relayout,
            Class::Relayout,
            Class::Block,
            Class::Relayout
        ]
    );
    let layers = r#"[
        ["addLayer", {"id": "new", "type": "line", "source": "v", "source-layer": "rail"}, "roads"],
        ["moveLayer", "bg"],
        ["removeLayer", "points"]
    ]"#;
    let before: Vec<_> = layer_keys(&doc);
    let patch = commands(&mut doc, layers);
    let after: Vec<_> = layer_keys(&doc);
    assert_eq!(after[0], before[1]);
    assert_eq!(after[2], before[2], "roads keeps its key");
    assert_eq!(after[3], before[0], "bg moved to the top");
    assert_eq!(
        patch.changes,
        [
            Change::LayerAdded(after[1]),
            Change::LayerRemoved(before[3]),
            Change::LayersReordered
        ]
    );
    let ids: Vec<_> = doc.style().layers.iter().map(|l| l.id.as_str()).collect();
    assert_eq!(ids, ["water", "new", "roads", "bg"]);
}

#[test]
fn failed_transactions_change_nothing() {
    let mut doc = doc();
    let tx = Transaction::from_commands(
        r#"[["setPaintProperty", "water", "fill-color", "green"],
            ["setPaintProperty", "water", "fill-opacity", "nope"]]"#,
    )
    .unwrap();
    assert!(doc.apply(&tx).is_err());
    assert_eq!(doc.version(), 1);
    let fresh = StyleDoc::parse(STYLE).unwrap();
    assert_eq!(doc.to_json(), fresh.to_json());
    let missing = Transaction::from_commands(r#"[["removeSource", "v"]]"#).unwrap();
    assert!(doc.apply(&missing).is_err(), "layers use it");
    assert!(Transaction::from_commands(r#"[["flyTo", {}]]"#).is_err());
}

/// A document's invalid value draws its default with a warning, but an
/// edit that writes one fails; an edit elsewhere in the layer, or one that
/// only reads it again, doesn't.
#[test]
fn edits_may_not_write_what_a_document_may_hold() {
    let invalid = STYLE
        .replace(
            r#""paint": {"fill-color": "blue"}"#,
            r#""paint": {"fill-color": "blue", "fill-opacity": "nope"}"#,
        )
        .replace(
            r#""paint": {"line-color": ["global-state", "accent"]}"#,
            r#""paint": {"line-color": ["global-state", "accent"]}, "layout": {"line-cap": "flat"}"#,
        );
    let mut doc = StyleDoc::parse(&invalid).unwrap();
    let paths: Vec<&str> = (doc.style().warnings.iter())
        .map(|w| w.path.as_str())
        .collect();
    assert_eq!(
        paths,
        ["layers[1].paint.fill-opacity", "layers[2].layout.line-cap"]
    );
    let fails = |doc: &mut StyleDoc, text: &str| {
        let tx = Transaction::from_commands(text).unwrap();
        doc.apply(&tx).unwrap_err().errors[0].path.clone()
    };
    assert_eq!(
        fails(
            &mut doc,
            r#"[["setPaintProperty", "water", "fill-color", "nope"]]"#
        ),
        "layers[1].paint.fill-color"
    );
    assert_eq!(
        fails(
            &mut doc,
            r#"[["addLayer", {"id": "more", "type": "fill", "source": "v", "source-layer": "x",
                "paint": {"fill-opacity": 2}}, "water"]]"#
        ),
        "layers[1].paint.fill-opacity"
    );
    assert_eq!(
        fails(&mut doc, r#"[["setLight", {"intensity": 2}]]"#),
        "light.intensity"
    );
    assert_eq!(doc.version(), 1);
    commands(
        &mut doc,
        r#"[["setPaintProperty", "water", "fill-color", "green"],
            ["setGlobalStateProperty", "accent", "green"],
            ["setLight", {"intensity": 1}]]"#,
    );
    let water = &doc.style().layers[1];
    assert_eq!(
        water.get(FILL_COLOR).constant(),
        Some(Color::parse("green").unwrap())
    );
    assert!(!water.get(FILL_OPACITY).is_set());
    // A new document is a document, whoever sets it.
    let patch = commands(&mut doc, &format!(r#"[["setStyle", {invalid}]]"#));
    assert_eq!(patch.warnings.len(), 2, "{:?}", patch.warnings);
}

/// A document's broken layer, source or terrain is dropped with a warning,
/// as in MapLibre Native, and the rest draws; an edit adding one fails.
#[test]
fn documents_drop_what_is_broken_but_edits_fail() {
    let broken = STYLE
        .replace(
            r#""g": {"type": "geojson""#,
            r#""bad": {"type": "nope"}, "g": {"type": "geojson""#,
        )
        .replace(
            r#"{"id": "points", "type": "circle", "source": "g"}"#,
            r#"{"id": "points", "type": "circle", "source": "g"}, {"id": "odd", "type": "nope"}"#,
        )
        .replace(
            r#""version": 8,"#,
            r#""version": 8, "terrain": {"source": "v"},"#,
        );
    let mut doc = StyleDoc::parse(&broken).unwrap();
    let dropped = |warnings: &[Diagnostic]| -> Vec<String> {
        (warnings.iter())
            .filter(|w| w.message.ends_with(DROPPED))
            .map(|w| w.path.clone())
            .collect()
    };
    assert_eq!(
        dropped(&doc.style().warnings),
        ["sources.bad.type", "layers[4].type", "terrain.source"]
    );
    assert_eq!(
        doc.layer_ids().collect::<Vec<_>>(),
        ["bg", "water", "roads", "points"]
    );
    assert!(doc.layer("odd").is_none() && doc.source("bad").is_none());
    assert!(doc.root(RootKey::Terrain).is_none());
    for edit in [
        r#"[["addLayer", {"id": "odd", "type": "nope"}]]"#,
        r#"[["addSource", "bad", {"type": "nope"}]]"#,
        r#"[["setTerrain", {"source": "v"}]]"#,
    ] {
        assert!(
            doc.apply(&Transaction::from_commands(edit).unwrap())
                .is_err(),
            "{edit}"
        );
    }
    // A new document drops its own, and its kept layers keep their keys.
    let keys = layer_keys(&doc);
    let patch = commands(&mut doc, &format!(r#"[["setStyle", {broken}]]"#));
    assert_eq!(dropped(&patch.warnings).len(), 3, "{:?}", patch.warnings);
    assert_eq!(layer_keys(&doc), keys);
    assert_eq!(doc.layer_ids().count(), doc.style().layers.len());
}

#[test]
fn global_state_changes_reach_the_layers_that_read_it() {
    let mut doc = doc();
    let roads = layer_keys(&doc)[2];
    let patch = commands(
        &mut doc,
        r#"[["setGlobalStateProperty", "accent", "green"]]"#,
    );
    assert_eq!(
        patch.changes,
        [Change::Paint {
            layer: roads,
            id: LINE_COLOR.id(),
            from: Binding::Constant,
            to: Binding::Constant
        }]
    );
    let line = &doc.style().layers[2];
    assert_eq!(
        line.get(crate::v8::LINE_COLOR).constant(),
        Some(crate::v8::Color::parse("green").unwrap())
    );
    // Loading the written document fresh gives the same values.
    let fresh = StyleDoc::parse(&doc.to_json()).unwrap();
    let fresh = &fresh.style().layers[2];
    assert_eq!(
        fresh.get(crate::v8::LINE_COLOR).constant(),
        line.get(crate::v8::LINE_COLOR).constant()
    );
    // Resetting returns to the default.
    let patch = commands(&mut doc, r#"[["setGlobalStateProperty", "accent", null]]"#);
    assert_eq!(classes(&patch), [Class::Block]);
}

#[test]
fn set_style_diffs_against_the_document() {
    let mut doc = doc();
    let keys: Vec<_> = layer_keys(&doc);
    let sources: Vec<_> = source_keys(&doc);
    let next = STYLE
        .replace(r#""fill-color": "blue""#, r#""fill-color": "teal""#)
        .replace(r#""source-layer": "roads""#, r#""source-layer": "streets""#)
        .replace(r#""features": []"#, r#""features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [0, 0]}}]"#);
    let patch = commands(&mut doc, &format!(r#"[["setStyle", {next}]]"#));
    let after: Vec<_> = layer_keys(&doc);
    let after_sources: Vec<_> = source_keys(&doc);
    assert_eq!(after[..2], keys[..2]);
    assert_ne!(after[2], keys[2], "a new source layer is a new layer");
    assert_ne!(after[3], keys[3], "a layer of a changed source is new");
    assert_eq!(after_sources[0], sources[0]);
    assert_ne!(after_sources[1], sources[1]);
    let mut classes = classes(&patch);
    classes.sort();
    assert_eq!(
        classes,
        [
            Class::Block,
            Class::Nodes,
            Class::Nodes,
            Class::Nodes,
            Class::Nodes,
            Class::Reload,
            Class::Reload
        ]
    );
    // The same document again changes nothing.
    let patch = commands(&mut doc, &format!(r#"[["setStyle", {next}]]"#));
    assert!(patch.changes.is_empty(), "{:?}", patch.changes);
}

#[test]
fn feature_state_is_validated_and_passed_through() {
    let mut doc = doc();
    let tx = Transaction::from_commands(
        r#"[["setFeatureState", {"source": "g", "id": 1}, {"hover": true}]]"#,
    )
    .unwrap();
    assert!(tx.only_feature_state());
    let patch = doc.apply(&tx).unwrap();
    assert_eq!(patch.version, 1, "no new snapshot");
    assert_eq!(
        patch.feature_state,
        [FeatureStateOp::Set {
            source: doc.source_key("g").unwrap(),
            source_layer: None,
            id: "1".into(),
            state: vec![("hover".into(), crate::v8::Value::Bool(true))],
        }]
    );
    let vector = Transaction::from_commands(
        r#"[["setFeatureState", {"source": "v", "id": 1}, {"hover": true}]]"#,
    )
    .unwrap();
    assert!(
        doc.apply(&vector).is_err(),
        "vector sources need a source layer"
    );
}

#[test]
fn sky_survives_layer_edits_and_sky_replacement_restyles() {
    let mut doc = StyleDoc::parse(r##"{"version":8,"sky":{"sky-color":"red"},"sources":{},"layers":[{"id":"bg","type":"background"}]}"##).unwrap();
    commands(
        &mut doc,
        r#"[["setPaintProperty","bg","background-color","green"]]"#,
    );
    assert_eq!(
        doc.style()
            .atmosphere
            .as_ref()
            .unwrap()
            .sky_color
            .evaluate(0.0)
            .straight(),
        [1.0, 0.0, 0.0, 1.0]
    );
    let patch = commands(
        &mut doc,
        r##"[["setStyle",{"version":8,"sky":{"sky-color":"blue"},"sources":{},"layers":[{"id":"bg","type":"background","paint":{"background-color":"green"}}]}]]"##,
    );
    assert!(!patch.changes.contains(&Change::Restyle));
    assert!(patch.changes.contains(&Change::Root));
    assert_eq!(
        doc.style()
            .atmosphere
            .as_ref()
            .unwrap()
            .sky_color
            .evaluate(0.0)
            .straight(),
        [0.0, 0.0, 1.0, 1.0]
    );
}

#[test]
fn sky_rebinds_changed_global_state() {
    let mut doc = StyleDoc::parse(r##"{"version":8,"state":{"sky":{"default":"red"}},"sky":{"sky-color":["global-state","sky"]},"sources":{},"layers":[]}"##).unwrap();
    let patch = commands(&mut doc, r#"[["setGlobalStateProperty","sky","blue"]]"#);
    assert!(!patch.changes.contains(&Change::Restyle));
    assert!(patch.changes.contains(&Change::Root));
    assert_eq!(
        doc.style()
            .atmosphere
            .as_ref()
            .unwrap()
            .sky_color
            .evaluate(0.0)
            .straight(),
        [0.0, 0.0, 1.0, 1.0]
    );
}

/// Light, sky and projection read the global state's overrides, and keep
/// reading them after an edit to another root value parses the root again.
#[test]
fn root_values_keep_global_state_overrides_across_root_edits() {
    let mut doc = StyleDoc::parse(
        r##"{"version":8,"state":{"night":{"default":false},"surface":{"default":"mercator"}},
            "light":{"color":["case",["global-state","night"],"blue","white"]},
            "sky":{"sky-color":["case",["global-state","night"],"blue","white"]},
            "projection":{"type":["global-state","surface"]},
            "sources":{},"layers":[]}"##,
    )
    .unwrap();
    assert_eq!(doc.style().globe, None);
    let night = |doc: &StyleDoc| {
        let style = doc.style();
        let sky = style.atmosphere.as_ref().unwrap();
        (
            style.light.color.evaluate(0.0).straight(),
            sky.sky_color.evaluate(0.0).straight(),
            style.globe,
        )
    };
    let blue = [0.0, 0.0, 1.0, 1.0];
    let patch = commands(
        &mut doc,
        r#"[["setGlobalStateProperty","night",true],
            ["setGlobalStateProperty","surface","vertical-perspective"]]"#,
    );
    assert!(patch.changes.contains(&Change::Root));
    assert!(patch.changes.contains(&Change::Projection));
    assert_eq!(night(&doc), (blue, blue, Some(orb_crs::Globe::SPHERE)));
    let patch = commands(&mut doc, r#"[["setTransition",{"duration":0,"delay":0}]]"#);
    assert!(!patch.changes.contains(&Change::Projection));
    assert_eq!(night(&doc), (blue, blue, Some(orb_crs::Globe::SPHERE)));
    let patch = commands(
        &mut doc,
        r#"[["setGlobalStateProperty","night",null],
            ["setGlobalStateProperty","surface",null]]"#,
    );
    assert!(patch.changes.contains(&Change::Projection));
    assert_eq!(night(&doc), ([1.0; 4], [1.0; 4], None));
}

#[test]
fn terrain_switches_without_a_reload_and_follows_its_source() {
    let mut doc = StyleDoc::parse(r##"{"version":8,"sources":{"dem":{"type":"raster-dem","tiles":["x/{z}/{x}/{y}.png"]}},"layers":[{"id":"bg","type":"background"}]}"##).unwrap();
    assert!(doc.style().terrain.is_none());
    let patch = commands(
        &mut doc,
        r#"[["setTerrain",{"source":"dem","exaggeration":2}]]"#,
    );
    assert!(!patch.changes.contains(&Change::Restyle));
    // Frames read it from the new snapshot.
    assert!(patch.changes.contains(&Change::Root));
    assert_eq!(doc.style().terrain.as_ref().unwrap().exaggeration, 2.0);
    // Layer edits keep it.
    commands(
        &mut doc,
        r#"[["setPaintProperty","bg","background-color","green"]]"#,
    );
    assert_eq!(doc.style().terrain.as_ref().unwrap().source, "dem");
    commands(&mut doc, r#"[["setTerrain",null]]"#);
    assert!(doc.style().terrain.is_none());
    let missing = Transaction::from_commands(r#"[["setTerrain",{"source":"nope"}]]"#).unwrap();
    assert!(doc.apply(&missing).is_err());
}

/// A projection to or from a coordinate reference system's plane, or a
/// source in another tile matrix set, plans the surface again; a globe
/// over the same plane doesn't.
#[test]
fn planes_and_tile_matrix_sets_restyle() {
    let mut doc = StyleDoc::parse_with(STYLE, Previews::ALL).unwrap();
    let patch = commands(&mut doc, r#"[["setProjection", {"type": "globe"}]]"#);
    assert!(!patch.changes.contains(&Change::Restyle));
    assert!(patch.changes.contains(&Change::Projection));
    let patch = commands(&mut doc, r#"[["setProjection", {"type": "EPSG:3413"}]]"#);
    assert!(patch.changes.contains(&Change::Restyle));
    assert_eq!(
        doc.style().projection,
        Some(orb_crs::Projection::from_descriptor("EPSG:3413").unwrap())
    );
    assert_eq!(doc.style().globe, None);
    let patch = commands(
        &mut doc,
        r#"[["addSource", "polar", {"type": "raster", "tiles": ["x/{z}/{x}/{y}.png"],
            "dev.sargunv.orbis:tileMatrixSet": "UPSArcticWGS84Quad"}]]"#,
    );
    assert!(patch.changes.contains(&Change::Restyle));
    assert_eq!(doc.style().tile_matrix_sets()[0].0, "polar");
    let patch = commands(&mut doc, r#"[["setProjection", {"type": "mercator"}]]"#);
    assert!(patch.changes.contains(&Change::Restyle));
    assert_eq!(doc.style().projection, None);
}

#[test]
fn source_data_is_a_new_revision_of_the_same_source() {
    let mut doc = doc();
    let keys: Vec<_> = source_keys(&doc);
    let patch = commands(
        &mut doc,
        r#"[["setSourceData", "g", {"type": "Feature", "properties": {},
            "geometry": {"type": "Point", "coordinates": [1, 2]}}]]"#,
    );
    assert_eq!(patch.changes, [Change::SourceData(keys[1])]);
    assert_eq!(classes(&patch), [Class::Relayout]);
    assert_eq!(source_keys(&doc), keys);
    let crate::v8::SourceKind::GeoJson(g) = &doc.style().sources[1].kind else {
        panic!("a GeoJSON source");
    };
    assert!(matches!(&g.data, crate::v8::GeoJsonData::Inline(f) if f.len() == 1));
    assert!(
        doc.source_json(keys[1])
            .unwrap()
            .to_string()
            .contains("[1,2]")
    );
    for bad in [
        r#"[["setSourceData", "v", {}]]"#,
        r#"[["setSourceData", "nope", {}]]"#,
    ] {
        assert!(
            doc.apply(&Transaction::from_commands(bad).unwrap())
                .is_err(),
            "{bad}"
        );
    }
}

#[test]
fn merged_patches_keep_diffs_only_of_sources_not_given_whole_data() {
    let mut doc = doc();
    let key = doc.source_key("g").unwrap();
    let data = r#"{"type":"FeatureCollection","features":[]}"#;
    let set = commands(&mut doc, &format!(r#"[["setSourceData", "g", {data}]]"#));
    let update = r#"[["updateSourceData", "g", {"remove": [1]}]]"#;
    let (first, second) = (commands(&mut doc, update), commands(&mut doc, update));
    // The whole data holds the diffs after it already.
    let merged = Patch::merge([set, first.clone()]);
    assert_eq!(merged.changes, [Change::SourceData(key)]);
    assert!(merged.source_diffs.is_empty());
    // Diffs alone go as one list, in order.
    let merged = Patch::merge([first, second]);
    assert_eq!(merged.source_diffs.len(), 1);
    assert_eq!(merged.source_diffs[0].1.len(), 2);
}

#[test]
fn source_diffs_change_the_record_and_the_features_alike() {
    let mut doc = doc();
    let point = |id: &str, x: f64, name: &str| {
        format!(
            r#"{{"type":"Feature","id":{id},"properties":{{"name":"{name}"}},"geometry":{{"type":"Point","coordinates":[{x},0]}}}}"#
        )
    };
    commands(
        &mut doc,
        &format!(
            r#"[["setSourceData", "g", {{"type":"FeatureCollection","features":[{},{},{}]}}]]"#,
            point("1", 1.0, "a"),
            point("2", 2.0, "b"),
            point("\"3\"", 3.0, "c")
        ),
    );
    let patch = commands(
        &mut doc,
        &format!(
            r#"[["updateSourceData", "g", {{"remove": [2], "add": [{}, {}],
                "update": [{{"id": "3", "addOrUpdateProperties": [{{"key": "name", "value": "z"}}],
                  "newGeometry": {{"type": "Point", "coordinates": [30, 0]}}}},
                  {{"id": 1, "removeAllProperties": true}}]}}]]"#,
            point("1", 10.0, "a2"),
            point("4", 4.0, "d")
        ),
    );
    let key = source_keys(&doc)[1];
    assert_eq!(patch.changes, [Change::SourceData(key)]);
    assert_eq!(patch.source_diffs.len(), 1);
    // The parsed features, as job hosts get them by the diff.
    let SourceKind::GeoJson(g) = &doc.style().sources[1].kind else {
        panic!()
    };
    let GeoJsonData::Inline(features) = &g.data else {
        panic!()
    };
    let summary: Vec<String> = features
        .iter()
        .map(|f| format!("{:?} {:?} {:?}", f.id, f.properties, f.geometry))
        .collect();
    // The record, parsed again from scratch, agrees.
    let reparsed =
        crate::v8::geojson::geojson_features(doc.source_json(key).unwrap().get("data").unwrap())
            .unwrap();
    let again: Vec<String> = reparsed
        .iter()
        .map(|f| format!("{:?} {:?} {:?}", f.id, f.properties, f.geometry))
        .collect();
    assert_eq!(summary, again);
    // 1 kept its place, re-added and then stripped; 2 went; 3 moved; 4 is new.
    assert_eq!(features.len(), 3, "{summary:?}");
    assert_eq!(features[0].id, Some(orb_tile::Value::F64(1.0)));
    assert!(features[0].properties.is_empty());
    assert_eq!(
        features[1].properties,
        [("name".to_string(), orb_tile::Value::Str("z".into()))]
    );
    assert_eq!(features[2].id, Some(orb_tile::Value::F64(4.0)));
    for bad in [
        r#"[["updateSourceData", "v", {}]]"#,
        r#"[["updateSourceData", "g", {"remove": [{}]}]]"#,
    ] {
        assert!(
            doc.apply(&Transaction::from_commands(bad).unwrap())
                .is_err(),
            "{bad}"
        );
    }
}

/// GL JS's `setStyle` with `diff: true` between two basemaps on one
/// source: the source and same-id layers keep their keys, paint shows as
/// paint changes, and another sprite reloads only the sprite.
#[test]
fn set_style_between_basemaps_keeps_shared_sources() {
    let style = |sprite: &str, color: &str| {
        format!(
            r##"{{"version":8,"sprite":"https://s/{sprite}","glyphs":"https://g/{{fontstack}}/{{range}}.pbf","sources":{{"v":{{"type":"vector","tiles":["https://t/{{z}}/{{x}}/{{y}}"]}}}},"layers":[{{"id":"bg","type":"background","paint":{{"background-color":"{color}"}}}},{{"id":"water","type":"fill","source":"v","source-layer":"water","paint":{{"fill-color":"{color}"}}}}]}}"##
        )
    };
    let mut doc = StyleDoc::parse(&style("light", "white")).unwrap();
    let sources: Vec<_> = source_keys(&doc);
    let layers: Vec<_> = layer_keys(&doc);
    let patch = commands(
        &mut doc,
        &format!(r#"[["setStyle",{}]]"#, style("dark", "black")),
    );
    assert_eq!(source_keys(&doc), sources);
    assert_eq!(layer_keys(&doc), layers);
    assert!(patch.changes.contains(&Change::Sprite));
    assert!(!patch.changes.contains(&Change::Restyle));
    assert!(
        !patch
            .changes
            .iter()
            .any(|c| matches!(c, Change::SourceAdded(_) | Change::SourceRemoved(_)))
    );
    assert_eq!(
        patch
            .changes
            .iter()
            .filter(|c| matches!(c, Change::Paint { .. }))
            .count(),
        2
    );
    // Other glyphs load every label again.
    let other = style("dark", "black").replace("https://g/", "https://h/");
    let patch = commands(&mut doc, &format!(r#"[["setStyle",{other}]]"#));
    assert!(patch.changes.contains(&Change::Restyle));
}
