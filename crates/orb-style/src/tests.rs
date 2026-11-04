use crate::*;

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

fn classes(patch: &Patch) -> Vec<Class> {
    patch.changes.iter().map(Change::class).collect()
}

#[test]
fn paint_edits_classify_by_binding() {
    let mut doc = doc();
    doc.set_cost_model(CostModel::ALL);
    let water = doc.layer_keys().nth(1).unwrap();
    let patch = commands(
        &mut doc,
        r#"[["setPaintProperty", "water", "fill-color", "green"]]"#,
    );
    assert_eq!(patch.version, 2);
    assert_eq!(
        patch.changes,
        [Change::Paint {
            layer: water,
            name: "fill-color".into(),
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
            from: Binding::Bytecode,
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
fn the_cost_model_picks_bytecode_or_rows() {
    let mut doc = doc();
    doc.set_cost_model(CostModel::ALL);
    let binding = |doc: &mut StyleDoc, value: &str| {
        let patch = commands(
            doc,
            &format!(r#"[["setPaintProperty", "water", "fill-color", {value}]]"#),
        );
        match patch.changes[0] {
            Change::Paint { to, .. } => to,
            _ => unreachable!("a paint change"),
        }
    };
    let flip = r#"["match", ["get", "c"], "a", "red", "blue"]"#;
    assert_eq!(binding(&mut doc, flip), Binding::Bytecode);
    // Strings the GPU can't compare stay rows.
    let upcase = r#"["match", ["upcase", ["get", "c"]], "A", "red", "blue"]"#;
    assert_eq!(binding(&mut doc, upcase), Binding::Source);
    doc.set_cost_model(CostModel::ROWS);
    assert_eq!(binding(&mut doc, flip), Binding::Source);
    // State and data together have no rows: interpreted whatever the cost.
    let mixed = r#"["case", ["boolean", ["feature-state", "hover"], false], "red", ["get", "c"]]"#;
    assert_eq!(binding(&mut doc, mixed), Binding::Bytecode);
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
    let before: Vec<_> = doc.layer_keys().collect();
    let patch = commands(&mut doc, layers);
    let after: Vec<_> = doc.layer_keys().collect();
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

#[test]
fn global_state_changes_reach_the_layers_that_read_it() {
    let mut doc = doc();
    let roads = doc.layer_keys().nth(2).unwrap();
    let patch = commands(
        &mut doc,
        r#"[["setGlobalStateProperty", "accent", "green"]]"#,
    );
    assert_eq!(
        patch.changes,
        [Change::Paint {
            layer: roads,
            name: "line-color".into(),
            from: Binding::Constant,
            to: Binding::Constant
        }]
    );
    let orb_v8::LayerKind::Line(line) = &doc.style().layers[2].kind else {
        unreachable!()
    };
    assert_eq!(
        line.paint.color.as_constant(),
        Some(&orb_v8::Color::parse("green").unwrap())
    );
    // Loading the written document fresh gives the same values.
    let fresh = StyleDoc::parse(&doc.to_json()).unwrap();
    let orb_v8::LayerKind::Line(fresh) = &fresh.style().layers[2].kind else {
        unreachable!()
    };
    assert_eq!(
        fresh.paint.color.as_constant(),
        line.paint.color.as_constant()
    );
    // Resetting returns to the default.
    let patch = commands(&mut doc, r#"[["setGlobalStateProperty", "accent", null]]"#);
    assert_eq!(classes(&patch), [Class::Block]);
}

#[test]
fn set_style_diffs_against_the_document() {
    let mut doc = doc();
    let keys: Vec<_> = doc.layer_keys().collect();
    let sources: Vec<_> = doc.source_keys().collect();
    let next = STYLE
        .replace(r#""fill-color": "blue""#, r#""fill-color": "teal""#)
        .replace(r#""source-layer": "roads""#, r#""source-layer": "streets""#)
        .replace(r#""features": []"#, r#""features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [0, 0]}}]"#);
    let patch = commands(&mut doc, &format!(r#"[["setStyle", {next}]]"#));
    let after: Vec<_> = doc.layer_keys().collect();
    let after_sources: Vec<_> = doc.source_keys().collect();
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
            state: vec![("hover".into(), orb_v8::Value::Bool(true))],
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
            .sky
            .as_ref()
            .unwrap()
            .color
            .evaluate(0.0)
            .straight(),
        [1.0, 0.0, 0.0, 1.0]
    );
    let patch = commands(
        &mut doc,
        r##"[["setStyle",{"version":8,"sky":{"sky-color":"blue"},"sources":{},"layers":[{"id":"bg","type":"background","paint":{"background-color":"green"}}]}]]"##,
    );
    assert!(patch.changes.contains(&Change::Restyle));
    assert_eq!(
        doc.style()
            .sky
            .as_ref()
            .unwrap()
            .color
            .evaluate(0.0)
            .straight(),
        [0.0, 0.0, 1.0, 1.0]
    );
}

#[test]
fn sky_rebinds_changed_global_state() {
    let mut doc = StyleDoc::parse(r##"{"version":8,"state":{"sky":{"default":"red"}},"sky":{"sky-color":["global-state","sky"]},"sources":{},"layers":[]}"##).unwrap();
    let patch = commands(&mut doc, r#"[["setGlobalStateProperty","sky","blue"]]"#);
    assert!(patch.changes.contains(&Change::Restyle));
    assert_eq!(
        doc.style()
            .sky
            .as_ref()
            .unwrap()
            .color
            .evaluate(0.0)
            .straight(),
        [0.0, 0.0, 1.0, 1.0]
    );
}
