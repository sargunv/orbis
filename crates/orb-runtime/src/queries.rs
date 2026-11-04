//! Asynchronous feature queries (plan §13): `queryRenderedFeatures` over
//! the last frame's picking index (labels included) and
//! `querySourceFeatures` over a source's drawn tiles. Properties and
//! geometry come from each tile's owner worker; no tile is decoded again
//! here.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use glam::DVec2;
use orb_crs::TileIndex;
use orb_v8::LayerKind;

use crate::features::{FeatureRequest, FeatureSelect, TileFeature, write_string};
use crate::jobs::TileRequest;
use crate::map::SourceState;
use crate::query::FeatureKey;

/// Called once with a query's features: a JSON array of GeoJSON features,
/// topmost first for rendered queries (GL JS's `MapGeoJSONFeature`s), or
/// an error.
pub type QueryDone = Box<dyn FnOnce(Result<String, String>) + Send + 'static>;

/// A feature a query reports, once its worker has read it.
struct Wanted {
    key: FeatureKey,
    /// The source layer, as the tile names it.
    tile_layer: String,
    /// The `"source":…` and following members, and the `"layer"` object for
    /// rendered features.
    members: String,
    source_key: u32,
    source_layer: Option<String>,
}

type Found = HashMap<(usize, TileIndex, String), Vec<TileFeature>>;

struct Pending {
    remaining: usize,
    found: Found,
    error: Option<String>,
}

fn layer_type(kind: &LayerKind) -> &str {
    match kind {
        LayerKind::Background(_) => "background",
        LayerKind::Fill(_) => "fill",
        LayerKind::FillExtrusion(_) => "fill-extrusion",
        LayerKind::Line(_) => "line",
        LayerKind::Circle(_) => "circle",
        LayerKind::Heatmap(_) => "heatmap",
        LayerKind::Raster(_) => "raster",
        LayerKind::Hillshade(_) => "hillshade",
        LayerKind::ColorRelief(_) => "color-relief",
        LayerKind::Symbol(_) => "symbol",
        LayerKind::Unsupported { ty } => ty,
    }
}

impl crate::Map {
    /// GL JS `queryRenderedFeatures`: the features drawn in the box
    /// `min..max` (logical pixels; a point when they are equal) in the last
    /// frame, topmost first, of `layers` or of every layer.
    pub fn query_rendered(
        &self,
        min: DVec2,
        max: DVec2,
        layers: Option<&[String]>,
        done: QueryDone,
    ) {
        let Some(style) = &self.style else {
            return done(Ok("[]".into()));
        };
        let plan = style.worker.plan();
        let (min, max) = (min.min(max), min.max(max));
        let mut seen = HashSet::new();
        let mut wanted = Vec::new();
        for hit in self.hits.iter().rev() {
            if layers.is_some_and(|l| !l.contains(&hit.hit.layer))
                || !hit.intersects(min, max)
                || !seen.insert((hit.order, hit.key))
            {
                continue;
            }
            let layer = &plan.style.layers[hit.order];
            let mut members = String::new();
            write_member(&mut members, "source", &hit.hit.source);
            if let Some(source_layer) = &hit.hit.source_layer {
                members.push(',');
                write_member(&mut members, "sourceLayer", source_layer);
            }
            members.push_str(",\"layer\":{");
            write_member(&mut members, "id", &layer.id);
            members.push(',');
            write_member(&mut members, "type", layer_type(&layer.kind));
            members.push(',');
            write_member(&mut members, "source", &hit.hit.source);
            if let Some(source_layer) = &hit.hit.source_layer {
                members.push(',');
                write_member(&mut members, "source-layer", source_layer);
            }
            members.push('}');
            wanted.push(Wanted {
                key: hit.key,
                tile_layer: hit
                    .hit
                    .source_layer
                    .clone()
                    .unwrap_or_else(|| orb_tile::geojson::LAYER_NAME.into()),
                members,
                source_key: plan.sources[hit.key.source].key,
                source_layer: hit.hit.source_layer.clone(),
            });
        }
        let mut groups: HashMap<(usize, TileIndex, String), Vec<u32>> = HashMap::new();
        for w in &wanted {
            groups
                .entry((w.key.source, w.key.tile, w.tile_layer.clone()))
                .or_default()
                .push(w.key.feature);
        }
        let requests = groups
            .into_iter()
            .filter_map(|((source, tile, layer), features)| {
                let request = self.tile_request(source, tile)?;
                Some((
                    (source, tile, layer.clone()),
                    FeatureRequest {
                        tile: request,
                        select: FeatureSelect::Indices { layer, features },
                    },
                ))
            })
            .collect();
        self.read_features(requests, Some(wanted), done);
    }

    /// GL JS `querySourceFeatures`: every feature in the tiles of `source`
    /// drawn last frame, of `source_layer` or of every layer, that a v8
    /// `filter` (JSON) selects. A feature crossing tiles comes once per tile.
    pub fn query_source(
        &self,
        source: &str,
        source_layer: Option<&str>,
        filter: Option<&str>,
        done: QueryDone,
    ) {
        let Some(style) = &self.style else {
            return done(Ok("[]".into()));
        };
        let plan = style.worker.plan();
        let Some(index) = plan.sources.iter().position(|s| s.id == source) else {
            return done(Err(format!("no source {source}")));
        };
        let requests = style.placed[index]
            .iter()
            .filter_map(|&tile| {
                let request = self.tile_request(index, tile)?;
                Some((
                    (index, tile, String::new()),
                    FeatureRequest {
                        tile: request,
                        select: FeatureSelect::Filter {
                            layer: source_layer.map(str::to_string),
                            filter: filter.map(str::to_string),
                        },
                    },
                ))
            })
            .collect();
        self.read_features(requests, None, done);
    }

    /// How tile `tile` of plan source `index` was loaded.
    fn tile_request(&self, index: usize, tile: TileIndex) -> Option<TileRequest> {
        let style = self.style.as_ref()?;
        let SourceState::Ready { endpoint, .. } = &style.sources[index] else {
            return None;
        };
        Some(TileRequest {
            style: style.id,
            source: style.worker.plan().sources[index].key,
            tile,
            zoom: endpoint.tms.style_zoom(tile),
            data_zoom: tile.z.min(endpoint.maxzoom),
            location: endpoint.location(endpoint.data_tile(tile)),
            encoding: endpoint.encoding,
            nodes: None,
            rows_only: false,
        })
    }

    /// Asks each tile's worker for its features, then answers `done` with
    /// `wanted`'s features in order, or with every feature found when
    /// `wanted` is `None`.
    fn read_features(
        &self,
        requests: Vec<((usize, TileIndex, String), FeatureRequest)>,
        wanted: Option<Vec<Wanted>>,
        done: QueryDone,
    ) {
        let style = self.style.as_ref().expect("queries need a style");
        let plan = style.worker.plan();
        let sources: Vec<u32> = requests.iter().map(|(_, r)| r.tile.source).collect();
        let states = style.states.snapshot(&sources);
        // Source features: their source's members, by plan index.
        let source_members: HashMap<usize, (String, u32, bool)> = requests
            .iter()
            .map(|((index, ..), _)| {
                let def = plan.source_def(*index);
                let mut members = String::new();
                write_member(&mut members, "source", &def.id);
                let geojson = matches!(def.kind, orb_v8::SourceKind::GeoJson(_));
                (*index, (members, plan.sources[*index].key, geojson))
            })
            .collect();
        if requests.is_empty() {
            return done(Ok("[]".into()));
        }
        let pending = Arc::new(Mutex::new(Pending {
            remaining: requests.len(),
            found: HashMap::new(),
            error: None,
        }));
        let finish = Arc::new(Mutex::new(Some((done, wanted, states, source_members))));
        for (group, request) in requests {
            let (pending, finish) = (Arc::clone(&pending), Arc::clone(&finish));
            self.services.jobs.features(
                request,
                Box::new(move |result| {
                    let mut p = pending.lock().expect("query poisoned");
                    match result {
                        Ok(features) => {
                            p.found.insert(group, features);
                        }
                        Err(e) => {
                            p.error.get_or_insert(e);
                        }
                    }
                    p.remaining -= 1;
                    if p.remaining > 0 {
                        return;
                    }
                    let Some((done, wanted, states, sources)) =
                        finish.lock().expect("query poisoned").take()
                    else {
                        return;
                    };
                    if let Some(e) = p.error.take() {
                        return done(Err(e));
                    }
                    let found = std::mem::take(&mut p.found);
                    drop(p);
                    done(Ok(assemble(found, wanted, &states, &sources)))
                }),
            );
        }
    }
}

fn write_member(out: &mut String, key: &str, value: &str) {
    write_string(out, key);
    out.push(':');
    write_string(out, value);
}

/// One GeoJSON feature: its id, geometry and properties, then the query's
/// members, and its state.
fn feature_json(out: &mut String, feature: &TileFeature, members: &str, state: Option<&String>) {
    out.push_str("{\"type\":\"Feature\",");
    if let Some(id) = &feature.id {
        out.push_str("\"id\":");
        out.push_str(id);
        out.push(',');
    }
    out.push_str(&feature.body);
    out.push(',');
    out.push_str(members);
    out.push_str(",\"state\":");
    out.push_str(state.map_or("{}", String::as_str));
    out.push('}');
}

type States = HashMap<(u32, Option<String>, String), String>;

fn assemble(
    found: Found,
    wanted: Option<Vec<Wanted>>,
    states: &States,
    sources: &HashMap<usize, (String, u32, bool)>,
) -> String {
    let state = |source: u32, layer: Option<String>, feature: &TileFeature| {
        feature
            .state_id
            .clone()
            .and_then(|id| states.get(&(source, layer, id)))
    };
    let mut out = String::from("[");
    let push = |out: &mut String, feature: &TileFeature, members: &str, state: Option<&String>| {
        if out.len() > 1 {
            out.push(',');
        }
        feature_json(out, feature, members, state);
    };
    match wanted {
        Some(wanted) => {
            let by_index: HashMap<(usize, TileIndex, &str, u32), &TileFeature> = found
                .iter()
                .flat_map(|((source, tile, _), features)| {
                    features
                        .iter()
                        .map(move |f| ((*source, *tile, f.layer.as_str(), f.index), f))
                })
                .collect();
            for w in &wanted {
                let key = (
                    w.key.source,
                    w.key.tile,
                    w.tile_layer.as_str(),
                    w.key.feature,
                );
                if let Some(feature) = by_index.get(&key) {
                    let state = state(w.source_key, w.source_layer.clone(), feature);
                    push(&mut out, feature, &w.members, state);
                }
            }
        }
        None => {
            let mut groups: Vec<_> = found.into_iter().collect();
            groups.sort_by_key(|((source, tile, _), _)| (*source, *tile));
            for ((source, ..), features) in groups {
                let (members, key, geojson) = &sources[&source];
                for feature in &features {
                    let layer = (!geojson).then(|| feature.layer.clone());
                    let mut members = members.clone();
                    if let Some(layer) = &layer {
                        members.push(',');
                        write_member(&mut members, "sourceLayer", layer);
                    }
                    let state = state(*key, layer, feature);
                    push(&mut out, feature, &members, state);
                }
            }
        }
    }
    out.push(']');
    out
}
