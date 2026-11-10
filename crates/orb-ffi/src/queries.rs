// Feature queries on a render session's map: rendered and source features,
// and a clustered source's clusters. Orbis answers a query as an event with
// GL JS's JSON; the map completes the query's completion when it arrives.

use glam::DVec2;
use orb_api::ClusterQuery;
use orb_expr::json::{self, Json, JsonValue};
use orb_expr::{EvaluationContext, Expression, Type, Value as ExprValue};

use crate::base::{Failure, Result, boundary, sized, text};
use crate::delivery::{Completion, Value};
use crate::handles;
use crate::runtime::State;
use crate::sys::*;

/// What a pending query reports when its answer arrives.
pub(crate) enum Pending {
    /// Rendered features, of these layers, that this filter keeps.
    Rendered(Option<Expression>),
    /// A source's features: the answers of one query per source layer,
    /// merged once the last arrives.
    Source {
        remaining: usize,
        features: Vec<Json>,
        filter: Option<Expression>,
        error: Option<String>,
    },
    /// A cluster's children or leaves, as a feature collection.
    Features,
    ExpansionZoom,
}

/// A pending query and its completion, by Orbis's query id.
pub(crate) struct Query {
    pub(crate) completion: Completion,
    pub(crate) pending: Pending,
}

/// The texts query results lend.
#[derive(Default)]
struct Lent(Vec<String>);

impl Lent {
    fn text(&mut self, text: String) -> mln_buffer_view {
        let mut text = text;
        let view = crate::base::lend(&mut text);
        self.0.push(text);
        view
    }
}

/// A feature's source layer, as the C API names it: none for GeoJSON a
/// custom geometry source gave.
fn source_layer(feature: &Json) -> String {
    match feature.get("sourceLayer").and_then(Json::as_str) {
        Some("_geojsonTileLayer") | None => String::new(),
        Some(layer) => layer.to_string(),
    }
}

/// `feature` without the members a query adds to a GeoJSON feature.
fn bare_feature(feature: &Json) -> String {
    let members: Vec<json::Member> = feature
        .as_object()
        .unwrap_or_default()
        .iter()
        .filter(|m| !matches!(m.key.as_str(), "source" | "sourceLayer" | "layer" | "state"))
        .cloned()
        .collect();
    Json::new(JsonValue::Object(members), json::Span::default()).to_string()
}

fn keeps(filter: Option<&Expression>, feature: &Json) -> bool {
    let Some(filter) = filter else {
        return true;
    };
    let context = EvaluationContext::default();
    matches!(
        filter.evaluate_geojson(&context, feature),
        Ok(ExprValue::Bool(true))
    )
}

/// Completes a query whose answer arrived. A source query waits for its
/// other source layers' answers: it returns it then.
pub(crate) fn answer(query: Query, result: std::result::Result<String, String>) -> Option<Query> {
    let Query {
        completion,
        pending,
    } = query;
    let parsed = result.and_then(|text| json::parse(&text).map_err(|e| e.to_string()));
    let outcome = match pending {
        Pending::Rendered(filter) => parsed.map(|features| {
            let mut lent = Lent::default();
            let mut out = Vec::new();
            for f in features.as_array().unwrap_or_default() {
                if !keeps(filter.as_ref(), f) {
                    continue;
                }
                let get = |k: &str| {
                    f.get(k)
                        .and_then(Json::as_str)
                        .unwrap_or_default()
                        .to_string()
                };
                let layer = f
                    .get("layer")
                    .and_then(|l| l.get("id"))
                    .and_then(Json::as_str)
                    .unwrap_or_default()
                    .to_string();
                let state = f
                    .get("state")
                    .map_or_else(|| "{}".to_string(), Json::to_string);
                out.push(mln_rendered_feature {
                    feature: lent.text(bare_feature(f)),
                    layer_id: lent.text(layer),
                    source_id: lent.text(get("source")),
                    source_layer_id: lent.text(source_layer(f)),
                    state: lent.text(state),
                });
            }
            Value::array(&out, Some(Box::new(lent)))
        }),
        Pending::Source {
            remaining,
            mut features,
            filter,
            mut error,
        } => {
            match parsed {
                Ok(found) => features.extend(found.as_array().unwrap_or_default().iter().cloned()),
                Err(e) => error = Some(e),
            }
            if remaining > 1 {
                return Some(Query {
                    completion,
                    pending: Pending::Source {
                        remaining: remaining - 1,
                        features,
                        filter,
                        error,
                    },
                });
            }
            match error {
                Some(e) => Err(e),
                None => {
                    let mut lent = Lent::default();
                    let out: Vec<mln_source_feature> = features
                        .iter()
                        .filter(|f| keeps(filter.as_ref(), f))
                        .map(|f| mln_source_feature {
                            feature: lent.text(bare_feature(f)),
                            source_layer_id: lent.text(source_layer(f)),
                        })
                        .collect();
                    Ok(Value::array(&out, Some(Box::new(lent))))
                }
            }
        }
        Pending::Features => parsed.map(|features| {
            let items: Vec<String> = features
                .as_array()
                .unwrap_or_default()
                .iter()
                .map(bare_feature)
                .collect();
            Value::Text(format!(
                "{{\"type\":\"FeatureCollection\",\"features\":[{}]}}",
                items.join(",")
            ))
        }),
        Pending::ExpansionZoom => parsed.and_then(|zoom| {
            zoom.as_f64()
                .map(|z| Value::of(z as u32))
                .ok_or_else(|| "the expansion zoom isn't a number".to_string())
        }),
    };
    completion.complete(outcome.map_err(Failure::argument), 0);
    None
}

/// Runs `start` on the session's map once its runtime gets to it; `start`
/// returns Orbis's query ids and what they report.
fn submit(
    session: mln_render_session,
    completion: *const mln_completion,
    start: impl FnOnce(&mut orb_api::Map) -> Result<(Vec<u64>, Pending)> + Send + 'static,
) -> Result {
    let handle = handles::session(session.0)?;
    // SAFETY: the caller's pointer.
    let completion = unsafe { Completion::required(completion) }?;
    let id = handle.session_id();
    handle.runtime.post(move |state: &mut State| {
        let Some(map) = crate::session::map_of(state, id) else {
            return state.complete(
                Some(completion),
                Err(Failure::state("the session is not attached to a map")),
                0,
            );
        };
        let entry = state.maps.get_mut(&map).expect("a session's map");
        match start(&mut entry.map) {
            Ok((ids, pending)) => {
                let Some((&first, rest)) = ids.split_first() else {
                    return state.complete(Some(completion), Ok(Value::array::<u8>(&[], None)), 0);
                };
                // A source query of several layers is answered under its
                // first id; the others forward to it.
                for &other in rest {
                    entry.query_aliases.insert(other, first);
                }
                entry.queries.insert(
                    first,
                    Query {
                        completion,
                        pending,
                    },
                );
            }
            Err(e) => state.complete(Some(completion), Err(e), 0),
        }
    })
}

/// A filter expression from the caller's JSON, or none when empty.
unsafe fn filter(view: mln_buffer_view) -> Result<Option<(String, Expression)>> {
    if view.data.is_null() && view.size != 0 {
        return Err(Failure::argument("filter is null"));
    }
    // SAFETY: the caller's view.
    let source = unsafe { text(view, "filter") }?;
    if source.is_empty() {
        return Ok(None);
    }
    let json = json::parse(&source)
        .map_err(|e| Failure::argument(format!("the filter is not valid JSON: {e}")))?;
    let expression = Expression::parse_value(&json, Some(&Type::Boolean)).map_err(|errors| {
        let messages: Vec<String> = errors.iter().map(ToString::to_string).collect();
        Failure::argument(format!("the filter is invalid: {}", messages.join("; ")))
    })?;
    Ok(Some((source, expression)))
}

unsafe fn ids(views: *const mln_buffer_view, count: usize, what: &str) -> Result<Vec<String>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if views.is_null() {
        return Err(Failure::argument(format!("{what} is null")));
    }
    // SAFETY: the caller's array.
    unsafe { std::slice::from_raw_parts(views, count) }
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if v.data.is_null() || v.size == 0 {
                return Err(Failure::argument(format!("{what}[{i}] must not be empty")));
            }
            // SAFETY: the caller's view.
            unsafe { text(*v, what) }
        })
        .collect()
}

fn point(p: mln_screen_point) -> Result<DVec2> {
    if !(p.x.is_finite() && p.y.is_finite()) {
        return Err(Failure::argument("the query point must be finite"));
    }
    Ok(DVec2::new(p.x, p.y))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_query_rendered_features(
    session: mln_render_session,
    geometry: *const mln_rendered_query_geometry,
    options: *const mln_rendered_feature_query_options,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            handles::session(session.0)?;
            if geometry.is_null() {
                return Err(Failure::argument("geometry is null"));
            }
            if ((*geometry).size as usize) < std::mem::size_of::<mln_rendered_query_geometry>() {
                return Err(Failure::argument("geometry.size is too small"));
            }
            let g = *geometry;
            let (min, max) = match g.type_ {
                MLN_RENDERED_QUERY_GEOMETRY_TYPE_POINT => {
                    let p = point(g.data.point)?;
                    (p, p)
                }
                MLN_RENDERED_QUERY_GEOMETRY_TYPE_BOX => {
                    (point(g.data.box_.min)?, point(g.data.box_.max)?)
                }
                MLN_RENDERED_QUERY_GEOMETRY_TYPE_LINE_STRING => {
                    let line = g.data.line_string;
                    if line.point_count == 0 {
                        return Err(Failure::argument(
                            "the query line string must contain points",
                        ));
                    }
                    if line.points.is_null() {
                        return Err(Failure::argument(
                            "the query line string's points must not be null",
                        ));
                    }
                    for p in std::slice::from_raw_parts(line.points, line.point_count) {
                        point(*p)?;
                    }
                    return Err(Failure::unsupported(
                        "Orbis queries rendered features in a point or a box, not along a line",
                    ));
                }
                _ => return Err(Failure::argument("the query geometry type is invalid")),
            };
            let mut layers = None;
            let mut filter_expression = None;
            if !options.is_null() {
                let o = sized(options, "options")?;
                if o.fields & !MLN_RENDERED_FEATURE_QUERY_OPTION_LAYER_IDS != 0 {
                    return Err(Failure::argument("options.fields has unknown bits"));
                }
                if o.fields & MLN_RENDERED_FEATURE_QUERY_OPTION_LAYER_IDS != 0 {
                    layers = Some(ids(o.layer_ids, o.layer_id_count, "options.layer_ids")?);
                }
                filter_expression = filter(o.filter)?.map(|(_, e)| e);
            }
            submit(session, completion, move |map| {
                let id = map.query_rendered_in(min, max, layers.as_deref());
                Ok((vec![id], Pending::Rendered(filter_expression)))
            })
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_query_source_features(
    session: mln_render_session,
    source_id: mln_buffer_view,
    options: *const mln_source_feature_query_options,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            handles::session(session.0)?;
            let source = source_id_of(source_id)?;
            let mut layers = Vec::new();
            let mut filter_expression = None;
            if !options.is_null() {
                let o = sized(options, "options")?;
                if o.fields & !MLN_SOURCE_FEATURE_QUERY_OPTION_SOURCE_LAYER_IDS != 0 {
                    return Err(Failure::argument("options.fields has unknown bits"));
                }
                if o.fields & MLN_SOURCE_FEATURE_QUERY_OPTION_SOURCE_LAYER_IDS != 0 {
                    layers = ids(
                        o.source_layer_ids,
                        o.source_layer_id_count,
                        "options.source_layer_ids",
                    )?;
                }
                filter_expression = filter(o.filter)?;
            }
            submit(session, completion, move |map| {
                let filter_text = filter_expression.as_ref().map(|(t, _)| t.clone());
                let ids: Vec<u64> = if layers.is_empty() {
                    vec![map.query_source(&source, None, filter_text.as_deref())]
                } else {
                    layers
                        .iter()
                        .map(|layer| map.query_source(&source, Some(layer), filter_text.as_deref()))
                        .collect()
                };
                let remaining = ids.len();
                Ok((
                    ids,
                    Pending::Source {
                        remaining,
                        features: Vec::new(),
                        // Orbis applied the filter.
                        filter: None,
                        error: None,
                    },
                ))
            })
        })
    }
}

unsafe fn source_id_of(view: mln_buffer_view) -> Result<String> {
    if view.data.is_null() && view.size != 0 {
        return Err(Failure::argument("source_id is null"));
    }
    // SAFETY: the caller's view.
    let id = unsafe { text(view, "source_id") }?;
    if id.is_empty() {
        return Err(Failure::argument("source_id must not be empty"));
    }
    Ok(id)
}

unsafe fn cluster(
    session: mln_render_session,
    source_id: mln_buffer_view,
    completion: *const mln_completion,
    query: ClusterQuery,
    pending: Pending,
) -> Result {
    handles::session(session.0)?;
    // SAFETY: the caller's view.
    let source = unsafe { source_id_of(source_id) }?;
    submit(session, completion, move |map| {
        Ok((vec![map.query_cluster(&source, query)], pending))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_get_cluster_children(
    session: mln_render_session,
    source_id: mln_buffer_view,
    cluster_id: u32,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            cluster(
                session,
                source_id,
                completion,
                ClusterQuery::Children(u64::from(cluster_id)),
                Pending::Features,
            )
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_get_cluster_leaves(
    session: mln_render_session,
    source_id: mln_buffer_view,
    cluster_id: u32,
    limit: u32,
    offset: u32,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            cluster(
                session,
                source_id,
                completion,
                ClusterQuery::Leaves {
                    id: u64::from(cluster_id),
                    limit,
                    offset,
                },
                Pending::Features,
            )
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_get_cluster_expansion_zoom(
    session: mln_render_session,
    source_id: mln_buffer_view,
    cluster_id: u32,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            cluster(
                session,
                source_id,
                completion,
                ClusterQuery::ExpansionZoom(u64::from(cluster_id)),
                Pending::ExpansionZoom,
            )
        })
    }
}
