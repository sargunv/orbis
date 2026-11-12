//! Offline regions over orb-api's: created, listed, read, deleted, and
//! merged from another cache file, Orbis's or MapLibre Native's.

use crate::base::{Failure, Result, boundary, bytes, sized, text};
use crate::delivery::{Completion, Value};
use crate::handles;
use crate::sys::*;

/// `regions` as C region infos, kept alive by what [`Value`] holds.
fn lend(regions: Vec<orb_api::RegionInfo>) -> Value {
    let mut texts: Vec<Box<[u8]>> = Vec::new();
    let mut keep = |bytes: &[u8]| -> mln_buffer_view {
        // Terminated, as the library lends text.
        let mut owned = bytes.to_vec();
        owned.push(0);
        let owned = owned.into_boxed_slice();
        let view = mln_buffer_view {
            data: owned.as_ptr().cast(),
            size: owned.len() - 1,
        };
        texts.push(owned);
        view
    };
    let infos: Vec<mln_offline_region_info> = regions
        .iter()
        .map(|region| {
            let d = &region.definition;
            let style_url = keep(d.style_url.as_bytes());
            let [west, south, east, north] = d.bounds;
            let [min_zoom, max_zoom] = d.zooms;
            let (type_, data) = match &d.geometry {
                Some(geometry) => (
                    MLN_OFFLINE_REGION_DEFINITION_GEOMETRY,
                    mln_offline_region_definition_data {
                        geometry: mln_offline_geometry_region_definition {
                            style_url,
                            geometry: keep(geometry.as_bytes()),
                            min_zoom,
                            max_zoom,
                            pixel_ratio: d.pixel_ratio,
                            include_ideographs: d.include_ideographs,
                        },
                    },
                ),
                None => (
                    MLN_OFFLINE_REGION_DEFINITION_TILE_PYRAMID,
                    mln_offline_region_definition_data {
                        tile_pyramid: mln_offline_tile_pyramid_region_definition {
                            style_url,
                            bounds: mln_lat_lng_bounds {
                                southwest: mln_lat_lng {
                                    latitude: south,
                                    longitude: west,
                                },
                                northeast: mln_lat_lng {
                                    latitude: north,
                                    longitude: east,
                                },
                            },
                            min_zoom,
                            max_zoom,
                            pixel_ratio: d.pixel_ratio,
                            include_ideographs: d.include_ideographs,
                        },
                    },
                ),
            };
            let metadata = keep(&d.metadata);
            mln_offline_region_info {
                id: region.id.into(),
                definition: mln_offline_region_definition {
                    size: std::mem::size_of::<mln_offline_region_definition>() as u32,
                    type_,
                    data,
                },
                metadata: if d.metadata.is_empty() {
                    std::ptr::null()
                } else {
                    metadata.data.cast()
                },
                metadata_size: d.metadata.len(),
            }
        })
        .collect();
    // The infos point into the texts' bytes, which stay where they are.
    Value::array(&infos, Some(Box::new(texts)))
}

/// Reads a style URL that must be there: a view of no bytes with a size is
/// none.
///
/// # Safety
/// As [`text`].
unsafe fn style(view: mln_buffer_view) -> Result<String> {
    if view.data.is_null() && view.size != 0 {
        return Err(Failure::argument("the definition's style_url is null"));
    }
    // SAFETY: the caller's contract.
    let url = unsafe { text(view, "style_url") }?;
    if url.is_empty() {
        return Err(Failure::argument("the definition has no style_url"));
    }
    Ok(url)
}

fn zooms(min: f64, max: f64) -> Result<[f64; 2]> {
    if !(min.is_finite() && min >= 0.0 && !max.is_nan() && min <= max) {
        return Err(Failure::argument("the definition's zoom range is invalid"));
    }
    Ok([min, max])
}

/// The region `definition` describes, holding `metadata`.
///
/// # Safety
/// `definition` is null or a readable definition whose views are readable.
unsafe fn read_definition(
    definition: *const mln_offline_region_definition,
    metadata: &[u8],
) -> Result<orb_api::RegionDefinition> {
    // SAFETY: the caller's contract.
    let d = unsafe { sized(definition, "definition") }?;
    let mut out = orb_api::RegionDefinition {
        metadata: metadata.to_vec(),
        ..Default::default()
    };
    match d.type_ {
        MLN_OFFLINE_REGION_DEFINITION_TILE_PYRAMID => {
            // SAFETY: the type names the union's member.
            let p = unsafe { d.data.tile_pyramid };
            let (sw, ne) = (p.bounds.southwest, p.bounds.northeast);
            let finite = [sw.latitude, sw.longitude, ne.latitude, ne.longitude]
                .iter()
                .all(|v| v.is_finite());
            if !finite
                || sw.latitude > ne.latitude
                || sw.longitude > ne.longitude
                || sw.latitude < -90.0
                || ne.latitude > 90.0
            {
                return Err(Failure::argument("the definition's bounds are invalid"));
            }
            // SAFETY: the caller's contract.
            out.style_url = unsafe { style(p.style_url) }?;
            out.bounds = [sw.longitude, sw.latitude, ne.longitude, ne.latitude];
            out.zooms = zooms(p.min_zoom, p.max_zoom)?;
            out.pixel_ratio = p.pixel_ratio;
            out.include_ideographs = p.include_ideographs;
        }
        MLN_OFFLINE_REGION_DEFINITION_GEOMETRY => {
            // SAFETY: the type names the union's member.
            let g = unsafe { d.data.geometry };
            // SAFETY: the caller's contract.
            out.style_url = unsafe { style(g.style_url) }?;
            // SAFETY: the caller's contract.
            let geometry = unsafe { text(g.geometry, "geometry") }?;
            let doc = orb_expr::json::parse(&geometry)
                .map_err(|e| Failure::argument(format!("the definition's geometry: {e}")))?;
            out.bounds = bounds(&doc)
                .ok_or_else(|| Failure::argument("the definition's geometry has no positions"))?;
            out.geometry = Some(geometry);
            out.zooms = zooms(g.min_zoom, g.max_zoom)?;
            out.pixel_ratio = g.pixel_ratio;
            out.include_ideographs = g.include_ideographs;
        }
        other => {
            return Err(Failure::argument(format!(
                "{other} names no offline region definition type"
            )));
        }
    }
    if !(out.pixel_ratio.is_finite() && out.pixel_ratio > 0.0) {
        return Err(Failure::argument("the definition's pixel_ratio is invalid"));
    }
    Ok(out)
}

/// The west, south, east and north of a GeoJSON geometry's positions.
fn bounds(geometry: &orb_expr::json::Json) -> Option<[f64; 4]> {
    use orb_expr::json::Json;
    fn visit(value: &Json, out: &mut [f64; 4]) {
        let Some(items) = value.as_array() else {
            if let Some(coordinates) = value.get("coordinates") {
                visit(coordinates, out);
            }
            if let Some(geometries) = value.get("geometries").and_then(Json::as_array) {
                geometries.iter().for_each(|g| visit(g, out));
            }
            return;
        };
        match (
            items.first().and_then(Json::as_f64),
            items.get(1).and_then(Json::as_f64),
        ) {
            (Some(lon), Some(lat)) => {
                *out = [
                    out[0].min(lon),
                    out[1].min(lat),
                    out[2].max(lon),
                    out[3].max(lat),
                ];
            }
            _ => items.iter().for_each(|i| visit(i, out)),
        }
    }
    let mut out = [f64::INFINITY, f64::INFINITY, -f64::INFINITY, -f64::INFINITY];
    visit(geometry, &mut out);
    out[0].is_finite().then_some(out)
}

/// Runs `job` with the runtime's Orbis runtime on its thread, and completes
/// with the regions it gives.
fn regions(
    runtime: mln_runtime,
    completion: *const mln_completion,
    job: impl FnOnce(&orb_api::Runtime) -> Result<Vec<orb_api::RegionInfo>> + Send + 'static,
) -> Result {
    let handle = handles::runtime(runtime.0)?;
    // SAFETY: the caller's pointer.
    let completion = unsafe { Completion::required(completion) }?;
    handle.post(move |state| {
        let result = job(state.runtime()).map(lend);
        state.complete(Some(completion), result, 0);
    })
}

/// The id of a region Orbis can have: Orbis numbers regions from 1.
fn id(region: mln_offline_region_id) -> Option<u32> {
    u32::try_from(region).ok().filter(|&id| id > 0)
}

fn listed(runtime: &orb_api::Runtime) -> Result<Vec<orb_api::RegionInfo>> {
    Ok(runtime.regions()?)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_create_offline_region(
    runtime: mln_runtime,
    definition: *const mln_offline_region_definition,
    metadata: *const u8,
    metadata_size: usize,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            handles::runtime(runtime.0)?;
            if metadata.is_null() && metadata_size != 0 {
                return Err(Failure::argument("metadata is null"));
            }
            let metadata = bytes(mln_buffer_view {
                data: metadata.cast(),
                size: metadata_size,
            });
            let definition = read_definition(definition, metadata)?;
            regions(runtime, completion, move |r| {
                Ok(vec![r.create_region(definition)?])
            })
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_get_offline_region(
    runtime: mln_runtime,
    region_id: mln_offline_region_id,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            regions(runtime, completion, move |r| {
                let mut found = listed(r)?;
                found.retain(|region| Some(region.id) == id(region_id));
                Ok(found)
            })
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_list_offline_regions(
    runtime: mln_runtime,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            regions(runtime, completion, |r| {
                let mut found = listed(r)?;
                found.sort_by_key(|region| region.id);
                Ok(found)
            })
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_delete_offline_region(
    runtime: mln_runtime,
    region_id: mln_offline_region_id,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = handles::runtime(runtime.0)?;
            let completion = Completion::required(completion)?;
            handle.post(move |state| {
                let r = state.runtime();
                let result = match listed(r) {
                    Ok(found) => match id(region_id).filter(|&id| found.iter().any(|f| f.id == id))
                    {
                        Some(id) => r
                            .delete_region(id)
                            .map(|()| Value::None)
                            .map_err(Failure::from),
                        None => Err(Failure::new(
                            MLN_STATUS_NOT_FOUND,
                            format!("the cache file has no region {region_id}"),
                        )),
                    },
                    Err(e) => Err(e),
                };
                state.complete(Some(completion), result, 0);
            })
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_merge_offline_regions(
    runtime: mln_runtime,
    side_database_path: mln_buffer_view,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            handles::runtime(runtime.0)?;
            if side_database_path.data.is_null() && side_database_path.size != 0 {
                return Err(Failure::argument("side_database_path is null"));
            }
            let path = text(side_database_path, "side_database_path")?;
            let source = side_store(&path)?;
            regions(runtime, completion, move |r| Ok(r.merge_regions(&source)?))
        })
    }
}

/// What a side database path names: an Orbis cache's state directory, or
/// a database file. Anything else isn't a readable database file.
fn side_store(path: &str) -> Result<std::path::PathBuf> {
    let unreadable = || Failure::argument(format!("{path:?} isn't a readable database file"));
    if path.is_empty() || path == ":memory:" || path.starts_with("file:") {
        return Err(unreadable());
    }
    let file = std::path::Path::new(path);
    let mut head = [0u8; 64];
    let read = std::fs::File::open(file)
        .and_then(|mut f| std::io::Read::read(&mut f, &mut head))
        .map_err(|_| unreadable())?;
    if !file.is_file() {
        return Err(unreadable());
    }
    // An Orbis cache file names the directory beside it.
    if head[..read].starts_with(crate::runtime::CACHE_MARKER.as_bytes()) {
        let dir = std::path::PathBuf::from(format!("{path}.orbis"));
        if dir.is_dir() {
            return Ok(dir);
        }
    }
    Ok(file.to_owned())
}
