// Projections: a map's view detached from the map, converting between
// coordinates and screen points on any thread for as long as the caller
// keeps it, and the spherical Mercator helpers.
//
// A view carries no projection or terrain (`mln_map_view` has no slot for
// them), so a projection is the flat Web Mercator view of its camera, as
// Native's is.

use std::sync::{Arc, Mutex};

use glam::DVec2;
use orb_camera::{CameraState, Padding, View, Viewport};
use orb_crs::{
    Geodetic, MERCATOR_MAX_LATITUDE as MAX_LATITUDE, Projection, Surface, TileMatrixSet,
};

use crate::base::{Failure, Result, boundary, fresh_out, put, sized};
use crate::handles::{self, Object};
use crate::sys::*;

/// Native's zoom limit.
const MAX_ZOOM: f64 = 25.5;
/// Native's tile size, which its zoom is defined on.
const TILE_SIZE: f64 = 512.0;
/// MapLibre's earth radius, Web Mercator's.
const EARTH_RADIUS_M: f64 = 6_378_137.0;

fn mercator() -> &'static Projection {
    TileMatrixSet::web_mercator().projection()
}

/// A projection's view, behind its handle.
pub(crate) struct ProjectionHandle {
    view: Mutex<mln_map_view>,
}

fn finite(v: f64, what: &str) -> Result<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(Failure::argument(format!("{what} must be finite")))
    }
}

fn check_coordinate(c: mln_lat_lng, what: &str) -> Result {
    finite(c.latitude, &format!("{what}.latitude"))?;
    finite(c.longitude, &format!("{what}.longitude"))?;
    if c.latitude.abs() > 90.0 {
        return Err(Failure::argument(format!(
            "{what}.latitude must be within [-90, 90]"
        )));
    }
    Ok(())
}

fn check_insets(p: mln_edge_insets, what: &str) -> Result {
    for v in [p.top, p.left, p.bottom, p.right] {
        if !(v.is_finite() && v >= 0.0) {
            return Err(Failure::argument(format!(
                "{what} must be finite and not negative"
            )));
        }
    }
    Ok(())
}

/// Checks a camera as a view holds it.
fn check_camera(c: &mln_camera) -> Result {
    check_coordinate(c.center, "camera.center")?;
    check_insets(c.padding, "camera.padding")?;
    for (v, what) in [
        (c.center_altitude, "camera.center_altitude"),
        (c.zoom, "camera.zoom"),
        (c.bearing, "camera.bearing"),
        (c.pitch, "camera.pitch"),
        (c.roll, "camera.roll"),
        (c.field_of_view, "camera.field_of_view"),
    ] {
        finite(v, what)?;
    }
    if !(0.0..=MAX_ZOOM).contains(&c.zoom) {
        return Err(Failure::argument(format!(
            "camera.zoom {} is outside the limits [0, {MAX_ZOOM}]",
            c.zoom
        )));
    }
    if !(0.0..90.0).contains(&c.pitch) {
        return Err(Failure::argument(format!(
            "camera.pitch {} is outside the limits [0, 90)",
            c.pitch
        )));
    }
    if !(c.field_of_view > 0.0 && c.field_of_view < 180.0) {
        return Err(Failure::argument(
            "camera.field_of_view is outside the limits (0, 180)",
        ));
    }
    Ok(())
}

fn check_view(v: &mln_map_view) -> Result {
    check_camera(&v.camera)?;
    if v.extent.width == 0
        || v.extent.height == 0
        || !(v.extent.scale_factor.is_finite() && v.extent.scale_factor > 0.0)
    {
        return Err(Failure::argument(
            "the view's extent must have a size and a positive scale factor",
        ));
    }
    let m = v.projection_mode;
    if !(m.x_skew.is_finite() && m.y_skew.is_finite()) {
        return Err(Failure::argument(
            "the projection mode's skews must be finite",
        ));
    }
    check_insets(v.viewport.frustum_offset, "the viewport's frustum offset")?;
    if v.viewport.north_orientation > MLN_NORTH_ORIENTATION_LEFT {
        return Err(Failure::argument(
            "viewport.north_orientation names no orientation",
        ));
    }
    Ok(())
}

/// The camera and viewport Orbis draws a view with.
fn resolve(v: &mln_map_view) -> View {
    let c = v.camera;
    // North pointing right turns the picture a quarter clockwise, as a
    // bearing of -90° does.
    let turn = f64::from(v.viewport.north_orientation) * 90.0;
    // An axonometric view centres the target in the whole extent.
    let padding = if v.projection_mode.axonometric {
        Padding::default()
    } else {
        Padding {
            top: c.padding.top,
            right: c.padding.right,
            bottom: c.padding.bottom,
            left: c.padding.left,
        }
    };
    let mut camera = CameraState::new(
        Geodetic::new(c.center.longitude, c.center.latitude),
        c.zoom,
        mercator(),
    );
    camera.bearing = c.bearing - turn;
    camera.pitch = c.pitch;
    camera.roll = c.roll;
    camera.fov = c.field_of_view;
    camera.elevation = c.center_altitude;
    camera.padding = padding;
    View::new(
        &camera,
        Viewport {
            width: f64::from(v.extent.width),
            height: f64::from(v.extent.height),
        },
        &Surface::plane(*mercator()),
    )
}

fn projection(handle: mln_map_projection) -> Result<Arc<ProjectionHandle>> {
    match handles::lookup(handle.0, handles::Kind::Projection)? {
        Object::Projection(p) => Ok(p),
        _ => unreachable!(),
    }
}

impl ProjectionHandle {
    fn view(&self) -> mln_map_view {
        *self
            .view
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn pixel(view: &View, c: mln_lat_lng) -> Result<mln_screen_point> {
    check_coordinate(c, "coordinate")?;
    let lat = c.latitude.clamp(-MAX_LATITUDE, MAX_LATITUDE);
    let world = mercator().forward(Geodetic::new(c.longitude, lat));
    let p = view
        .project(world)
        .ok_or_else(|| Failure::argument("the coordinate is behind the camera"))?;
    Ok(mln_screen_point { x: p.x, y: p.y })
}

fn coordinate(view: &View, p: mln_screen_point, unwrapped: bool) -> Result<mln_lat_lng> {
    finite(p.x, "point.x")?;
    finite(p.y, "point.y")?;
    let world = view
        .unproject(DVec2::new(p.x, p.y))
        .ok_or_else(|| Failure::argument("the point is above the horizon"))?;
    let mut g = mercator().inverse(world);
    if !unwrapped {
        g = g.wrapped();
    }
    Ok(mln_lat_lng {
        latitude: g.lat,
        longitude: g.lon,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_projection_create(
    view: *const mln_map_view,
    out_projection: *mut mln_map_projection,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let view = *sized(view, "view")?;
            fresh_out(out_projection, "out_projection")?;
            check_view(&view)?;
            let id = handles::insert(Object::Projection(Arc::new(ProjectionHandle {
                view: Mutex::new(view),
            })));
            put(out_projection, mln_map_projection(id), "out_projection")
        })
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mln_map_projection_release(projection: mln_map_projection) {
    if let Ok(object) = handles::lookup(projection.0, handles::Kind::Projection) {
        drop(object);
        handles::remove(projection.0);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_projection_get_camera(
    projection: mln_map_projection,
    out_result: *mut mln_map_projection_camera_result,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let p = self::projection(projection)?;
            let size = sized(out_result.cast_const(), "out_result")?.size;
            let camera = p.view().camera;
            out_result.write(mln_map_projection_camera_result { size, camera });
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_projection_set_camera(
    projection: mln_map_projection,
    camera: *const mln_camera_options,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let p = self::projection(projection)?;
            let o = *sized(camera, "camera")?;
            let mut view = p
                .view
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut c = view.camera;
            let has = |bit| o.fields & bit != 0;
            if has(MLN_CAMERA_OPTION_CENTER) {
                c.center = o.center;
            }
            if has(MLN_CAMERA_OPTION_CENTER_ALTITUDE) {
                c.center_altitude = o.center_altitude;
            }
            if has(MLN_CAMERA_OPTION_PADDING) {
                c.padding = o.padding;
            }
            if has(MLN_CAMERA_OPTION_ZOOM) {
                c.zoom = o.zoom;
            }
            if has(MLN_CAMERA_OPTION_BEARING) {
                c.bearing = o.bearing;
            }
            if has(MLN_CAMERA_OPTION_PITCH) {
                c.pitch = o.pitch;
            }
            if has(MLN_CAMERA_OPTION_ROLL) {
                c.roll = o.roll;
            }
            if has(MLN_CAMERA_OPTION_FOV) {
                c.field_of_view = o.field_of_view;
            }
            check_camera(&c)?;
            view.camera = c;
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_projection_pixel_for_lat_lng(
    projection: mln_map_projection,
    coordinate: mln_lat_lng,
    out_point: *mut mln_screen_point,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let p = self::projection(projection)?;
            if out_point.is_null() {
                return Err(Failure::argument("out_point is null"));
            }
            let point = pixel(&resolve(&p.view()), coordinate)?;
            put(out_point, point, "out_point")
        })
    }
}

unsafe fn lat_lng_for_pixel(
    projection: mln_map_projection,
    point: mln_screen_point,
    out: *mut mln_lat_lng,
    unwrapped: bool,
) -> Result {
    let p = self::projection(projection)?;
    if out.is_null() {
        return Err(Failure::argument("out_coordinate is null"));
    }
    let c = coordinate(&resolve(&p.view()), point, unwrapped)?;
    // SAFETY: the caller's pointer.
    unsafe { put(out, c, "out_coordinate") }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_projection_lat_lng_for_pixel(
    projection: mln_map_projection,
    point: mln_screen_point,
    out_coordinate: *mut mln_lat_lng,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            lat_lng_for_pixel(projection, point, out_coordinate, false)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_projection_lat_lng_for_pixel_unwrapped(
    projection: mln_map_projection,
    point: mln_screen_point,
    out_coordinate: *mut mln_lat_lng,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            lat_lng_for_pixel(projection, point, out_coordinate, true)
        })
    }
}

/// Converts a batch whole or not at all: every input is checked before
/// any output is written.
unsafe fn batch<I: Copy, O: Copy>(
    inputs: *const I,
    count: usize,
    outputs: *mut O,
    names: (&str, &str),
    convert: impl Fn(I) -> Result<O>,
) -> Result {
    if count == 0 {
        return Ok(());
    }
    if inputs.is_null() {
        return Err(Failure::argument(format!("{} is null", names.0)));
    }
    if outputs.is_null() {
        return Err(Failure::argument(format!("{} is null", names.1)));
    }
    // SAFETY: the caller's arrays of `count`.
    let inputs = unsafe { std::slice::from_raw_parts(inputs, count) };
    let converted = inputs
        .iter()
        .enumerate()
        .map(|(i, &input)| {
            convert(input)
                .map_err(|f| Failure::new(f.status, format!("{}[{i}]: {}", names.0, f.message)))
        })
        .collect::<Result<Vec<O>>>()?;
    // SAFETY: as above.
    unsafe { std::ptr::copy_nonoverlapping(converted.as_ptr(), outputs, count) };
    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_projection_pixels_for_lat_lngs(
    projection: mln_map_projection,
    coordinates: *const mln_lat_lng,
    coordinate_count: usize,
    out_points: *mut mln_screen_point,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let view = resolve(&self::projection(projection)?.view());
            batch(
                coordinates,
                coordinate_count,
                out_points,
                ("coordinates", "out_points"),
                |c| pixel(&view, c),
            )
        })
    }
}

unsafe fn lat_lngs_for_pixels(
    projection: mln_map_projection,
    points: *const mln_screen_point,
    count: usize,
    out: *mut mln_lat_lng,
    unwrapped: bool,
) -> Result {
    let view = resolve(&self::projection(projection)?.view());
    // SAFETY: the caller's pointers.
    unsafe {
        batch(points, count, out, ("points", "out_coordinates"), |p| {
            coordinate(&view, p, unwrapped)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_projection_lat_lngs_for_pixels(
    projection: mln_map_projection,
    points: *const mln_screen_point,
    point_count: usize,
    out_coordinates: *mut mln_lat_lng,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            lat_lngs_for_pixels(projection, points, point_count, out_coordinates, false)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_projection_lat_lngs_for_pixels_unwrapped(
    projection: mln_map_projection,
    points: *const mln_screen_point,
    point_count: usize,
    out_coordinates: *mut mln_lat_lng,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            lat_lngs_for_pixels(projection, points, point_count, out_coordinates, true)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_projection_meters_per_pixel_at_latitude(
    projection: mln_map_projection,
    latitude: f64,
    out_meters_per_pixel: *mut f64,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let zoom = self::projection(projection)?.view().camera.zoom;
            check_coordinate(
                mln_lat_lng {
                    latitude,
                    longitude: 0.0,
                },
                "coordinate",
            )?;
            let lat = latitude.clamp(-MAX_LATITUDE, MAX_LATITUDE).to_radians();
            let meters =
                lat.cos() * 2.0 * std::f64::consts::PI * EARTH_RADIUS_M / zoom.exp2() / TILE_SIZE;
            put(out_meters_per_pixel, meters, "out_meters_per_pixel")
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_projected_meters_for_lat_lng(
    coordinate: mln_lat_lng,
    out_meters: *mut mln_projected_meters,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            if out_meters.is_null() {
                return Err(Failure::argument("out_meters is null"));
            }
            check_coordinate(coordinate, "coordinate")?;
            let lat = coordinate.latitude.clamp(-MAX_LATITUDE, MAX_LATITUDE);
            let p = mercator().forward(Geodetic::new(coordinate.longitude, lat));
            put(
                out_meters,
                mln_projected_meters {
                    northing: p.y,
                    easting: p.x,
                },
                "out_meters",
            )
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_lat_lng_for_projected_meters(
    meters: mln_projected_meters,
    out_coordinate: *mut mln_lat_lng,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            if out_coordinate.is_null() {
                return Err(Failure::argument("out_coordinate is null"));
            }
            finite(meters.northing, "meters.northing")?;
            finite(meters.easting, "meters.easting")?;
            let g = mercator().inverse(DVec2::new(meters.easting, meters.northing));
            put(
                out_coordinate,
                mln_lat_lng {
                    latitude: g.lat,
                    longitude: g.lon,
                },
                "out_coordinate",
            )
        })
    }
}
