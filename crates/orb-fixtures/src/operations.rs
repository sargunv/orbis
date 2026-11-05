//! GL JS render-test `operations`: camera calls, `wait`/`idle`, style
//! images from the assets, and map calls applied as style transactions.
//! `Metadata` skips what this can't apply; the Chrome runner applies the
//! same calls in `orb-web`.

use std::path::Path;

use orb_runtime::{Map, StyleImage, Transaction};
use serde_json::Value;

use crate::images::Image;

/// What the runner does after an operation, as GL JS's harness does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Next {
    /// Go on to the next operation.
    Continue,
    /// `["wait"]`: render until the map has loaded (GL JS `loaded()`).
    Loaded,
    /// `["idle"]`: render until it is idle, fades done.
    Idle,
    /// `["wait", ms]`: keep rendering for `ms` of wall time, then render.
    Wait(f64),
}

/// Applies one operation and says what the runner does next.
/// `assets` is the directory `addImage` paths are in.
pub(crate) fn apply(map: &mut Map, op: &Value, assets: &Path) -> Result<Next, String> {
    let name = op
        .get(0)
        .and_then(Value::as_str)
        .ok_or("unnamed operation")?;
    let number = |i: usize| op.get(i).and_then(Value::as_f64).ok_or("expected a number");
    match name {
        "wait" => {
            return Ok(match op.get(1).and_then(Value::as_f64) {
                Some(ms) => Next::Wait(ms),
                None => Next::Loaded,
            });
        }
        "idle" => return Ok(Next::Idle),
        "setZoom" => {
            let (zoom, projection) = (number(1)?, map.projection());
            move_camera(map, |c| c.set_zoom(&projection, zoom));
        }
        "setBearing" => {
            let bearing = number(1)?;
            move_camera(map, |c| c.bearing = bearing);
        }
        "setPitch" => {
            let pitch = number(1)?;
            move_camera(map, |c| c.pitch = pitch);
        }
        "setRoll" => {
            let roll = number(1)?;
            move_camera(map, |c| c.roll = roll);
        }
        "setCenterElevation" => map.set_center_elevation(number(1)?),
        "setCenterClampedToGround" => {
            let clamped = op
                .get(1)
                .and_then(Value::as_bool)
                .ok_or("expected a boolean")?;
            map.set_center_clamped_to_ground(clamped);
        }
        "setCenter" => {
            let center = op
                .get(1)
                .and_then(Value::as_array)
                .ok_or("expected a centre")?;
            let at = |i: usize| {
                center
                    .get(i)
                    .and_then(Value::as_f64)
                    .ok_or("expected a number")
            };
            let target = orb_crs::Geodetic::new(at(0)?, at(1)?);
            move_camera(map, |c| c.target = target);
        }
        // `["addImage", name, path, options?]`.
        "addImage" => {
            let text = |i: usize| op.get(i).and_then(Value::as_str).ok_or("expected a string");
            let path = text(2)?;
            let image = Image::read(&assets.join(path.trim_start_matches("./")))?;
            let options = op.get(3).map_or_else(String::new, Value::to_string);
            let image =
                StyleImage::with_options(image.width, image.height, image.pixels, &options)?;
            map.add_image(text(1)?, image)?;
        }
        "removeImage" => {
            let name = op.get(1).and_then(Value::as_str).ok_or("expected a name")?;
            map.remove_image(name)?;
        }
        _ => {
            let tx = Transaction::from_commands(&format!("[{op}]"))?;
            map.edit(&tx)?;
        }
    }
    Ok(Next::Continue)
}

/// Jumps the camera to a changed copy of itself, as GL JS's harness's
/// setters do.
fn move_camera(map: &mut Map, change: impl FnOnce(&mut orb_runtime::CameraState)) {
    let mut camera = *map.camera();
    change(&mut camera);
    map.jump_to_state(camera);
}
