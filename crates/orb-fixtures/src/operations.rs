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
            map.camera_mut().set_zoom(&projection, zoom)
        }
        "setBearing" => map.camera_mut().bearing = number(1)?,
        "setPitch" => map.camera_mut().pitch = number(1)?,
        "setRoll" => map.camera_mut().roll = number(1)?,
        "setCenterElevation" => map.camera_mut().elevation = number(1)?,
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
            map.camera_mut().target = orb_geo::Geodetic::new(at(0)?, at(1)?);
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
