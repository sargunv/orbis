//! GL JS render-test `operations`: camera calls, `wait`/`idle`, and map
//! calls applied as style transactions. `Metadata` skips what this can't
//! apply; the Chrome runner applies the same calls in `orb-web`.

use orb_runtime::{Map, Transaction};
use serde_json::Value;

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
pub(crate) fn apply(map: &mut Map, op: &Value) -> Result<Next, String> {
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
        "setZoom" => map.camera_mut().set_zoom(number(1)?),
        "setBearing" => map.camera_mut().bearing = number(1)?,
        "setPitch" => map.camera_mut().pitch = number(1)?,
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
        _ => {
            let tx = Transaction::from_commands(&format!("[{op}]"))?;
            map.edit(&tx)?;
        }
    }
    Ok(Next::Continue)
}
