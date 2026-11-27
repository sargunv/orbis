//! When still output is done, a rule orb-api's still output and orb-web's
//! snapshots share.

use crate::{FrameReport, Map};

/// Whether a still of `map` is done with the frame `report` describes
/// (`None` for one that drew nothing): the first frame that rests
/// ([`FrameReport::rests`]) shows everything, as a continuous map whose
/// noise moves never idles. An error says why it never will: a style that
/// failed to load.
pub fn settled(map: &Map, report: Option<&FrameReport>) -> Result<bool, String> {
    match map.style_error() {
        Some(message) => Err(message.to_string()),
        None => Ok(report.is_some_and(FrameReport::rests)),
    }
}
