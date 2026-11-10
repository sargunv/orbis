//! When still output is done, a rule orb-api's still output and orb-web's
//! snapshots share.

use crate::{FrameReport, Map};

/// A still's frames so far: it has settled after two idle frames in a
/// row, the second drawing what the first's replies brought.
#[derive(Clone, Copy, Debug, Default)]
pub struct Settle {
    /// Whether the last frame was idle.
    idle: bool,
}

impl Settle {
    /// Counts a frame of `map` (`None` for one that drew nothing):
    /// whether the still has settled, or why it never will, a style that
    /// failed to load.
    pub fn frame(&mut self, map: &Map, report: Option<&FrameReport>) -> Result<bool, String> {
        if let Some(message) = map.style_error() {
            return Err(message.to_string());
        }
        let idle = report.is_some_and(|r| r.idle);
        let settled = idle && self.idle;
        self.idle = idle;
        Ok(settled)
    }
}
