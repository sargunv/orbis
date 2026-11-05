//! `video` sources (decision 0039). The core has no decoder: the host plays
//! each video. The map asks for playback with [`MapEvent::Video`], and the
//! host hands it frames with [`Map::video_frame`](crate::Map::video_frame).
//! The newest frame fills the source's one tile, and later frames are
//! written over its texture layer.

use std::collections::HashMap;

use orb_crs::TileIndex;
use orb_hal::HostImageId;
use orb_v8::{ImageContent, SourceKind};

use crate::MapEvent;
use crate::jobs::{ImagePixels, TileImage};
use crate::style::StylePlan;

/// What the map asks of the host's player for a video source.
#[derive(Clone, Debug, PartialEq)]
pub enum VideoRequest {
    /// Play the first of `urls` the host can, muted, looping, from the
    /// start (GL JS plays video sources as they're added).
    Open {
        urls: Vec<String>,
    },
    Play,
    Pause,
    /// Seek to a time in seconds.
    Seek(f64),
    /// The source is gone: stop and let go of the video.
    Close,
}

/// Why a video source draws nothing on a host that plays no video.
pub const NO_VIDEO: &str =
    "video sources need a host that plays video, which this one doesn't (#77)";

/// The video sources' state, by source key.
#[derive(Debug, Default)]
pub(crate) struct Videos {
    /// The host plays video ([`Map::set_plays_video`](crate::Map::set_plays_video)).
    pub(crate) enabled: bool,
    open: HashMap<u32, Open>,
    /// Host frames no source holds any more, to release with the GPU.
    released: Vec<HostImageId>,
}

#[derive(Debug)]
struct Open {
    id: String,
    urls: Vec<String>,
    /// The newest frame, premultiplied, kept for a tile requested again.
    last: Option<TileImage>,
    /// `last` isn't in the resident tile's layer yet.
    fresh: bool,
    failed: Option<String>,
    /// Requested tiles, and whether their arrival was counted.
    wanted: Vec<(TileIndex, bool)>,
}

/// One video source's work in a frame.
pub(crate) struct VideoWork<'a> {
    pub(crate) key: u32,
    /// The newest frame, if the resident tile's layer needs it.
    pub(crate) fresh: Option<&'a TileImage>,
    /// Requested tiles that can have the newest frame, or the failure.
    pub(crate) wanted: Vec<(TileIndex, bool)>,
    pub(crate) frame: Option<&'a TileImage>,
    pub(crate) failed: Option<&'a str>,
}

impl Videos {
    /// Opens the plan's video sources and closes the others, asking the
    /// host through `events`.
    pub(crate) fn sync(&mut self, plan: Option<&StylePlan>, events: &mut Vec<MapEvent>) {
        let mut sources = HashMap::new();
        if let Some(plan) = plan {
            for (index, source) in plan.sources.iter().enumerate() {
                let def = plan.source_def(index);
                if let SourceKind::Image(image) = &def.kind
                    && let ImageContent::Video(urls) = &image.content
                {
                    sources.insert(source.key, (def.id.as_str(), urls));
                }
            }
        }
        let released = &mut self.released;
        self.open.retain(|key, open| {
            let keep = sources
                .get(key)
                .is_some_and(|(id, urls)| open.id == *id && open.urls == **urls);
            if !keep {
                released.extend(open.last.take().and_then(host));
                if open.failed.is_none() {
                    events.push(MapEvent::Video {
                        source: open.id.clone(),
                        request: VideoRequest::Close,
                    });
                }
            }
            keep
        });
        for (key, (id, urls)) in sources {
            self.open.entry(key).or_insert_with(|| {
                let failed = (!self.enabled).then(|| NO_VIDEO.to_owned());
                if failed.is_none() {
                    events.push(MapEvent::Video {
                        source: id.to_owned(),
                        request: VideoRequest::Open { urls: urls.clone() },
                    });
                }
                Open {
                    id: id.to_owned(),
                    urls: urls.clone(),
                    last: None,
                    fresh: false,
                    failed,
                    wanted: Vec::new(),
                }
            });
        }
    }

    fn get(&mut self, id: &str) -> Result<&mut Open, String> {
        self.open
            .values_mut()
            .find(|o| o.id == id)
            .ok_or_else(|| format!("no video source \"{id}\""))
    }

    /// Whether source `key` is a video's; then its tile waits for a frame.
    pub(crate) fn want(&mut self, key: u32, tile: TileIndex) -> bool {
        let Some(open) = self.open.get_mut(&key) else {
            return false;
        };
        if !open.wanted.iter().any(|(t, _)| *t == tile) {
            open.wanted.push((tile, false));
        }
        true
    }

    /// The host's newest frame of source `id`, straight alpha.
    pub(crate) fn frame(&mut self, id: &str, mut frame: TileImage) -> Result<(), String> {
        if let ImagePixels::Core(pixels) = &mut frame.pixels {
            crate::jobs::premultiply(pixels);
        }
        let open = match self.get(id) {
            Ok(open) => open,
            Err(e) => {
                self.released.extend(host(frame));
                return Err(e);
            }
        };
        let old = open.last.replace(frame).and_then(host);
        open.fresh = true;
        self.released.extend(old);
        Ok(())
    }

    pub(crate) fn failed(&mut self, id: &str, message: String) -> Result<(), String> {
        self.get(id)?.failed = Some(message);
        Ok(())
    }

    /// Asks the host to play, pause or seek source `id`.
    pub(crate) fn request(
        &mut self,
        id: &str,
        request: VideoRequest,
        events: &mut Vec<MapEvent>,
    ) -> Result<(), String> {
        let open = self.get(id)?;
        if open.failed.is_none() {
            events.push(MapEvent::Video {
                source: open.id.clone(),
                request,
            });
        }
        Ok(())
    }

    /// Host frames to release.
    pub(crate) fn released(&mut self) -> Vec<HostImageId> {
        std::mem::take(&mut self.released)
    }

    /// This frame's work per video source.
    pub(crate) fn work(&mut self) -> Vec<VideoWork<'_>> {
        self.open
            .iter_mut()
            .map(|(&key, open)| {
                let fresh = std::mem::take(&mut open.fresh);
                let ready = open.last.is_some() || open.failed.is_some();
                VideoWork {
                    key,
                    fresh: open.last.as_ref().filter(|_| fresh),
                    wanted: if ready {
                        std::mem::take(&mut open.wanted)
                    } else {
                        Vec::new()
                    },
                    frame: open.last.as_ref(),
                    failed: open.failed.as_deref(),
                }
            })
            .collect()
    }

    /// Asks again for a tile that couldn't take its frame yet.
    pub(crate) fn retry(&mut self, key: u32, tile: TileIndex) {
        if let Some(open) = self.open.get_mut(&key) {
            open.wanted.push((tile, true));
        }
    }
}

fn host(frame: TileImage) -> Option<HostImageId> {
    match frame.pixels {
        ImagePixels::Host(id) => Some(id),
        ImagePixels::Core(_) => None,
    }
}

/// A frame as a tile's texels; a host frame stays the source's.
pub(crate) fn tile_image(frame: &TileImage) -> TileImage {
    TileImage {
        width: frame.width,
        height: frame.height,
        pixels: match &frame.pixels {
            ImagePixels::Core(pixels) => ImagePixels::Core(pixels.clone()),
            &ImagePixels::Host(id) => ImagePixels::Host(id),
        },
    }
}
