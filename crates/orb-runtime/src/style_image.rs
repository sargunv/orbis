//! Style images a host adds at run time (GL JS `addImage`, `updateImage`,
//! `removeImage`, `hasImage`): beside the sprites' images in the style's
//! image namespace, for icons, text images and patterns alike.
//!
//! An edit shows in layout at once (`has_image`, and the symbol layers lay
//! out again in a new revision); its pixels reach the atlas on the next
//! frame. Rows keep their atlas ids across edits, so tiles holding a name's
//! id read the new pixels, or none, without being laid out again.

use rustc_hash::FxHashSet as HashSet;
use std::sync::Arc;

use orb_hal::Gpu;
use orb_render::{ImageKey, ImageMetadata, ImageUpload};

use crate::events::EventQueue;
use crate::jobs::TileResult;
use crate::map::Map;
use crate::{MapEvent, SpriteIcons};

/// An image for the style: straight (not premultiplied) RGBA8 rows, top
/// first, and how icons and patterns read it (`pixel_ratio`, `sdf`,
/// stretches, content box, text fit).
#[derive(Clone, Debug, PartialEq)]
pub struct StyleImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    pub metadata: ImageMetadata,
}

impl StyleImage {
    /// An image with GL JS `addImage` options as JSON, such as
    /// `{"pixelRatio": 2, "sdf": true}`, in sprite metadata's spelling;
    /// empty for the defaults.
    pub fn with_options(
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        options: &str,
    ) -> Result<Self, String> {
        let metadata = if options.trim().is_empty() {
            ImageMetadata::default()
        } else {
            let options = orb_expr::json::parse(options).map_err(|e| e.to_string())?;
            if options.as_object().is_none() {
                return Err("image options are not an object".into());
            }
            crate::sprite::metadata(&options)?
        };
        let image = Self {
            width,
            height,
            pixels,
            metadata,
        };
        image.check()?;
        Ok(image)
    }

    fn check(&self) -> Result<(), String> {
        let ratio = self.metadata.pixel_ratio;
        if self.width == 0
            || self.height == 0
            || self.pixels.len() as u64 != u64::from(self.width) * u64::from(self.height) * 4
        {
            return Err(format!(
                "a {}x{} image needs {} bytes of RGBA, not {}",
                self.width,
                self.height,
                u64::from(self.width) * u64::from(self.height) * 4,
                self.pixels.len()
            ));
        }
        if !ratio.is_finite() || ratio <= 0.0 {
            return Err(format!("pixel ratio {ratio} is not positive"));
        }
        Ok(())
    }
}

/// A change to the atlas that waits for the next frame's GPU.
#[derive(Debug)]
pub(crate) enum AtlasEdit {
    /// New pixels and metadata for a name.
    Put(String, Arc<StyleImage>),
    /// New pixels for a name, keeping its metadata.
    Pixels(String, Arc<Vec<u8>>),
    Remove(String),
}

impl Map {
    /// Adds an image to the style (GL JS `addImage`); an error when the
    /// style has one by that name, from its sprites or added.
    pub fn add_image(&mut self, name: &str, image: StyleImage) -> Result<(), String> {
        image.check()?;
        let style = self.style.as_mut().ok_or("no style is loaded")?;
        if style.icons.contains(name) {
            return Err(format!("an image named {name:?} already exists"));
        }
        style
            .icons
            .add(name, [image.width, image.height], &image.metadata);
        style.reported_missing.remove(name);
        let image = Arc::new(image);
        style.added_images.insert(name.into(), Arc::clone(&image));
        style.image_edits.push(AtlasEdit::Put(name.into(), image));
        self.icons_changed();
        Ok(())
    }

    /// Replaces an image's pixels (GL JS `updateImage`): the same size, and
    /// its metadata stays.
    pub fn update_image(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    ) -> Result<(), String> {
        let style = self.style.as_mut().ok_or("no style is loaded")?;
        let image = style
            .icons
            .images
            .get(name)
            .ok_or_else(|| format!("no image named {name:?}"))?;
        if image.size != [width, height] {
            return Err(format!(
                "image {name:?} is {}x{}, not {width}x{height}",
                image.size[0], image.size[1]
            ));
        }
        if pixels.len() as u64 != u64::from(width) * u64::from(height) * 4 {
            return Err(format!("a {width}x{height} image needs its RGBA bytes"));
        }
        if let Some(added) = style.added_images.get_mut(name) {
            Arc::make_mut(added).pixels.clone_from(&pixels);
        }
        style
            .image_edits
            .push(AtlasEdit::Pixels(name.into(), Arc::new(pixels)));
        self.events.wake();
        Ok(())
    }

    /// Removes an image from the style (GL JS `removeImage`); an error when
    /// it has none by that name.
    pub fn remove_image(&mut self, name: &str) -> Result<(), String> {
        let style = self.style.as_mut().ok_or("no style is loaded")?;
        if !style.icons.remove(name) {
            return Err(format!("no image named {name:?}"));
        }
        style.added_images.remove(name);
        style.image_edits.push(AtlasEdit::Remove(name.into()));
        self.icons_changed();
        Ok(())
    }

    /// Whether the style has an image by this name (GL JS `hasImage`):
    /// from its sprites once they loaded, or added.
    pub fn has_image(&self, name: &str) -> bool {
        self.style.as_ref().is_some_and(|s| s.icons.contains(name))
    }

    /// The jobs lay icons out with the new set, once the sprites are in
    /// (their arrival hands the jobs every image until then).
    fn icons_changed(&mut self) {
        let style = self.style.as_mut().expect("checked by the caller");
        if style.sprites.loading > 0 {
            return;
        }
        style.icons.epoch += 1;
        self.services
            .jobs
            .set_icons(style.id, &Arc::new(style.icons.clone()), &style.worker);
        self.relayout_symbols();
    }

    /// Applies the queued image edits to the atlas, before the frame's
    /// tile results.
    pub(crate) fn apply_image_edits(&mut self, gpu: &mut dyn Gpu) {
        let Some(style) = &mut self.style else {
            return;
        };
        let Some(namespace) = style.image_namespace else {
            return;
        };
        if style.image_edits.is_empty() {
            return;
        }
        let mut images = self.services.images.0.lock().expect("images poisoned");
        let atlas = images.atlas.as_mut().expect("initialized");
        for edit in std::mem::take(&mut style.image_edits) {
            let (name, size, pixels, metadata) = match &edit {
                AtlasEdit::Remove(name) => {
                    atlas.clear(gpu, namespace, name);
                    continue;
                }
                AtlasEdit::Put(name, image) => (
                    name,
                    [image.width, image.height],
                    &image.pixels[..],
                    image.metadata.clone(),
                ),
                AtlasEdit::Pixels(name, pixels) => {
                    let id = atlas.resolve_icon(namespace, name);
                    let Some(metadata) = id.ok().and_then(|id| atlas.metadata(id)).cloned() else {
                        // Removed since, or never uploaded.
                        continue;
                    };
                    let size = id.ok().and_then(|id| atlas.size(id)).expect("uploaded");
                    (name, size, &pixels[..], metadata)
                }
            };
            atlas.clear(gpu, namespace, name);
            // Patterns and icons pad differently, as sprites upload.
            let mut upload = |pattern: bool| {
                let id = if pattern {
                    atlas.resolve(namespace, name)?
                } else {
                    atlas.resolve_icon(namespace, name)?
                };
                atlas.upload(
                    gpu,
                    id,
                    ImageUpload {
                        size,
                        pixels,
                        metadata: metadata.clone(),
                        pattern,
                    },
                )
            };
            if let Err(e) = upload(true).and_then(|()| upload(false)) {
                self.warnings.push(format!("image {name}: {e}"));
            }
        }
        style.paint.invalidate();
    }
}

/// Reports the images a tile's layers name that the style lacks (GL JS
/// `styleimagemissing`), once per name until it is added. Patterns are
/// checked here, symbols in their layout; nothing is missing before the
/// sprites are in.
pub(crate) fn report_missing(
    reported: &mut HashSet<String>,
    icons: &SpriteIcons,
    sprites_loading: u32,
    build: &TileResult,
    events: &mut EventQueue,
) {
    if sprites_loading > 0 {
        return;
    }
    let patterns = build.meshes.iter().flat_map(|m| &m.images).filter_map(|r| {
        let ImageKey::Name(name) = &r.key else {
            return None;
        };
        (!name.is_empty() && !icons.contains(name)).then_some(name.as_str())
    });
    for name in build
        .missing_images
        .iter()
        .map(String::as_str)
        .chain(patterns)
    {
        if !reported.contains(name) {
            reported.insert(name.to_owned());
            events.push(MapEvent::StyleImageMissing { name: name.into() });
        }
    }
}
