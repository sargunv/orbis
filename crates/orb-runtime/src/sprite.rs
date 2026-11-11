//! Sprite I/O stays above the scene. Both resources load concurrently;
//! decoding is a host service, and the render thread receives straight RGBA8.
//! The metadata goes ahead of the pixels: icon layout needs only the JSON,
//! so it doesn't wait for the image's decode and upload.

use std::sync::{Arc, Mutex};

use orb_expr::json::{self, Json};
use orb_host::{Abort, DecodedImage, Fetch, ImageDecode, ResourceKind};
use orb_render::ImageMetadata;
use orb_style::Sprite;

/// A sprite image as its JSON describes it: what icon layout needs, and
/// where its pixels are in the sprite's image.
#[derive(Clone, Debug)]
pub(crate) struct SpriteIcon {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub metadata: ImageMetadata,
    origin: [u32; 2],
}

#[derive(Debug)]
pub(crate) struct SpriteImage {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    pub metadata: ImageMetadata,
}

type Icons = Box<dyn FnOnce(Result<Vec<SpriteIcon>, String>) + Send>;
type Done = Box<dyn FnOnce(Result<Vec<SpriteImage>, String>) + Send>;
type Fetched = Result<Vec<u8>, String>;
struct Pending {
    /// The JSON's icons once parsed, and the image once fetched.
    icons: Option<Result<Vec<SpriteIcon>, String>>,
    image: Option<Fetched>,
    done: Option<Done>,
}

/// Loads `sprite`: `icons` hears its JSON's images as soon as it is
/// parsed, and `done` their pixels once the image is decoded too. A sprite
/// whose JSON fails says so to `icons` alone; `done` then gets no images.
pub(crate) fn load(
    fetch: &dyn Fetch,
    decode: Arc<dyn ImageDecode>,
    sprite: &Sprite,
    pixel_ratio: f32,
    abort: &Abort,
    icons: Icons,
    done: Done,
) {
    let pending = Arc::new(Mutex::new(Pending {
        icons: None,
        image: None,
        done: Some(done),
    }));
    let mut icons = Some(icons);
    for (index, extension) in [".json", ".png"].iter().enumerate() {
        let pending = pending.clone();
        let decode = decode.clone();
        let id = sprite.id.clone();
        let heard = (index == 0).then(|| icons.take().expect("one JSON"));
        let url = resource_url(&sprite.url, pixel_ratio, extension);
        let kind = if index == 0 {
            ResourceKind::SpriteJson
        } else {
            ResourceKind::SpriteImage
        };
        fetch.get(
            kind,
            &url,
            abort,
            Box::new(move |result| {
                let result = result.map_err(|e| e.to_string());
                let ready = {
                    let mut p = pending.lock().expect("sprite load poisoned");
                    match heard {
                        Some(heard) => {
                            let parsed = result.and_then(|json| parse_icons(&id, &json));
                            heard(parsed.clone());
                            p.icons = Some(parsed);
                        }
                        None => p.image = Some(result),
                    }
                    match (&p.icons, &p.image) {
                        (Some(_), Some(_)) => Some((
                            p.icons.take().unwrap(),
                            p.image.take().unwrap(),
                            p.done.take().unwrap(),
                        )),
                        _ => None,
                    }
                };
                let Some((icons, image, done)) = ready else {
                    return;
                };
                match (icons, image) {
                    // The JSON's failure went with its icons.
                    (Err(_), _) => done(Ok(Vec::new())),
                    (Ok(_), Err(e)) => done(Err(e)),
                    (Ok(icons), Ok(image)) => decode.decode(
                        image,
                        Box::new(move |image| done(image.and_then(|image| extract(icons, image)))),
                    ),
                }
            }),
        );
    }
}

fn resource_url(base: &str, ratio: f32, extension: &str) -> String {
    let split = base.find(['?', '#']).unwrap_or(base.len());
    let (path, query) = base.split_at(split);
    let scale = if ratio > 1.0 { "@2x" } else { "" };
    format!("{path}{scale}{extension}{query}")
}

fn bad() -> String {
    "invalid sprite metadata or rectangle".to_string()
}

/// Sprite `id`'s images, as its JSON `bytes` describe them.
fn parse_icons(id: &str, bytes: &[u8]) -> Result<Vec<SpriteIcon>, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| bad())?;
    let doc = json::parse(text).map_err(|_| bad())?;
    let members = doc.as_object().ok_or_else(bad)?;
    let mut out = Vec::with_capacity(members.len());
    for m in members {
        let d = &m.value;
        let number = |key| d.get(key).and_then(Json::as_f64).ok_or_else(bad);
        let integer = |key| {
            let n = number(key)?;
            if n < 0.0 || n > f64::from(u32::MAX) || n.fract() != 0.0 {
                return Err(bad());
            }
            Ok(n as u32)
        };
        let (x, y, width, height) = (
            integer("x")?,
            integer("y")?,
            integer("width")?,
            integer("height")?,
        );
        if width == 0 || height == 0 {
            return Err(bad());
        }
        let name = if id == "default" {
            m.key.clone()
        } else {
            format!("{id}:{}", m.key)
        };
        out.push(SpriteIcon {
            name,
            width,
            height,
            metadata: metadata(d).map_err(|_| bad())?,
            origin: [x, y],
        });
    }
    Ok(out)
}

/// The icons' pixels, from the sprite's decoded `image`; a rectangle past
/// its edge fails the sprite.
fn extract(icons: Vec<SpriteIcon>, image: DecodedImage) -> Result<Vec<SpriteImage>, String> {
    if image.pixels.len() as u64 != u64::from(image.width) * u64::from(image.height) * 4 {
        return Err(bad());
    }
    let mut out = Vec::with_capacity(icons.len());
    for icon in icons {
        let ([x, y], width, height) = (icon.origin, icon.width, icon.height);
        if u64::from(x) + u64::from(width) > u64::from(image.width)
            || u64::from(y) + u64::from(height) > u64::from(image.height)
        {
            return Err(bad());
        }
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for row in y..y + height {
            let at = (u64::from(row) * u64::from(image.width) + u64::from(x)) as usize * 4;
            pixels.extend_from_slice(&image.pixels[at..at + width as usize * 4]);
        }
        out.push(SpriteImage {
            name: icon.name,
            width,
            height,
            pixels,
            metadata: icon.metadata,
        });
    }
    Ok(out)
}

/// An image's metadata as sprite JSON and GL JS `addImage` options spell
/// it: `pixelRatio`, `sdf`, `stretchX`, `stretchY`, `content`,
/// `textFitWidth`, `textFitHeight`.
pub(crate) fn metadata(d: &Json) -> Result<ImageMetadata, String> {
    let ratio = match d.get("pixelRatio") {
        None => 1.0,
        Some(r) => r.as_f64().ok_or("pixelRatio is not a number")? as f32,
    };
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(format!("pixelRatio {ratio} is not positive"));
    }
    let stretch = |key| -> Result<Vec<[f32; 2]>, String> {
        d.get(key).map_or(Ok(Vec::new()), |j| {
            j.as_array()
                .ok_or_else(|| format!("{key} is not an array"))?
                .iter()
                .map(numbers::<2>)
                .collect()
        })
    };
    let text = |key| d.get(key).and_then(Json::as_str).map(str::to_owned);
    Ok(ImageMetadata {
        pixel_ratio: ratio,
        sdf: d.get("sdf").and_then(Json::as_bool).unwrap_or(false),
        stretch_x: stretch("stretchX")?,
        stretch_y: stretch("stretchY")?,
        content: d.get("content").map(numbers::<4>).transpose()?,
        text_fit_width: text("textFitWidth"),
        text_fit_height: text("textFitHeight"),
    })
}

fn numbers<const N: usize>(j: &Json) -> Result<[f32; N], String> {
    let a = j
        .as_array()
        .filter(|a| a.len() == N)
        .ok_or("invalid sprite array")?;
    let mut out = [0.0; N];
    for (o, v) in out.iter_mut().zip(a) {
        *o = v.as_f64().ok_or("invalid sprite number")? as f32;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selects_retina_and_preserves_query() {
        assert_eq!(
            resource_url("test://sprite?token=x#f", 2.0, ".png"),
            "test://sprite@2x.png?token=x#f"
        );
        assert_eq!(
            resource_url("test://sprite", 1.0, ".json"),
            "test://sprite.json"
        );
    }
    #[test]
    fn extracts_namespaces_metadata_and_rejects_out_of_bounds() {
        let json = br#"{"p":{"x":1,"y":0,"width":1,"height":1,"pixelRatio":2,"sdf":true,"stretchX":[[0,1]],"content":[0,0,1,1]}}"#;
        let image = DecodedImage {
            width: 2,
            height: 1,
            pixels: vec![0, 0, 0, 0, 255, 0, 0, 128],
        };
        let icons = parse_icons("extra", json).unwrap();
        assert_eq!(icons[0].name, "extra:p");
        assert!(icons[0].metadata.sdf);
        assert_eq!(icons[0].metadata.stretch_x, [[0.0, 1.0]]);
        assert_eq!(icons[0].metadata.content, Some([0.0, 0.0, 1.0, 1.0]));
        let images = extract(icons, image.clone()).unwrap();
        assert_eq!(images[0].name, "extra:p");
        assert_eq!(images[0].pixels, [255, 0, 0, 128]);
        assert_eq!(parse_icons("default", json).unwrap()[0].name, "p");
        // The metadata alone is fine; its rectangle is past the image's edge.
        let past = String::from_utf8_lossy(json).replace("\"x\":1", "\"x\":2");
        let icons = parse_icons("default", past.as_bytes()).unwrap();
        assert!(extract(icons, image).is_err());
        assert!(parse_icons("default", br#"{"p":{"x":0,"y":0,"width":0,"height":1}}"#).is_err());
    }

    /// The JSON's icons arrive without waiting for the image, which
    /// brings their pixels after.
    #[test]
    fn icons_go_ahead_of_the_pixels() {
        use orb_host::{FetchDone, ImageDone};
        struct Answers(Mutex<Vec<(String, FetchDone)>>);
        impl Fetch for Answers {
            fn get(
                &self,
                _: orb_host::ResourceKind,
                url: &str,
                _: &orb_host::Abort,
                done: FetchDone,
            ) {
                self.0.lock().unwrap().push((url.to_string(), done));
            }
            fn get_range(
                &self,
                _: orb_host::ResourceKind,
                _: &str,
                _: u64,
                _: u64,
                _: &orb_host::Abort,
                _: FetchDone,
            ) {
                unreachable!("sprites are read whole")
            }
        }
        struct Decode;
        impl ImageDecode for Decode {
            fn decode(&self, _: Vec<u8>, done: ImageDone) {
                done(Ok(DecodedImage {
                    width: 1,
                    height: 1,
                    pixels: vec![1, 2, 3, 4],
                }));
            }
        }
        let fetch = Answers(Mutex::new(Vec::new()));
        let heard = Arc::new(Mutex::new(Vec::new()));
        let sprite = Sprite {
            id: "default".into(),
            url: "test://s".into(),
        };
        let (icons, pixels) = (heard.clone(), heard.clone());
        load(
            &fetch,
            Arc::new(Decode),
            &sprite,
            1.0,
            &Abort::NEVER,
            Box::new(move |r| {
                icons
                    .lock()
                    .unwrap()
                    .push(format!("icons {}", r.unwrap()[0].name))
            }),
            Box::new(move |r| {
                pixels
                    .lock()
                    .unwrap()
                    .push(format!("pixels {:?}", r.unwrap()[0].pixels))
            }),
        );
        let mut answers = std::mem::take(&mut *fetch.0.lock().unwrap());
        let (json_url, json) = answers.remove(0);
        assert_eq!(json_url, "test://s.json");
        json(Ok(br#"{"a":{"x":0,"y":0,"width":1,"height":1}}"#.to_vec()));
        assert_eq!(*heard.lock().unwrap(), ["icons a"]);
        let (_, png) = answers.remove(0);
        png(Ok(Vec::new()));
        assert_eq!(*heard.lock().unwrap(), ["icons a", "pixels [1, 2, 3, 4]"]);
    }
}
