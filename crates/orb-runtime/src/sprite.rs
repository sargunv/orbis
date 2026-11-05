//! Sprite I/O stays above the scene. Both resources load concurrently;
//! decoding is a host service, and the render thread receives straight RGBA8.

use std::sync::{Arc, Mutex};

use orb_expr::json::{self, Json};
use orb_host::{DecodedImage, Fetch, ImageDecode};
use orb_render::ImageMetadata;
use orb_v8::Sprite;

#[derive(Debug)]
pub(crate) struct SpriteImage {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    pub metadata: ImageMetadata,
}

type Done = Box<dyn FnOnce(Result<Vec<SpriteImage>, String>) + Send>;
type Fetched = Result<Vec<u8>, String>;
struct Pending {
    resources: [Option<Fetched>; 2],
    done: Option<Done>,
}

pub(crate) fn load(
    fetch: &dyn Fetch,
    decode: Arc<dyn ImageDecode>,
    sprite: &Sprite,
    pixel_ratio: f32,
    done: Done,
) {
    let pending = Arc::new(Mutex::new(Pending {
        resources: [None, None],
        done: Some(done),
    }));
    for (index, extension) in [".json", ".png"].iter().enumerate() {
        let pending = pending.clone();
        let decode = decode.clone();
        let id = sprite.id.clone();
        let url = resource_url(&sprite.url, pixel_ratio, extension);
        fetch.get(
            &url,
            Box::new(move |result| {
                let ready = {
                    let mut p = pending.lock().expect("sprite load poisoned");
                    p.resources[index] = Some(result.map_err(|e| e.to_string()));
                    if p.resources.iter().all(Option::is_some) {
                        Some((
                            p.resources[0].take().unwrap(),
                            p.resources[1].take().unwrap(),
                            p.done.take().unwrap(),
                        ))
                    } else {
                        None
                    }
                };
                let Some((json, image, done)) = ready else {
                    return;
                };
                match json.and_then(|json| image.map(|image| (json, image))) {
                    Err(e) => done(Err(e)),
                    Ok((json, image)) => decode.decode(
                        image,
                        Box::new(move |image| {
                            done(image.and_then(|image| extract(&id, &json, image)));
                        }),
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

fn extract(id: &str, bytes: &[u8], image: DecodedImage) -> Result<Vec<SpriteImage>, String> {
    let bad = || "invalid sprite metadata or rectangle".to_string();
    let text = std::str::from_utf8(bytes).map_err(|_| bad())?;
    let doc = json::parse(text).map_err(|_| bad())?;
    let members = doc.as_object().ok_or_else(bad)?;
    if image.pixels.len() as u64 != u64::from(image.width) * u64::from(image.height) * 4 {
        return Err(bad());
    }
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
        if width == 0
            || height == 0
            || u64::from(x) + u64::from(width) > u64::from(image.width)
            || u64::from(y) + u64::from(height) > u64::from(image.height)
        {
            return Err(bad());
        }
        let metadata = metadata(d).map_err(|_| bad())?;
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for row in y..y + height {
            let at = (u64::from(row) * u64::from(image.width) + u64::from(x)) as usize * 4;
            pixels.extend_from_slice(&image.pixels[at..at + width as usize * 4]);
        }
        let name = if id == "default" {
            m.key.clone()
        } else {
            format!("{id}:{}", m.key)
        };
        out.push(SpriteImage {
            name,
            width,
            height,
            pixels,
            metadata,
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
        let icons = extract("extra", json, image.clone()).unwrap();
        assert_eq!(icons[0].name, "extra:p");
        assert_eq!(icons[0].pixels, [255, 0, 0, 128]);
        assert!(icons[0].metadata.sdf);
        assert_eq!(icons[0].metadata.stretch_x, [[0.0, 1.0]]);
        assert_eq!(icons[0].metadata.content, Some([0.0, 0.0, 1.0, 1.0]));
        assert_eq!(
            extract("default", json, image.clone()).unwrap()[0].name,
            "p"
        );
        assert!(
            extract(
                "default",
                &String::from_utf8_lossy(json)
                    .replace("\"x\":1", "\"x\":2")
                    .into_bytes(),
                image
            )
            .is_err()
        );
    }
}
