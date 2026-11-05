//! Batteries: engine-drawn markers and the location layer,
//! built on the map's public API alone. Each is a managed GeoJSON source
//! with symbol and circle layers on top of the style, and images of its
//! own; data changes are `setSourceData` edits, so nothing blinks, and
//! [`Batteries::sync`] puts them back when the style is replaced.
//!
//! Markers are features with the marker's id: hit tests and rendered
//! queries name them in the `orbis-markers` layer. Location fixes are
//! data only the app has: it pushes them with
//! [`Batteries::set_location`].

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::{Map, StyleImage, Transaction};

/// The markers' source and layer.
pub const MARKERS: &str = "orbis-markers";
/// The location layer's source, and the prefix of its layers.
pub const LOCATION: &str = "orbis-location";
/// The built-in marker image: a pin, anchored at its point.
pub const PIN: &str = "orbis-pin";
const HEADING: &str = "orbis-location-heading";

/// A marker: an image at a point.
#[derive(Clone, Debug, PartialEq)]
pub struct Marker {
    /// Longitude and latitude, degrees.
    pub position: [f64; 2],
    /// An image of the style (`add_image`, or its sprite's); `None` for
    /// the built-in pin.
    pub image: Option<String>,
    /// Which part of the image sits on the point (`icon-anchor`):
    /// `"center"`, `"bottom"`, `"top-left"` and so on; the image's default
    /// (the pin's tip, the others' centre) when `None`.
    pub anchor: Option<String>,
}

/// A location fix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Location {
    /// Longitude and latitude, degrees.
    pub position: [f64; 2],
    /// The horizontal accuracy, metres (the radius drawn around the
    /// point); 0 for none.
    pub accuracy: f64,
    /// Degrees clockwise from north the device faces; `None` hides the
    /// heading.
    pub heading: Option<f64>,
}

/// The markers and location of one map.
#[derive(Debug, Default)]
pub struct Batteries {
    markers: BTreeMap<u64, Marker>,
    next_marker: u64,
    location: Option<Location>,
}

impl Batteries {
    /// Adds a marker; its id names it to the other calls and in hits.
    pub fn add_marker(&mut self, map: &mut Map, marker: Marker) -> u64 {
        self.next_marker += 1;
        self.markers.insert(self.next_marker, marker);
        self.publish_markers(map);
        self.next_marker
    }

    /// Replaces a marker; an error if there is none with this id.
    pub fn update_marker(&mut self, map: &mut Map, id: u64, marker: Marker) -> Result<(), String> {
        let slot = self
            .markers
            .get_mut(&id)
            .ok_or_else(|| format!("no marker {id}"))?;
        *slot = marker;
        self.publish_markers(map);
        Ok(())
    }

    /// Removes a marker; an error if there is none with this id.
    pub fn remove_marker(&mut self, map: &mut Map, id: u64) -> Result<(), String> {
        self.markers
            .remove(&id)
            .ok_or_else(|| format!("no marker {id}"))?;
        self.publish_markers(map);
        Ok(())
    }

    /// Shows a location fix, or (`None`) hides the location layer.
    pub fn set_location(&mut self, map: &mut Map, location: Option<Location>) {
        self.location = location;
        if installed(map, LOCATION) {
            let data = location_data(self.location.as_ref());
            edit(map, &format!(r#"[["setSourceData","{LOCATION}",{data}]]"#));
        } else {
            self.sync(map);
        }
    }

    /// Puts the sources, layers and images back after the style changed;
    /// call before each frame. Nothing to do while they're in place, or
    /// before a style has loaded.
    pub fn sync(&self, map: &mut Map) {
        if map.style_doc().is_none() {
            return;
        }
        if self.location.is_some() && !installed(map, LOCATION) {
            add_image(map, HEADING, heading_image());
            let data = location_data(self.location.as_ref());
            edit(
                map,
                &format!(
                    r##"[["addSource","{LOCATION}",{{"type":"geojson","data":{data}}}],
                ["addLayer",{{"id":"{LOCATION}-accuracy","type":"circle","source":"{LOCATION}",
                  "paint":{{"circle-radius":["interpolate",["exponential",2],["zoom"],
                      0,["get","r0"],24,["*",["get","r0"],16777216]],
                    "circle-color":"#1a73e8","circle-opacity":0.15,
                    "circle-stroke-color":"#1a73e8","circle-stroke-opacity":0.3,"circle-stroke-width":1,
                    "circle-pitch-alignment":"map","circle-pitch-scale":"map"}}}}],
                ["addLayer",{{"id":"{LOCATION}-heading","type":"symbol","source":"{LOCATION}",
                  "filter":["has","heading"],
                  "layout":{{"icon-image":"{HEADING}","icon-rotate":["get","heading"],
                    "icon-rotation-alignment":"map","icon-pitch-alignment":"map",
                    "icon-allow-overlap":true,"icon-ignore-placement":true}},
                  "paint":{{"icon-opacity":0.6}}}}],
                ["addLayer",{{"id":"{LOCATION}","type":"circle","source":"{LOCATION}",
                  "paint":{{"circle-radius":7,"circle-color":"#1a73e8",
                    "circle-stroke-color":"#ffffff","circle-stroke-width":3,
                    "circle-pitch-alignment":"map"}}}}]]"##
                ),
            );
        }
        if !self.markers.is_empty() && !installed(map, MARKERS) {
            add_image(map, PIN, pin_image());
            let data = markers_data(&self.markers);
            edit(
                map,
                &format!(
                    r#"[["addSource","{MARKERS}",{{"type":"geojson","data":{data}}}],
                ["addLayer",{{"id":"{MARKERS}","type":"symbol","source":"{MARKERS}",
                  "layout":{{"icon-image":["get","image"],"icon-anchor":["get","anchor"],
                    "icon-allow-overlap":true,"symbol-sort-key":["get","order"]}}}}]]"#
                ),
            );
        }
    }

    fn publish_markers(&self, map: &mut Map) {
        if installed(map, MARKERS) {
            let data = markers_data(&self.markers);
            edit(map, &format!(r#"[["setSourceData","{MARKERS}",{data}]]"#));
        } else {
            self.sync(map);
        }
    }
}

/// Whether the style has the source `id`.
fn installed(map: &Map, id: &str) -> bool {
    map.style_doc()
        .is_some_and(|doc| doc.style().sources.iter().any(|s| s.id == id))
}

/// Applies an edit the batteries made: its JSON and ids are their own, so
/// a failure is a bug.
fn edit(map: &mut Map, commands: &str) {
    let tx = Transaction::from_commands(commands).expect("battery commands parse");
    if let Err(e) = map.edit(&tx) {
        debug_assert!(false, "battery edit failed: {e}");
    }
}

fn add_image(map: &mut Map, name: &str, image: StyleImage) {
    if !map.has_image(name) {
        let _ = map.add_image(name, image);
    }
}

/// A JSON string literal.
fn string(text: &str) -> String {
    let mut out = String::from('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn markers_data(markers: &BTreeMap<u64, Marker>) -> String {
    let features: Vec<String> = markers
        .iter()
        .enumerate()
        .map(|(order, (id, m))| {
            let (image, anchor) = match &m.image {
                Some(image) => (image.as_str(), m.anchor.as_deref().unwrap_or("center")),
                None => (PIN, m.anchor.as_deref().unwrap_or("bottom")),
            };
            format!(
                r#"{{"type":"Feature","id":{id},"properties":{{"image":{},"anchor":{},"order":{order}}},"geometry":{{"type":"Point","coordinates":[{},{}]}}}}"#,
                string(image),
                string(anchor),
                m.position[0],
                m.position[1],
            )
        })
        .collect();
    format!(
        r#"{{"type":"FeatureCollection","features":[{}]}}"#,
        features.join(",")
    )
}

/// The fix as one point, with the accuracy radius in pixels at zoom 0 at
/// its latitude (`r0`), which the accuracy circle scales by zoom.
fn location_data(location: Option<&Location>) -> String {
    let Some(l) = location else {
        return r#"{"type":"FeatureCollection","features":[]}"#.into();
    };
    let metres_per_px = 40_075_016.686 * l.position[1].to_radians().cos() / 512.0;
    let mut properties = format!(r#""r0":{}"#, l.accuracy.max(0.0) / metres_per_px);
    if let Some(heading) = l.heading {
        let _ = write!(properties, r#","heading":{heading}"#);
    }
    format!(
        r#"{{"type":"Feature","properties":{{{properties}}},"geometry":{{"type":"Point","coordinates":[{},{}]}}}}"#,
        l.position[0], l.position[1]
    )
}

/// An image drawn at twice its size in pixels by `coverage(x, y)` (in
/// logical pixels from the top left), 4×4 samples a pixel.
fn draw(width: u32, height: u32, colour: impl Fn(f64, f64) -> Option<[u8; 3]>) -> StyleImage {
    let (w, h) = (width * 2, height * 2);
    let mut pixels = vec![0; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let (mut sum, mut hits) = ([0u32; 3], 0u32);
            for sy in 0..4 {
                for sx in 0..4 {
                    let px = (f64::from(x) + (f64::from(sx) + 0.5) / 4.0) / 2.0;
                    let py = (f64::from(y) + (f64::from(sy) + 0.5) / 4.0) / 2.0;
                    if let Some(c) = colour(px, py) {
                        for (s, c) in sum.iter_mut().zip(c) {
                            *s += u32::from(c);
                        }
                        hits += 1;
                    }
                }
            }
            let at = ((y * w + x) * 4) as usize;
            for (i, s) in sum.iter().enumerate() {
                pixels[at + i] = s.checked_div(hits).unwrap_or(0) as u8;
            }
            pixels[at + 3] = (hits * 255 / 16) as u8;
        }
    }
    StyleImage::with_options(w, h, pixels, r#"{"pixelRatio":2}"#).expect("a valid image")
}

/// A map pin, 24×34: a red head with a white dot over a point.
fn pin_image() -> StyleImage {
    let (cx, cy, r) = (12.0, 12.0, 11.0);
    draw(24, 34, |x, y| {
        let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
        // The point: tangents from the tip at (12, 33) to the head.
        let tip = 33.0;
        let below = y > cy && {
            let half = r * (tip - y) / (tip - cy);
            (x - cx).abs() <= half.max(0.0) * 0.82
        };
        if d <= 4.0 {
            Some([255, 255, 255])
        } else if d <= r || below {
            Some([0xd9, 0x30, 0x25])
        } else {
            None
        }
    })
}

/// The heading cone, 40×40: a translucent wedge above the centre, which
/// the dot covers, pointing north before rotation.
fn heading_image() -> StyleImage {
    let (cx, cy) = (20.0, 20.0);
    draw(40, 40, |x, y| {
        let (dx, dy) = (x - cx, cy - y);
        let (d, angle) = ((dx * dx + dy * dy).sqrt(), dx.atan2(dy).to_degrees().abs());
        (dy > 0.0 && d <= 19.0 && angle <= 35.0).then_some([0x1a, 0x73, 0xe8])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_escapes_names_and_scales_accuracy() {
        let mut markers = BTreeMap::new();
        markers.insert(
            3,
            Marker {
                position: [1.5, -2.0],
                image: Some("a \"quoted\" image".into()),
                anchor: None,
            },
        );
        let data = markers_data(&markers);
        assert!(orb_expr::json::parse(&data).is_ok(), "{data}");
        assert!(data.contains(r#""id":3"#) && data.contains(r#""anchor":"center""#));
        // 1 km at the equator is 0.0128 pixels at zoom 0 (a 512-pixel
        // world), 12.8 at zoom 10.
        let fix = Location {
            position: [0.0, 0.0],
            accuracy: 1000.0,
            heading: Some(90.0),
        };
        let json = orb_expr::json::parse(&location_data(Some(&fix))).unwrap();
        let properties = json.get("properties").unwrap();
        let r0 = properties.get("r0").unwrap().as_f64().unwrap();
        assert!((r0 - 0.012_776).abs() < 1e-5, "{r0}");
        assert_eq!(properties.get("heading").unwrap().as_f64(), Some(90.0));
    }
}
