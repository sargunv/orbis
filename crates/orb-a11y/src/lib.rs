//! The semantic snapshot (plan §13, M2): what a placement commit shows, as
//! assistive technology needs it. Each shown label or icon becomes a
//! [`Node`]: its text, a role, its importance, its bounds in the viewport,
//! and its bearing and distance from the view's center. A reading order and
//! a summary of the region go with them. M10 maps the snapshot to AccessKit,
//! platform accessibility trees and an ARIA list; nothing here depends on
//! them.

#![forbid(unsafe_code)]

use std::fmt::Write;

/// What a node is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    /// A label along a line: a road, a river.
    Road,
    /// A point with an icon: a shop, a station.
    Poi,
    /// Any other point label: a place, a park, an address.
    Label,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Road => "road",
            Self::Poi => "poi",
            Self::Label => "label",
        }
    }
}

/// One shown label or icon.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// What it reads: the label's text as the data has it (before
    /// `text-transform` and shaping), or an icon's image name.
    pub text: String,
    pub role: Role,
    /// The style layer it belongs to.
    pub layer: String,
    /// The layer's place in the style, topmost first: labels of upper
    /// layers are placed first, so they matter more.
    pub layer_rank: u32,
    /// `symbol-sort-key`, where the layer sorts by it: lower places first.
    pub sort_key: Option<f64>,
    /// Viewport pixels: left, top, right, bottom.
    pub bounds: [f64; 4],
    /// Degrees clockwise from north, from the view's center.
    pub bearing: f64,
    /// Meters from the view's center.
    pub distance: f64,
}

/// A region summary: where the view is and what it shows most.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Summary {
    /// Longitude and latitude of the view's center.
    pub center: [f64; 2],
    pub zoom: f64,
    /// Shown nodes per role.
    pub counts: Vec<(Role, u32)>,
    /// The first few texts in reading order.
    pub highlights: Vec<String>,
}

/// A placement commit's semantic snapshot.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snapshot {
    /// Which commit it describes, counting from 1.
    pub commit: u64,
    pub nodes: Vec<Node>,
    /// Indices into `nodes`: most important first (upper layers, then
    /// lower sort keys), then top to bottom and left to right.
    pub reading_order: Vec<u32>,
    pub summary: Summary,
}

/// How many texts a summary highlights.
const HIGHLIGHTS: usize = 5;

impl Snapshot {
    /// Assembles a snapshot from the shown nodes. A text drawn twice at the
    /// same place (a label on two tiles' shared edge) is one node.
    pub fn new(commit: u64, mut nodes: Vec<Node>, center: [f64; 2], zoom: f64) -> Self {
        nodes.dedup_by(|a, b| a.text == b.text && a.role == b.role && a.bounds == b.bounds);
        let mut order: Vec<u32> = (0..nodes.len() as u32).collect();
        order.sort_by(|&a, &b| {
            let (a, b) = (&nodes[a as usize], &nodes[b as usize]);
            a.layer_rank
                .cmp(&b.layer_rank)
                .then(
                    a.sort_key
                        .unwrap_or(0.0)
                        .total_cmp(&b.sort_key.unwrap_or(0.0)),
                )
                .then(a.bounds[1].total_cmp(&b.bounds[1]))
                .then(a.bounds[0].total_cmp(&b.bounds[0]))
        });
        let mut counts: Vec<(Role, u32)> = Vec::new();
        for node in &nodes {
            match counts.iter_mut().find(|(r, _)| *r == node.role) {
                Some((_, n)) => *n += 1,
                None => counts.push((node.role, 1)),
            }
        }
        counts.sort();
        let mut highlights: Vec<String> = Vec::new();
        for &i in &order {
            let text = &nodes[i as usize].text;
            if highlights.len() == HIGHLIGHTS {
                break;
            }
            if !highlights.contains(text) {
                highlights.push(text.clone());
            }
        }
        Self {
            commit,
            nodes,
            reading_order: order,
            summary: Summary {
                center,
                zoom,
                counts,
                highlights,
            },
        }
    }

    /// The snapshot as JSON, for hosts that hand it on (an ARIA list, a
    /// test).
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        let _ = write!(out, "{{\"commit\":{},\"nodes\":[", self.commit);
        for (i, n) in self.nodes.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str("{\"text\":");
            string(&mut out, &n.text);
            out.push_str(",\"role\":");
            string(&mut out, n.role.as_str());
            out.push_str(",\"layer\":");
            string(&mut out, &n.layer);
            let _ = write!(out, ",\"layerRank\":{}", n.layer_rank);
            out.push_str(",\"sortKey\":");
            number(&mut out, n.sort_key);
            out.push_str(",\"bounds\":[");
            for (j, v) in n.bounds.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                number(&mut out, Some(*v));
            }
            out.push_str("],\"bearing\":");
            number(&mut out, Some(n.bearing));
            out.push_str(",\"distance\":");
            number(&mut out, Some(n.distance));
            out.push('}');
        }
        out.push_str("],\"readingOrder\":[");
        for (i, v) in self.reading_order.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(out, "{v}");
        }
        let s = &self.summary;
        out.push_str("],\"summary\":{\"center\":[");
        number(&mut out, Some(s.center[0]));
        out.push(',');
        number(&mut out, Some(s.center[1]));
        out.push_str("],\"zoom\":");
        number(&mut out, Some(s.zoom));
        out.push_str(",\"counts\":{");
        for (i, (role, n)) in s.counts.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            string(&mut out, role.as_str());
            let _ = write!(out, ":{n}");
        }
        out.push_str("},\"highlights\":[");
        for (i, h) in s.highlights.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            string(&mut out, h);
        }
        out.push_str("]}}");
        out
    }
}

fn string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
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
}

fn number(out: &mut String, v: Option<f64>) {
    match v {
        Some(v) if v.is_finite() => {
            // Whole pixels and meters to a hundredth are plenty.
            let _ = write!(out, "{}", (v * 100.0).round() / 100.0);
        }
        _ => out.push_str("null"),
    }
}

/// The great-circle bearing (degrees clockwise from north) and distance
/// (meters) from `from` to `to`, longitude and latitude in degrees.
pub fn bearing_distance(from: [f64; 2], to: [f64; 2]) -> (f64, f64) {
    const EARTH_RADIUS: f64 = 6_371_008.8;
    let (l1, p1) = (from[0].to_radians(), from[1].to_radians());
    let (l2, p2) = (to[0].to_radians(), to[1].to_radians());
    let dl = l2 - l1;
    let y = dl.sin() * p2.cos();
    let x = p1.cos() * p2.sin() - p1.sin() * p2.cos() * dl.cos();
    let bearing = y.atan2(x).to_degrees().rem_euclid(360.0);
    let h = ((p2 - p1) / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    (bearing, 2.0 * EARTH_RADIUS * h.sqrt().asin())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(text: &str, role: Role, rank: u32, key: Option<f64>, top: f64) -> Node {
        Node {
            text: text.into(),
            role,
            layer: "l".into(),
            layer_rank: rank,
            sort_key: key,
            bounds: [0.0, top, 10.0, top + 10.0],
            bearing: 0.0,
            distance: 0.0,
        }
    }

    #[test]
    fn reads_upper_layers_and_lower_keys_first() {
        let s = Snapshot::new(
            3,
            vec![
                node("Street", Role::Road, 1, None, 5.0),
                node("Cafe", Role::Poi, 0, Some(2.0), 50.0),
                node("City", Role::Label, 0, Some(1.0), 90.0),
                node("City", Role::Label, 0, Some(1.0), 90.0),
            ],
            [31.2, 30.0],
            13.0,
        );
        assert_eq!(s.nodes.len(), 3, "a label on two tiles is one node");
        assert_eq!(s.reading_order, vec![2, 1, 0]);
        assert_eq!(
            s.summary.counts,
            vec![(Role::Road, 1), (Role::Poi, 1), (Role::Label, 1)]
        );
        assert_eq!(s.summary.highlights, vec!["City", "Cafe", "Street"]);
        let json = s.to_json();
        assert!(
            json.starts_with("{\"commit\":3,\"nodes\":[{\"text\":\"Street\""),
            "{json}"
        );
        assert!(
            json.ends_with("\"highlights\":[\"City\",\"Cafe\",\"Street\"]}}"),
            "{json}"
        );
    }

    #[test]
    fn bearings_and_distances_follow_the_great_circle() {
        let (bearing, distance) = bearing_distance([0.0, 0.0], [0.0, 1.0]);
        assert!((bearing - 0.0).abs() < 1e-9);
        assert!((distance - 111_195.0).abs() < 1.0, "{distance}");
        let (bearing, _) = bearing_distance([0.0, 0.0], [1.0, 0.0]);
        assert!((bearing - 90.0).abs() < 1e-9);
        let (bearing, _) = bearing_distance([0.0, 0.0], [-1.0, -1.0]);
        assert!(bearing > 180.0 && bearing < 270.0);
    }
}
