//! Joining lines that continue each other under the same label (GL JS
//! `merge_lines.ts`, the linear version of maplibre-gl-js#8436).

use rustc_hash::FxHashMap as HashMap;

use crate::geometry::Point;

/// A feature kept by [`merge_lines`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Merged {
    /// Index of the feature in the input.
    pub(crate) feature: usize,
    /// Its first line joined with the lines merged into it, if any were.
    pub(crate) joined: Option<Vec<Point>>,
}

/// One end of a line, keyed as GL JS's `${text}:${x}:${y}`.
type EndKey<'a> = (&'a str, u64, u64);

fn end_key<'a>(text: &'a str, p: Point) -> EndKey<'a> {
    // `+ 0.0` folds -0 into 0, as the number's string form does.
    (text, (p.x + 0.0).to_bits(), (p.y + 0.0).to_bits())
}

struct Line {
    feature: usize,
    first: usize,
    last: usize,
}

/// Merges, in input order, the first lines of features whose ends meet and
/// whose label text is the same. `firsts[i]` is feature `i`'s first line
/// and `texts[i]` its label text, empty for none. Returns the surviving
/// features in GL JS's order.
pub(crate) fn merge_lines(texts: &[&str], firsts: &[&[Point]]) -> Vec<Merged> {
    let mut left_index: HashMap<EndKey<'_>, usize> = HashMap::default();
    let mut right_index: HashMap<EndKey<'_>, usize> = HashMap::default();
    let mut merged: Vec<Line> = Vec::new();
    // Each feature's line is one chunk, numbered by feature.
    let mut next: Vec<Option<usize>> = vec![None; firsts.len()];
    let mut removed = vec![false; firsts.len()];

    let right_key = |text, merged: &[Line], i: usize| {
        let points = firsts[merged[i].last];
        end_key(text, points[points.len() - 1])
    };

    for (feature, &first) in firsts.iter().enumerate() {
        let line = Line {
            feature,
            first: feature,
            last: feature,
        };
        let text = texts[feature];
        if text.is_empty() {
            merged.push(line);
            continue;
        }
        let lk = end_key(text, first[0]);
        let rk = end_key(text, first[first.len() - 1]);

        let from_right = right_index.get(&lk).copied();
        let from_left = left_index.get(&rk).copied();
        match (from_right, from_left) {
            (Some(i), Some(j)) if i != j => {
                // Lines with this text meet both ends: join all three.
                left_index.remove(&rk);
                left_index.insert(lk, j);
                next[line.last] = Some(merged[j].first);
                merged[j].first = line.first;

                right_index.remove(&lk);
                right_index.insert(rk, i);
                next[merged[i].last] = Some(merged[j].first);
                merged[i].last = merged[j].last;

                left_index.remove(&lk);
                right_index.remove(&rk);
                let key = right_key(text, &merged, i);
                right_index.insert(key, i);
                removed[merged[j].feature] = true;
            }
            (Some(i), _) => {
                right_index.remove(&lk);
                right_index.insert(rk, i);
                next[merged[i].last] = Some(line.first);
                merged[i].last = line.last;
            }
            (None, Some(j)) => {
                left_index.remove(&rk);
                left_index.insert(lk, j);
                next[line.last] = Some(merged[j].first);
                merged[j].first = line.first;
            }
            (None, None) => {
                let i = merged.len();
                merged.push(line);
                left_index.insert(lk, i);
                right_index.insert(rk, i);
            }
        }
    }

    merged
        .iter()
        .filter(|line| !removed[line.feature])
        .map(|line| Merged {
            feature: line.feature,
            joined: next[line.first].map(|_| join_chunks(firsts, &next, line.first)),
        })
        .collect()
}

fn join_chunks(firsts: &[&[Point]], next: &[Option<usize>], first: usize) -> Vec<Point> {
    let mut points = firsts[first].to_vec();
    let mut chunk = next[first];
    while let Some(c) = chunk {
        points.extend_from_slice(&firsts[c][1..]);
        chunk = next[c];
    }
    points
}
