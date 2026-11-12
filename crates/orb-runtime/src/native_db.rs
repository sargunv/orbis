//! MapLibre Native's offline databases, read to merge their regions into
//! this runtime's ([`crate::offline::import`]).
//!
//! Native keeps a region's definition as JSON in `regions`, and the
//! resources and tiles it holds in `resources` and `tiles`, linked by
//! `region_resources` and `region_tiles`. A body may be zlib-compressed
//! (`compressed`). Tiles keep their URL template, pixel ratio and the
//! `z/x/y` of their URL; those of a template with only `{z}`, `{x}` and
//! `{y}` go into the region's extract, and the rest are kept by URL.

use orb_crs::TileIndex;
use orb_expr::json::{self, Json};
use orb_sqlite::{Database, Pages, Row, Table};

use crate::offline::{Import, RegionDefinition};

/// The most a compressed body inflates to.
const MAX_BODY: usize = 256 << 20;

/// A MapLibre Native offline database, opened.
#[derive(Debug)]
pub struct NativeDatabase<'p, P: Pages + ?Sized> {
    pages: &'p P,
    db: Database,
    regions: Table,
    resources: Table,
    tiles: Table,
    region_resources: Table,
    region_tiles: Table,
}

fn column(table: &Table, name: &str) -> Result<usize, String> {
    table
        .column(name)
        .ok_or_else(|| format!("its {} table has no {name} column", table.name))
}

fn text(row: &Row, at: usize) -> Option<&str> {
    row.values[at].as_text()
}

/// A body as Native stored it, inflated if it was compressed.
fn body(row: &mut Row, data: usize, compressed: usize) -> Result<Option<Vec<u8>>, String> {
    let deflated = row.values[compressed].as_integer().unwrap_or(0) != 0;
    let Some(bytes) = row.values[data].take_bytes() else {
        return Ok(None);
    };
    if !deflated {
        return Ok(Some(bytes));
    }
    miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&bytes, MAX_BODY)
        .map(Some)
        .map_err(|e| format!("a compressed body didn't inflate: {e:?}"))
}

impl<'p, P: Pages + ?Sized> NativeDatabase<'p, P> {
    /// Reads the database's schema: it must have Native's offline tables.
    pub fn open(pages: &'p P) -> Result<Self, String> {
        let db = Database::open(pages).map_err(|e| e.to_string())?;
        let table = |name: &str| {
            db.table(name)
                .cloned()
                .ok_or_else(|| format!("not a MapLibre Native offline database: no {name} table"))
        };
        Ok(Self {
            regions: table("regions")?,
            resources: table("resources")?,
            tiles: table("tiles")?,
            region_resources: table("region_resources")?,
            region_tiles: table("region_tiles")?,
            pages,
            db,
        })
    }

    fn rows(
        &self,
        table: &Table,
        mut each: impl FnMut(Row) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut failed = None;
        self.db
            .rows(self.pages, table, |row| match each(row) {
                Ok(()) => Ok(true),
                Err(e) => {
                    failed = Some(e);
                    Ok(false)
                }
            })
            .map_err(|e| e.to_string())?;
        failed.map_or(Ok(()), Err)
    }

    /// Its regions: each one's id and definition.
    pub fn regions(&self) -> Result<Vec<(i64, RegionDefinition)>, String> {
        let definition = column(&self.regions, "definition")?;
        let description = column(&self.regions, "description")?;
        let mut out = Vec::new();
        self.rows(&self.regions, |row| {
            let id = row.rowid;
            let text =
                text(&row, definition).ok_or_else(|| format!("region {id} has no definition"))?;
            let mut parsed = parse_definition(text)
                .ok_or_else(|| format!("region {id}'s definition isn't one: {text}"))?;
            parsed.metadata = row.values[description]
                .as_bytes()
                .unwrap_or_default()
                .to_vec();
            out.push((id, parsed));
            Ok(())
        })?;
        Ok(out)
    }

    /// Writes region `region`'s resources and tiles into `import`; their
    /// bodies' bytes, and how many there were.
    pub fn copy(&self, region: i64, import: &Import) -> Result<(u64, u64), String> {
        let (mut bytes, mut count) = (0u64, 0u64);
        let linked = |table: &Table, key: &str| -> Result<Vec<i64>, String> {
            let (region_id, id) = (column(table, "region_id")?, column(table, key)?);
            let mut ids = Vec::new();
            self.rows(table, |row| {
                if row.values[region_id].as_integer() == Some(region) {
                    ids.extend(row.values[id].as_integer());
                }
                Ok(())
            })?;
            Ok(ids)
        };
        let resources = &self.resources;
        let (url, data, compressed) = (
            column(resources, "url")?,
            column(resources, "data")?,
            column(resources, "compressed")?,
        );
        for id in linked(&self.region_resources, "resource_id")? {
            let Some(mut row) = self
                .db
                .row(self.pages, resources, id)
                .map_err(|e| e.to_string())?
            else {
                continue;
            };
            let Some(link) = text(&row, url).map(str::to_string) else {
                continue;
            };
            if let Some(body) = body(&mut row, data, compressed)? {
                bytes += body.len() as u64;
                count += 1;
                import.resource(&link, &body);
            }
        }
        let tiles = &self.tiles;
        let [template, ratio, z, x, y, data, compressed] = [
            "url_template",
            "pixel_ratio",
            "z",
            "x",
            "y",
            "data",
            "compressed",
        ]
        .map(|c| column(tiles, c));
        let (template, ratio, z, x, y, data, compressed) =
            (template?, ratio?, z?, x?, y?, data?, compressed?);
        for id in linked(&self.region_tiles, "tile_id")? {
            let Some(mut row) = self
                .db
                .row(self.pages, tiles, id)
                .map_err(|e| e.to_string())?
            else {
                continue;
            };
            let int = |at: usize| row.values[at].as_integer();
            let (Some(z), Some(x), Some(y)) = (int(z), int(x), int(y)) else {
                continue;
            };
            let (Ok(z), Ok(x), Ok(y)) = (u8::try_from(z), u32::try_from(x), u32::try_from(y))
            else {
                continue;
            };
            if z > 30 || x >> z != 0 || y >> z != 0 {
                continue;
            }
            let ratio = int(ratio).unwrap_or(1);
            let Some(template) = text(&row, template).map(str::to_string) else {
                continue;
            };
            let Some(body) = body(&mut row, data, compressed)? else {
                continue;
            };
            bytes += body.len() as u64;
            count += 1;
            let rest = template
                .replace("{z}", "")
                .replace("{x}", "")
                .replace("{y}", "");
            if rest.contains('{') {
                import.resource(&tile_url(&template, ratio, z, x, y), &body);
            } else {
                import.tile(&template, TileIndex::new(z, x, y), body);
            }
        }
        Ok((bytes, count))
    }
}

/// A tile's URL from its template, as MapLibre Native makes it.
fn tile_url(template: &str, ratio: i64, z: u8, x: u32, y: u32) -> String {
    let quadkey: String = (1..=z)
        .rev()
        .map(|i| {
            let mask = 1 << (i - 1);
            char::from(b'0' + u8::from(x & mask != 0) + 2 * u8::from(y & mask != 0))
        })
        .collect();
    template
        .replace("{z}", &z.to_string())
        .replace("{x}", &x.to_string())
        .replace("{y}", &y.to_string())
        .replace("{ratio}", if ratio > 1 { "@2x" } else { "" })
        .replace("{prefix}", &format!("{:x}{:x}", x % 16, y % 16))
        .replace("{quadkey}", &quadkey)
}

/// A region definition as MapLibre Native encodes it: a tile pyramid's
/// `bounds` (south, west, north, east) or a `geometry`, with the style,
/// zooms (`max_zoom` left out for no limit), pixel ratio and whether it
/// holds ideographs.
fn parse_definition(text: &str) -> Option<RegionDefinition> {
    let doc = json::parse(text).ok()?;
    let number = |key: &str| doc.get(key).and_then(Json::as_f64);
    let style_url = doc.get("style_url")?.as_str()?.to_string();
    let min_zoom = number("min_zoom").unwrap_or(0.0);
    // Without a limit, as far as any source's tiles go.
    let max_zoom = number("max_zoom").unwrap_or(f64::from(MAX_ZOOM));
    let geometry = doc.get("geometry").map(|g| g.to_string());
    let bounds = match (&geometry, doc.get("bounds").and_then(Json::as_array)) {
        (_, Some(bounds)) => {
            let b: Vec<f64> = bounds.iter().filter_map(Json::as_f64).collect();
            let [south, west, north, east] = b[..] else {
                return None;
            };
            [west, south, east, north]
        }
        (Some(_), None) => geometry_bounds(doc.get("geometry")?)?,
        (None, None) => return None,
    };
    Some(RegionDefinition {
        style_url,
        bounds,
        zooms: [min_zoom, max_zoom],
        pixel_ratio: number("pixel_ratio").unwrap_or(1.0) as f32,
        include_ideographs: doc
            .get("include_ideographs")
            .and_then(Json::as_bool)
            .unwrap_or(false),
        geometry,
        ..RegionDefinition::default()
    })
}

/// The zoom a region without a limit stands at.
const MAX_ZOOM: u8 = 25;

/// The west, south, east and north of a GeoJSON geometry's positions.
fn geometry_bounds(geometry: &Json) -> Option<[f64; 4]> {
    fn visit(value: &Json, out: &mut [f64; 4]) {
        let Some(items) = value.as_array() else {
            if let Some(members) = value.get("coordinates") {
                visit(members, out);
            }
            if let Some(geometries) = value.get("geometries").and_then(Json::as_array) {
                geometries.iter().for_each(|g| visit(g, out));
            }
            return;
        };
        match (
            items.first().and_then(Json::as_f64),
            items.get(1).and_then(Json::as_f64),
        ) {
            (Some(lon), Some(lat)) => {
                *out = [
                    out[0].min(lon),
                    out[1].min(lat),
                    out[2].max(lon),
                    out[3].max(lat),
                ];
            }
            _ => items.iter().for_each(|i| visit(i, out)),
        }
    }
    let mut out = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    visit(geometry, &mut out);
    out[0].is_finite().then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definitions_read_as_native_writes_them() {
        let d = parse_definition(
            r#"{"style_url":"custom://s.json","bounds":[1.0,2.0,3.0,4.0],"min_zoom":0.0,"max_zoom":0.0,"pixel_ratio":2.0,"include_ideographs":true}"#,
        )
        .unwrap();
        assert_eq!(d.bounds, [2.0, 1.0, 4.0, 3.0]);
        assert_eq!(
            (d.zooms, d.pixel_ratio, d.include_ideographs),
            ([0.0, 0.0], 2.0, true)
        );
        let g = parse_definition(
            r#"{"style_url":"s","geometry":{"type":"LineString","coordinates":[[2,1],[4,3]]},"min_zoom":5,"pixel_ratio":1}"#,
        )
        .unwrap();
        assert_eq!(g.bounds, [2.0, 1.0, 4.0, 3.0]);
        assert_eq!(g.zooms, [5.0, 25.0]);
        assert!(g.geometry.unwrap().contains("LineString"));
        assert!(parse_definition(r#"{"bounds":[1,2,3,4]}"#).is_none());
    }

    #[test]
    fn tile_urls_fill_natives_tokens() {
        assert_eq!(
            tile_url(
                "https://{prefix}.x/{z}/{x}/{y}{ratio}.png?q={quadkey}",
                2,
                3,
                5,
                2
            ),
            "https://52.x/3/5/2@2x.png?q=121"
        );
    }
}
