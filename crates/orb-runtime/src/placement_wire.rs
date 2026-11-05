//! The placement lane's messages as bytes, for hosts that place in a worker
//! of their own (the web): a bucket's tables, a frame, and a commit. Both
//! ends are this build, so the layout carries no version.

use bytemuck::Pod;
use orb_symbol::{
    BucketStates, Camera, Committed, Frame, LayerFrame, LayerParams, Overlap, PlacedTile,
    PlacementTables, SizeKind, TileGlobe, TileId, TileView, ZoomSize,
};
use orb_text::WritingMode;

/// A tile's globe: a flag, then its series and camera.
fn write_globe(w: &mut Writer, globe: Option<&TileGlobe>) {
    let Some(g) = globe else {
        w.bool(false);
        return;
    };
    w.bool(true);
    let s = &*g.series;
    for v in s.anchor.to_array() {
        w.f64(v);
    }
    w.u32(s.cells);
    w.f64(s.max_segment);
    w.f64(s.scale);
    for p in &s.patches {
        for v in p.points.iter().flat_map(|v| v.to_array()) {
            w.f64(v);
        }
    }
    for v in g.target.to_array().into_iter().chain(g.eye.to_array()) {
        w.f64(v);
    }
    for v in g.view_proj.to_cols_array() {
        w.f64(v);
    }
}

fn read_globe(r: &mut Reader<'_>) -> Result<Option<TileGlobe>, String> {
    if !r.bool()? {
        return Ok(None);
    }
    let mut vec3 =
        || -> Result<glam::DVec3, String> { Ok(glam::DVec3::new(r.f64()?, r.f64()?, r.f64()?)) };
    let anchor = vec3()?;
    let cells = r.u32()?;
    if cells == 0 || cells > 256 {
        return Err("placement message: bad globe series".into());
    }
    let (max_segment, scale) = (r.f64()?, r.f64()?);
    let patches = (0..cells * cells)
        .map(|_| {
            let mut points = [glam::DVec3::ZERO; 9];
            for p in &mut points {
                *p = glam::DVec3::new(r.f64()?, r.f64()?, r.f64()?);
            }
            Ok(orb_crs::Patch { points })
        })
        .collect::<Result<_, String>>()?;
    let target = glam::DVec3::new(r.f64()?, r.f64()?, r.f64()?);
    let eye = glam::DVec3::new(r.f64()?, r.f64()?, r.f64()?);
    let mut m = [0.0; 16];
    for v in &mut m {
        *v = r.f64()?;
    }
    Ok(Some(TileGlobe {
        series: std::sync::Arc::new(orb_crs::Series {
            anchor,
            cells,
            patches,
            max_segment,
            scale,
        }),
        target,
        view_proj: glam::DMat4::from_cols_array(&m),
        eye,
    }))
}

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn u32(&mut self, v: u32) {
        self.0.extend(v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend(v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend(v.to_le_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.0.push(u8::from(v));
    }
    fn rows<T: Pod>(&mut self, rows: &[T]) {
        self.u32(rows.len() as u32);
        self.0.extend_from_slice(bytemuck::cast_slice(rows));
    }
    fn size(&mut self, kind: SizeKind) {
        match kind {
            SizeKind::Constant => self.u32(0),
            SizeKind::Source => self.u32(1),
            SizeKind::Camera => self.u32(2),
            SizeKind::Composite { zooms } => {
                self.u32(3);
                for z in zooms {
                    self.f64(z);
                }
            }
        }
    }
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        if self.0.len() < n {
            return Err("placement message: truncated".into());
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn bool(&mut self) -> Result<bool, String> {
        Ok(self.take(1)?[0] != 0)
    }
    fn rows<T: Pod>(&mut self) -> Result<Vec<T>, String> {
        let len = self.u32()? as usize;
        let bytes = self.take(len * std::mem::size_of::<T>())?;
        let mut rows = vec![T::zeroed(); len];
        bytemuck::cast_slice_mut::<T, u8>(&mut rows).copy_from_slice(bytes);
        Ok(rows)
    }
    fn size(&mut self) -> Result<SizeKind, String> {
        Ok(match self.u32()? {
            0 => SizeKind::Constant,
            1 => SizeKind::Source,
            2 => SizeKind::Camera,
            _ => SizeKind::Composite {
                zooms: [self.f64()?, self.f64()?],
            },
        })
    }
    fn done(&self) -> Result<(), String> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err("placement message: trailing bytes".into())
        }
    }
}

pub fn pack_tables(t: &PlacementTables) -> Vec<u8> {
    let mut w = Writer::default();
    w.rows(&t.instances);
    w.rows(&t.text_placed);
    w.rows(&t.icon_placed);
    w.rows(&t.text_glyph_offsets);
    w.rows(&t.lines);
    w.rows(&t.anchor_offsets);
    w.rows(&t.sort_key_ranges);
    w.size(t.text_size);
    w.bool(t.allow_vertical_placement);
    w.0
}

pub fn unpack_tables(bytes: &[u8]) -> Result<PlacementTables, String> {
    let mut r = Reader(bytes);
    let t = PlacementTables {
        instances: r.rows()?,
        text_placed: r.rows()?,
        icon_placed: r.rows()?,
        text_glyph_offsets: r.rows()?,
        lines: r.rows()?,
        anchor_offsets: r.rows()?,
        sort_key_ranges: r.rows()?,
        text_size: r.size()?,
        allow_vertical_placement: r.bool()?,
    };
    r.done()?;
    Ok(t)
}

fn overlap(o: Overlap) -> u32 {
    match o {
        Overlap::Never => 0,
        Overlap::Always => 1,
        Overlap::Cooperative => 2,
    }
}

fn read_overlap(v: u32) -> Overlap {
    match v {
        1 => Overlap::Always,
        2 => Overlap::Cooperative,
        _ => Overlap::Never,
    }
}

fn write_params(w: &mut Writer, p: &LayerParams) {
    w.u32(p.layer);
    w.u32(p.source);
    w.f64(p.minzoom);
    w.f64(p.maxzoom);
    w.u32(overlap(p.text_overlap));
    w.u32(overlap(p.icon_overlap));
    for flag in [
        p.text_optional,
        p.icon_optional,
        p.text_ignore_placement,
        p.icon_ignore_placement,
        p.rotate_with_map,
        p.pitch_with_map,
        p.icon_text_fit,
        p.z_order_viewport_y,
        p.draw_by_viewport_y,
        p.sort_across_tiles,
        p.height_anchor_ground,
        p.variable_placement,
    ] {
        w.bool(flag);
    }
    w.f64(p.text_padding);
    for (t, viewport) in [p.text_translate, p.icon_translate] {
        for v in t {
            w.f64(v);
        }
        w.bool(viewport);
    }
    w.u32(p.writing_modes.len() as u32);
    for m in &p.writing_modes {
        w.bool(*m == WritingMode::Vertical);
    }
}

fn read_params(r: &mut Reader<'_>) -> Result<LayerParams, String> {
    let (layer, source) = (r.u32()?, r.u32()?);
    let (minzoom, maxzoom) = (r.f64()?, r.f64()?);
    let (text_overlap, icon_overlap) = (read_overlap(r.u32()?), read_overlap(r.u32()?));
    let mut flags = [false; 12];
    for f in &mut flags {
        *f = r.bool()?;
    }
    let text_padding = r.f64()?;
    let mut translate =
        || -> Result<([f64; 2], bool), String> { Ok(([r.f64()?, r.f64()?], r.bool()?)) };
    let (text_translate, icon_translate) = (translate()?, translate()?);
    let modes = r.u32()?;
    let writing_modes = (0..modes)
        .map(|_| {
            Ok(if r.bool()? {
                WritingMode::Vertical
            } else {
                WritingMode::Horizontal
            })
        })
        .collect::<Result<_, String>>()?;
    let [
        text_optional,
        icon_optional,
        text_ignore_placement,
        icon_ignore_placement,
        rotate_with_map,
        pitch_with_map,
        icon_text_fit,
        z_order_viewport_y,
        draw_by_viewport_y,
        sort_across_tiles,
        height_anchor_ground,
        variable_placement,
    ] = flags;
    Ok(LayerParams {
        layer,
        source,
        minzoom,
        maxzoom,
        text_overlap,
        icon_overlap,
        text_optional,
        icon_optional,
        text_ignore_placement,
        icon_ignore_placement,
        rotate_with_map,
        pitch_with_map,
        icon_text_fit,
        z_order_viewport_y,
        draw_by_viewport_y,
        sort_across_tiles,
        height_anchor_ground,
        text_padding,
        text_translate,
        icon_translate,
        writing_modes,
        variable_placement,
    })
}

pub fn pack_frame(f: &Frame) -> Vec<u8> {
    let mut w = Writer::default();
    let c = &f.camera;
    for v in [
        c.width,
        c.height,
        c.zoom,
        c.bearing,
        c.pitch,
        c.roll,
        c.camera_to_center_distance,
        f.now,
        f.fade_duration,
        f.lng,
    ] {
        w.f64(v);
    }
    w.bool(f.cross_source_collisions);
    w.u32(f.layers.len() as u32);
    for layer in &f.layers {
        write_params(&mut w, &layer.params);
        w.u32(layer.tiles.len() as u32);
        for t in &layer.tiles {
            w.u64(t.bucket);
            let id = t.view.id;
            w.u32(u32::from(id.z));
            w.u32(id.x);
            w.u32(id.y);
            w.u32(id.wrap as u32);
            for v in t.view.matrix {
                w.f64(v);
            }
            write_globe(&mut w, t.view.globe.as_ref());
            w.f64(t.text_size.size);
            w.f64(t.text_size.t);
        }
    }
    w.0
}

pub fn unpack_frame(bytes: &[u8]) -> Result<Frame, String> {
    let mut r = Reader(bytes);
    let mut v = [0.0; 10];
    for x in &mut v {
        *x = r.f64()?;
    }
    let [
        width,
        height,
        zoom,
        bearing,
        pitch,
        roll,
        camera_to_center_distance,
        now,
        fade,
        lng,
    ] = v;
    let cross_source_collisions = r.bool()?;
    let layers = (0..r.u32()?)
        .map(|_| {
            let params = read_params(&mut r)?;
            let tiles = (0..r.u32()?)
                .map(|_| {
                    let bucket = r.u64()?;
                    let id = TileId {
                        z: r.u32()? as u8,
                        x: r.u32()?,
                        y: r.u32()?,
                        wrap: r.u32()? as i32,
                    };
                    let mut matrix = [0.0; 16];
                    for m in &mut matrix {
                        *m = r.f64()?;
                    }
                    let globe = read_globe(&mut r)?;
                    Ok(PlacedTile {
                        bucket,
                        view: TileView { id, matrix, globe },
                        text_size: ZoomSize {
                            size: r.f64()?,
                            t: r.f64()?,
                        },
                    })
                })
                .collect::<Result<_, String>>()?;
            Ok(LayerFrame { params, tiles })
        })
        .collect::<Result<_, String>>()?;
    r.done()?;
    Ok(Frame {
        camera: Camera {
            width,
            height,
            zoom,
            bearing,
            pitch,
            roll,
            camera_to_center_distance,
        },
        layers,
        now,
        fade_duration: fade,
        cross_source_collisions,
        lng,
        ground: None,
    })
}

pub fn pack_committed(c: &Committed) -> Vec<u8> {
    let mut w = Writer::default();
    w.f64(c.commit_time);
    w.f64(c.fade_duration);
    w.f64(c.zoom_adjustment);
    w.bool(c.placement_changed);
    w.u32(c.changed.len() as u32);
    for (bucket, wrap, s) in &c.changed {
        w.u64(*bucket);
        w.u32(*wrap as u32);
        w.rows(&s.text);
        w.rows(&s.icon);
        w.rows(&s.text_order);
        w.rows(&s.icon_order);
    }
    w.0
}

pub fn unpack_committed(bytes: &[u8]) -> Result<Committed, String> {
    let mut r = Reader(bytes);
    let commit_time = r.f64()?;
    let fade_duration = r.f64()?;
    let zoom_adjustment = r.f64()?;
    let placement_changed = r.bool()?;
    let changed = (0..r.u32()?)
        .map(|_| {
            Ok((
                r.u64()?,
                r.u32()? as i32,
                BucketStates {
                    text: r.rows()?,
                    icon: r.rows()?,
                    text_order: r.rows()?,
                    icon_order: r.rows()?,
                },
            ))
        })
        .collect::<Result<_, String>>()?;
    r.done()?;
    Ok(Committed {
        changed,
        commit_time,
        fade_duration,
        zoom_adjustment,
        placement_changed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    #[test]
    fn messages_round_trip() {
        let tables = PlacementTables {
            instances: vec![orb_symbol::SymbolInstance::zeroed(); 2],
            text_placed: vec![orb_symbol::PlacedSymbol::zeroed()],
            text_glyph_offsets: vec![1.5, -2.0],
            lines: vec![[3, -4]],
            text_size: SizeKind::Composite {
                zooms: [12.0, 13.0],
            },
            allow_vertical_placement: true,
            ..PlacementTables::default()
        };
        assert_eq!(unpack_tables(&pack_tables(&tables)).unwrap(), tables);

        let frame = Frame {
            camera: Camera {
                width: 512.0,
                height: 256.0,
                zoom: 3.5,
                bearing: 0.5,
                pitch: 0.25,
                roll: 0.125,
                camera_to_center_distance: 2.0,
            },
            layers: vec![LayerFrame {
                params: LayerParams {
                    layer: 7,
                    source: 2,
                    minzoom: 1.0,
                    maxzoom: 20.0,
                    text_overlap: Overlap::Cooperative,
                    icon_overlap: Overlap::Always,
                    text_optional: true,
                    icon_optional: false,
                    text_ignore_placement: false,
                    icon_ignore_placement: true,
                    rotate_with_map: true,
                    pitch_with_map: false,
                    icon_text_fit: true,
                    z_order_viewport_y: false,
                    draw_by_viewport_y: true,
                    sort_across_tiles: false,
                    height_anchor_ground: true,
                    text_padding: 2.0,
                    text_translate: ([1.0, -1.0], true),
                    icon_translate: ([0.0, 3.0], false),
                    writing_modes: vec![WritingMode::Vertical, WritingMode::Horizontal],
                    variable_placement: true,
                },
                tiles: vec![PlacedTile {
                    bucket: 9,
                    view: TileView {
                        id: TileId {
                            z: 4,
                            x: 5,
                            y: 6,
                            wrap: -1,
                        },
                        matrix: std::array::from_fn(|i| i as f64),
                        globe: orb_crs::sphere::series(
                            &orb_crs::TileMatrixSet::by_id("WebMercatorQuad").unwrap(),
                            orb_crs::TileIndex::new(4, 5, 6),
                            1.0 / 2048.0,
                        )
                        .map(|series| TileGlobe {
                            series: std::sync::Arc::new(series),
                            target: glam::DVec3::new(1.0, 2.0, 3.0),
                            view_proj: glam::DMat4::from_cols_array(&std::array::from_fn(|i| {
                                i as f64 * 0.5
                            })),
                            eye: glam::DVec3::new(4.0, 5.0, 6.0),
                        }),
                    },
                    text_size: ZoomSize { size: 16.0, t: 0.5 },
                }],
            }],
            now: 100.0,
            fade_duration: 300.0,
            cross_source_collisions: true,
            lng: -170.0,
            ground: None,
        };
        assert_eq!(unpack_frame(&pack_frame(&frame)).unwrap(), frame);

        let committed = Committed {
            changed: vec![(
                9,
                -1,
                BucketStates {
                    text: vec![1, 2, 3],
                    icon: vec![],
                    text_order: vec![0],
                    icon_order: vec![],
                },
            )],
            commit_time: 5.0,
            fade_duration: 300.0,
            zoom_adjustment: 0.25,
            placement_changed: true,
        };
        assert_eq!(
            unpack_committed(&pack_committed(&committed)).unwrap(),
            committed
        );
    }
}
