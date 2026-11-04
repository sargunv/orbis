//! MapLibre GL JS 2b68611a's LineAtlas SDF construction (BSD-3-Clause).
//! License terms are in `crates/orb-layout/src/LICENSE-MapLibre`.
//! A separate 256 × 512 R8 texture avoids expanding scalar SDFs into RGBA.
//! Keys use exact f32 bits; rows are stable and recycled with their namespace.

use orb_hal::*;

pub(crate) const WIDTH: u32 = 256;
const HEIGHT: u32 = 512;

#[derive(Debug)]
pub(crate) struct LineAtlas {
    pub texture: TextureId,
    used: Vec<bool>,
}

impl LineAtlas {
    pub(crate) fn new(gpu: &mut dyn Gpu) -> Self {
        Self {
            texture: gpu.create_texture(&TextureDesc {
                label: "line SDF atlas",
                width: WIDTH,
                height: HEIGHT,
                array_layers: 1,
                format: TextureFormat::R8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                sample_count: 1,
                mip_levels: 1,
            }),
            used: vec![false; HEIGHT as usize],
        }
    }

    pub(crate) fn insert(
        &mut self,
        gpu: &mut dyn Gpu,
        array: &[u32],
        round: bool,
    ) -> Result<([u32; 2], f32), &'static str> {
        let array: Vec<f64> = array
            .iter()
            .map(|&v| f64::from(f32::from_bits(v)))
            .collect();
        if array.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("invalid dash array");
        }
        let height = if round { 15 } else { 1 };
        let y = self
            .used
            .windows(height)
            .position(|rows| rows.iter().all(|v| !v))
            .ok_or("line atlas is full")?;
        let pixels = sdf(&array, round);
        self.used[y..y + height].fill(true);
        gpu.write_texture(
            &TextureRegion {
                texture: self.texture,
                x: 0,
                y: y as u32,
                layer: 0,
                width: WIDTH,
                height: height as u32,
            },
            &pixels,
        );
        Ok(([y as u32, height as u32], array.iter().sum::<f64>() as f32))
    }

    pub(crate) fn release(&mut self, [y, height]: [u32; 2]) {
        self.used[y as usize..(y + height) as usize].fill(false);
    }
}

#[derive(Clone, Copy)]
struct Range {
    left: f64,
    right: f64,
    dash: bool,
    zero: bool,
}

fn sdf(array: &[f64], round: bool) -> Vec<u8> {
    let height = if round { 15 } else { 1 };
    let mut pixels = vec![0; WIDTH as usize * height];
    let length: f64 = array.iter().sum();
    if length == 0.0 {
        return pixels;
    }
    let stretch = f64::from(WIDTH) / length;
    let mut left = if array.len() % 2 == 1 {
        -array[array.len() - 1] * stretch
    } else {
        0.0
    };
    let mut position = 0.0;
    let mut ranges = Vec::new();
    for (i, &v) in array.iter().enumerate() {
        position += v;
        let right = position * stretch;
        ranges.push(Range {
            left,
            right,
            dash: i % 2 == 0,
            zero: v == 0.0,
        });
        left = right;
    }
    if !round {
        for i in (0..ranges.len()).rev() {
            if ranges[i].zero {
                ranges.remove(i);
            } else if i + 1 < ranges.len() && ranges[i].dash == ranges[i + 1].dash {
                ranges[i + 1].left = ranges[i].left;
                ranges.remove(i);
            }
        }
        if ranges.is_empty() {
            return pixels;
        }
        let last = ranges.len() - 1;
        if ranges[0].dash == ranges[last].dash {
            let (left, right) = (
                ranges[last].left - f64::from(WIDTH),
                ranges[0].right + f64::from(WIDTH),
            );
            ranges[0].left = left;
            ranges[last].right = right;
        }
    }
    for y in 0..height {
        let mut index = 0;
        for x in 0..WIDTH as usize {
            // The reference advances once at each sample, including zero ranges.
            if x as f64 / ranges[index].right > 1.0 && index + 1 < ranges.len() {
                index += 1;
            }
            let r = ranges[index];
            let d = (x as f64 - r.left).abs().min((x as f64 - r.right).abs());
            let distance = if round {
                let middle = (y as f64 - 7.0) / 7.0 * (stretch / 2.0 + 1.0);
                if r.dash {
                    d.hypot(stretch / 2.0 - middle.abs())
                } else {
                    stretch / 2.0 - d.hypot(middle)
                }
            } else if r.dash {
                d
            } else {
                -d
            };
            pixels[y * WIDTH as usize + x] = (distance + 128.0).clamp(0.0, 255.0) as u8;
        }
    }
    pixels
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sdf_wraps_and_collapses_zero_gaps() {
        let p = sdf(&[1.0, 1.0], false);
        assert_eq!((p[0], p[64], p[128], p[192]), (128, 192, 128, 64));
        assert_eq!(sdf(&[1.0, 0.0, 1.0, 1.0], false), sdf(&[2.0, 1.0], false));
        assert!(sdf(&[0.0, 0.0], true).iter().all(|v| *v == 0));
        let dots = sdf(&[0.0, 2.0], true);
        assert!(dots[7 * WIDTH as usize] > 128);
        assert!(dots[7 * WIDTH as usize + 128] < 128);
    }
}
