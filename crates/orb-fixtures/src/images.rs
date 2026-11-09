use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

#[derive(Debug)]
pub(crate) struct Image {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixels: Vec<u8>,
}

impl Image {
    pub(crate) fn read(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut decoder = png::Decoder::new(BufReader::new(file));
        decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut reader = decoder
            .read_info()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let size = reader.output_buffer_size().ok_or("PNG is too large")?;
        let mut bytes = vec![0; size];
        let info = reader
            .next_frame(&mut bytes)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let bytes = &bytes[..info.buffer_size()];
        let pixels = match info.color_type {
            png::ColorType::Rgba => bytes.to_vec(),
            png::ColorType::Rgb => bytes
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|c| [c[0], c[1], c[2], 255])
                .collect(),
            png::ColorType::Grayscale => bytes.iter().flat_map(|&c| [c, c, c, 255]).collect(),
            png::ColorType::GrayscaleAlpha => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|c| [c[0], c[0], c[0], c[1]])
                .collect(),
            png::ColorType::Indexed => return Err("PNG palette was not expanded".into()),
        };
        Ok(Self {
            width: info.width,
            height: info.height,
            pixels,
        })
    }

    pub(crate) fn write(&self, path: &Path) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut encoder = png::Encoder::new(BufWriter::new(file), self.width, self.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .and_then(|mut w| w.write_image_data(&self.pixels))
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Unassociates premultiplied alpha as GL JS's harness wrote the
    /// expected images: dividing in place in a `Uint8Array`, which
    /// truncates and wraps modulo 256. Light added over a transparent
    /// pixel (the globe's atmosphere in space) has more colour than alpha,
    /// so its expected bytes wrapped, and a saturating division would
    /// differ from them wherever the renderers agree. Wgpu readback
    /// already has top-to-bottom scanlines.
    pub(crate) fn unpremultiply(&mut self) {
        for pixel in self.pixels.as_chunks_mut::<4>().0 {
            let alpha = f64::from(pixel[3]) / 255.0;
            if alpha != 0.0 {
                for c in &mut pixel[..3] {
                    *c = (f64::from(*c) / alpha) as u32 as u8;
                }
            }
        }
    }
}
