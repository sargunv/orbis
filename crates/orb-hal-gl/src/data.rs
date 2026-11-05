//! Storage buffers as data textures.
//!
//! WebGL2 and GLES 3.0 have no storage buffers. A buffer with storage use
//! is an R32UI texture [`DATA_WIDTH`] words wide, word `i` at texel
//! `(i % DATA_WIDTH, i / DATA_WIDTH)`, and `fetch_compat.wgsl` reads it
//! with `texelFetch`. Writes land as row spans.

use glow::HasContext;

use crate::state::State;

/// `fetch_compat.wgsl`'s `DATA_WIDTH_LOG2`.
pub const DATA_WIDTH_LOG2: u32 = 12;
/// Words per data texture row.
pub const DATA_WIDTH: u32 = 1 << DATA_WIDTH_LOG2;

pub(crate) struct DataTexture {
    pub(crate) texture: glow::Texture,
}

impl DataTexture {
    pub(crate) fn new(gl: &glow::Context, state: &mut State, size: u64, label: &str) -> Self {
        let words = size.div_ceil(4).max(1);
        let rows = words.div_ceil(u64::from(DATA_WIDTH)) as i32;
        // SAFETY: a new texture on the current context.
        let texture = unsafe {
            let texture = gl.create_texture().expect("data texture");
            state.bind_texture(gl, 0, glow::TEXTURE_2D, Some(texture));
            gl.tex_storage_2d(glow::TEXTURE_2D, 1, glow::R32UI, DATA_WIDTH as i32, rows);
            texture
        };
        let _ = label;
        Self { texture }
    }

    /// Writes `data`, whole words, at byte `offset`.
    pub(crate) fn write(&self, gl: &glow::Context, state: &mut State, offset: u64, data: &[u8]) {
        assert!(
            offset.is_multiple_of(4) && data.len().is_multiple_of(4),
            "storage writes are whole words"
        );
        let width = u64::from(DATA_WIDTH);
        let mut word = offset / 4;
        let mut data = data;
        // SAFETY: row spans inside the texture, which holds the buffer.
        unsafe {
            state.bind_texture(gl, 0, glow::TEXTURE_2D, Some(self.texture));
            let span = |x: u64, y: u64, w: u64, h: u64, bytes: &[u8]| {
                gl.tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    x as i32,
                    y as i32,
                    w as i32,
                    h as i32,
                    glow::RED_INTEGER,
                    glow::UNSIGNED_INT,
                    glow::PixelUnpackData::Slice(Some(bytes)),
                );
            };
            // The first row's tail.
            let x = word % width;
            if x != 0 {
                let n = (width - x).min(data.len() as u64 / 4);
                span(x, word / width, n, 1, &data[..n as usize * 4]);
                word += n;
                data = &data[n as usize * 4..];
            }
            // Whole rows.
            let rows = data.len() as u64 / 4 / width;
            if rows > 0 {
                let bytes = (rows * width * 4) as usize;
                span(0, word / width, width, rows, &data[..bytes]);
                word += rows * width;
                data = &data[bytes..];
            }
            // The last row's head.
            if !data.is_empty() {
                span(0, word / width, data.len() as u64 / 4, 1, data);
            }
        }
    }
}

/// Copies `size` bytes at byte `from` of data texture `source` to byte
/// `to` of `destination`: the same columns, so whole rows and a partial
/// first and last row, read through a framebuffer.
pub(crate) fn copy(
    gl: &glow::Context,
    state: &mut State,
    source: glow::Texture,
    from: u64,
    destination: glow::Texture,
    to: u64,
    size: u64,
) {
    let width = u64::from(DATA_WIDTH);
    let (from, to, words) = (from / 4, to / 4, size / 4);
    assert_eq!(
        from % width,
        to % width,
        "data texture copies keep their columns"
    );
    // SAFETY: a temporary framebuffer over the source; spans inside both
    // textures.
    unsafe {
        let fbo = gl.create_framebuffer().expect("framebuffer");
        state.read_framebuffer(gl, Some(fbo));
        gl.framebuffer_texture_2d(
            glow::READ_FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(source),
            0,
        );
        state.bind_texture(gl, 0, glow::TEXTURE_2D, Some(destination));
        let span = |word: u64, n: u64, rows: u64| {
            let (x, y) = ((word % width) as i32, (word / width) as i32);
            let (dx, dy) = (
                ((to + word - from) % width) as i32,
                ((to + word - from) / width) as i32,
            );
            gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, dx, dy, x, y, n as i32, rows as i32);
        };
        let (mut word, end) = (from, from + words);
        if word % width != 0 {
            let n = (width - word % width).min(end - word);
            span(word, n, 1);
            word += n;
        }
        let rows = (end - word) / width;
        if rows > 0 {
            span(word, width, rows);
            word += rows * width;
        }
        if word < end {
            span(word, end - word, 1);
        }
        state.read_framebuffer(gl, None);
        gl.delete_framebuffer(fbo);
    }
}
