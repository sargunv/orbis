// The shaders' WGSL, put together from the catalog's modules. Shared with
// the build script; WebGL 2's web build (`glsl_only`) leaves it out, since
// its pipelines compile the build's GLSL.

use crate::catalog::{ATTRIBUTE_ZERO, CLIP_PLANES, Shader};

/// The modules a WebGL 2 build swaps (#140), modern first, then each's
/// GL counterpart: built-ins written out, and tables read from data
/// textures where the modern path reads storage buffers.
const GL_SWAPS: &[(&str, &str)] = &[
    (wgsl!("builtins"), wgsl!("builtins_gl")),
    (
        wgsl!("fetch"),
        concat!(
            wgsl!("data_texture"),
            wgsl!("fetch_texture"),
            generated!("rows_texture")
        ),
    ),
    (
        wgsl!("fetch_terrain"),
        concat!(
            wgsl!("data_texture"),
            wgsl!("fetch_terrain_texture"),
            generated!("terrain_rows_texture")
        ),
    ),
    (
        wgsl!("fetch_drape"),
        concat!(wgsl!("data_texture"), wgsl!("fetch_drape_texture")),
    ),
    (
        wgsl!("fetch_regions"),
        concat!(wgsl!("data_texture"), wgsl!("fetch_regions_texture")),
    ),
];

impl Shader {
    pub fn source(&self) -> String {
        let mut source: String = self.parts.concat();
        if self.parts.contains(&CLIP_PLANES) {
            source = source.replace("//planes ", "");
        }
        source
    }

    /// Its source for WebGL 2 (#140), its swapped modules (`GL_SWAPS`)
    /// replaced; `None` for what WebGL 2 can't run: compute, and clip
    /// distances, which a device without them draws another way.
    pub fn gl_source(&self) -> Option<String> {
        if !self.compute.is_empty() || self.parts.contains(&CLIP_PLANES) {
            return None;
        }
        Some(with_attribute_zero(&self.swapped(GL_SWAPS)))
    }

    /// Its source for a device that reads the scene's and the terrain's
    /// tables from data textures and has the rest of WebGPU: WebGL 2's
    /// path run natively (orb-hal-wgpu's `force_cpu_driver`), the
    /// built-ins and those tables' `fetch` modules swapped
    /// ([`gl_source`](Self::gl_source)'s first three).
    pub fn texture_tables_source(&self) -> String {
        self.swapped(&GL_SWAPS[..3])
    }

    fn swapped(&self, swaps: &[(&str, &str)]) -> String {
        let mut source = self.source();
        for (modern, gl) in swaps {
            source = source.replace(modern, gl);
        }
        source
    }
}

/// `source` with every vertex entry point taking a `u32` at
/// `@location(0)`, which the HAL binds instance-stepped over zeros, and
/// OR-ing it into the vertex index so no compiler drops it (#179). Every
/// vertex is pulled, so programs would otherwise have no attribute, and
/// Firefox on desktop GL then fakes attribute 0 with an array as long as
/// a draw's largest index: compacted indices pass its 10M limit, and it
/// refuses the draw.
fn with_attribute_zero(source: &str) -> String {
    const INDEX: &str = "@builtin(vertex_index)";
    let mut out = String::with_capacity(source.len() + 1024);
    let mut rest = source;
    while let Some(at) = rest.find("@vertex") {
        let (head, tail) = rest.split_at(at);
        out.push_str(head);
        let open = tail.find('(').expect("a vertex entry point's parameters");
        let body = tail.find('{').expect("a vertex entry point's body");
        let signature = &tail[..body];
        let index = signature
            .find(INDEX)
            .unwrap_or_else(|| panic!("`{signature}` takes no {INDEX}"));
        let name = index + INDEX.len();
        let name_end = name + signature[name..].find(':').expect("a typed parameter");
        out.push_str(&tail[..=open]);
        out.push_str("@location(0) ");
        out.push_str(ATTRIBUTE_ZERO);
        out.push_str(": u32, ");
        out.push_str(&tail[open + 1..name]);
        out.push_str(" orb_vertex_index");
        out.push_str(&tail[name_end..=body]);
        out.push_str("\n    let ");
        out.push_str(signature[name..name_end].trim());
        out.push_str(" = orb_vertex_index | ");
        out.push_str(ATTRIBUTE_ZERO);
        out.push(';');
        rest = &tail[body + 1..];
    }
    out.push_str(rest);
    out
}
