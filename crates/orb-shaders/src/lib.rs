//! WGSL modules and their composition.
//!
//! Each pipeline's source is the shared scene types, the bindings of its
//! kind (draw or cull) and its own module, concatenated.

#![forbid(unsafe_code)]

/// A WGSL module as the build script wrote it: without comments or
/// indentation, which only the source tree needs.
macro_rules! wgsl {
    ($name:literal) => {
        include_str!(concat!(env!("OUT_DIR"), "/wgsl/", $name, ".wgsl"))
    };
}

mod catalog;
pub use catalog::*;

#[cfg(feature = "ir")]
mod ir {
    use std::sync::OnceLock;

    /// Where each of a shader's modules lies in the inflated whole, by
    /// entry point.
    type Ranges = &'static [(&'static str, std::ops::Range<usize>)];

    include!(concat!(env!("OUT_DIR"), "/ir.rs"));

    /// A shader's modules by entry point.
    type Modules = Vec<(&'static str, &'static [u8])>;

    /// Every shader's modules by label, inflated the first time a shader
    /// asks (a fraction of a millisecond).
    pub(crate) fn modules() -> &'static [(&'static str, Modules)] {
        static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
        static MODULES: OnceLock<Vec<(&str, Modules)>> = OnceLock::new();
        MODULES.get_or_init(|| {
            let bytes = BYTES.get_or_init(|| {
                let deflated = include_bytes!(concat!(env!("OUT_DIR"), "/ir.deflate"));
                miniz_oxide::inflate::decompress_to_vec(deflated)
                    .expect("inflates the build's modules")
            });
            (IR.iter())
                .map(|&(label, entries)| {
                    let modules = entries
                        .iter()
                        .map(|(entry, at)| (*entry, &bytes[at.clone()]));
                    (label, modules.collect())
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod split;

/// A shader's naga modules by entry point, postcard-encoded.
pub type EntryModules = &'static [(&'static str, &'static [u8])];

impl Shader {
    /// Its naga modules, which the build script parsed from
    /// [`source`](Self::source), one per entry point with only what it
    /// uses, postcard-encoded (the `ir` feature): devices that compile
    /// with naga make each stage's module from these rather than parse the
    /// WGSL and split it per stage.
    pub fn ir(&self) -> Option<EntryModules> {
        #[cfg(feature = "ir")]
        {
            ir::modules()
                .iter()
                .find(|(label, _)| *label == self.label)
                .map(|(_, modules)| modules.as_slice())
        }
        #[cfg(not(feature = "ir"))]
        None
    }
}

/// The scene tables' layouts, reflected from `types.wgsl`: each struct's
/// words and each member's first word.
pub mod layout {
    include!(concat!(env!("OUT_DIR"), "/layout.rs"));
}

/// A uniform's words, written at its struct's members as the shaders
/// declare them (`orb_shaders::layout`).
#[derive(Debug)]
pub struct Block(pub Vec<u32>);

impl Block {
    pub fn new(words: usize) -> Self {
        Self(vec![0; words])
    }

    pub fn u(&mut self, at: usize, values: &[u32]) -> &mut Self {
        self.0[at..at + values.len()].copy_from_slice(values);
        self
    }

    pub fn f(&mut self, at: usize, values: &[f32]) -> &mut Self {
        for (word, v) in self.0[at..at + values.len()].iter_mut().zip(values) {
            *word = v.to_bits();
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(shader: &Shader) -> naga::Module {
        parse(shader.label, &shader.source()).0
    }

    fn parse(label: &str, source: &str) -> (naga::Module, naga::valid::ModuleInfo) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(source)));
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::CLIP_DISTANCES,
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(source)));
        (module, info)
    }

    /// The cull module's functions keep no function-scope arrays: neither
    /// array variables nor dynamic indexing into an array value, which
    /// compilers lower to the same. An Adreno 830's Vulkan driver (Chrome
    /// 156) miscompiled the cull dispatch with them, writing no draws or
    /// losing the device. Shared helpers it doesn't call are compiled
    /// out.
    #[test]
    fn cull_shader_has_no_function_arrays() {
        let (module, info) = parse(CULL_SHADER.label, &CULL_SHADER.source());
        let is_array = |inner: &naga::TypeInner| matches!(inner, naga::TypeInner::Array { .. });
        let functions = module
            .functions
            .iter()
            .map(|(handle, f)| (f, &info[handle]))
            .chain(
                module
                    .entry_points
                    .iter()
                    .enumerate()
                    .map(|(i, e)| (&e.function, info.get_entry_point(i))),
            );
        for (function, function_info) in functions {
            let name = function.name.as_deref().unwrap_or("?");
            if !CULL.contains(&format!("fn {name}(")) {
                continue;
            }
            for (_, local) in function.local_variables.iter() {
                assert!(
                    !is_array(&module.types[local.ty].inner),
                    "{name}: function-scope array `{}`",
                    local.name.as_deref().unwrap_or("?"),
                );
            }
            for (_, expression) in function.expressions.iter() {
                if let naga::Expression::Access { base, .. } = *expression {
                    let base = function_info[base].ty.inner_with(&module.types);
                    assert!(!is_array(base), "{name}: indexes an array value");
                }
            }
        }
    }

    /// FNV-1a: a hash that stays the same across Rust releases.
    fn fnv(text: &str) -> u64 {
        text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Each pipeline stage as the modern path compiles it, its override
    /// constants at their defaults, written out by naga's Metal back end
    /// and hashed, against `src/stage-hashes.txt`: a change to what any
    /// stage computes shows as a hash to review, and changes meant to
    /// keep the modern path's code (the WebGL 2 path's shared fixes, #140)
    /// prove they do. `UPDATE_GOLDEN=1` rewrites the file.
    #[test]
    fn stages_keep_their_hashes() {
        let mut text =
            String::from("# Each pipeline stage's Metal, hashed (`stages_keep_their_hashes`).\n");
        for shader in SHADERS {
            for (entry, one) in split::entry_modules(&validate(shader)) {
                let info = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::CLIP_DISTANCES,
                )
                .validate(&one)
                .unwrap_or_else(|e| panic!("{} {entry}: {e}", shader.label));
                let (one, info) = naga::back::pipeline_constants::process_overrides(
                    &one,
                    &info,
                    None,
                    &Default::default(),
                )
                .unwrap_or_else(|e| panic!("{} {entry}: {e}", shader.label));
                let options = naga::back::msl::Options {
                    lang_version: (2, 4),
                    ..Default::default()
                };
                let (msl, _) = naga::back::msl::write_string(
                    &one,
                    &info,
                    &options,
                    &naga::back::msl::PipelineOptions::default(),
                )
                .unwrap_or_else(|e| panic!("{} {entry}: {e}", shader.label));
                text.push_str(&format!("{} {entry} {:016x}\n", shader.label, fnv(&msl)));
            }
        }
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/stage-hashes.txt");
        if std::env::var_os("UPDATE_GOLDEN").is_some() {
            std::fs::write(path, &text).unwrap();
            return;
        }
        let golden = std::fs::read_to_string(path).unwrap_or_default();
        let changed: Vec<&str> = text
            .lines()
            .filter(|l| !golden.lines().any(|g| g == *l))
            .collect();
        assert!(
            changed.is_empty() && golden.lines().count() == text.lines().count(),
            "stages changed from {path}: {changed:?}; UPDATE_GOLDEN=1 rewrites it, once the change is the one meant"
        );
    }

    #[test]
    fn shaders_validate() {
        for &shader in SHADERS {
            let module = validate(&shader);
            for entry in [shader.vertex, shader.fragment, shader.compute] {
                if !entry.is_empty() {
                    assert!(
                        module.entry_points.iter().any(|e| e.name == entry),
                        "{}: no entry point {entry}",
                        shader.label
                    );
                }
            }
        }
    }

    /// Each build-made module is the one the shader's WGSL parses to for
    /// its entry point, every entry point has one, and each decodes to a
    /// valid module.
    #[cfg(feature = "ir")]
    #[test]
    fn modules_are_their_sources() {
        for shader in SHADERS {
            let ir = shader
                .ir()
                .unwrap_or_else(|| panic!("{}: no modules", shader.label));
            let parsed = split::entry_modules(&validate(shader));
            assert_eq!(ir.len(), parsed.len(), "{}", shader.label);
            for (&(entry, bytes), (name, one)) in ir.iter().zip(&parsed) {
                assert_eq!(entry, name, "{}", shader.label);
                assert!(
                    postcard::to_allocvec(one).unwrap() == bytes,
                    "{} {entry}: the build's module differs",
                    shader.label
                );
                let made: naga::Module = postcard::from_bytes(bytes).expect("decodes");
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::CLIP_DISTANCES,
                )
                .validate(&made)
                .unwrap_or_else(|e| panic!("{} {entry}: {e}", shader.label));
            }
        }
    }
}
