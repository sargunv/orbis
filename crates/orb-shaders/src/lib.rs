//! WGSL modules and their composition (plan §8.4).
//!
//! Each pipeline's source is the shared scene types, the bindings of its
//! kind (draw or cull) and its own module, concatenated.

#![forbid(unsafe_code)]

mod catalog;
pub use catalog::*;

/// Compat's reads of the scene tables' rows, reflected from `types.wgsl`.
fn rows_compat() -> &'static str {
    include_str!(concat!(env!("OUT_DIR"), "/rows_compat.wgsl"))
}

/// The scene tables' layouts, reflected from `types.wgsl`: each struct's
/// words and each member's first word.
pub mod layout {
    include!(concat!(env!("OUT_DIR"), "/layout.rs"));
}

#[cfg(feature = "glsl")]
mod glsl {
    include!(concat!(env!("OUT_DIR"), "/glsl.rs"));
}

/// A Compat shader's offline GLSL ES 3.00 program (plan §8.4).
#[cfg(feature = "glsl")]
pub fn glsl(shader: &Shader) -> &'static orb_hal::GlslProgram<'static> {
    glsl::PROGRAMS
        .iter()
        .find(|(label, _)| *label == shader.label)
        .map(|(.., program)| program)
        .unwrap_or_else(|| panic!("{} has no Compat program", shader.label))
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
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(source)));
        (module, info)
    }

    /// The cull module's functions keep no function-scope arrays: neither
    /// array variables nor dynamic indexing into an array value, which
    /// compilers lower to the same. An Adreno 830's Vulkan driver (Chrome
    /// 156) miscompiled the cull dispatch with them, writing no draws or
    /// losing the device (#23). Shared helpers it doesn't call are compiled
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

    #[test]
    fn shaders_validate() {
        for shader in [
            SKY_SHADER,
            SCATTER_SHADER,
            COMPOSITE_OVER_SHADER,
            COMPOSITE_REPLACE_SHADER,
            TERRAIN_SHADER,
            DRAPE_SHADER,
            EXTRUSION_SHADER,
            EXTRUSION_CASTER_SHADER,
            SHADOW_GROUND_SHADER,
            OCCLUSION_SHADERS[0],
            OCCLUSION_SHADERS[1],
            FILL_SHADER,
            FILL_OUTLINE_SHADER,
            LINE_SHADER,
            CIRCLE_SHADER,
            MODEL_SHADER,
            SYMBOL_SHADER,
            HEATMAP_SHADER,
            RASTER_SHADER,
            HILLSHADE_SHADER,
            COLOR_RELIEF_SHADER,
            PREPARE_PAD_SHADER,
            PREPARE_SLOPE_SHADER,
            IMAGE_MIPS_SHADER,
            CULL_SHADER,
        ] {
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
}
