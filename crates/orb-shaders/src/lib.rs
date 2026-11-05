//! WGSL modules and their composition (plan §8.4).
//!
//! Each pipeline's source is the shared scene types, the bindings of its
//! kind (draw or cull) and its own module, concatenated. Shaders that read
//! paint come in two classes: the table class with stand-ins for the GPU
//! expression interpreter, and the bytecode class with the interpreter
//! (`expr.wgsl`), compiled only into pipelines of nodes that need it.

#![forbid(unsafe_code)]

mod catalog;
pub use catalog::*;

#[cfg(feature = "glsl")]
mod glsl {
    include!(concat!(env!("OUT_DIR"), "/glsl.rs"));
}

/// A Compat shader's offline GLSL ES 3.00 program (plan §8.4): its table
/// class, or with `bytecode` the bytecode class.
#[cfg(feature = "glsl")]
pub fn glsl(shader: &Shader, bytecode: bool) -> &'static orb_hal::GlslProgram<'static> {
    glsl::PROGRAMS
        .iter()
        .find(|(label, b, _)| *label == shader.label && *b == bytecode)
        .map(|(.., program)| program)
        .unwrap_or_else(|| panic!("{} has no Compat program", shader.label))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(shader: &Shader) -> naga::Module {
        if shader.paint {
            parse(shader.label, &shader.bytecode_source());
        }
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

    /// The shading modules validate with a receiver that calls them over
    /// bindings of its own.
    #[test]
    fn pbr_and_shadow_modules_validate() {
        let receiver = r#"
            @group(0) @binding(0) var<uniform> cascades: ShadowCascades;
            @group(0) @binding(1) var shadow_map: texture_depth_2d_array;
            @group(0) @binding(2) var shadow_cmp: sampler_comparison;
            struct In {
                @builtin(position) position: vec4<f32>,
                @location(0) world: vec3<f32>,
                @location(1) @interpolate(flat) normal: u32,
                @location(2) uv: vec2<f32>,
            };
            @fragment
            fn fs(in: In) -> @location(0) vec4<f32> {
                let n = turn_enu(octahedral_normal(in.normal), vec3<f32>(0.0, 0.0, 30.0));
                let p = perturb_normal(n, mat2x3<f32>(dpdx(in.world), dpdy(in.world)),
                    mat2x2<f32>(dpdx(in.uv), dpdy(in.uv)), vec3<f32>(0.5, 0.5, 1.0), 1.0);
                var lights: PbrLights;
                lights.ambient = vec3<f32>(0.4);
                lights.directional = vec3<f32>(1.0);
                lights.toward_light = normalize(vec3<f32>(1.0, 1.0, 1.0));
                lights.toward_eye = vec3<f32>(0.0, 0.0, 1.0);
                lights.shadow = shadow_visibility(shadow_map, shadow_cmp, cascades, in.world, p, in.position.w);
                let base = srgb_to_linear(vec3<f32>(wrap_uv(in.uv, 6u), 0.5));
                return vec4<f32>(linear_to_srgb(pbr_shade(base, 0.5, 0.5, 1.0, p, lights)), 1.0);
            }
            @vertex
            fn vs(@location(0) world: vec3<f32>) -> @builtin(position) vec4<f32> {
                return shadow_clip(cascades, 0u, world);
            }
        "#;
        parse(
            "pbr and shadow",
            &[PBR_MODULE, SHADOW_MODULE, receiver].concat(),
        );
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
            LAYER_COMPOSITE_SHADER,
            LAYER_BLEND_SHADER,
            TERRAIN_SHADER,
            DRAPE_SHADER,
            TERRAIN_STACK_SHADER,
            EXTRUSION_SHADER,
            EXTRUSION_COMPOSITE_SHADER,
            FILL_SHADER,
            FILL_OUTLINE_SHADER,
            LINE_SHADER,
            CIRCLE_SHADER,
            SYMBOL_SHADER,
            HEATMAP_SHADER,
            HEATMAP_COLOR_SHADER,
            RASTER_SHADER,
            HILLSHADE_SHADER,
            COLOR_RELIEF_SHADER,
            PREPARE_PAD_SHADER,
            PREPARE_SLOPE_SHADER,
            IMAGE_MIPS_SHADER,
            PREPARE_PAD_DRAW_SHADER,
            PREPARE_SLOPE_DRAW_SHADER,
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
