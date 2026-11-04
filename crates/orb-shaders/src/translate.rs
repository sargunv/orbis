// WGSL → GLSL ES 3.00 with naga, for the build script and this crate's
// tests: one vertex and one fragment shader per program, overrides
// specialized, and the uniforms naga named after each WGSL binding.

use std::collections::BTreeMap;

/// A sampler uniform: name, its texture's (group, binding), its sampler's.
pub(crate) type Texture = (String, (u32, u32), Option<(u32, u32)>);

pub(crate) struct Translated {
    pub(crate) vertex: String,
    pub(crate) fragment: String,
    /// Uniform blocks: name, group, binding.
    pub(crate) blocks: Vec<(String, u32, u32)>,
    /// Sampler uniforms: name, texture's (group, binding), sampler's.
    pub(crate) textures: Vec<Texture>,
}

pub(crate) fn glsl(
    label: &str,
    source: &str,
    vertex: &str,
    fragment: &str,
    constants: &[(&str, f64)],
) -> Result<Translated, String> {
    use naga::back::glsl;
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    let mut module = naga::front::wgsl::parse_str(source)
        .map_err(|e| format!("{label}: {}", e.emit_to_string(source)))?;
    split_samplers(&mut module);
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .map_err(|e| format!("{label}: {}", e.emit_to_string(source)))?;
    let constants: naga::back::PipelineConstants =
        constants.iter().map(|&(k, v)| (k.to_string(), v)).collect();
    let options = glsl::Options {
        version: glsl::Version::Embedded {
            version: 300,
            is_webgl: true,
        },
        // WebGPU's clip space and framebuffer orientation: GL renders
        // upside down into textures, which orb-hal-gl flips on present.
        writer_flags: glsl::WriterFlags::ADJUST_COORDINATE_SPACE,
        binding_map: Default::default(),
        zero_initialize_workgroup_memory: false,
    };
    let mut blocks = BTreeMap::new();
    let mut textures = BTreeMap::new();
    let mut stage = |stage: naga::ShaderStage, entry: &str| -> Result<String, String> {
        let (module, info) = naga::back::pipeline_constants::process_overrides(
            &module,
            &info,
            Some((stage, entry)),
            &constants,
        )
        .map_err(|e| format!("{label} {entry}: {e}"))?;
        let pipeline = glsl::PipelineOptions {
            shader_stage: stage,
            entry_point: entry.to_string(),
            multiview: None,
        };
        let mut out = String::new();
        let mut writer = glsl::Writer::new(
            &mut out,
            &module,
            &info,
            &options,
            &pipeline,
            naga::proc::BoundsCheckPolicies::default(),
        )
        .map_err(|e| format!("{label} {entry}: {e}"))?;
        let reflection = writer
            .write()
            .map_err(|e| format!("{label} {entry}: {e}"))?;
        let binding = |handle: naga::Handle<naga::GlobalVariable>| {
            let b = module.global_variables[handle]
                .binding
                .as_ref()
                .expect("resources have bindings");
            (b.group, b.binding % ALIAS)
        };
        for (handle, name) in &reflection.uniforms {
            if module.global_variables[*handle].space == naga::AddressSpace::Uniform {
                let (group, binding) = binding(*handle);
                blocks.insert(name.clone(), (group, binding));
            }
        }
        for (name, mapping) in &reflection.texture_mapping {
            textures.insert(
                name.clone(),
                (binding(mapping.texture), mapping.sampler.map(binding)),
            );
        }
        Ok(out)
    };
    let vertex = stage(naga::ShaderStage::Vertex, vertex)?;
    let fragment = stage(naga::ShaderStage::Fragment, fragment)?;
    Ok(Translated {
        vertex,
        fragment,
        blocks: blocks.into_iter().map(|(n, (g, b))| (n, g, b)).collect(),
        textures: textures.into_iter().map(|(n, (t, s))| (n, t, s)).collect(),
    })
}

/// Bindings of [`split_samplers`]'s aliases: the texture's plus a multiple
/// of this.
const ALIAS: u32 = 1000;

/// GLSL ES has only combined samplers, and naga names one per texture: a
/// texture sampled with several samplers gets an alias per extra sampler,
/// bound to the same texture.
fn split_samplers(module: &mut naga::Module) {
    use naga::{Expression, Handle};

    enum At {
        Function(Handle<naga::Function>),
        Entry(usize),
    }
    let global = |f: &naga::Function, e: Handle<Expression>| match f.expressions[e] {
        Expression::GlobalVariable(g) => Some(g),
        _ => None,
    };
    let mut samples = Vec::new();
    let functions = module
        .functions
        .iter()
        .map(|(h, f)| (At::Function(h), f))
        .chain(
            module
                .entry_points
                .iter()
                .enumerate()
                .map(|(i, e)| (At::Entry(i), &e.function)),
        );
    for (at, f) in functions {
        for (handle, expression) in f.expressions.iter() {
            if let Expression::ImageSample { image, sampler, .. } = *expression
                && let (Some(texture), Some(sampler)) = (global(f, image), global(f, sampler))
            {
                let _ = handle;
                samples.push((
                    match at {
                        At::Function(h) => At::Function(h),
                        At::Entry(i) => At::Entry(i),
                    },
                    image,
                    texture,
                    sampler,
                ));
            }
        }
    }
    // Each use of a texture is its own expression, so a sample's image
    // expression can name the alias in place.
    let mut first = std::collections::HashMap::new();
    let mut aliases = std::collections::HashMap::new();
    let mut renamed = std::collections::HashMap::new();
    for (at, expression, image, sampler) in samples {
        let key = (
            match at {
                At::Function(h) => Some(h),
                At::Entry(_) => None,
            },
            match at {
                At::Entry(i) => i,
                At::Function(_) => 0,
            },
            expression,
        );
        if let Some(&previous) = renamed.get(&key) {
            assert_eq!(previous, sampler, "one use of a texture sampled two ways");
            continue;
        }
        renamed.insert(key, sampler);
        if *first.entry(image).or_insert(sampler) == sampler {
            continue;
        }
        let count = aliases.len() as u32 + 1;
        let alias = *aliases.entry((image, sampler)).or_insert_with(|| {
            let mut variable = module.global_variables[image].clone();
            let binding = variable.binding.as_mut().expect("textures have bindings");
            binding.binding += ALIAS * count;
            variable.name = variable.name.map(|n| format!("{n}_alias{count}"));
            module
                .global_variables
                .append(variable, naga::Span::UNDEFINED)
        });
        let f = match at {
            At::Function(h) => &mut module.functions[h],
            At::Entry(i) => &mut module.entry_points[i].function,
        };
        f.expressions[expression] = Expression::GlobalVariable(alias);
    }
}
