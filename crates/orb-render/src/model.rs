//! Model layers: glTF models drawn once per visible point of a
//! model node ([`NodeDraw::Instances`]), into the node's own 3D target over
//! the depth 3D layers share, then composited at the layer's opacity as
//! extrusions are (`offscreen`). A node draws one model: the map gives each a model
//! index, and uploads models by it ([`Renderer::set_model`]).
//!
//! [`NodeDraw::Instances`]: orb_scene::NodeDraw::Instances
//! [`Renderer::set_model`]: crate::Renderer::set_model

use crate::release::Release;
use std::collections::HashMap;

use orb_3d::gpu::{DrawClass, GpuModel};
use orb_hal::*;
use orb_scene::SceneBuffers;
use orb_shaders::{MODEL_MASKED_SHADER, MODEL_SHADER};

/// Bytes between a model node's draws, one per [`DrawClass`].
const CLASS_ARGS: u64 = 32;

/// A model's GPU residency.
#[derive(Debug)]
struct Resident {
    /// Its vertex words, then its material words.
    words: BufferId,
    indices: BufferId,
    texture: Option<TextureId>,
    /// `model.wgsl`'s `ModelDraw`: where its materials start.
    draw: BufferId,
    group: BindGroupId,
    /// Its first index, then its indices' counts per [`DrawClass`].
    draws: (u32, [u32; 4]),
    bytes: u64,
}

#[derive(Debug)]
pub(crate) struct Models {
    resident: HashMap<u32, Resident>,
    /// Group 2: a model's words, textures, sampler and draw.
    layout: BindGroupLayoutId,
    /// Scene, node and model groups.
    layouts: [BindGroupLayoutId; 3],
    /// Per [`DrawClass`], at one sample and at four.
    pipelines: [[Option<RenderPipelineId>; 4]; 2],
    sampler: SamplerId,
    /// Bound for models without textures.
    blank: TextureId,
}

impl Models {
    pub(crate) fn new(
        gpu: &mut dyn Gpu,
        scene: BindGroupLayoutId,
        node: BindGroupLayoutId,
    ) -> Self {
        let vf = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "model",
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vf,
                    ty: BindingType::Storage { read_only: true },
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample: TextureSampleType::Float { filterable: true },
                        dimension: TextureViewDimension::D2Array,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler { filtering: true },
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: vf,
                    ty: BindingType::Uniform,
                },
            ],
        });
        // Wrap modes other than repeating are applied in the shader.
        let sampler = gpu.create_sampler(&SamplerDesc {
            label: "model",
            filter: FilterMode::Linear,
            address: AddressMode::Repeat,
            mipmap_filter: FilterMode::Linear,
            anisotropy: 4,
            compare: None,
        });
        let blank = gpu.create_texture(&TextureDesc {
            label: "no model texture",
            width: 1,
            height: 1,
            array_layers: 1,
            dimension: TextureViewDimension::D2Array,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            mip_levels: 1,
        });
        Self {
            resident: HashMap::new(),
            layout,
            layouts: [scene, node, layout],
            pipelines: [[None; 4]; 2],
            sampler,
            blank,
        }
    }

    /// Uploads model `index`, replacing what was there.
    pub(crate) fn set(&mut self, gpu: &mut dyn Gpu, index: u32, model: &GpuModel) {
        self.remove(gpu, index);
        let mut words = model.vertices.clone();
        let material_base = words.len() as u32;
        words.extend(&model.materials);
        // A buffer of at least one word: storage bindings can't be empty.
        if words.is_empty() {
            words.push(0);
        }
        let buffer = |gpu: &mut dyn Gpu, label, usage, data: &[u32]| {
            let id = gpu.create_buffer(&BufferDesc {
                label,
                size: (data.len().max(1) * 4) as u64,
                usage: usage | BufferUsages::COPY_DST,
            });
            gpu.write_buffer(id, 0, bytemuck::cast_slice(data));
            id
        };
        let words_buffer = buffer(gpu, "model words", BufferUsages::STORAGE, &words);
        let indices = buffer(gpu, "model indices", BufferUsages::INDEX, &model.indices);
        let draw = buffer(
            gpu,
            "model draw",
            BufferUsages::UNIFORM,
            &[0, material_base, 0, 0],
        );
        let mut bytes = (words.len() + model.indices.len() + 4) as u64 * 4;
        let texture = (model.texture_size > 0 && !model.layers.is_empty()).then(|| {
            let size = model.texture_size;
            let levels = 32 - size.leading_zeros();
            let texture = gpu.create_texture(&TextureDesc {
                label: "model textures",
                width: size,
                height: size,
                array_layers: model.layers.len() as u32,
                dimension: TextureViewDimension::D2Array,
                // sRGB images decode in the shader: other maps are linear.
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                sample_count: 1,
                mip_levels: levels,
            });
            for (layer, texels) in model.layers.iter().enumerate() {
                let mut at = 0;
                for level in 0..levels {
                    let side = size >> level;
                    let len = (side * side * 4) as usize;
                    gpu.write_texture(
                        &TextureRegion {
                            texture,
                            x: 0,
                            y: 0,
                            layer: layer as u32,
                            width: side,
                            height: side,
                            level,
                        },
                        &texels[at..at + len],
                    );
                    at += len;
                }
                bytes += texels.len() as u64;
            }
            texture
        });
        let group = gpu.create_bind_group(&BindGroupDesc {
            label: "model",
            layout: self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: words_buffer,
                        offset: 0,
                        size: None,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Texture {
                        texture: texture.unwrap_or(self.blank),
                        dimension: TextureViewDimension::D2Array,
                    },
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(self.sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer {
                        buffer: draw,
                        offset: 0,
                        size: None,
                    },
                },
            ],
        });
        let draws = (
            model.draws[0].start,
            model.draws.clone().map(|r| r.end - r.start),
        );
        self.resident.insert(
            index,
            Resident {
                words: words_buffer,
                indices,
                texture,
                draw,
                group,
                draws,
                bytes,
            },
        );
    }

    pub(crate) fn remove(&mut self, gpu: &mut dyn Gpu, index: u32) {
        if let Some(r) = self.resident.remove(&index) {
            (r.group, r.words, r.indices, r.draw, r.texture).release(gpu);
        }
    }

    /// Model `index`'s first index and its indices' counts per
    /// [`DrawClass`], when it is resident.
    pub(crate) fn draws(&self, index: u32) -> Option<(u32, [u32; 4])> {
        self.resident.get(&index).map(|r| r.draws)
    }

    /// GPU bytes of the resident models.
    pub(crate) fn bytes(&self) -> u64 {
        self.resident.values().map(|r| r.bytes).sum()
    }

    /// Makes the pipelines at `samples`, for the first model node: one
    /// per [`DrawClass`]. Opaque classes shade with `fs_model`, which never
    /// discards, so a tile-based GPU keeps hidden-surface removal; single-
    /// sided opaque materials' back faces are culled by the rasterizer.
    pub(crate) fn make(&mut self, gpu: &mut dyn Gpu, samples: u32) {
        let slot = usize::from(samples == 4);
        if self.pipelines[slot][0].is_some() {
            return;
        }
        // The two shaders differ only in their fragment entry point.
        let module = crate::shader_module(gpu, &MODEL_SHADER);
        for (made, class) in self.pipelines[slot].iter_mut().zip(DrawClass::ALL) {
            let shader = match class {
                DrawClass::Opaque | DrawClass::DoubleSided => &MODEL_SHADER,
                DrawClass::Mask | DrawClass::Blend => &MODEL_MASKED_SHADER,
            };
            let blended = class == DrawClass::Blend;
            *made = Some(gpu.create_render_pipeline(&RenderPipelineDesc {
                label: shader.label,
                bind_group_layouts: &self.layouts,
                vertex: ShaderStage {
                    module,
                    entry_point: shader.vertex,
                    constants: &[],
                },
                fragment: ShaderStage {
                    module,
                    entry_point: shader.fragment,
                    constants: &[],
                },
                targets: &[ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: blended.then_some(BlendState::PremultipliedAlpha),
                }],
                // The other classes draw both faces: masked and blended
                // single-sided materials discard their backs.
                cull_mode: (class == DrawClass::Opaque).then_some(Face::Back),
                depth: Some(DepthState {
                    format: TextureFormat::Depth32Float,
                    write: !blended,
                    compare: CompareFunction::GreaterEqual,
                }),
                topology: PrimitiveTopology::TriangleList,
                sample_count: samples,
            }));
        }
        gpu.destroy_shader_module(module);
    }

    pub(crate) fn pipelines(&self, samples: u32) -> Vec<PipelineId> {
        self.pipelines[usize::from(samples == 4)]
            .iter()
            .flatten()
            .map(|&p| p.into())
            .collect()
    }

    /// Whether the pipelines at `samples` compiled.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu, samples: u32) -> bool {
        self.pipelines[usize::from(samples == 4)][0].is_some()
            && self
                .pipelines(samples)
                .into_iter()
                .all(|p| gpu.pipeline_ready(p))
    }

    /// Group 2's layout: a model's words, textures, sampler and draw.
    pub(crate) fn layout(&self) -> BindGroupLayoutId {
        self.layout
    }

    /// Model node `index`'s draws of model `model` (node group `group`)
    /// into a shadow cascade, after the caster's pipeline and scene and
    /// pass groups: its opaque materials, or with `masked` its masked ones,
    /// which alone discard; none while the model isn't resident.
    pub(crate) fn caster_commands(
        &self,
        group: BindGroupId,
        buffers: &SceneBuffers,
        index: u32,
        model: Option<u32>,
        masked: bool,
    ) -> Vec<RenderCommand> {
        let classes = if masked {
            [DrawClass::Mask].as_slice()
        } else {
            &[DrawClass::Opaque, DrawClass::DoubleSided]
        };
        let Some(r) = model.and_then(|m| self.resident.get(&m)) else {
            return Vec::new();
        };
        let draws: Vec<RenderCommand> = classes
            .iter()
            .map(|&c| c as usize)
            .filter(|&c| r.draws.1[c] > 0)
            .map(|c| RenderCommand::DrawIndexedIndirect {
                buffer: buffers.args,
                offset: crate::args_offset(index) + c as u64 * CLASS_ARGS,
            })
            .collect();
        if draws.is_empty() {
            return Vec::new();
        }
        let mut commands = vec![
            RenderCommand::SetBindGroup { index: 1, group },
            RenderCommand::SetBindGroup {
                index: 2,
                group: r.group,
            },
            RenderCommand::SetIndexBuffer {
                buffer: r.indices,
                format: IndexFormat::Uint32,
                offset: 0,
            },
        ];
        commands.extend(draws);
        commands
    }

    /// Model node `index`'s draws of model `model`, as its 3D pass records
    /// them: none while the model isn't resident.
    pub(crate) fn commands(
        &self,
        samples: u32,
        (scene, group): (BindGroupId, BindGroupId),
        buffers: &SceneBuffers,
        index: u32,
        model: Option<u32>,
    ) -> Vec<RenderCommand> {
        let Some(r) = model.and_then(|m| self.resident.get(&m)) else {
            return Vec::new();
        };
        let pipelines = self.pipelines[usize::from(samples == 4)]
            .map(|p| p.expect("made before its nodes prepare"));
        let mut commands = vec![
            RenderCommand::SetBindGroup {
                index: 0,
                group: scene,
            },
            RenderCommand::SetBindGroup { index: 1, group },
            RenderCommand::SetBindGroup {
                index: 2,
                group: r.group,
            },
            RenderCommand::SetIndexBuffer {
                buffer: r.indices,
                format: IndexFormat::Uint32,
                offset: 0,
            },
        ];
        let args = crate::args_offset(index);
        for ((pipeline, count), offset) in pipelines
            .into_iter()
            .zip(r.draws.1)
            .zip((0..).step_by(CLASS_ARGS as usize))
        {
            if count > 0 {
                commands.extend([
                    RenderCommand::SetPipeline(pipeline),
                    RenderCommand::DrawIndexedIndirect {
                        buffer: buffers.args,
                        offset: args + offset,
                    },
                ]);
            }
        }
        commands
    }
}

impl crate::release::Release for Models {
    fn release(self, gpu: &mut dyn Gpu) {
        let resident: Vec<_> = (self.resident.into_values())
            .map(|r| (r.words, r.indices, r.texture, r.draw, r.group))
            .collect();
        (
            resident,
            self.layout,
            self.pipelines,
            self.sampler,
            self.blank,
        )
            .release(gpu);
    }
}
