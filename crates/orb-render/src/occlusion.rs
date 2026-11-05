//! Symbols hidden by 3D content (plan §8.5 `Symbols`, depth-tested with
//! `symbol-occlusion`). After the 3D passes, the depth they share (terrain
//! and 3D layers, reversed-Z) is resolved to its nearest sample in a colour
//! texture, which symbols and circles read where they otherwise read the
//! terrain's own depth (`terrain_visibility`).

use orb_hal::*;
use orb_shaders::OCCLUSION_SHADERS;

#[derive(Debug)]
pub(crate) struct Occlusion {
    /// Labels are hidden by 3D layers, not by the terrain alone.
    pub(crate) enabled: bool,
    /// From one sample and from several.
    layouts: [BindGroupLayoutId; 2],
    pipelines: [Option<RenderPipelineId>; 2],
    target: Option<TextureId>,
    size: [u32; 2],
    /// The group over the 3D depth it last read.
    group: Option<(TextureId, BindGroupId)>,
}

impl Occlusion {
    pub(crate) fn new(gpu: &mut dyn Gpu) -> Self {
        let mut layout = |multisampled| {
            gpu.create_bind_group_layout(&BindGroupLayoutDesc {
                label: "occlusion",
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample: if multisampled {
                            TextureSampleType::DepthMultisampled
                        } else {
                            TextureSampleType::Depth
                        },
                        dimension: TextureViewDimension::D2,
                    },
                }],
            })
        };
        let layouts = [layout(false), layout(true)];
        Self {
            enabled: false,
            layouts,
            pipelines: [None; 2],
            target: None,
            size: [0; 2],
            group: None,
        }
    }

    /// The resolved depth, once a frame sized it.
    pub(crate) fn texture(&self) -> Option<TextureId> {
        self.target.filter(|_| self.enabled)
    }

    /// Sizes the resolved depth to the target; whether it changed.
    pub(crate) fn resize(&mut self, gpu: &mut dyn Gpu, size: [u32; 2]) -> bool {
        if !self.enabled || self.size == size {
            return false;
        }
        self.release(gpu);
        self.target = Some(gpu.create_texture(&TextureDesc {
            label: "3d occlusion",
            width: size[0],
            height: size[1],
            array_layers: 1,
            dimension: TextureViewDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            mip_levels: 1,
        }));
        self.size = size;
        true
    }

    pub(crate) fn release(&mut self, gpu: &mut dyn Gpu) {
        if let Some(target) = self.target.take() {
            gpu.destroy_texture(target);
        }
        if let Some((_, group)) = self.group.take() {
            gpu.destroy_bind_group(group);
        }
        self.size = [0; 2];
    }

    /// The pass that resolves `depth` (of `samples`), once its pipeline
    /// compiled; `None` while it compiles or nothing is enabled.
    pub(crate) fn pass(
        &mut self,
        gpu: &mut dyn Gpu,
        depth: TextureId,
        samples: u32,
    ) -> Option<Pass> {
        let target = self.target.filter(|_| self.enabled)?;
        let slot = usize::from(samples > 1);
        let pipeline = *self.pipelines[slot].get_or_insert_with(|| {
            let shader = &OCCLUSION_SHADERS[slot];
            let module = crate::shader_module(gpu, shader);
            gpu.create_render_pipeline(&RenderPipelineDesc {
                label: shader.label,
                bind_group_layouts: &[self.layouts[slot]],
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
                    format: TextureFormat::R32Float,
                    blend: None,
                }],
                depth: None,
                topology: PrimitiveTopology::TriangleList,
                cull_mode: None,
                sample_count: 1,
            })
        });
        if !gpu.pipeline_ready(pipeline.into()) {
            return None;
        }
        if self.group.is_none_or(|(read, _)| read != depth) {
            if let Some((_, group)) = self.group.take() {
                gpu.destroy_bind_group(group);
            }
            let group = gpu.create_bind_group(&BindGroupDesc {
                label: "occlusion",
                layout: self.layouts[slot],
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Texture {
                        texture: depth,
                        dimension: TextureViewDimension::D2,
                    },
                }],
            });
            self.group = Some((depth, group));
        }
        let (_, group) = self.group.expect("made above");
        Some(Pass::Render(RenderPass {
            label: "3d occlusion".into(),
            color: Some(ColorAttachment {
                target,
                resolve: None,
                store: StoreOp::Store,
                load: LoadOp::Clear([0.0; 4]),
            }),
            depth: None,
            commands: vec![
                RenderCommand::SetPipeline(pipeline),
                RenderCommand::SetBindGroup { index: 0, group },
                RenderCommand::Draw {
                    vertices: 0..3,
                    instances: 0..1,
                },
            ],
        }))
    }
}
