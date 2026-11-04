//! Per-layer density targets and colour ramps. Density is prepared before
//! the main pass; its fullscreen composite occupies the layer's bundle slot.
//! Targets persist until viewport resize or node replacement.

use crate::{TargetFormat, draw_pipeline};
use orb_hal::*;
use orb_shaders::{HEATMAP_COLOR_SHADER, HEATMAP_SHADER};

#[derive(Debug)]
pub(crate) struct Heatmaps {
    pub(crate) density_pipeline: Option<RenderPipelineId>,
    pub(crate) color_pipeline: RenderPipelineId,
    layout: BindGroupLayoutId,
    sampler: SamplerId,
}

#[derive(Debug)]
pub(crate) struct NodeHeatmap {
    ramp: TextureId,
    colors: Vec<u32>,
    target: Option<TextureId>,
    size: [u32; 2],
    pub(crate) group: Option<BindGroupId>,
    pub(crate) bundle: Option<RenderBundleId>,
}

impl NodeHeatmap {
    /// Drops the recorded draw, so the next frame records it over the
    /// current bind groups.
    pub(crate) fn forget_bundle(&mut self, gpu: &mut dyn Gpu) {
        if let Some(bundle) = self.bundle.take() {
            gpu.destroy_render_bundle(bundle);
        }
    }
}

impl Heatmaps {
    pub(crate) fn new(
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        scene: BindGroupLayoutId,
        node: BindGroupLayoutId,
    ) -> Self {
        let fragment = ShaderStages::FRAGMENT;
        let texture = |binding| BindGroupLayoutEntry {
            binding,
            visibility: fragment,
            ty: BindingType::Texture {
                sample: TextureSampleType::Float { filterable: true },
                dimension: TextureViewDimension::D2,
            },
        };
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "heatmap composite",
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | fragment,
                    ty: BindingType::Uniform,
                },
                texture(1),
                texture(2),
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: fragment,
                    ty: BindingType::Sampler { filtering: true },
                },
            ],
        });
        let density_pipeline = if gpu.caps().rgba16float_blendable {
            let shader = HEATMAP_SHADER;
            let module = gpu.create_shader_module(&ShaderModuleDesc {
                label: shader.label,
                wgsl: shader.source().into(),
            });
            Some(gpu.create_render_pipeline(&RenderPipelineDesc {
                cull_mode: None,
                label: shader.label,
                bind_group_layouts: &[scene, node],
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
                    format: TextureFormat::Rgba16Float,
                    blend: Some(BlendState::Additive),
                }],
                depth: None,
                topology: PrimitiveTopology::TriangleList,
                sample_count: 1,
            }))
        } else {
            None
        };
        let color_pipeline = draw_pipeline(
            gpu,
            &HEATMAP_COLOR_SHADER,
            false,
            PrimitiveTopology::TriangleList,
            target,
            &[scene, layout],
        );
        let sampler = gpu.create_sampler(&SamplerDesc {
            label: "heatmap linear",
            filter: FilterMode::Linear,
            address: AddressMode::ClampToEdge,
            mipmap_filter: FilterMode::Nearest,
        });
        Self {
            density_pipeline,
            color_pipeline,
            layout,
            sampler,
        }
    }

    pub(crate) fn node(gpu: &mut dyn Gpu) -> NodeHeatmap {
        let ramp = gpu.create_texture(&TextureDesc {
            label: "heatmap ramp",
            width: 256,
            height: 1,
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            sample_count: 1,
            mip_levels: 1,
        });
        NodeHeatmap {
            ramp,
            colors: Vec::new(),
            target: None,
            size: [0; 2],
            group: None,
            bundle: None,
        }
    }

    pub(crate) fn release(gpu: &mut dyn Gpu, node: NodeHeatmap) {
        if let Some(bundle) = node.bundle {
            gpu.destroy_render_bundle(bundle);
        }
        if let Some(group) = node.group {
            gpu.destroy_bind_group(group);
        }
        if let Some(target) = node.target {
            gpu.destroy_texture(target);
        }
        gpu.destroy_texture(node.ramp);
    }

    pub(crate) fn set_ramp(gpu: &mut dyn Gpu, node: &mut NodeHeatmap, colors: &[u32]) {
        assert_eq!(colors.len(), 256);
        if colors == node.colors {
            return;
        }
        gpu.write_texture(
            &TextureRegion {
                texture: node.ramp,
                x: 0,
                y: 0,
                layer: 0,
                width: 256,
                height: 1,
            },
            bytemuck::cast_slice(colors),
        );
        node.colors = colors.to_vec();
    }

    /// Returns true if the main bundle's texture bindings changed.
    pub(crate) fn resize(
        &self,
        gpu: &mut dyn Gpu,
        node: &mut NodeHeatmap,
        uniform: BindingResource,
        size: [u32; 2],
    ) -> bool {
        if self.density_pipeline.is_none() || node.size == size {
            return false;
        }
        if let Some(group) = node.group.take() {
            gpu.destroy_bind_group(group);
        }
        if let Some(target) = node.target.take() {
            gpu.destroy_texture(target);
        }
        let target = gpu.create_texture(&TextureDesc {
            label: "heatmap density",
            width: size[0],
            height: size[1],
            array_layers: 1,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            mip_levels: 1,
        });
        node.target = Some(target);
        node.size = size;
        let texture = |texture| BindingResource::Texture {
            texture,
            dimension: TextureViewDimension::D2,
        };
        node.group = Some(gpu.create_bind_group(&BindGroupDesc {
            label: "heatmap composite",
            layout: self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: texture(target),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: texture(node.ramp),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(self.sampler),
                },
            ],
        }));
        true
    }

    pub(crate) fn prepare(
        &self,
        gpu: &mut dyn Gpu,
        node: &mut NodeHeatmap,
        scene: BindGroupId,
        group: BindGroupId,
        args: BufferId,
        index: u32,
    ) -> Option<RenderPass> {
        let pipeline = self.density_pipeline?;
        let target = node.target?;
        let bundle = *node.bundle.get_or_insert_with(|| {
            gpu.create_render_bundle(
                &RenderBundleDesc {
                    label: "heatmap density",
                    color_formats: &[TextureFormat::Rgba16Float],
                    depth_format: None,
                    sample_count: 1,
                },
                &[
                    RenderCommand::SetPipeline(pipeline),
                    RenderCommand::SetBindGroup {
                        index: 0,
                        group: scene,
                    },
                    RenderCommand::SetBindGroup { index: 1, group },
                    RenderCommand::DrawIndirect {
                        buffer: args,
                        offset: crate::args_offset(index),
                    },
                ],
            )
        });
        Some(RenderPass {
            label: "heatmap density".into(),
            color: ColorAttachment {
                store: StoreOp::Store,
                target,
                resolve: None,
                load: LoadOp::Clear([0.0; 4]),
            },
            depth: None,
            timestamps: None,
            commands: vec![RenderCommand::ExecuteBundles(vec![bundle])],
        })
    }
}
