//! `fill-layer-opacity` and `line-layer-opacity` (style-spec 26): a style
//! layer's nodes draw into a texture of their own, which composites into
//! the map once at the layer's opacity, so its overlaps don't accumulate
//! (GL JS `draw_layer_opacity.ts`). A group is a run of consecutive nodes
//! of one layer whose opacity is below 1; at 1 the layer draws as usual.

use std::ops::Range;

use crate::{TargetFormat, draw_pipeline};
use orb_hal::*;
use orb_shaders::LAYER_OPACITY_SHADER;

#[derive(Debug)]
pub(crate) struct LayerOpacities {
    composite: RenderPipelineId,
    layout: BindGroupLayoutId,
    target: TargetFormat,
    pub(crate) groups: Vec<Group>,
}

#[derive(Debug)]
pub(crate) struct Group {
    pub(crate) nodes: Range<usize>,
    uniform: BufferId,
    opacity: f32,
    color: TextureId,
    multisample: Option<TextureId>,
    group: BindGroupId,
    pub(crate) bundle: Option<RenderBundleId>,
}

impl LayerOpacities {
    pub(crate) fn new(gpu: &mut dyn Gpu, target: TargetFormat) -> Self {
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "layer opacity",
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Uniform,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample: TextureSampleType::Float { filterable: false },
                        dimension: TextureViewDimension::D2,
                    },
                },
            ],
        });
        let composite = draw_pipeline(
            gpu,
            &LAYER_OPACITY_SHADER,
            false,
            PrimitiveTopology::TriangleList,
            target,
            &[layout],
        );
        Self {
            composite,
            layout,
            target,
            groups: Vec::new(),
        }
    }

    /// Replaces the groups with `runs` of node indices and their
    /// opacities, each with a texture of `size`.
    pub(crate) fn regroup(
        &mut self,
        gpu: &mut dyn Gpu,
        runs: Vec<(Range<usize>, f32)>,
        size: [u32; 2],
    ) {
        self.clear(gpu);
        for (nodes, opacity) in runs {
            let texture = |gpu: &mut dyn Gpu, label, samples, usage| {
                gpu.create_texture(&TextureDesc {
                    label,
                    width: size[0],
                    height: size[1],
                    array_layers: 1,
                    dimension: TextureViewDimension::D2,
                    format: self.target.color,
                    usage,
                    sample_count: samples,
                    mip_levels: 1,
                })
            };
            let color = texture(
                gpu,
                "layer opacity",
                1,
                TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            );
            let multisample = (self.target.sample_count > 1).then(|| {
                texture(
                    gpu,
                    "layer opacity multisample",
                    self.target.sample_count,
                    TextureUsages::RENDER_ATTACHMENT | TextureUsages::TRANSIENT_ATTACHMENT,
                )
            });
            let uniform = gpu.create_buffer(&BufferDesc {
                label: "layer opacity",
                size: 16,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });
            gpu.write_buffer(uniform, 0, bytemuck::cast_slice(&[opacity, 0.0, 0.0, 0.0]));
            let group = gpu.create_bind_group(&BindGroupDesc {
                label: "layer opacity",
                layout: self.layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::buffer(uniform),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Texture {
                            texture: color,
                            dimension: TextureViewDimension::D2,
                        },
                    },
                ],
            });
            self.groups.push(Group {
                nodes,
                uniform,
                opacity,
                color,
                multisample,
                group,
                bundle: None,
            });
        }
    }

    /// Writes a group's opacity, if it changed.
    pub(crate) fn set_opacity(&mut self, gpu: &mut dyn Gpu, node: usize, opacity: f32) {
        if let Some(group) = self.groups.iter_mut().find(|g| g.nodes.contains(&node))
            && group.opacity != opacity
        {
            group.opacity = opacity;
            gpu.write_buffer(
                group.uniform,
                0,
                bytemuck::cast_slice(&[opacity, 0.0, 0.0, 0.0]),
            );
        }
    }

    /// The group node `i` starts, if any, and whether it belongs to one.
    pub(crate) fn at(&self, i: usize) -> (Option<&Group>, bool) {
        match self.groups.iter().find(|g| g.nodes.contains(&i)) {
            Some(g) => ((g.nodes.start == i).then_some(g), true),
            None => (None, false),
        }
    }

    /// A group's composite into the map.
    pub(crate) fn composite_commands(&self, group: &Group) -> [RenderCommand; 3] {
        [
            RenderCommand::SetPipeline(self.composite),
            RenderCommand::SetBindGroup {
                index: 0,
                group: group.group,
            },
            RenderCommand::Draw {
                vertices: 0..3,
                instances: 0..1,
            },
        ]
    }

    /// Each group's pass: its nodes' draws into its texture.
    pub(crate) fn passes(&self) -> impl Iterator<Item = Pass> + '_ {
        self.groups.iter().filter_map(|g| {
            Some(Pass::Render(RenderPass {
                label: "layer opacity".into(),
                color: ColorAttachment {
                    target: g.multisample.unwrap_or(g.color),
                    resolve: g.multisample.map(|_| g.color),
                    store: if g.multisample.is_some() {
                        StoreOp::Discard
                    } else {
                        StoreOp::Store
                    },
                    load: LoadOp::Clear([0.0; 4]),
                },
                depth: None,
                timestamps: None,
                commands: vec![RenderCommand::ExecuteBundles(vec![g.bundle?])],
            }))
        })
    }

    pub(crate) fn clear(&mut self, gpu: &mut dyn Gpu) {
        for g in self.groups.drain(..) {
            if let Some(bundle) = g.bundle {
                gpu.destroy_render_bundle(bundle);
            }
            gpu.destroy_bind_group(g.group);
            gpu.destroy_buffer(g.uniform);
            for texture in [Some(g.color), g.multisample].into_iter().flatten() {
                gpu.destroy_texture(texture);
            }
        }
    }
}
