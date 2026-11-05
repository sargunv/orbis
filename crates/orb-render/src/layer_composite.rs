//! Layer composites: `*-layer-opacity` and `*-layer-blend` (style-spec 26).
//! A style layer's nodes draw into a texture of their own. It composites
//! into the map once, at the layer's opacity, so its overlaps don't
//! accumulate (GL JS `draw_layer_opacity.ts`). A group is a run of
//! consecutive nodes of one layer below opacity 1 or in a blend mode other
//! than normal. Otherwise the layer draws as usual.
//!
//! Normal groups composite with fixed-function premultiplied blending,
//! inside the main pass. Every other mode needs the map beneath it. The
//! main pass splits at each such group into canvases, two textures used in
//! turn. Each canvas pass after the first opens with a full-screen
//! draw: the group's layer over the previous canvas, by W3C Compositing 1's
//! formula. Then the map's draws continue on top. The last pass draws
//! into the target, so a map without such groups keeps its one pass.

use std::ops::Range;

use crate::{TargetFormat, draw_pipeline_blending};
use orb_hal::*;
use orb_shaders::{LAYER_BLEND_SHADER, LAYER_COMPOSITE_SHADER};

/// How a layer composited once combines with the map beneath it: W3C
/// Compositing 1's modes, as `fill-layer-blend` and `line-layer-blend`
/// name them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Blend {
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
    Color,
    Luminosity,
    PlusLighter,
}

impl Blend {
    /// The shader's mode number (`blend_mix` in `layer_composite.wgsl`).
    fn mode(self) -> f32 {
        match self {
            Blend::Normal => 0.0,
            Blend::Multiply => 1.0,
            Blend::Screen => 2.0,
            Blend::Overlay => 3.0,
            Blend::Color => 4.0,
            Blend::Luminosity => 5.0,
            Blend::PlusLighter => 6.0,
        }
    }
}

#[derive(Debug)]
pub(crate) struct LayerComposites {
    composite: RenderPipelineId,
    blend: RenderPipelineId,
    composite_layout: BindGroupLayoutId,
    blend_layout: BindGroupLayoutId,
    target: TargetFormat,
    pub(crate) groups: Vec<Group>,
    /// The two canvases the main pass splits into, while a group blends
    /// in a mode other than normal.
    canvases: Vec<Attachment>,
}

/// A colour texture to draw into and sample, with its multisampled
/// attachment when the target has one.
#[derive(Debug, Clone, Copy)]
struct Attachment {
    color: TextureId,
    multisample: Option<TextureId>,
}

impl Attachment {
    fn color_attachment(self, load: LoadOp<[f64; 4]>) -> ColorAttachment {
        ColorAttachment {
            target: self.multisample.unwrap_or(self.color),
            resolve: self.multisample.map(|_| self.color),
            store: if self.multisample.is_some() {
                StoreOp::Discard
            } else {
                StoreOp::Store
            },
            load,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Group {
    pub(crate) nodes: Range<usize>,
    pub(crate) blend: Blend,
    uniform: BufferId,
    opacity: f32,
    layer: Attachment,
    group: BindGroupId,
    pub(crate) bundle: Option<RenderBundleId>,
}

impl LayerComposites {
    pub(crate) fn new(gpu: &mut dyn Gpu, target: TargetFormat) -> Self {
        let entry = |binding, ty| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::FRAGMENT,
            ty,
        };
        let texture = BindingType::Texture {
            sample: TextureSampleType::Float { filterable: false },
            dimension: TextureViewDimension::D2,
        };
        let composite_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "layer composite",
            entries: &[entry(0, BindingType::Uniform), entry(1, texture)],
        });
        let blend_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "layer blend",
            entries: &[
                entry(0, BindingType::Uniform),
                entry(1, texture),
                entry(2, texture),
            ],
        });
        let pipeline = |gpu: &mut dyn Gpu, shader, blend, layout| {
            draw_pipeline_blending(
                gpu,
                shader,
                false,
                PrimitiveTopology::TriangleList,
                target,
                &[layout],
                blend,
            )
        };
        let composite = pipeline(
            gpu,
            &LAYER_COMPOSITE_SHADER,
            Some(BlendState::PremultipliedAlpha),
            composite_layout,
        );
        // The blend writes the composited result over the canvas.
        let blend = pipeline(gpu, &LAYER_BLEND_SHADER, None, blend_layout);
        Self {
            composite,
            blend,
            composite_layout,
            blend_layout,
            target,
            groups: Vec::new(),
            canvases: Vec::new(),
        }
    }

    fn attachment(&self, gpu: &mut dyn Gpu, label: &'static str, size: [u32; 2]) -> Attachment {
        let texture = |gpu: &mut dyn Gpu, samples, usage| {
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
        Attachment {
            color: texture(
                gpu,
                1,
                TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            ),
            multisample: (self.target.sample_count > 1).then(|| {
                texture(
                    gpu,
                    self.target.sample_count,
                    TextureUsages::RENDER_ATTACHMENT | TextureUsages::TRANSIENT_ATTACHMENT,
                )
            }),
        }
    }

    /// Replaces the groups with `runs` of node indices, their opacities
    /// and blend modes, each with a texture of `size`.
    pub(crate) fn regroup(
        &mut self,
        gpu: &mut dyn Gpu,
        runs: Vec<(Range<usize>, f32, Blend)>,
        size: [u32; 2],
    ) {
        self.clear(gpu);
        if runs.iter().any(|&(_, _, b)| b != Blend::Normal) {
            self.canvases = vec![
                self.attachment(gpu, "map canvas", size),
                self.attachment(gpu, "map canvas", size),
            ];
        }
        let mut backdrops = 0;
        for (nodes, opacity, blend) in runs {
            let layer = self.attachment(gpu, "layer composite", size);
            let uniform = gpu.create_buffer(&BufferDesc {
                label: "layer composite",
                size: 16,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });
            gpu.write_buffer(
                uniform,
                0,
                bytemuck::cast_slice(&[opacity, blend.mode(), 0.0, 0.0]),
            );
            let mut entries = vec![
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::buffer(uniform),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Texture {
                        texture: layer.color,
                        dimension: TextureViewDimension::D2,
                    },
                },
            ];
            // The k-th blending group reads the canvas pass k drew.
            if blend != Blend::Normal {
                entries.push(BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Texture {
                        texture: self.canvases[backdrops % 2].color,
                        dimension: TextureViewDimension::D2,
                    },
                });
                backdrops += 1;
            }
            let group = gpu.create_bind_group(&BindGroupDesc {
                label: "layer composite",
                layout: if blend == Blend::Normal {
                    self.composite_layout
                } else {
                    self.blend_layout
                },
                entries: &entries,
            });
            self.groups.push(Group {
                nodes,
                blend,
                uniform,
                opacity,
                layer,
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
                bytemuck::cast_slice(&[opacity, group.blend.mode(), 0.0, 0.0]),
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

    /// A group's composite into the map: in the main pass's bundle for
    /// normal blending, or opening the canvas pass after a split.
    pub(crate) fn composite_commands(&self, group: &Group) -> [RenderCommand; 3] {
        [
            RenderCommand::SetPipeline(if group.blend == Blend::Normal {
                self.composite
            } else {
                self.blend
            }),
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
                label: "layer composite".into(),
                color: g.layer.color_attachment(LoadOp::Clear([0.0; 4])),
                depth: None,
                timestamps: None,
                commands: vec![RenderCommand::ExecuteBundles(vec![g.bundle?])],
            }))
        })
    }

    /// The map's passes: `first` (the sky) and the `segments` of the main
    /// bundle, split at each group blending other than normally, then
    /// `last` (the scattering), ending in `target`.
    pub(crate) fn main_passes(
        &self,
        mut first: Vec<RenderCommand>,
        segments: &[RenderBundleId],
        last: Vec<RenderCommand>,
        target: TextureId,
        clear: [f64; 4],
    ) -> Vec<Pass> {
        let mut blending = self.groups.iter().filter(|g| g.blend != Blend::Normal);
        let mut last = Some(last);
        segments
            .iter()
            .enumerate()
            .map(|(k, &segment)| {
                let mut commands = if k == 0 {
                    std::mem::take(&mut first)
                } else {
                    let group = blending.next().expect("a segment per blending group");
                    self.composite_commands(group).to_vec()
                };
                commands.push(RenderCommand::ExecuteBundles(vec![segment]));
                // The blend draw covers every pixel: only the first clears.
                let load = LoadOp::Clear(if k == 0 { clear } else { [0.0; 4] });
                let color = if k + 1 == segments.len() {
                    commands.extend(last.take().expect("one last pass"));
                    ColorAttachment {
                        store: StoreOp::Store,
                        target,
                        resolve: None,
                        load,
                    }
                } else {
                    self.canvases[k % 2].color_attachment(load)
                };
                Pass::Render(RenderPass {
                    label: "main".into(),
                    color,
                    depth: None,
                    timestamps: None,
                    commands,
                })
            })
            .collect()
    }

    pub(crate) fn clear(&mut self, gpu: &mut dyn Gpu) {
        for g in self.groups.drain(..) {
            if let Some(bundle) = g.bundle {
                gpu.destroy_render_bundle(bundle);
            }
            gpu.destroy_bind_group(g.group);
            gpu.destroy_buffer(g.uniform);
            destroy(gpu, g.layer);
        }
        for canvas in self.canvases.drain(..) {
            destroy(gpu, canvas);
        }
    }
}

fn destroy(gpu: &mut dyn Gpu, attachment: Attachment) {
    for texture in [Some(attachment.color), attachment.multisample]
        .into_iter()
        .flatten()
    {
        gpu.destroy_texture(texture);
    }
}
