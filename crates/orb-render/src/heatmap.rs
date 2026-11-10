//! Per-layer density targets and colour ramps. Density is prepared before
//! the main pass; its fullscreen composite occupies the layer's bundle slot.
//! Targets persist until viewport resize or node replacement.

use crate::offscreen::{Apart, Composites, Mode, Offscreen};
use orb_hal::*;
use orb_shaders::HEATMAP_SHADER;

#[derive(Debug)]
pub(crate) struct Heatmaps {
    /// The density pipeline, made for the first heatmap node.
    pipeline: Option<RenderPipelineId>,
    layouts: [BindGroupLayoutId; 2],
}

#[derive(Debug)]
pub(crate) struct NodeHeatmap {
    ramp: TextureId,
    /// Its ramp holds colours in linear light.
    linear: bool,
    colors: Vec<u32>,
    /// Its density, drawn apart; the composite draws nothing while the
    /// node is hidden or has nothing to draw.
    pub(crate) apart: Apart,
}

impl Heatmaps {
    pub(crate) fn new(scene: BindGroupLayoutId, node: BindGroupLayoutId) -> Self {
        Self {
            pipeline: None,
            layouts: [scene, node],
        }
    }

    /// Makes the density pipeline, for the first heatmap node.
    pub(crate) fn make(&mut self, gpu: &mut dyn Gpu) {
        if self.pipeline.is_some() {
            return;
        }
        let shader = HEATMAP_SHADER;
        let module = crate::shader_module(gpu, &shader);
        self.pipeline = Some(gpu.create_render_pipeline(&RenderPipelineDesc {
            cull_mode: None,
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
                format: TextureFormat::Rgba16Float,
                blend: Some(BlendState::Additive),
            }],
            depth: None,
            topology: PrimitiveTopology::TriangleList,
            sample_count: 1,
        }));
        gpu.destroy_shader_module(module);
    }

    /// Whether the density pipeline and the composites' compiled.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu, composites: &Composites) -> bool {
        self.pipeline.is_some_and(|p| gpu.pipeline_ready(p.into()))
            && composites.ready(gpu, Mode::Ramp)
    }

    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        self.pipeline.iter().map(|&p| p.into()).collect()
    }

    /// A heatmap's node; `linear` when colours mix in linear light, whose
    /// ramp the composite samples decoded from an sRGB texture.
    pub(crate) fn node(gpu: &mut dyn Gpu, linear: bool) -> NodeHeatmap {
        let ramp = gpu.create_texture(&TextureDesc {
            label: "heatmap ramp",
            width: 256,
            height: 1,
            array_layers: 1,
            dimension: TextureViewDimension::D2,
            format: if linear {
                TextureFormat::Rgba8UnormSrgb
            } else {
                TextureFormat::Rgba8Unorm
            },
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            sample_count: 1,
            mip_levels: 1,
            view_formats: &[],
        });
        NodeHeatmap {
            ramp,
            linear,
            colors: Vec::new(),
            apart: Apart::default(),
        }
    }

    pub(crate) fn set_ramp(gpu: &mut dyn Gpu, node: &mut NodeHeatmap, colors: &[u32]) {
        assert_eq!(colors.len(), 256);
        if colors == node.colors {
            return;
        }
        // In linear light each texel holds its colour decoded and
        // premultiplied, encoded as an sRGB texture stores it, so sampling
        // decodes and filters it in linear light.
        let linear: Vec<u32>;
        let texels = if node.linear {
            linear = colors.iter().map(|&c| encoded_linear(c)).collect();
            &linear
        } else {
            colors
        };
        gpu.write_texture(
            &TextureRegion {
                texture: node.ramp,
                x: 0,
                y: 0,
                layer: 0,
                width: 256,
                height: 1,
                level: 0,
            },
            bytemuck::cast_slice(texels),
        );
        node.colors = colors.to_vec();
    }

    /// Gives a node its density target of `size` and its composite;
    /// whether the main bundle's bindings changed.
    pub(crate) fn resize(
        &self,
        gpu: &mut dyn Gpu,
        composites: &mut Composites,
        node: &mut NodeHeatmap,
        size: [u32; 2],
    ) -> bool {
        Offscreen::fit(
            &mut node.apart.offscreen,
            gpu,
            composites,
            ("heatmap density", &[TextureFormat::Rgba16Float]),
            (size, 1),
            (Mode::Ramp, Some(node.ramp)),
            node.apart.opacity,
        )
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
        let pipeline = self.pipeline?;
        let target = &node.apart.offscreen.as_ref()?.target;
        let bundle = *node.apart.bundle.get_or_insert_with(|| {
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
            colors: vec![target.attachment(0, LoadOp::Clear([0.0; 4]))],
            depth: None,
            commands: vec![RenderCommand::ExecuteBundles(vec![bundle])],
        })
    }
}

impl crate::release::Release for Heatmaps {
    fn release(self, gpu: &mut dyn Gpu) {
        self.pipeline.release(gpu);
    }
}

impl crate::release::Release for NodeHeatmap {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.apart, self.ramp).release(gpu);
    }
}

/// A premultiplied rgba8 colour in linear light, each component encoded as
/// an sRGB texture stores it.
fn encoded_linear(c: u32) -> u32 {
    let [r, g, b, a] = c.to_le_bytes().map(|v| f32::from(v) / 255.0);
    let [r, g, b, a] = crate::color_in([r, g, b, a], true);
    let encode = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        let e = if v <= 0.003_130_8 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (e * 255.0).round() as u8
    };
    u32::from_le_bytes([encode(r), encode(g), encode(b), (a * 255.0).round() as u8])
}
