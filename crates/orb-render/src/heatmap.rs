//! Per-layer density targets and colour ramps. Density is prepared before
//! the main pass; its fullscreen composite occupies the layer's bundle slot.
//! Targets persist until viewport resize or node replacement.

use crate::offscreen::{Apart, Composites, Mode, Offscreen};
use orb_hal::*;
use orb_shaders::HEATMAP_SHADER;

#[derive(Debug)]
pub(crate) struct Heatmaps {
    /// The density pipeline, made for the first heatmap node; none without
    /// blendable half-float targets, which Compat may lack.
    pipeline: Option<RenderPipelineId>,
    density: bool,
    layouts: [BindGroupLayoutId; 2],
}

#[derive(Debug)]
pub(crate) struct NodeHeatmap {
    ramp: TextureId,
    colors: Vec<u32>,
    /// Its density, drawn apart; the composite draws nothing while the
    /// node is hidden or has nothing to draw.
    pub(crate) apart: Apart,
}

impl Heatmaps {
    pub(crate) fn new(
        gpu: &mut dyn Gpu,
        scene: BindGroupLayoutId,
        node: BindGroupLayoutId,
    ) -> Self {
        Self {
            pipeline: None,
            density: gpu.caps().rgba16float_blendable,
            layouts: [scene, node],
        }
    }

    /// Makes the density pipeline, for the first heatmap node.
    pub(crate) fn make(&mut self, gpu: &mut dyn Gpu) {
        if self.pipeline.is_some() || !self.density {
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
    }

    /// Whether the density pipeline and the composites' compiled.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu, composites: &Composites) -> bool {
        self.pipeline.is_some_and(|p| gpu.pipeline_ready(p.into()))
            && composites.ready(gpu, Mode::Ramp)
    }

    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        self.pipeline.iter().map(|&p| p.into()).collect()
    }

    pub(crate) fn node(gpu: &mut dyn Gpu) -> NodeHeatmap {
        let ramp = gpu.create_texture(&TextureDesc {
            label: "heatmap ramp",
            width: 256,
            height: 1,
            array_layers: 1,
            dimension: TextureViewDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            sample_count: 1,
            mip_levels: 1,
        });
        NodeHeatmap {
            ramp,
            colors: Vec::new(),
            apart: Apart::default(),
        }
    }

    pub(crate) fn release(gpu: &mut dyn Gpu, node: NodeHeatmap) {
        node.apart.release(gpu);
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
                level: 0,
            },
            bytemuck::cast_slice(colors),
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
        self.density
            && Offscreen::fit(
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
