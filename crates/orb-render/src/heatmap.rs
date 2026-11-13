//! Per-layer density targets and colour ramps. Density is prepared before
//! the main pass; its fullscreen composite occupies the layer's bundle slot.
//! Targets persist until viewport resize or node replacement.

use crate::TargetFormat;
use crate::offscreen::{Apart, Composites, Mode, Offscreen};
use crate::pipeline::Draw;
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
    colors: Vec<u32>,
    /// Its density, drawn apart; the composite draws nothing while the
    /// node is hidden or has nothing to draw.
    pub(crate) apart: Apart,
}

/// The density target's format: half floats where they blend, else 8 bits
/// ([`Gap::HeatmapPrecision`](crate::Gap::HeatmapPrecision)).
fn density_format(gpu: &dyn Gpu) -> TextureFormat {
    if gpu.caps().float_targets {
        TextureFormat::Rgba16Float
    } else {
        TextureFormat::Rgba8Unorm
    }
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
        let target = TargetFormat {
            color: density_format(gpu),
            sample_count: 1,
        };
        let blend = Some(BlendState::Additive);
        let draw = Draw::new(&HEATMAP_SHADER, &self.layouts, target, blend);
        self.pipeline = Some(draw.make(gpu));
    }

    /// Whether the density pipeline and the composites' compiled.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu, composites: &Composites) -> bool {
        self.pipeline.is_some_and(|p| gpu.pipeline_ready(p.into()))
            && composites.ready(gpu, Mode::Ramp)
    }

    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        self.pipeline.iter().map(|&p| p.into()).collect()
    }

    /// A heatmap's node.
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
        let format = density_format(gpu);
        Offscreen::fit(
            &mut node.apart.offscreen,
            gpu,
            composites,
            ("heatmap density", format),
            (size, 1),
            (Mode::Ramp, Some(node.ramp)),
            node.apart.opacity,
        )
    }

    /// The node's density pass: its draw in each scene block, over the
    /// block's group (`draws`).
    pub(crate) fn prepare(
        &self,
        gpu: &mut dyn Gpu,
        node: &mut NodeHeatmap,
        group: BindGroupId,
        draws: &[(BindGroupId, RenderCommand)],
    ) -> Option<RenderPass> {
        let pipeline = self.pipeline?;
        let target = &node.apart.offscreen.as_ref()?.target;
        let bundle = *node.apart.bundle.get_or_insert_with(|| {
            let mut commands = vec![
                RenderCommand::SetPipeline(pipeline),
                RenderCommand::SetBindGroup { index: 1, group },
            ];
            for (scene, draw) in draws {
                commands.extend([
                    RenderCommand::SetBindGroup {
                        index: 0,
                        group: *scene,
                    },
                    draw.clone(),
                ]);
            }
            gpu.create_render_bundle(
                &RenderBundleDesc {
                    label: "heatmap density",
                    color_formats: &[density_format(gpu)],
                    depth_format: None,
                    sample_count: 1,
                },
                &commands,
            )
        });
        Some(RenderPass {
            label: "heatmap density",
            colors: vec![target.attachment(LoadOp::Clear([0.0; 4]))],
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
