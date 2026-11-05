//! The lit map (#91, decision 0048): the map's surface layers draw their
//! colours as albedo, and their nodes' materials, into a material buffer
//! (`map_out_material.wgsl`); one full-screen draw in the main pass then
//! shades it under the style's lights (`map_light.wgsl`), and labels draw
//! after it, unlit.

use orb_hal::*;
use orb_shaders::{MAP_LIGHT_SHADER, Shader};

use crate::offscreen::Target;
use crate::{Family, TargetFormat};

/// The material buffer's attachments: albedo, then roughness, metalness and
/// the self-lit share; both premultiplied.
pub(crate) const MATERIAL_FORMATS: [TextureFormat; 2] = [TextureFormat::Rgba8Unorm; 2];

/// What a map layer is made of under the lit map's lights: its colours are
/// its albedo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeMaterial {
    /// 0 mirror-smooth to 1 matte.
    pub roughness: f32,
    /// 0 a dielectric to 1 a metal.
    pub metalness: f32,
    /// The share of its colour that shows whatever the lights.
    pub self_lit: f32,
}

impl Default for NodeMaterial {
    fn default() -> Self {
        Self {
            roughness: 1.0,
            metalness: 0.0,
            self_lit: 0.0,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Lit {
    /// Lit pipelines' group 2: the nodes' materials, by node index.
    pub(crate) material_layout: BindGroupLayoutId,
    materials: Vec<[f32; 4]>,
    buffer: Option<(BufferId, BindGroupId, usize)>,
    dirty: bool,
    light_layout: BindGroupLayoutId,
    light: Option<RenderPipelineId>,
    /// The material buffer, its samples, and the lighting pass's group.
    target: Option<(Target, [u32; 2], u32, BindGroupId)>,
    /// The fill, outline, line and circle pipelines, and the image
    /// families', into the material buffer.
    paint: [Option<RenderPipelineId>; 4],
    images: [Option<RenderPipelineId>; 3],
    /// The surface layers' draws, recorded with the main bundle.
    pub(crate) surface: Option<RenderBundleId>,
}

impl Lit {
    pub(crate) fn new(gpu: &mut dyn Gpu) -> Self {
        let material_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "node materials",
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Storage { read_only: true },
            }],
        });
        let texture = |binding| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample: TextureSampleType::Float { filterable: false },
                dimension: TextureViewDimension::D2,
            },
        };
        let light_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "map light",
            entries: &[texture(0), texture(1)],
        });
        Self {
            material_layout,
            materials: Vec::new(),
            buffer: None,
            dirty: true,
            light_layout,
            light: None,
            target: None,
            paint: [None; 4],
            images: [None; 3],
            surface: None,
        }
    }

    /// Node `node`'s material; written with the next frame.
    pub(crate) fn set_material(&mut self, node: usize, m: NodeMaterial) {
        if self.materials.len() <= node {
            self.materials.resize(node + 1, [1.0, 0.0, 0.0, 0.0]);
        }
        let words = [m.roughness, m.metalness, m.self_lit, 0.0];
        if self.materials[node] != words {
            self.materials[node] = words;
            self.dirty = true;
        }
    }

    /// Writes the materials of `nodes` nodes; whether the group changed,
    /// so the surface records again.
    pub(crate) fn upload(&mut self, gpu: &mut dyn Gpu, nodes: usize) -> bool {
        if self.materials.len() < nodes {
            self.materials.resize(nodes, [1.0, 0.0, 0.0, 0.0]);
            self.dirty = true;
        }
        let mut regrouped = false;
        if self
            .buffer
            .is_none_or(|(_, _, capacity)| capacity < nodes.max(1))
        {
            if let Some((buffer, group, _)) = self.buffer.take() {
                gpu.destroy_bind_group(group);
                gpu.destroy_buffer(buffer);
            }
            let capacity = nodes.max(1).next_power_of_two();
            let buffer = gpu.create_buffer(&BufferDesc {
                label: "node materials",
                size: capacity as u64 * 16,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
            let group = gpu.create_bind_group(&BindGroupDesc {
                label: "node materials",
                layout: self.material_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::buffer(buffer),
                }],
            });
            self.buffer = Some((buffer, group, capacity));
            self.dirty = true;
            regrouped = true;
        }
        if std::mem::take(&mut self.dirty) && !self.materials.is_empty() {
            let (buffer, _, _) = self.buffer.expect("made above");
            gpu.write_buffer(buffer, 0, bytemuck::cast_slice(&self.materials));
        }
        regrouped
    }

    /// The materials' group, once uploaded.
    pub(crate) fn material_group(&self) -> Option<BindGroupId> {
        self.buffer.map(|(_, group, _)| group)
    }

    /// Sizes the material buffer; whether it changed (the surface's bundle
    /// and the lighting group follow it).
    pub(crate) fn resize(&mut self, gpu: &mut dyn Gpu, size: [u32; 2], samples: u32) -> bool {
        if self
            .target
            .as_ref()
            .is_some_and(|(_, s, n, _)| *s == size && *n == samples)
        {
            return false;
        }
        self.release_target(gpu);
        let target = Target::new(gpu, "material", &MATERIAL_FORMATS, size, samples);
        let texture = |binding, texture| BindGroupEntry {
            binding,
            resource: BindingResource::Texture {
                texture,
                dimension: TextureViewDimension::D2,
            },
        };
        let group = gpu.create_bind_group(&BindGroupDesc {
            label: "map light",
            layout: self.light_layout,
            entries: &[texture(0, target.texture(0)), texture(1, target.texture(1))],
        });
        self.target = Some((target, size, samples, group));
        true
    }

    fn release_target(&mut self, gpu: &mut dyn Gpu) {
        if let Some((target, _, _, group)) = self.target.take() {
            gpu.destroy_bind_group(group);
            target.destroy(gpu);
        }
    }

    /// Makes the lighting pipeline into the map's `target`.
    pub(crate) fn make_light(
        &mut self,
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        draw_layout: BindGroupLayoutId,
    ) {
        if self.light.is_none() {
            self.light = Some(crate::draw_pipeline(
                gpu,
                &MAP_LIGHT_SHADER,
                PrimitiveTopology::TriangleList,
                target,
                &[draw_layout, self.light_layout],
            ));
        }
    }

    /// The lighting pass's draw over the scene group `scene`, once its
    /// pipeline compiled and the buffer is sized.
    pub(crate) fn light_commands(
        &self,
        gpu: &mut dyn Gpu,
        scene: BindGroupId,
    ) -> Option<[RenderCommand; 4]> {
        let light = self.light.filter(|&p| gpu.pipeline_ready(p.into()))?;
        let (_, _, _, group) = self.target.as_ref()?;
        Some([
            RenderCommand::SetPipeline(light),
            RenderCommand::SetBindGroup {
                index: 0,
                group: scene,
            },
            RenderCommand::SetBindGroup {
                index: 1,
                group: *group,
            },
            RenderCommand::Draw {
                vertices: 0..3,
                instances: 0..1,
            },
        ])
    }

    /// The pass that draws the surface's bundle into the material buffer.
    pub(crate) fn surface_pass(&self) -> Option<Pass> {
        let (target, ..) = self.target.as_ref()?;
        let bundle = self.surface?;
        Some(Pass::Render(RenderPass {
            label: "lit surface".into(),
            colors: (0..MATERIAL_FORMATS.len())
                .map(|i| target.attachment(i, LoadOp::Clear([0.0; 4])))
                .collect(),
            depth: None,
            commands: vec![RenderCommand::ExecuteBundles(vec![bundle])],
        }))
    }

    pub(crate) fn forget_surface(&mut self, gpu: &mut dyn Gpu) {
        if let Some(bundle) = self.surface.take() {
            gpu.destroy_render_bundle(bundle);
        }
    }

    /// `family`'s pipeline into the material buffer at `samples`, made for
    /// its first node; `layouts` are its scene and node groups'.
    pub(crate) fn pipeline(
        &mut self,
        gpu: &mut dyn Gpu,
        family: Family,
        samples: u32,
        layouts: [BindGroupLayoutId; 2],
    ) -> Option<RenderPipelineId> {
        use orb_shaders::*;
        let (slot, shader, topology): (_, Shader, _) = match family {
            Family::Fill => (
                &mut self.paint[0],
                FILL_SHADER,
                PrimitiveTopology::TriangleList,
            ),
            Family::FillOutline => (
                &mut self.paint[1],
                FILL_OUTLINE_SHADER,
                PrimitiveTopology::LineList,
            ),
            Family::Line => (
                &mut self.paint[2],
                LINE_SHADER,
                PrimitiveTopology::TriangleList,
            ),
            Family::Circle => (
                &mut self.paint[3],
                CIRCLE_SHADER,
                PrimitiveTopology::TriangleList,
            ),
            Family::Raster => (
                &mut self.images[0],
                RASTER_SHADER,
                PrimitiveTopology::TriangleList,
            ),
            Family::Hillshade => (
                &mut self.images[1],
                HILLSHADE_SHADER,
                PrimitiveTopology::TriangleList,
            ),
            Family::ColorRelief => (
                &mut self.images[2],
                COLOR_RELIEF_SHADER,
                PrimitiveTopology::TriangleList,
            ),
            _ => return None,
        };
        let material_layout = self.material_layout;
        Some(*slot.get_or_insert_with(|| {
            material_pipeline(
                gpu,
                &shader.lit(),
                topology,
                samples,
                &[layouts[0], layouts[1], material_layout],
            )
        }))
    }

    /// `family`'s pipeline, if made.
    pub(crate) fn get(&self, family: Family) -> Option<RenderPipelineId> {
        match family {
            Family::Fill => self.paint[0],
            Family::FillOutline => self.paint[1],
            Family::Line => self.paint[2],
            Family::Circle => self.paint[3],
            Family::Raster => self.images[0],
            Family::Hillshade => self.images[1],
            Family::ColorRelief => self.images[2],
            _ => None,
        }
    }
}

/// A pipeline drawing `shader` (its lit composition) into the material
/// buffer, each attachment blended as premultiplied colour is.
pub(crate) fn material_pipeline(
    gpu: &mut dyn Gpu,
    shader: &Shader,
    topology: PrimitiveTopology,
    samples: u32,
    layouts: &[BindGroupLayoutId],
) -> RenderPipelineId {
    let module = crate::shader_module(gpu, shader);
    let target = ColorTargetState {
        format: TextureFormat::Rgba8Unorm,
        blend: Some(BlendState::PremultipliedAlpha),
    };
    gpu.create_render_pipeline(&RenderPipelineDesc {
        cull_mode: None,
        label: shader.label,
        bind_group_layouts: layouts,
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
        targets: &[target; 2],
        depth: None,
        topology,
        sample_count: samples,
    })
}
