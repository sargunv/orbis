//! The lit map (#91, decision 0048): under a scene's lights, the map's
//! layers shade their fragments in place (`map_out_lit.wgsl`), each with
//! its node's material, in the main pass where they draw unlit otherwise.
//! Their lit pipelines take the nodes' materials as a group of their own.

use crate::release::Release;
use orb_hal::*;

use crate::{Family, TargetFormat};

/// What a map layer is made of under a scene's lights: its colours are its
/// albedo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeMaterial {
    /// 0 mirror-smooth to 1 matte.
    pub roughness: f32,
    /// 0 a dielectric to 1 a metal.
    pub metalness: f32,
    /// The share of its colour that shows whatever the lights.
    pub self_lit: f32,
    /// An extrusion's windows: the share of them lit.
    pub windows: f32,
}

impl Default for NodeMaterial {
    fn default() -> Self {
        Self {
            roughness: 1.0,
            metalness: 0.0,
            self_lit: 0.0,
            windows: 0.0,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Lit {
    /// Lit pipelines' group 2: the nodes' materials, by node index.
    layout: BindGroupLayoutId,
    materials: Vec<[f32; 4]>,
    buffer: Option<(BufferId, BindGroupId, usize)>,
    dirty: bool,
    /// The fill, outline, line and circle pipelines, then the image
    /// families'.
    pipelines: [Option<RenderPipelineId>; 7],
}

impl Lit {
    pub(crate) fn new(gpu: &mut dyn Gpu) -> Self {
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "node materials",
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Storage { read_only: true },
            }],
        });
        Self {
            layout,
            materials: Vec::new(),
            buffer: None,
            dirty: true,
            pipelines: [None; 7],
        }
    }

    /// Node `node`'s material; written with the next frame.
    pub(crate) fn set_material(&mut self, node: usize, m: NodeMaterial) {
        if self.materials.len() <= node {
            self.materials.resize(node + 1, [1.0, 0.0, 0.0, 0.0]);
        }
        let words = [m.roughness, m.metalness, m.self_lit, m.windows];
        if self.materials[node] != words {
            self.materials[node] = words;
            self.dirty = true;
        }
    }

    /// Writes the materials of `nodes` nodes; whether the group changed,
    /// so the bundles record again.
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
            self.buffer
                .take()
                .map(|(buffer, group, _)| (buffer, group))
                .release(gpu);
            let capacity = nodes.max(1).next_power_of_two();
            let buffer = gpu.create_buffer(&BufferDesc {
                label: "node materials",
                size: capacity as u64 * 16,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
            let group = gpu.create_bind_group(&BindGroupDesc {
                label: "node materials",
                layout: self.layout,
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

    /// The materials' group's layout, for lit pipelines' group 2.
    pub(crate) fn layout(&self) -> BindGroupLayoutId {
        self.layout
    }

    /// The materials' group, once uploaded.
    pub(crate) fn group(&self) -> Option<BindGroupId> {
        self.buffer.map(|(_, group, _)| group)
    }

    fn slot(family: Family) -> Option<usize> {
        Some(match family {
            Family::Fill => 0,
            Family::FillOutline => 1,
            Family::Line => 2,
            Family::Circle => 3,
            Family::Raster => 4,
            Family::Hillshade => 5,
            Family::ColorRelief => 6,
            _ => return None,
        })
    }

    /// Makes `family`'s lit pipeline into `target`, for its first node;
    /// `layouts` are its scene and node groups'.
    pub(crate) fn make(
        &mut self,
        gpu: &mut dyn Gpu,
        family: Family,
        target: TargetFormat,
        layouts: [BindGroupLayoutId; 2],
    ) {
        use orb_shaders::*;
        let Some(slot) = Self::slot(family) else {
            return;
        };
        let (shader, topology): (Shader, _) = match family {
            Family::Fill => (FILL_SHADER, PrimitiveTopology::TriangleList),
            Family::FillOutline => (FILL_OUTLINE_SHADER, PrimitiveTopology::LineList),
            Family::Line => (LINE_SHADER, PrimitiveTopology::TriangleList),
            Family::Circle => (CIRCLE_SHADER, PrimitiveTopology::TriangleList),
            Family::Raster => (RASTER_SHADER, PrimitiveTopology::TriangleList),
            Family::Hillshade => (HILLSHADE_SHADER, PrimitiveTopology::TriangleList),
            _ => (COLOR_RELIEF_SHADER, PrimitiveTopology::TriangleList),
        };
        let layout = self.layout;
        self.pipelines[slot].get_or_insert_with(|| {
            crate::draw_pipeline(
                gpu,
                &shader.lit(),
                topology,
                target,
                &[layouts[0], layouts[1], layout],
            )
        });
    }

    /// `family`'s lit pipeline, if made.
    pub(crate) fn get(&self, family: Family) -> Option<RenderPipelineId> {
        self.pipelines[Self::slot(family)?]
    }
}

impl Release for Lit {
    fn release(self, gpu: &mut dyn Gpu) {
        let buffer = self.buffer.map(|(buffer, group, _)| (buffer, group));
        (self.layout, buffer, self.pipelines).release(gpu);
    }
}
