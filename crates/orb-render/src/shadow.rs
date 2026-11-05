//! Shadows from the directional light.
//!
//! Each frame fits two cascades of a depth map to slices of the view, as
//! orthographic views along the light. The extrusions and models draw into
//! them from the light with the cull's arguments, recorded once as a bundle
//! per cascade. Receivers read the cascades and maps in the scene group
//! (`shadow_receive.wgsl`): extrusions and models per fragment (models in
//! `pbr_light`'s directional term), and the ground in one
//! full-screen multiply at the first 3D layer's place. Planes only.

use crate::release::Release;
use glam::{DMat4, DVec3};
use orb_hal::*;
use orb_scene::{FrameBlock, FrameShadow, SceneBuffers};
use orb_shaders::{
    EXTRUSION_CASTER_SHADER, MODEL_CASTER_MASKED_SHADER, MODEL_CASTER_SHADER, SHADOW_GROUND_SHADER,
    Shader,
};

use crate::{TargetFormat, draw_pipeline_blending, shader_module};

/// Texels per side of a cascade.
const MAP_PX: u32 = 2048;
const CASCADES: usize = 2;
/// Where each cascade ends, in view depths of the camera's target.
const SPLITS: [f64; CASCADES] = [1.5, 4.0];

#[derive(Debug)]
pub(crate) struct Shadows {
    uniform: BufferId,
    /// The words last written, to skip unchanged writes.
    written: Vec<u32>,
    /// While shadows are cast: the cascades' depth layers, and a view of
    /// each for its caster pass.
    map: Option<(TextureId, [TextureId; CASCADES])>,
    /// Bound in the map's place without shadows.
    no_map: TextureId,
    compare: SamplerId,
    pass_layout: BindGroupLayoutId,
    /// Each cascade's caster group: which cascade it draws.
    passes: [BindGroupId; CASCADES],
    pass_buffer: BufferId,
    /// Made when extrusions (whole, cut), then models (opaque, masked)
    /// first cast.
    casters: [Option<RenderPipelineId>; CASTERS],
    /// Made when shadows are first cast.
    ground: Option<RenderPipelineId>,
    /// The caster passes' bundles, recorded with the main pass's.
    bundles: Vec<RenderBundleId>,
}

impl Shadows {
    pub(crate) fn new(gpu: &mut dyn Gpu) -> Self {
        let uniform = gpu.create_buffer(&BufferDesc {
            label: "shadow",
            size: orb_shaders::layout::shadow::WORDS as u64 * 4,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let no_map = gpu.create_texture(&TextureDesc {
            label: "no shadow map",
            width: 1,
            height: 1,
            array_layers: CASCADES as u32,
            dimension: TextureViewDimension::D2Array,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            mip_levels: 1,
        });
        let compare = gpu.create_sampler(&SamplerDesc {
            label: "shadow compare",
            filter: FilterMode::Linear,
            address: AddressMode::ClampToEdge,
            mipmap_filter: FilterMode::Nearest,
            anisotropy: 1,
            compare: Some(CompareFunction::LessEqual),
        });
        let pass_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "shadow pass",
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Uniform,
            }],
        });
        let stride = u64::from(gpu.caps().limits.min_uniform_buffer_offset_alignment).max(16);
        let pass_buffer = gpu.create_buffer(&BufferDesc {
            label: "shadow passes",
            size: CASCADES as u64 * stride,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let passes = [0u32, 1].map(|cascade| {
            let offset = u64::from(cascade) * stride;
            gpu.write_buffer(
                pass_buffer,
                offset,
                bytemuck::cast_slice(&[cascade, 0, 0, 0]),
            );
            gpu.create_bind_group(&BindGroupDesc {
                label: "shadow pass",
                layout: pass_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: pass_buffer,
                        offset,
                        size: Some(16),
                    },
                }],
            })
        });
        Self {
            uniform,
            written: Vec::new(),
            map: None,
            no_map,
            compare,
            pass_layout,
            passes,
            pass_buffer,
            casters: [None; CASTERS],
            ground: None,
            bundles: Vec::new(),
        }
    }

    /// The scene group's bindings 16–18: the receivers' uniform, the map
    /// and its comparison sampler.
    pub(crate) fn bindings(&self) -> (BufferId, TextureId, SamplerId) {
        let map = self.map.map_or(self.no_map, |(map, _)| map);
        (self.uniform, map, self.compare)
    }

    /// [`bindings`](Self::bindings) for the caster passes' scene group,
    /// which can't bind the map they draw into.
    pub(crate) fn caster_bindings(&self) -> (BufferId, TextureId, SamplerId) {
        (self.uniform, self.no_map, self.compare)
    }

    /// Makes or frees the map, and makes the pipelines the first time,
    /// as extrusions and models cast or not. The scene group binds the
    /// map: it must be bound again when [`bindings`](Self::bindings)
    /// change. Models' casters bind their model group (`model`) in group
    /// 2 and the cascade in group 3.
    pub(crate) fn set_casting(
        &mut self,
        gpu: &mut dyn Gpu,
        casters: [bool; 2],
        target: TargetFormat,
        [scene, node, model]: [BindGroupLayoutId; 3],
    ) {
        let casting = casters.contains(&true);
        let extrusions = vec![scene, node, self.pass_layout];
        let models = vec![scene, node, model, self.pass_layout];
        let shaders = [
            (&EXTRUSION_CASTER_SHADER, &extrusions),
            (&MODEL_CASTER_SHADER, &models),
            (&MODEL_CASTER_MASKED_SHADER, &models),
        ];
        let casts = caster_kinds(casters);
        for ((made, casts), (shader, layouts)) in self.casters.iter_mut().zip(casts).zip(shaders) {
            if casts && made.is_none() {
                *made = Some(caster_pipeline(gpu, shader, layouts));
            }
        }
        if casting && self.ground.is_none() {
            self.ground = Some(draw_pipeline_blending(
                gpu,
                &SHADOW_GROUND_SHADER,
                PrimitiveTopology::TriangleList,
                target,
                &[scene],
                Some(BlendState::Multiply),
            ));
        }
        match (casting, self.map) {
            (true, None) => {
                let map = gpu.create_texture(&TextureDesc {
                    label: "shadow map",
                    width: MAP_PX,
                    height: MAP_PX,
                    array_layers: CASCADES as u32,
                    dimension: TextureViewDimension::D2Array,
                    format: TextureFormat::Depth32Float,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    sample_count: 1,
                    mip_levels: 1,
                });
                let layers = [0, 1].map(|layer| gpu.create_layer_view(map, layer));
                self.map = Some((map, layers));
            }
            (false, Some((map, layers))) => {
                layers.release(gpu);
                gpu.destroy_texture(map);
                self.map = None;
                self.forget_bundles(gpu);
            }
            _ => {}
        }
    }

    /// The pipelines shadows draw with, once made.
    pub(crate) fn pipelines(&self) -> impl Iterator<Item = PipelineId> + '_ {
        (self.casters.iter().chain([&self.ground]))
            .flatten()
            .map(|&p| PipelineId::from(p))
    }

    /// Whether the map exists and the pipelines `casters` (extrusions',
    /// models') need compiled.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu, casters: [bool; 2]) -> bool {
        self.map.is_some()
            && (self.casters.iter().zip(caster_kinds(casters)))
                .filter(|&(_, casts)| casts)
                .map(|(&p, _)| p)
                .chain([self.ground])
                .all(|p| p.is_some_and(|p| gpu.pipeline_ready(p.into())))
    }

    /// GPU bytes of the map.
    pub(crate) fn bytes(&self) -> u64 {
        self.map
            .map_or(0, |_| u64::from(MAP_PX * MAP_PX) * 4 * CASCADES as u64)
    }

    pub(crate) fn forget_bundles(&mut self, gpu: &mut dyn Gpu) {
        std::mem::take(&mut self.bundles).release(gpu);
    }

    /// Writes the receivers' uniform for this frame: the cascades fitted
    /// to its camera when `cast`, else none.
    pub(crate) fn write(
        &mut self,
        gpu: &mut dyn Gpu,
        frame: &FrameBlock,
        cast: Option<(FrameShadow, [f32; 3])>,
    ) {
        use orb_shaders::layout::{shadow as n, shadow_cascades as c};
        let mut b = orb_shaders::Block::new(n::WORDS);
        let view_proj = DMat4::from_cols_array_2d(&frame.view_proj.map(|c| c.map(f64::from)));
        if let Some((shadow, light)) = cast {
            let light = DVec3::from(light.map(f64::from));
            let at = n::CASCADES;
            for (i, cascade) in fit(frame, view_proj, shadow.heights, light)
                .iter()
                .enumerate()
            {
                b.f(
                    at + c::VIEW_PROJ + i * 16,
                    &cascade.view_proj.as_mat4().to_cols_array(),
                );
                b.f(at + c::SPLITS + i, &[cascade.end as f32])
                    .f(at + c::TEXEL + i, &[cascade.texel as f32]);
            }
            b.f(at + c::LIGHT, &light.as_vec3().to_array())
                .u(at + c::COUNT, &[CASCADES as u32]);
            b.f(at + c::STRENGTH, &[shadow.intensity])
                .f(at + c::GROUND, &[ground_share(frame)]);
        }
        b.f(n::INVERSE, &view_proj.inverse().as_mat4().to_cols_array());
        let words = b.0;
        if words != self.written {
            gpu.write_buffer(self.uniform, 0, bytemuck::cast_slice(&words));
            self.written = words;
        }
    }

    /// The caster passes, one per cascade, recording their bundles if
    /// they aren't: the `extrusions`' draws (node index and group), then
    /// the `models`' opaque and masked materials' (their groups, index
    /// buffers and draws, as
    /// [`Models::caster_commands`](crate::model::Models::caster_commands)
    /// records them).
    pub(crate) fn passes(
        &mut self,
        gpu: &mut dyn Gpu,
        scene: BindGroupId,
        extrusions: &[(u32, BindGroupId)],
        models: &[Vec<RenderCommand>; 2],
        buffers: &SceneBuffers,
    ) -> Vec<Pass> {
        let Some((_, layers)) = self.map else {
            return Vec::new();
        };
        if self.bundles.is_empty() {
            for &group in &self.passes {
                let mut commands = Vec::new();
                if let Some(caster) = self.casters[0].filter(|_| !extrusions.is_empty()) {
                    commands.extend([
                        RenderCommand::SetPipeline(caster),
                        RenderCommand::SetBindGroup {
                            index: 0,
                            group: scene,
                        },
                        RenderCommand::SetBindGroup { index: 2, group },
                        RenderCommand::SetIndexBuffer {
                            buffer: buffers.compacted,
                            format: IndexFormat::Uint32,
                            offset: 0,
                        },
                    ]);
                    for &(index, node) in extrusions {
                        commands.extend([
                            RenderCommand::SetBindGroup {
                                index: 1,
                                group: node,
                            },
                            RenderCommand::DrawIndexedIndirect {
                                buffer: buffers.args,
                                offset: crate::args_offset(index),
                            },
                        ]);
                    }
                }
                for (caster, models) in self.casters[1..].iter().zip(models) {
                    let Some(caster) = caster.filter(|_| !models.is_empty()) else {
                        continue;
                    };
                    commands.extend([
                        RenderCommand::SetPipeline(caster),
                        RenderCommand::SetBindGroup {
                            index: 0,
                            group: scene,
                        },
                        RenderCommand::SetBindGroup { index: 3, group },
                    ]);
                    commands.extend_from_slice(models);
                }
                self.bundles.push(gpu.create_render_bundle(
                    &RenderBundleDesc {
                        label: "shadow casters",
                        color_formats: &[],
                        depth_format: Some(TextureFormat::Depth32Float),
                        sample_count: 1,
                    },
                    &commands,
                ));
            }
        }
        layers
            .iter()
            .zip(&self.bundles)
            .map(|(&layer, &bundle)| {
                Pass::Render(RenderPass {
                    label: "shadow".into(),
                    colors: Vec::new(),
                    depth: Some(DepthAttachment {
                        target: layer,
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    commands: vec![RenderCommand::ExecuteBundles(vec![bundle])],
                })
            })
            .collect()
    }

    /// The ground's shadows, for the main pass, once compiled.
    pub(crate) fn ground_commands(
        &self,
        gpu: &mut dyn Gpu,
        scene: BindGroupId,
    ) -> Option<[RenderCommand; 3]> {
        let ground = self.ground.filter(|&g| gpu.pipeline_ready(g.into()))?;
        Some([
            RenderCommand::SetPipeline(ground),
            RenderCommand::SetBindGroup {
                index: 0,
                group: scene,
            },
            RenderCommand::Draw {
                vertices: 0..3,
                instances: 0..1,
            },
        ])
    }
}

/// A caster pipeline: depth only, from the light, both faces (walls
/// alone leave a box's middle unshadowed from a high light).
fn caster_pipeline(
    gpu: &mut dyn Gpu,
    shader: &Shader,
    layouts: &[BindGroupLayoutId],
) -> RenderPipelineId {
    let module = shader_module(gpu, shader);
    let pipeline = gpu.create_render_pipeline(&RenderPipelineDesc {
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
        targets: &[],
        depth: Some(DepthState {
            format: TextureFormat::Depth32Float,
            write: true,
            compare: CompareFunction::Less,
        }),
        topology: PrimitiveTopology::TriangleList,
        cull_mode: None,
        sample_count: 1,
    });
    gpu.destroy_shader_module(module);
    pipeline
}

/// Extrusions' caster, then models' solid and masked ones.
const CASTERS: usize = 3;

/// Which casters a style needs, from whether it has extrusions and models.
fn caster_kinds([extrusions, models]: [bool; 2]) -> [bool; CASTERS] {
    [extrusions, models, models]
}

/// One cascade as fitted: camera-relative world → the light's clip
/// space, where it ends in view depth, and world units per texel.
struct Cascade {
    view_proj: DMat4,
    end: f64,
    texel: f64,
}

/// The cascades for a frame's camera. Each takes a slice of the view's
/// depth (with the last tenth of the one before, which blends into it),
/// cut to the heights shadows fall between, and bounds it by a
/// sphere: its size depends on the camera's zoom and pitch alone, not its
/// bearing. Seen along the light, the sphere's centre snaps to whole texels
/// of the world, so shadows hold still as the camera pans and turns. Toward
/// the light the view reaches past the sphere to casters as tall as the
/// heights allow.
fn fit(frame: &FrameBlock, view_proj: DMat4, [low, high]: [f32; 2], light: DVec3) -> [Cascade; 2] {
    let (low, high) = (f64::from(low), f64::from(high));
    let inverse = view_proj.inverse();
    let depth_row = view_proj.row(3);
    let depth = |p: DVec3| depth_row.dot(p.extend(1.0));
    let rays = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]].map(|[x, y]| {
        (
            inverse.project_point3(DVec3::new(x, y, 0.0)),
            inverse.project_point3(DVec3::new(x, y, 1.0)),
        )
    });
    // Clip w is affine along each ray.
    let at = |(a, b): (DVec3, DVec3), d: f64| a.lerp(b, (d - depth(a)) / (depth(b) - depth(a)));
    let (near, far) = (depth(rays[0].0), depth(rays[0].1));
    let target = depth(DVec3::ZERO).max(near);
    let away = -light;
    let right = if away.z.abs() < 0.999 {
        away.cross(DVec3::Z).normalize()
    } else {
        DVec3::X
    };
    let up = right.cross(away);
    let camera = DVec3::new(frame.camera[0], frame.camera[1], 0.0);
    let reach = (high - low) / light.z;
    let mut start = near;
    SPLITS.map(|split| {
        let end = (split * target).min(far);
        let near_corners = rays.map(|r| at(r, start));
        let far_corners = rays.map(|r| at(r, end));
        // The next cascade also holds this one's last tenth, where
        // receivers blend into it (`shadow_visibility`).
        start = start + 0.9 * (end - start);
        let mut points = Vec::with_capacity(24);
        for k in 0..4 {
            let next = (k + 1) % 4;
            for (a, b) in [
                (near_corners[k], far_corners[k]),
                (near_corners[k], near_corners[next]),
                (far_corners[k], far_corners[next]),
            ] {
                points.extend(clip_heights(a, b, low, high).into_iter().flatten());
            }
        }
        if points.is_empty() {
            points.extend(near_corners.iter().chain(&far_corners));
        }
        let centre = points.iter().sum::<DVec3>() / points.len() as f64;
        let radius = points
            .iter()
            .map(|p| p.distance(centre))
            .fold(1e-6, f64::max);
        let texel = 2.0 * radius / f64::from(MAP_PX);
        let snap = |axis: DVec3| {
            let world = axis.dot(centre + camera);
            (world / texel).round() * texel - axis.dot(camera)
        };
        let (x, y) = (snap(right), snap(up));
        let z = away.dot(centre);
        let (z0, z1) = (z - radius - reach, z + radius);
        let rows = [
            (right / radius).extend(-x / radius),
            (up / radius).extend(-y / radius),
            (away / (z1 - z0)).extend(-z0 / (z1 - z0)),
            DVec3::ZERO.extend(1.0),
        ];
        Cascade {
            view_proj: DMat4::from_cols(rows[0], rows[1], rows[2], rows[3]).transpose(),
            end,
            texel,
        }
    })
}

/// The part of the segment `a`–`b` between heights `low` and `high`.
fn clip_heights(a: DVec3, b: DVec3, low: f64, high: f64) -> Option<[DVec3; 2]> {
    let rise = b.z - a.z;
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    if rise.abs() < 1e-12 {
        if a.z < low || a.z > high {
            return None;
        }
    } else {
        let (ta, tb) = ((low - a.z) / rise, (high - a.z) / rise);
        t0 = t0.max(ta.min(tb));
        t1 = t1.min(ta.max(tb));
    }
    (t0 <= t1).then(|| [a.lerp(b, t0), a.lerp(b, t1)])
}

/// The directional light's share of the light on level ground: what a
/// shadow there takes away at full intensity (at mid luminance).
fn ground_share(frame: &FrameBlock) -> f32 {
    let Some(l) = frame.lights else {
        return 0.0;
    };
    let luminance = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let ambient = l.ambient_level * luminance(l.ambient);
    let lit = (l.ambient_level + l.directional_level - 0.5).max(1.0);
    let directional = lit * luminance(l.directional) * l.position[2].clamp(0.0, 1.0);
    if ambient + directional <= 0.0 {
        return 0.0;
    }
    directional / (ambient + directional)
}

impl crate::release::Release for Shadows {
    fn release(self, gpu: &mut dyn Gpu) {
        self.pass_buffer.release(gpu);
        let fixed = (
            self.uniform,
            self.no_map,
            self.compare,
            self.pass_layout,
            self.passes,
        );
        (fixed, self.map, self.casters, self.ground, self.bundles).release(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(bearing: f64, camera: [f64; 2]) -> FrameBlock {
        // A camera 1000 units from its target at pitch 60, turned, as
        // orb-camera places it, with depth in [0, 1].
        let view = DMat4::from_translation(DVec3::new(0.0, 0.0, -1000.0))
            * DMat4::from_rotation_x(-60f64.to_radians())
            * DMat4::from_rotation_z(bearing.to_radians());
        let (f, aspect, near, far) = (1.0 / 0.32f64.tan(), 1.5, 1.0, 20_000.0);
        let proj = DMat4::from_cols_array(&[
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            far / (near - far),
            -1.0,
            0.0,
            0.0,
            near * far / (near - far),
            0.0,
        ]);
        FrameBlock {
            view_proj: (proj * view).as_mat4().to_cols_array_2d(),
            camera,
            world_span: 4e7,
            viewport: [1200.0, 800.0],
            world_per_px: 1.0,
            pixel_ratio: 1.0,
            zoom: 15.0,
            bearing: bearing.to_radians() as f32,
            align: [0.0; 2],
            extrusion_scale: 1.0,
            ground_horizon: 0.0,
            symbol_fade_change: 1.0,
            pitch: 1.05,
            roll: 0.0,
            camera_to_center: 1.0,
            time: 0.0,
            terrain: None,
            lights: None,
            globe: None,
        }
    }

    fn cascades(f: &FrameBlock) -> [Cascade; 2] {
        let vp = DMat4::from_cols_array_2d(&f.view_proj.map(|c| c.map(f64::from)));
        fit(f, vp, [0.0, 300.0], DVec3::new(0.3, -0.5, 0.8).normalize())
    }

    /// A world point's texel in a cascade.
    fn texel_of(c: &Cascade, world: DVec3) -> [f64; 2] {
        let p = c.view_proj.project_point3(world);
        [
            (p.x + 1.0) * 0.5 * f64::from(MAP_PX),
            (p.y + 1.0) * 0.5 * f64::from(MAP_PX),
        ]
    }

    #[test]
    fn cascades_cover_the_view_and_its_target() {
        let f = frame(0.0, [0.0; 2]);
        let [first, second] = cascades(&f);
        assert!(first.end < second.end);
        assert!(
            first.texel < second.texel,
            "{} {}",
            first.texel,
            second.texel
        );
        let p = first.view_proj.project_point3(DVec3::ZERO);
        assert!(
            p.x.abs() < 1.0 && p.y.abs() < 1.0 && (0.0..1.0).contains(&p.z),
            "{p}"
        );
        // A caster's roof stands nearer the light than the ground below it.
        let roof = first.view_proj.project_point3(DVec3::new(0.0, 0.0, 200.0));
        assert!(roof.z < p.z);
    }

    #[test]
    fn panning_moves_texels_by_whole_steps() {
        // The camera pans by a fraction of a texel: a fixed world point
        // stays on the same texel phase, so shadow edges don't crawl.
        let world = DVec3::new(10_123.37, 20_456.81, 0.0);
        let a = frame(0.0, [10_000.0, 20_000.0]);
        let b = frame(0.0, [10_037.3, 20_011.9]);
        let [ca, _] = cascades(&a);
        let [cb, _] = cascades(&b);
        assert!((ca.texel - cb.texel).abs() < 1e-9);
        let ta = texel_of(&ca, world - DVec3::new(a.camera[0], a.camera[1], 0.0));
        let tb = texel_of(&cb, world - DVec3::new(b.camera[0], b.camera[1], 0.0));
        for (x, y) in ta.into_iter().zip(tb) {
            let shift = x - y;
            assert!((shift - shift.round()).abs() < 1e-3, "{ta:?} {tb:?}");
        }
    }

    #[test]
    fn turning_keeps_the_cascades_size() {
        let [a, _] = cascades(&frame(0.0, [0.0; 2]));
        let [b, _] = cascades(&frame(73.0, [0.0; 2]));
        assert!(
            (a.texel - b.texel).abs() < 1e-6 * a.texel,
            "{} {}",
            a.texel,
            b.texel
        );
    }
}
