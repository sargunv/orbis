//! [`Recording`]: a [`Gpu`] that passes everything to another and writes
//! down the passes it submits, for golden tests of a renderer's command
//! stream.
//!
//! Ids are named by their kind, label and order of first appearance in
//! the text, not by their numbers, and each named resource, layout,
//! group, pipeline and bundle is described once where it first appears:
//! the text depends on what a frame submits, not on the order its
//! resources were made in.

use std::collections::HashMap;
use std::fmt::Write;

use crate::*;

/// A [`Gpu`] that records what it submits while [`start`](Self::start)ed.
#[derive(Debug)]
pub struct Recording<G> {
    inner: G,
    on: bool,
    defs: HashMap<Ref, Def>,
    passes: Vec<Vec<Pass>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Ref {
    Buffer(u32),
    Texture(u32),
    Sampler(u32),
    Module(u32),
    Layout(u32),
    Group(u32),
    Render(u32),
    Compute(u32),
    Bundle(u32),
}

impl Ref {
    fn kind(self) -> &'static str {
        match self {
            Self::Buffer(_) => "buffer",
            Self::Texture(_) => "texture",
            Self::Sampler(_) => "sampler",
            Self::Module(_) => "module",
            Self::Layout(_) => "layout",
            Self::Group(_) => "group",
            Self::Render(_) => "render",
            Self::Compute(_) => "compute",
            Self::Bundle(_) => "bundle",
        }
    }
}

/// What made an id.
#[derive(Debug)]
enum Def {
    /// A resource whose description names no other id.
    Plain { label: String, text: String },
    /// A view of a texture.
    View { of: TextureId, text: String },
    Group {
        label: String,
        layout: BindGroupLayoutId,
        entries: Vec<BindGroupEntry>,
    },
    Pipeline {
        label: String,
        layouts: Vec<BindGroupLayoutId>,
        /// Each stage's module, entry point and constants.
        stages: Vec<(ShaderModuleId, String, String)>,
        text: String,
    },
    Bundle {
        label: String,
        text: String,
        commands: Vec<RenderCommand>,
    },
}

impl<G: Gpu> Recording<G> {
    pub fn new(inner: G) -> Self {
        Self {
            inner,
            on: false,
            defs: HashMap::new(),
            passes: Vec::new(),
        }
    }

    pub fn inner(&self) -> &G {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut G {
        &mut self.inner
    }

    /// Records the submissions from now on.
    pub fn start(&mut self) {
        self.on = true;
        self.passes.clear();
    }

    /// Stops recording, and gives the submissions since
    /// [`start`](Self::start) as text.
    pub fn finish(&mut self) -> String {
        self.on = false;
        let passes = std::mem::take(&mut self.passes);
        let mut w = Writer {
            defs: &self.defs,
            names: HashMap::new(),
            counts: HashMap::new(),
            described: Vec::new(),
        };
        let mut out = String::new();
        for (i, submission) in passes.iter().enumerate() {
            let _ = writeln!(out, "submit {i}");
            for pass in submission {
                w.pass(&mut out, pass);
            }
        }
        out.push('\n');
        for line in &w.described {
            out.push_str(line);
            out.push('\n');
        }
        out
    }

    fn texture_made(&mut self, desc: &TextureDesc<'_>, id: TextureId) {
        let text = format!(
            "{:?} {:?} samples {} {:?}",
            desc.format, desc.dimension, desc.sample_count, desc.usage
        );
        let label = desc.label.to_string();
        self.def(Ref::Texture(id.0), Def::Plain { label, text });
    }

    fn def(&mut self, id: Ref, def: Def) {
        self.defs.insert(id, def);
    }
}

fn stage(stage: &ShaderStage<'_>) -> (ShaderModuleId, String, String) {
    (
        stage.module,
        stage.entry_point.to_string(),
        format!("{:?}", stage.constants),
    )
}

/// Names ids as the text first meets them.
struct Writer<'a> {
    defs: &'a HashMap<Ref, Def>,
    names: HashMap<Ref, String>,
    counts: HashMap<(&'static str, String), u32>,
    described: Vec<String>,
}

impl Writer<'_> {
    fn name(&mut self, id: Ref) -> String {
        if let Some(name) = self.names.get(&id) {
            return name.clone();
        }
        let label = match self.defs.get(&id) {
            Some(
                Def::Plain { label, .. }
                | Def::Group { label, .. }
                | Def::Pipeline { label, .. }
                | Def::Bundle { label, .. },
            ) => label.clone(),
            Some(Def::View { of, .. }) => {
                let of = self.name(Ref::Texture(of.0));
                format!("view of {of}")
            }
            None => "?".into(),
        };
        let count = self.counts.entry((id.kind(), label.clone())).or_default();
        let name = format!("{} {label:?}#{count}", id.kind());
        *count += 1;
        self.names.insert(id, name.clone());
        let text = self.describe(id);
        self.described.push(format!("{name}: {text}"));
        name
    }

    fn describe(&mut self, id: Ref) -> String {
        let Some(def) = self.defs.get(&id) else {
            return "made elsewhere".into();
        };
        match def {
            Def::Plain { text, .. } | Def::View { text, .. } => text.clone(),
            Def::Group {
                layout, entries, ..
            } => {
                let mut text = format!("layout {}", self.name(Ref::Layout(layout.0)));
                for e in entries {
                    let resource = match e.resource {
                        BindingResource::Buffer {
                            buffer,
                            offset,
                            size,
                        } => format!(
                            "{} at {offset} size {size:?}",
                            self.name(Ref::Buffer(buffer.0))
                        ),
                        BindingResource::Texture { texture, dimension } => {
                            format!("{} as {dimension:?}", self.name(Ref::Texture(texture.0)))
                        }
                        BindingResource::Sampler(s) => self.name(Ref::Sampler(s.0)),
                    };
                    let _ = write!(text, "; {} {resource}", e.binding);
                }
                text
            }
            Def::Pipeline {
                layouts,
                stages,
                text,
                ..
            } => {
                let mut out = String::from("layouts [");
                for (i, l) in layouts.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&self.name(Ref::Layout(l.0)));
                }
                out.push(']');
                for (module, entry, constants) in stages {
                    let module = self.name(Ref::Module(module.0));
                    let _ = write!(out, "; {module} {entry} {constants}");
                }
                let _ = write!(out, "; {text}");
                out
            }
            Def::Bundle { text, commands, .. } => {
                let mut out = text.clone();
                for c in commands {
                    out.push_str("\n    ");
                    out.push_str(&self.command(c));
                }
                out
            }
        }
    }

    fn pass(&mut self, out: &mut String, pass: &Pass) {
        match pass {
            Pass::Compute(p) => {
                let _ = writeln!(out, "  compute pass {:?}", p.label);
                for c in &p.commands {
                    let text = match c {
                        ComputeCommand::SetPipeline(p) => {
                            format!("pipeline {}", self.name(Ref::Compute(p.0)))
                        }
                        ComputeCommand::SetBindGroup { index, group } => {
                            format!("group {index} {}", self.name(Ref::Group(group.0)))
                        }
                        ComputeCommand::Dispatch { x, y, z } => format!("dispatch {x} {y} {z}"),
                    };
                    let _ = writeln!(out, "    {text}");
                }
            }
            Pass::Render(p) => {
                let _ = writeln!(out, "  render pass {:?}", p.label);
                for c in &p.colors {
                    let target = self.name(Ref::Texture(c.target.0));
                    let resolve = c.resolve.map(|r| self.name(Ref::Texture(r.0)));
                    let _ = writeln!(
                        out,
                        "    colour {target} resolve {resolve:?} {:?} {:?}",
                        c.load, c.store
                    );
                }
                if let Some(d) = &p.depth {
                    let target = self.name(Ref::Texture(d.target.0));
                    let _ = writeln!(out, "    depth {target} {:?} {:?}", d.load, d.store);
                }
                // A scissor and what it limits, repeated, write once: the
                // rects atlases placed next to each other merge, as their
                // placement follows arrival order.
                let mut lines: Vec<String> = Vec::new();
                for c in &p.commands {
                    lines.push(self.command(c));
                    let n = lines.len();
                    if n >= 4
                        && lines[n - 2].starts_with("scissor")
                        && lines[n - 2..] == lines[n - 4..n - 2]
                    {
                        lines.truncate(n - 2);
                    }
                }
                for line in lines {
                    let _ = writeln!(out, "    {line}");
                }
            }
            Pass::CopyTexture {
                source,
                destination,
                copies,
            } => {
                let (s, d) = (
                    self.name(Ref::Texture(source.0)),
                    self.name(Ref::Texture(destination.0)),
                );
                let _ = writeln!(out, "  copy {s} to {d}: {copies:?}");
            }
            Pass::CopyBuffer {
                source,
                source_offset,
                destination,
                destination_offset,
                size,
            } => {
                let (s, d) = (
                    self.name(Ref::Buffer(source.0)),
                    self.name(Ref::Buffer(destination.0)),
                );
                let _ = writeln!(
                    out,
                    "  copy {s} at {source_offset} to {d} at {destination_offset}, {size} bytes"
                );
            }
            Pass::ClearBuffer {
                buffer,
                offset,
                size,
            } => {
                let b = self.name(Ref::Buffer(buffer.0));
                let _ = writeln!(out, "  clear {b} at {offset} size {size:?}");
            }
            Pass::PushDebugGroup(label) => {
                let _ = writeln!(out, "  push {label:?}");
            }
            Pass::PopDebugGroup => {
                let _ = writeln!(out, "  pop");
            }
        }
    }

    fn command(&mut self, c: &RenderCommand) -> String {
        match c {
            RenderCommand::SetPipeline(p) => format!("pipeline {}", self.name(Ref::Render(p.0))),
            RenderCommand::SetBindGroup { index, group } => {
                format!("group {index} {}", self.name(Ref::Group(group.0)))
            }
            RenderCommand::Draw {
                vertices,
                instances,
            } => format!("draw {vertices:?} {instances:?}"),
            RenderCommand::SetIndexBuffer { buffer, offset } => {
                format!("index {} at {offset}", self.name(Ref::Buffer(buffer.0)))
            }
            RenderCommand::DrawIndexed {
                indices,
                base_vertex,
                instances,
            } => format!("draw indexed {indices:?} {base_vertex} {instances:?}"),
            RenderCommand::DrawIndirect { buffer, offset } => {
                format!(
                    "draw indirect {} at {offset}",
                    self.name(Ref::Buffer(buffer.0))
                )
            }
            RenderCommand::DrawIndexedIndirect { buffer, offset } => format!(
                "draw indexed indirect {} at {offset}",
                self.name(Ref::Buffer(buffer.0))
            ),
            RenderCommand::ExecuteBundles(bundles) => {
                let names: Vec<String> = bundles
                    .iter()
                    .map(|b| self.name(Ref::Bundle(b.0)))
                    .collect();
                format!("bundles [{}]", names.join(", "))
            }
            // Where it is depends on where atlases placed things, which
            // follows arrival order.
            RenderCommand::SetScissor(_) => "scissor".into(),
        }
    }
}

impl<G: Gpu> Gpu for Recording<G> {
    fn caps(&self) -> &Caps {
        self.inner.caps()
    }

    fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> BufferId {
        let id = self.inner.create_buffer(desc);
        let text = format!("{:?}", desc.usage);
        let label = desc.label.to_string();
        self.def(Ref::Buffer(id.0), Def::Plain { label, text });
        id
    }

    fn destroy_buffer(&mut self, id: BufferId) {
        self.inner.destroy_buffer(id);
    }

    fn write_buffer(&mut self, id: BufferId, offset: u64, data: &[u8]) {
        self.inner.write_buffer(id, offset, data);
    }

    fn write_buffer_from_host(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: HostBytesId,
        range: std::ops::Range<u64>,
    ) {
        self.inner.write_buffer_from_host(id, offset, bytes, range);
    }

    fn read_host_bytes(&self, bytes: HostBytesId, range: std::ops::Range<u64>) -> Vec<u8> {
        self.inner.read_host_bytes(bytes, range)
    }

    fn release_host_bytes(&mut self, bytes: HostBytesId) {
        self.inner.release_host_bytes(bytes);
    }

    fn texture_fits(&self, desc: &TextureDesc<'_>) -> bool {
        self.inner.texture_fits(desc)
    }

    fn try_create_texture(&mut self, desc: &TextureDesc<'_>) -> Option<TextureId> {
        let id = self.inner.try_create_texture(desc)?;
        self.texture_made(desc, id);
        Some(id)
    }

    fn create_texture(&mut self, desc: &TextureDesc<'_>) -> TextureId {
        let id = self.inner.create_texture(desc);
        self.texture_made(desc, id);
        id
    }

    fn create_texture_view(
        &mut self,
        texture: TextureId,
        levels: std::ops::Range<u32>,
    ) -> TextureId {
        let id = self.inner.create_texture_view(texture, levels.clone());
        let text = format!("levels {levels:?}");
        self.def(Ref::Texture(id.0), Def::View { of: texture, text });
        id
    }

    fn create_layer_view(&mut self, texture: TextureId, layer: u32, level: u32) -> TextureId {
        let id = self.inner.create_layer_view(texture, layer, level);
        let level = match level {
            0 => String::new(),
            level => format!(" level {level}"),
        };
        let text = format!("layer {layer}{level}");
        self.def(Ref::Texture(id.0), Def::View { of: texture, text });
        id
    }

    fn destroy_texture(&mut self, id: TextureId) {
        self.inner.destroy_texture(id);
    }

    fn write_texture(&mut self, region: &TextureRegion, data: &[u8]) {
        self.inner.write_texture(region, data);
    }

    fn write_texture_from_host(
        &mut self,
        region: &TextureRegion,
        image: HostImageId,
        premultiply: bool,
    ) {
        self.inner
            .write_texture_from_host(region, image, premultiply);
    }

    fn release_host_image(&mut self, image: HostImageId) {
        self.inner.release_host_image(image);
    }

    fn create_sampler(&mut self, desc: &SamplerDesc<'_>) -> SamplerId {
        let id = self.inner.create_sampler(desc);
        let text = format!(
            "{:?} mips {:?} anisotropy {}",
            desc.filter, desc.mipmap_filter, desc.anisotropy
        );
        let label = desc.label.to_string();
        self.def(Ref::Sampler(id.0), Def::Plain { label, text });
        id
    }

    fn destroy_sampler(&mut self, id: SamplerId) {
        self.inner.destroy_sampler(id);
    }

    fn create_shader_module(&mut self, desc: &ShaderModuleDesc<'_>) -> ShaderModuleId {
        let id = self.inner.create_shader_module(desc);
        let label = desc.label.to_string();
        let text = "wgsl".to_string();
        self.def(Ref::Module(id.0), Def::Plain { label, text });
        id
    }

    fn destroy_shader_module(&mut self, id: ShaderModuleId) {
        self.inner.destroy_shader_module(id);
    }

    fn create_bind_group_layout(&mut self, desc: &BindGroupLayoutDesc<'_>) -> BindGroupLayoutId {
        let id = self.inner.create_bind_group_layout(desc);
        let mut text = String::new();
        for e in desc.entries {
            let _ = write!(text, "{} {:?} {:?}; ", e.binding, e.visibility, e.ty);
        }
        let label = desc.label.to_string();
        self.def(Ref::Layout(id.0), Def::Plain { label, text });
        id
    }

    fn destroy_bind_group_layout(&mut self, id: BindGroupLayoutId) {
        self.inner.destroy_bind_group_layout(id);
    }

    fn create_bind_group(&mut self, desc: &BindGroupDesc<'_>) -> BindGroupId {
        let id = self.inner.create_bind_group(desc);
        self.def(
            Ref::Group(id.0),
            Def::Group {
                label: desc.label.to_string(),
                layout: desc.layout,
                entries: desc.entries.to_vec(),
            },
        );
        id
    }

    fn destroy_bind_group(&mut self, id: BindGroupId) {
        self.inner.destroy_bind_group(id);
    }

    fn create_render_pipeline(&mut self, desc: &RenderPipelineDesc<'_>) -> RenderPipelineId {
        let id = self.inner.create_render_pipeline(desc);
        let text = format!(
            "{:?} {:?} {:?} cull back {} samples {}",
            desc.targets, desc.depth, desc.topology, desc.cull_back, desc.sample_count
        );
        self.def(
            Ref::Render(id.0),
            Def::Pipeline {
                label: desc.label.to_string(),
                layouts: desc.bind_group_layouts.to_vec(),
                stages: vec![stage(&desc.vertex), stage(&desc.fragment)],
                text,
            },
        );
        id
    }

    fn create_compute_pipeline(&mut self, desc: &ComputePipelineDesc<'_>) -> ComputePipelineId {
        let id = self.inner.create_compute_pipeline(desc);
        self.def(
            Ref::Compute(id.0),
            Def::Pipeline {
                label: desc.label.to_string(),
                layouts: desc.bind_group_layouts.to_vec(),
                stages: vec![stage(&desc.compute)],
                text: String::new(),
            },
        );
        id
    }

    fn destroy_pipeline(&mut self, id: PipelineId) {
        self.inner.destroy_pipeline(id);
    }

    fn pipeline_ready(&mut self, id: PipelineId) -> bool {
        self.inner.pipeline_ready(id)
    }

    fn poll_pipelines(&mut self) -> PipelineCounts {
        self.inner.poll_pipelines()
    }

    fn prioritize_pipeline(&mut self, id: PipelineId) {
        self.inner.prioritize_pipeline(id);
    }

    fn set_background_compiles(&mut self, on: bool) {
        self.inner.set_background_compiles(on);
    }

    fn create_render_bundle(
        &mut self,
        desc: &RenderBundleDesc<'_>,
        commands: &[RenderCommand],
    ) -> RenderBundleId {
        let id = self.inner.create_render_bundle(desc, commands);
        let text = format!(
            "{:?} depth {:?} samples {}",
            desc.color_formats, desc.depth_format, desc.sample_count
        );
        self.def(
            Ref::Bundle(id.0),
            Def::Bundle {
                label: desc.label.to_string(),
                text,
                commands: commands.to_vec(),
            },
        );
        id
    }

    fn destroy_render_bundle(&mut self, id: RenderBundleId) {
        self.inner.destroy_render_bundle(id);
    }

    fn submit(&mut self, passes: &[Pass]) {
        if self.on {
            self.passes.push(passes.to_vec());
        }
        self.inner.submit(passes);
    }

    fn live_resources(&self) -> usize {
        self.inner.live_resources()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device that makes ids and does nothing else.
    #[derive(Default)]
    struct Ids {
        next: u32,
        caps: Option<Caps>,
    }

    impl Ids {
        fn id(&mut self) -> u32 {
            self.next += 1;
            self.next
        }
    }

    impl Gpu for Ids {
        fn caps(&self) -> &Caps {
            self.caps.as_ref().expect("unused")
        }
        fn create_buffer(&mut self, _: &BufferDesc<'_>) -> BufferId {
            BufferId(self.id())
        }
        fn destroy_buffer(&mut self, _: BufferId) {}
        fn write_buffer(&mut self, _: BufferId, _: u64, _: &[u8]) {}
        fn write_buffer_from_host(
            &mut self,
            _: BufferId,
            _: u64,
            _: HostBytesId,
            _: std::ops::Range<u64>,
        ) {
        }
        fn read_host_bytes(&self, _: HostBytesId, _: std::ops::Range<u64>) -> Vec<u8> {
            Vec::new()
        }
        fn release_host_bytes(&mut self, _: HostBytesId) {}
        fn create_texture(&mut self, _: &TextureDesc<'_>) -> TextureId {
            TextureId(self.id())
        }
        fn create_texture_view(&mut self, _: TextureId, _: std::ops::Range<u32>) -> TextureId {
            TextureId(self.id())
        }
        fn create_layer_view(&mut self, _: TextureId, _: u32, _: u32) -> TextureId {
            TextureId(self.id())
        }
        fn destroy_texture(&mut self, _: TextureId) {}
        fn write_texture(&mut self, _: &TextureRegion, _: &[u8]) {}
        fn write_texture_from_host(&mut self, _: &TextureRegion, _: HostImageId, _: bool) {}
        fn release_host_image(&mut self, _: HostImageId) {}
        fn create_sampler(&mut self, _: &SamplerDesc<'_>) -> SamplerId {
            SamplerId(self.id())
        }
        fn destroy_sampler(&mut self, _: SamplerId) {}
        fn create_shader_module(&mut self, _: &ShaderModuleDesc<'_>) -> ShaderModuleId {
            ShaderModuleId(self.id())
        }
        fn destroy_shader_module(&mut self, _: ShaderModuleId) {}
        fn create_bind_group_layout(&mut self, _: &BindGroupLayoutDesc<'_>) -> BindGroupLayoutId {
            BindGroupLayoutId(self.id())
        }
        fn destroy_bind_group_layout(&mut self, _: BindGroupLayoutId) {}
        fn create_bind_group(&mut self, _: &BindGroupDesc<'_>) -> BindGroupId {
            BindGroupId(self.id())
        }
        fn destroy_bind_group(&mut self, _: BindGroupId) {}
        fn create_render_pipeline(&mut self, _: &RenderPipelineDesc<'_>) -> RenderPipelineId {
            RenderPipelineId(self.id())
        }
        fn create_compute_pipeline(&mut self, _: &ComputePipelineDesc<'_>) -> ComputePipelineId {
            ComputePipelineId(self.id())
        }
        fn destroy_pipeline(&mut self, _: PipelineId) {}
        fn pipeline_ready(&mut self, _: PipelineId) -> bool {
            true
        }
        fn poll_pipelines(&mut self) -> PipelineCounts {
            PipelineCounts::default()
        }
        fn prioritize_pipeline(&mut self, _: PipelineId) {}
        fn create_render_bundle(
            &mut self,
            _: &RenderBundleDesc<'_>,
            _: &[RenderCommand],
        ) -> RenderBundleId {
            RenderBundleId(self.id())
        }
        fn destroy_render_bundle(&mut self, _: RenderBundleId) {}
        fn submit(&mut self, _: &[Pass]) {}
        fn live_resources(&self) -> usize {
            0
        }
    }

    /// Two devices that made the same things in other orders record the
    /// same text.
    #[test]
    fn names_follow_the_text_not_the_ids() {
        let record = |skew: u32| {
            let mut gpu = Recording::new(Ids {
                next: skew,
                ..Ids::default()
            });
            let desc = |label| BufferDesc {
                label,
                size: 4,
                usage: BufferUsages::INDIRECT,
            };
            let (a, b) = if skew == 0 {
                let a = gpu.create_buffer(&desc("args"));
                (a, gpu.create_buffer(&desc("args")))
            } else {
                let b = gpu.create_buffer(&desc("args"));
                (gpu.create_buffer(&desc("args")), b)
            };
            let commands = [
                RenderCommand::DrawIndirect {
                    buffer: a,
                    offset: 0,
                },
                RenderCommand::DrawIndirect {
                    buffer: b,
                    offset: 32,
                },
            ];
            let bundle = gpu.create_render_bundle(
                &RenderBundleDesc {
                    label: "layers",
                    color_formats: &[TextureFormat::Rgba8Unorm],
                    depth_format: None,
                    sample_count: 1,
                },
                &commands,
            );
            gpu.start();
            gpu.submit(&[Pass::Render(RenderPass {
                label: "main",
                colors: Vec::new(),
                depth: None,
                commands: vec![RenderCommand::ExecuteBundles(vec![bundle])],
            })]);
            gpu.finish()
        };
        let text = record(0);
        assert_eq!(text, record(100));
        assert!(
            text.contains("draw indirect buffer \"args\"#1 at 32"),
            "{text}"
        );
    }
}
