//! Executing passes: render passes into framebuffers made per attachment,
//! direct draws, indirect draws from CPU copies, bundles replayed.

use std::collections::HashMap;

use glow::HasContext;
use orb_hal::*;

use crate::state::State;
use crate::{GlGpu, Texture, TextureObject};

/// What a framebuffer attaches: colour object, level and layer, and depth.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    color: Object,
    level: u32,
    layer: u32,
    depth: Option<Object>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Object {
    Texture(glow::Texture),
    Renderbuffer(glow::Renderbuffer),
}

impl From<TextureObject> for Object {
    fn from(o: TextureObject) -> Self {
        match o {
            TextureObject::Texture(t) => Self::Texture(t),
            TextureObject::Renderbuffer(rb) => Self::Renderbuffer(rb),
        }
    }
}

/// Framebuffers by attachment, made on first use.
#[derive(Default)]
pub(crate) struct Framebuffers {
    made: HashMap<Key, glow::Framebuffer>,
}

impl Framebuffers {
    /// A framebuffer with `texture`'s `level` and `layer` as its colour.
    pub(crate) fn color(
        &mut self,
        gl: &glow::Context,
        state: &mut State,
        texture: &Texture,
        level: u32,
        layer: u32,
    ) -> glow::Framebuffer {
        self.get(gl, state, texture, level, layer, None)
    }

    fn get(
        &mut self,
        gl: &glow::Context,
        state: &mut State,
        color: &Texture,
        level: u32,
        layer: u32,
        depth: Option<&Texture>,
    ) -> glow::Framebuffer {
        let key = Key {
            color: color.object.into(),
            level,
            layer,
            depth: depth.map(|d| d.object.into()),
        };
        *self.made.entry(key).or_insert_with(|| {
            // SAFETY: a new framebuffer over objects of this context,
            // bound to both targets while attaching.
            unsafe {
                let fbo = gl.create_framebuffer().expect("framebuffer object");
                state.unknown_framebuffers();
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                attach(gl, glow::COLOR_ATTACHMENT0, color, level, layer);
                if let Some(depth) = depth {
                    attach(gl, glow::DEPTH_ATTACHMENT, depth, 0, 0);
                }
                let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
                assert_eq!(
                    status,
                    glow::FRAMEBUFFER_COMPLETE,
                    "incomplete framebuffer ({status:#x})"
                );
                fbo
            }
        })
    }

    /// Deletes every framebuffer.
    pub(crate) fn clear(&mut self, gl: &glow::Context) {
        for (_, fbo) in self.made.drain() {
            // SAFETY: framebuffers this cache made.
            unsafe { gl.delete_framebuffer(fbo) };
        }
    }

    /// Drops the framebuffers that attach `texture`.
    pub(crate) fn forget(&mut self, gl: &glow::Context, state: &mut State, texture: &Texture) {
        let object: Object = texture.object.into();
        self.made.retain(|key, fbo| {
            let keep = key.color != object && key.depth != Some(object);
            if !keep {
                state.forget_framebuffer(*fbo);
                // SAFETY: a framebuffer this cache made.
                unsafe { gl.delete_framebuffer(*fbo) };
            }
            keep
        });
    }
}

/// Attaches level `level` (and `layer`, for arrays) of `texture`.
unsafe fn attach(gl: &glow::Context, attachment: u32, texture: &Texture, level: u32, layer: u32) {
    // SAFETY: the caller's framebuffer is bound to FRAMEBUFFER.
    unsafe {
        match texture.object {
            TextureObject::Renderbuffer(rb) => gl.framebuffer_renderbuffer(
                glow::FRAMEBUFFER,
                attachment,
                glow::RENDERBUFFER,
                Some(rb),
            ),
            TextureObject::Texture(t) => match texture.dimension {
                TextureViewDimension::D2 => gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    attachment,
                    glow::TEXTURE_2D,
                    Some(t),
                    level as i32,
                ),
                TextureViewDimension::D2Array => gl.framebuffer_texture_layer(
                    glow::FRAMEBUFFER,
                    attachment,
                    Some(t),
                    level as i32,
                    layer as i32,
                ),
            },
        }
    }
}

/// The draw state a render pass carries between commands.
#[derive(Default)]
struct Encoder {
    pipeline: Option<RenderPipelineId>,
    groups: [Option<BindGroupId>; 4],
    index: Option<(BufferId, IndexFormat, u64)>,
    /// Pipeline or groups changed since resources were last bound.
    stale: bool,
}

impl GlGpu {
    pub(crate) fn execute(&mut self, pass: &Pass) {
        match pass {
            Pass::Render(pass) => self.render_pass(pass),
            Pass::Compute(pass) => panic!("{}: Compat has no compute", pass.label),
            Pass::CopyTextureLayers {
                source,
                destination,
                layers,
            } => self.copy_layers(*source, *destination, *layers),
            Pass::ClearBuffer {
                buffer,
                offset,
                size,
            } => {
                let end = size.map_or(self.buffers[buffer.0].size, |s| offset + s);
                let zeros = vec![0u8; (end - offset) as usize];
                self.write_buffer(*buffer, *offset, &zeros);
            }
            Pass::ResolveQuerySet { .. } => panic!("Compat has no timestamp queries"),
        }
    }

    fn render_pass(&mut self, pass: &RenderPass) {
        let color = self.textures[pass.color.target.0];
        let depth = pass.depth.map(|d| self.textures[d.target.0]);
        let fbo = self.framebuffers.get(
            &self.gl,
            &mut self.state,
            &color,
            color.base_level,
            color.layer,
            depth.as_ref(),
        );
        self.state.draw_framebuffer(&self.gl, Some(fbo));
        self.state
            .viewport(&self.gl, [0, 0, color.width as i32, color.height as i32]);
        // SAFETY: clears of the bound framebuffer.
        unsafe {
            if let LoadOp::Clear(c) = pass.color.load {
                let c = [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32];
                self.gl.clear_buffer_f32_slice(glow::COLOR, 0, &c);
            }
            if let Some(DepthAttachment {
                load: LoadOp::Clear(d),
                ..
            }) = pass.depth
            {
                self.state.depth_writes(&self.gl);
                self.gl.clear_buffer_f32_slice(glow::DEPTH, 0, &[d]);
            }
        }
        let mut encoder = Encoder::default();
        for command in &pass.commands {
            self.command(&mut encoder, command);
        }
        if let Some(resolve) = pass.color.resolve {
            let target = self.textures[resolve.0];
            let to = self.framebuffers.color(
                &self.gl,
                &mut self.state,
                &target,
                target.base_level,
                target.layer,
            );
            self.state.read_framebuffer(&self.gl, Some(fbo));
            self.state.draw_framebuffer(&self.gl, Some(to));
            // SAFETY: a resolve blit of equal sizes.
            unsafe {
                self.gl.blit_framebuffer(
                    0,
                    0,
                    color.width as i32,
                    color.height as i32,
                    0,
                    0,
                    target.width as i32,
                    target.height as i32,
                    glow::COLOR_BUFFER_BIT,
                    glow::NEAREST,
                );
            }
        }
        let mut discard = Vec::new();
        if pass.color.store == StoreOp::Discard {
            discard.push(glow::COLOR_ATTACHMENT0);
        }
        if pass.depth.is_some_and(|d| d.store == StoreOp::Discard) {
            discard.push(glow::DEPTH_ATTACHMENT);
        }
        if !discard.is_empty() {
            self.state.draw_framebuffer(&self.gl, Some(fbo));
            // SAFETY: a hint about the bound framebuffer's attachments.
            unsafe {
                self.gl
                    .invalidate_framebuffer(glow::DRAW_FRAMEBUFFER, &discard)
            };
        }
    }

    fn command(&mut self, encoder: &mut Encoder, command: &RenderCommand) {
        match command {
            RenderCommand::SetPipeline(p) => {
                encoder.pipeline = Some(*p);
                encoder.stale = true;
            }
            RenderCommand::SetBindGroup { index, group } => {
                encoder.groups[*index as usize] = Some(*group);
                encoder.stale = true;
            }
            RenderCommand::SetIndexBuffer {
                buffer,
                format,
                offset,
            } => encoder.index = Some((*buffer, *format, *offset)),
            RenderCommand::Draw {
                vertices,
                instances,
            } => self.draw(encoder, None, vertices.clone(), instances.clone()),
            RenderCommand::DrawIndexed {
                indices,
                base_vertex,
                instances,
            } => {
                assert_eq!(*base_vertex, 0, "GLES 3.0 has no base vertex");
                self.draw(
                    encoder,
                    Some(indices.start),
                    indices.clone(),
                    instances.clone(),
                );
            }
            RenderCommand::DrawIndirect { buffer, offset } => {
                let [count, instances, first, first_instance] = self.args(*buffer, *offset);
                self.draw(
                    encoder,
                    None,
                    first..first + count,
                    first_instance..first_instance + instances,
                );
            }
            RenderCommand::DrawIndexedIndirect { buffer, offset } => {
                let [count, instances, first, base_vertex] = self.args(*buffer, *offset);
                assert_eq!(base_vertex, 0, "GLES 3.0 has no base vertex");
                self.draw(encoder, Some(first), first..first + count, 0..instances);
            }
            RenderCommand::ExecuteBundles(bundles) => {
                for &bundle in bundles {
                    // Bundles start from empty state, as WebGPU's do.
                    let mut inner = Encoder::default();
                    let commands = std::mem::take(&mut self.bundles[bundle.0]);
                    for command in &commands {
                        self.command(&mut inner, command);
                    }
                    self.bundles[bundle.0] = commands;
                }
                // And leave the pass's state unset after them.
                *encoder = Encoder::default();
            }
        }
    }

    /// The first four argument words at `offset` of an indirect buffer.
    fn args(&self, buffer: BufferId, offset: u64) -> [u32; 4] {
        let shadow = self.read_buffer(buffer);
        let at = offset as usize;
        bytemuck::pod_read_unaligned(&shadow[at..at + 16])
    }

    /// Draws `vertices` (or with `indexed`, the indices from that one) of
    /// `instances`.
    fn draw(
        &mut self,
        encoder: &mut Encoder,
        indexed: Option<u32>,
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    ) {
        if vertices.is_empty() || instances.is_empty() {
            return;
        }
        let id = encoder.pipeline.expect("a pipeline is set before drawing");
        if encoder.stale {
            self.bind(id, &encoder.groups);
            encoder.stale = false;
        }
        let pipeline = &self.pipelines[id.0];
        if let Some(location) = &pipeline.first_instance {
            self.state
                .first_instance(&self.gl, pipeline.program, location, instances.start);
        } else {
            assert_eq!(
                instances.start, 0,
                "a draw past instance 0 reads no instance"
            );
        }
        let mode = pipeline.topology;
        let count = (vertices.end - vertices.start) as i32;
        let instances = (instances.end - instances.start) as i32;
        // SAFETY: draws over resources bound above; the index buffer, if
        // any, is one of this context's.
        unsafe {
            match indexed {
                None => {
                    self.gl
                        .draw_arrays_instanced(mode, vertices.start as i32, count, instances)
                }
                Some(first) => {
                    let (buffer, format, offset) = encoder
                        .index
                        .expect("an index buffer is set before indexed draws");
                    let (gl_buffer, _) = self.buffers[buffer.0]
                        .gl
                        .expect("index buffers are GL buffers");
                    self.state
                        .bind_buffer(&self.gl, glow::ELEMENT_ARRAY_BUFFER, Some(gl_buffer));
                    let (ty, size) = match format {
                        IndexFormat::Uint16 => (glow::UNSIGNED_SHORT, 2),
                        IndexFormat::Uint32 => (glow::UNSIGNED_INT, 4),
                    };
                    self.gl.draw_elements_instanced(
                        mode,
                        count,
                        ty,
                        (offset + u64::from(first) * size) as i32,
                        instances,
                    );
                }
            }
        }
    }

    /// Makes `pipeline` current with its uniforms and textures from
    /// `groups`.
    fn bind(&mut self, id: RenderPipelineId, groups: &[Option<BindGroupId>; 4]) {
        let pipeline = &self.pipelines[id.0];
        self.state.use_program(&self.gl, pipeline.program);
        self.state.blend(&self.gl, pipeline.blend);
        self.state
            .depth(&self.gl, pipeline.depth.map(|d| (d.write, d.compare)));
        self.state.cull(&self.gl, pipeline.cull);
        let resource = |(group, binding): (u32, u32)| -> BindingResource {
            let group = groups[group as usize].expect("pipelines' groups are set before drawing");
            self.groups[group.0]
                .iter()
                .find(|e| e.binding == binding)
                .unwrap_or_else(|| panic!("no binding {binding} in group"))
                .resource
        };
        for (point, &at) in pipeline.blocks.iter().enumerate() {
            let BindingResource::Buffer {
                buffer,
                offset,
                size,
            } = resource(at)
            else {
                panic!("uniform blocks bind buffers");
            };
            let b = &self.buffers[buffer.0];
            let (gl_buffer, _) = b.gl.expect("uniform buffers are GL buffers");
            let size = size.unwrap_or(b.size - offset);
            self.state.bind_uniform(
                &self.gl,
                point as u32,
                gl_buffer,
                offset as i32,
                size as i32,
            );
        }
        for (unit, &(texture, sampler)) in pipeline.textures.iter().enumerate() {
            let unit = unit as u32;
            match resource(texture) {
                BindingResource::Texture { texture, .. } => {
                    let t = self.textures[texture.0];
                    let TextureObject::Texture(object) = t.object else {
                        panic!("multisampled attachments aren't sampled");
                    };
                    self.state
                        .bind_texture(&self.gl, unit, t.target(), Some(object));
                    let sampler = match sampler.map(resource) {
                        Some(BindingResource::Sampler(s)) => self.samplers[s.0],
                        // Loads ignore filtering; integer textures need
                        // nearest.
                        _ => self.data_sampler,
                    };
                    self.state.bind_sampler(&self.gl, unit, sampler);
                }
                BindingResource::Buffer { buffer, offset, .. } => {
                    assert_eq!(offset, 0, "data textures bind whole buffers");
                    let data = self.buffers[buffer.0]
                        .data
                        .as_ref()
                        .expect("storage bindings are of storage buffers");
                    self.state
                        .bind_texture(&self.gl, unit, glow::TEXTURE_2D, Some(data.texture));
                    self.state.bind_sampler(&self.gl, unit, self.data_sampler);
                }
                BindingResource::Sampler(_) => panic!("samplers combine with their textures"),
            }
        }
    }

    fn copy_layers(&mut self, source: TextureId, destination: TextureId, layers: u32) {
        let from = self.textures[source.0];
        let to = self.textures[destination.0];
        for layer in 0..layers {
            let read = self
                .framebuffers
                .color(&self.gl, &mut self.state, &from, 0, layer);
            let draw = self
                .framebuffers
                .color(&self.gl, &mut self.state, &to, 0, layer);
            self.state.read_framebuffer(&self.gl, Some(read));
            self.state.draw_framebuffer(&self.gl, Some(draw));
            // SAFETY: a blit between same-sized layers.
            unsafe {
                self.gl.blit_framebuffer(
                    0,
                    0,
                    from.width as i32,
                    from.height as i32,
                    0,
                    0,
                    from.width as i32,
                    from.height as i32,
                    glow::COLOR_BUFFER_BIT,
                    glow::NEAREST,
                );
            }
        }
    }
}
