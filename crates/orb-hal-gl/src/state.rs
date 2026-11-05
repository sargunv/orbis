//! A GL state cache after GL JS's `gl/value.ts` (BSD-3-Clause): each piece
//! of state the executor sets is remembered, and a set to the current value
//! issues no call.

use glow::HasContext;
use orb_hal::{BlendState, CompareFunction, Face};

#[derive(Default)]
pub(crate) struct State {
    program: Option<glow::Program>,
    read_framebuffer: Option<Option<glow::Framebuffer>>,
    draw_framebuffer: Option<Option<glow::Framebuffer>>,
    viewport: Option<[i32; 4]>,
    blend: Option<Option<BlendState>>,
    depth: Option<Option<(bool, CompareFunction)>>,
    cull: Option<Option<Face>>,
    active_unit: Option<u32>,
    /// Per texture unit: target and texture.
    textures: Vec<Option<(u32, glow::Texture)>>,
    samplers: Vec<Option<glow::Sampler>>,
    /// Per uniform buffer binding point: buffer, offset, size.
    uniforms: Vec<Option<(glow::Buffer, i32, i32)>>,
    /// Buffers bound at the targets the executor uses.
    array_buffer: Option<Option<glow::Buffer>>,
    element_buffer: Option<Option<glow::Buffer>>,
    uniform_buffer: Option<Option<glow::Buffer>>,
    first_instance: Option<(glow::Program, u32)>,
}

fn grow<T: Clone>(v: &mut Vec<Option<T>>, i: usize) -> &mut Option<T> {
    if v.len() <= i {
        v.resize(i + 1, None);
    }
    &mut v[i]
}

impl State {
    pub(crate) fn use_program(&mut self, gl: &glow::Context, program: glow::Program) {
        if self.program != Some(program) {
            self.program = Some(program);
            // SAFETY: a program of this context.
            unsafe { gl.use_program(Some(program)) };
        }
    }

    /// Something else made a program current.
    pub(crate) fn forget_program(&mut self) {
        self.program = None;
    }

    pub(crate) fn read_framebuffer(&mut self, gl: &glow::Context, fbo: Option<glow::Framebuffer>) {
        if self.read_framebuffer != Some(fbo) {
            self.read_framebuffer = Some(fbo);
            // SAFETY: a framebuffer of this context, or the default.
            unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, fbo) };
        }
    }

    pub(crate) fn draw_framebuffer(&mut self, gl: &glow::Context, fbo: Option<glow::Framebuffer>) {
        if self.draw_framebuffer != Some(fbo) {
            self.draw_framebuffer = Some(fbo);
            // SAFETY: as above.
            unsafe { gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, fbo) };
        }
    }

    /// Framebuffers were bound behind the cache.
    pub(crate) fn unknown_framebuffers(&mut self) {
        self.read_framebuffer = None;
        self.draw_framebuffer = None;
    }

    pub(crate) fn forget_framebuffer(&mut self, fbo: glow::Framebuffer) {
        if self.read_framebuffer == Some(Some(fbo)) {
            self.read_framebuffer = None;
        }
        if self.draw_framebuffer == Some(Some(fbo)) {
            self.draw_framebuffer = None;
        }
    }

    pub(crate) fn viewport(&mut self, gl: &glow::Context, rect: [i32; 4]) {
        if self.viewport != Some(rect) {
            self.viewport = Some(rect);
            // SAFETY: plain state.
            unsafe { gl.viewport(rect[0], rect[1], rect[2], rect[3]) };
        }
    }

    pub(crate) fn blend(&mut self, gl: &glow::Context, blend: Option<BlendState>) {
        if self.blend == Some(blend) {
            return;
        }
        // SAFETY: plain state.
        unsafe {
            match blend {
                None => gl.disable(glow::BLEND),
                Some(state) => {
                    if !matches!(self.blend, Some(Some(_))) {
                        gl.enable(glow::BLEND);
                    }
                    let dst = match state {
                        BlendState::PremultipliedAlpha => glow::ONE_MINUS_SRC_ALPHA,
                        BlendState::Additive => glow::ONE,
                    };
                    gl.blend_func(glow::ONE, dst);
                }
            }
        }
        self.blend = Some(blend);
    }

    /// Depth test with `(write, compare)`, or none.
    pub(crate) fn depth(&mut self, gl: &glow::Context, depth: Option<(bool, CompareFunction)>) {
        if self.depth == Some(depth) {
            return;
        }
        // SAFETY: plain state.
        unsafe {
            match depth {
                None => gl.disable(glow::DEPTH_TEST),
                Some((write, compare)) => {
                    gl.enable(glow::DEPTH_TEST);
                    gl.depth_mask(write);
                    gl.depth_func(match compare {
                        CompareFunction::Always => glow::ALWAYS,
                        CompareFunction::Less => glow::LESS,
                        CompareFunction::LessEqual => glow::LEQUAL,
                        CompareFunction::Greater => glow::GREATER,
                        CompareFunction::GreaterEqual => glow::GEQUAL,
                    });
                }
            }
        }
        self.depth = Some(depth);
    }

    /// The depth mask, for clears: GL clears only writable depth.
    pub(crate) fn depth_writes(&mut self, gl: &glow::Context) {
        if let Some(Some((true, _))) = self.depth {
            return;
        }
        // SAFETY: plain state.
        unsafe { gl.depth_mask(true) };
        if let Some(Some((false, compare))) = self.depth {
            // The mask no longer matches the cached state; set it next time.
            let _ = compare;
            self.depth = None;
        }
    }

    pub(crate) fn cull(&mut self, gl: &glow::Context, cull: Option<Face>) {
        if self.cull == Some(cull) {
            return;
        }
        // SAFETY: plain state.
        unsafe {
            match cull {
                None => gl.disable(glow::CULL_FACE),
                Some(face) => {
                    gl.enable(glow::CULL_FACE);
                    gl.cull_face(match face {
                        Face::Front => glow::FRONT,
                        Face::Back => glow::BACK,
                    });
                }
            }
        }
        self.cull = Some(cull);
    }

    fn active_unit(&mut self, gl: &glow::Context, unit: u32) {
        if self.active_unit != Some(unit) {
            self.active_unit = Some(unit);
            // SAFETY: plain state.
            unsafe { gl.active_texture(glow::TEXTURE0 + unit) };
        }
    }

    pub(crate) fn bind_texture(
        &mut self,
        gl: &glow::Context,
        unit: u32,
        target: u32,
        texture: Option<glow::Texture>,
    ) {
        // Uploads bind to the active unit, so it's made active even when
        // the texture is already bound there.
        self.active_unit(gl, unit);
        let slot = grow(&mut self.textures, unit as usize);
        let wanted = texture.map(|t| (target, t));
        if *slot == wanted && wanted.is_some() {
            return;
        }
        let previous = *slot;
        *slot = wanted;
        // SAFETY: a texture of this context at its own target.
        unsafe {
            // A unit holds one texture per target; clear the other.
            if let Some((old_target, _)) = previous
                && old_target != target
            {
                gl.bind_texture(old_target, None);
            }
            gl.bind_texture(target, texture);
        }
    }

    pub(crate) fn forget_texture(&mut self, texture: glow::Texture) {
        for slot in &mut self.textures {
            if slot.is_some_and(|(_, t)| t == texture) {
                *slot = None;
            }
        }
    }

    pub(crate) fn bind_sampler(&mut self, gl: &glow::Context, unit: u32, sampler: glow::Sampler) {
        let slot = grow(&mut self.samplers, unit as usize);
        if *slot != Some(sampler) {
            *slot = Some(sampler);
            // SAFETY: a sampler of this context.
            unsafe { gl.bind_sampler(unit, Some(sampler)) };
        }
    }

    pub(crate) fn bind_buffer(
        &mut self,
        gl: &glow::Context,
        target: u32,
        buffer: Option<glow::Buffer>,
    ) {
        let slot = match target {
            glow::ARRAY_BUFFER => &mut self.array_buffer,
            glow::ELEMENT_ARRAY_BUFFER => &mut self.element_buffer,
            glow::UNIFORM_BUFFER => &mut self.uniform_buffer,
            _ => unreachable!("unused buffer target {target:#x}"),
        };
        if *slot != Some(buffer) {
            *slot = Some(buffer);
            // SAFETY: a buffer of this context at its target.
            unsafe { gl.bind_buffer(target, buffer) };
        }
    }

    pub(crate) fn forget_buffer(&mut self, target: u32, buffer: glow::Buffer) {
        for slot in [
            &mut self.array_buffer,
            &mut self.element_buffer,
            &mut self.uniform_buffer,
        ] {
            if *slot == Some(Some(buffer)) {
                *slot = None;
            }
        }
        if target == glow::UNIFORM_BUFFER {
            for slot in &mut self.uniforms {
                if slot.is_some_and(|(b, ..)| b == buffer) {
                    *slot = None;
                }
            }
        }
    }

    pub(crate) fn bind_uniform(
        &mut self,
        gl: &glow::Context,
        point: u32,
        buffer: glow::Buffer,
        offset: i32,
        size: i32,
    ) {
        let slot = grow(&mut self.uniforms, point as usize);
        if *slot != Some((buffer, offset, size)) {
            *slot = Some((buffer, offset, size));
            // Binding a range binds the generic target too.
            self.uniform_buffer = Some(Some(buffer));
            // SAFETY: a range inside a uniform buffer of this context.
            unsafe {
                gl.bind_buffer_range(glow::UNIFORM_BUFFER, point, Some(buffer), offset, size)
            };
        }
    }

    /// Sets `naga_vs_first_instance` for draws whose instances don't start
    /// at 0 (GLES 3.0 has no base instance).
    pub(crate) fn first_instance(
        &mut self,
        gl: &glow::Context,
        program: glow::Program,
        location: &glow::UniformLocation,
        first: u32,
    ) {
        if self.first_instance != Some((program, first)) {
            self.first_instance = Some((program, first));
            // SAFETY: the current program's uniform.
            unsafe { gl.uniform_1_u32(Some(location), first) };
        }
    }
}
