// Borrowed rings: the host's textures a session renders into, the frames it
// hands the host, and what the host does with them.
//
// A ring's slots live in the session's handle, so the host acquires,
// releases and discards frames on its own threads while the runtime's
// thread picks the slot each frame draws into. Each frame is done on the
// GPU before the host hears of it (MLN_GPU_SYNC_CPU_COMPLETE): Orbis
// submits on a queue of its own, which no host shares.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::base::{Failure, Result, UserData, boundary, put, sized};
use crate::delivery::Completion;
use crate::handles::{self, Object};
use crate::session::{SessionHandle, TargetSpec};
use crate::sys::*;

/// One of the host's textures.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SlotInfo {
    texture: UserData,
    device: UserData,
    pixel_format: u64,
    width: u32,
    height: u32,
    /// Physical pixels per logical pixel.
    scale: f64,
}

#[derive(Debug)]
enum SlotState {
    Free,
    /// The runtime's thread draws into it.
    Drawing,
    /// It holds the newest frame, which the host hasn't acquired.
    Ready,
    /// The host holds its frame, by this handle.
    Acquired(u64),
    /// The host gave its frame up without saying when its GPU stops
    /// reading it, so nothing draws into it again.
    Quarantined,
}

#[derive(Debug)]
struct Slot {
    info: SlotInfo,
    state: SlotState,
    /// The result of the frame it holds last.
    frame: mln_render_frame_result,
    /// When it was last drawn into, in draws of the ring.
    drawn: u64,
}

/// A session's ring of the host's textures.
#[derive(Debug, Default)]
pub(crate) struct Ring {
    slots: Vec<Slot>,
    /// Grows with each set of textures, so a frame of an earlier set gives
    /// back nothing.
    generation: u64,
    draws: u64,
}

/// Where a frame draws.
pub(crate) enum Begin {
    Slot(usize),
    /// The host holds every usable slot.
    Wait,
    /// No slot will take a frame again.
    Quarantined,
}

impl SlotInfo {
    pub(crate) fn pixel_format(&self) -> u64 {
        self.pixel_format
    }
}

impl Ring {
    pub(crate) fn new(slots: Vec<SlotInfo>) -> Self {
        let mut ring = Self::default();
        ring.replace(slots);
        ring
    }

    pub(crate) fn depth(&self) -> u32 {
        self.slots.len() as u32
    }

    /// New textures: every slot is free again, the quarantined ones too.
    pub(crate) fn replace(&mut self, slots: Vec<SlotInfo>) {
        self.generation += 1;
        self.slots = slots
            .into_iter()
            .map(|info| Slot {
                info,
                state: SlotState::Free,
                frame: mln_render_frame_result::default(),
                drawn: 0,
            })
            .collect();
    }

    /// The host took the textures back: no frame is left to acquire.
    pub(crate) fn release_target(&mut self) {
        self.generation += 1;
        for slot in &mut self.slots {
            slot.state = SlotState::Free;
        }
    }

    /// Whether the ring holds these textures, in this order.
    pub(crate) fn holds(&self, slots: &[SlotInfo]) -> bool {
        self.slots.len() == slots.len()
            && (self.slots.iter().zip(slots)).all(|(s, t)| s.info.texture.get() == t.texture.get())
    }

    /// The pixel format of its textures.
    pub(crate) fn pixel_format(&self) -> Option<u64> {
        self.slots.first().map(|s| s.info.pixel_format)
    }

    /// The session's textures are gone: its frames give back nothing.
    pub(crate) fn clear(&mut self) {
        if !self.slots.is_empty() {
            self.generation += 1;
            self.slots.clear();
        }
    }

    /// Whether a demand can still render into the ring.
    pub(crate) fn usable(&self) -> Result {
        if !self.slots.is_empty()
            && self
                .slots
                .iter()
                .all(|s| matches!(s.state, SlotState::Quarantined))
        {
            return Err(Failure::state(
                "every texture of the session's ring is quarantined: the host discarded its frames",
            ));
        }
        Ok(())
    }

    /// Picks the slot the next frame draws into: the free one drawn into
    /// longest ago, or the one holding a frame the host hasn't taken,
    /// which the new frame replaces.
    pub(crate) fn begin(&mut self) -> Begin {
        let free = (self.slots.iter().enumerate())
            .filter(|(_, s)| matches!(s.state, SlotState::Free))
            .min_by_key(|(_, s)| s.drawn)
            .map(|(i, _)| i);
        let pick = free.or_else(|| {
            self.slots
                .iter()
                .position(|s| matches!(s.state, SlotState::Ready))
        });
        match pick {
            Some(i) => {
                self.draws += 1;
                let slot = &mut self.slots[i];
                slot.drawn = self.draws;
                slot.state = SlotState::Drawing;
                Begin::Slot(i)
            }
            None if self.usable().is_err() || self.slots.is_empty() => Begin::Quarantined,
            None => Begin::Wait,
        }
    }

    /// Ends a frame drawn into `slot`: it holds `frame`, which drops an
    /// older one the host didn't take, or nothing when the frame failed.
    pub(crate) fn end(&mut self, slot: usize, frame: Option<mln_render_frame_result>) {
        if frame.is_some() {
            for s in &mut self.slots {
                if matches!(s.state, SlotState::Ready) {
                    s.state = SlotState::Free;
                }
            }
        }
        if let Some(s) = self.slots.get_mut(slot) {
            s.state = match frame {
                Some(frame) => {
                    s.frame = frame;
                    SlotState::Ready
                }
                None => SlotState::Free,
            };
        }
    }
}

/// A frame the host acquired.
pub(crate) struct FrameHandle {
    session: Arc<SessionHandle>,
    ring_generation: u64,
    slot: usize,
    info: SlotInfo,
    result: mln_render_frame_result,
    views: AtomicUsize,
}

impl FrameHandle {
    /// Whether the session still backs the objects the frame lends.
    fn backed(&self) -> Result {
        if self.session.lifecycle() != MLN_RENDER_SESSION_STATE_ATTACHED
            || self.session.ring.lock().unwrap().generation != self.ring_generation
        {
            return Err(Failure::new(
                MLN_STATUS_TARGET_LOST,
                "the frame's session failed, or was detached, abandoned or disposed",
            ));
        }
        Ok(())
    }

    /// Gives the frame's slot back to the session, free or quarantined.
    fn give_back(&self, quarantine: bool) {
        {
            let mut ring = self.session.ring.lock().unwrap();
            if ring.generation == self.ring_generation
                && let Some(slot) = ring.slots.get_mut(self.slot)
            {
                slot.state = if quarantine {
                    SlotState::Quarantined
                } else {
                    SlotState::Free
                };
            }
        }
        self.session.count_acquired(-1);
        self.session.wake_runtime();
    }
}

/// Ends every frame the host holds of a session that goes: their handles
/// go stale, and their slots take no frame again.
pub(crate) fn end_frames(session: &SessionHandle) {
    let mut ring = session.ring.lock().unwrap();
    for slot in &mut ring.slots {
        if let SlotState::Acquired(frame) = slot.state {
            handles::remove(frame);
            slot.state = SlotState::Quarantined;
        }
    }
    session.state_mut(|s| s.acquired_frame_count = 0);
}

fn frame(frame: mln_acquired_frame) -> Result<Arc<FrameHandle>> {
    handles::frame(frame.0)
}

/// Validates the host's ring of Metal textures, as every Metal attach and
/// replacement gives it.
///
/// # Safety
/// `textures` is null or holds `count` entries.
unsafe fn metal_ring(
    logical: mln_logical_size,
    physical_width: u32,
    physical_height: u32,
    textures: *const mln_metal_borrowed_texture,
    count: usize,
) -> Result<Vec<SlotInfo>> {
    if logical.width == 0 || logical.height == 0 {
        return Err(Failure::argument(
            "the logical size's width and height must be positive",
        ));
    }
    if physical_width == 0 || physical_height == 0 {
        return Err(Failure::argument(
            "the physical texture dimensions must be positive",
        ));
    }
    if textures.is_null() || count == 0 {
        return Err(Failure::argument("a ring takes at least one texture"));
    }
    if count > MLN_TEXTURE_RING_MAX_DEPTH as usize {
        return Err(Failure::argument(format!(
            "a ring holds at most MLN_TEXTURE_RING_MAX_DEPTH ({MLN_TEXTURE_RING_MAX_DEPTH}) textures"
        )));
    }
    // SAFETY: the caller's contract.
    let textures = unsafe { std::slice::from_raw_parts(textures, count) };
    for (i, t) in textures.iter().enumerate() {
        if t.texture.is_null() {
            return Err(Failure::argument(format!("textures[{i}].texture is null")));
        }
        if textures[..i].iter().any(|u| u.texture == t.texture) {
            return Err(Failure::argument("a ring takes distinct textures"));
        }
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let mut slots: Vec<SlotInfo> = Vec::with_capacity(count);
        for (i, t) in textures.iter().enumerate() {
            let texture = std::ptr::NonNull::new(t.texture).expect("checked above");
            // SAFETY: the C API's contract: a live MTLTexture.
            let info = unsafe { orb_api::metal_texture(texture) };
            if (info.width, info.height) != (physical_width, physical_height) {
                return Err(Failure::argument(format!(
                    "textures[{i}] is {}x{}: its size must match descriptor physical size \
                     {physical_width}x{physical_height}",
                    info.width, info.height
                )));
            }
            if let Some(first) = slots.first()
                && (first.device.get() != info.device.as_ptr()
                    || first.pixel_format != info.pixel_format)
            {
                return Err(Failure::argument(
                    "the textures of a ring must share a device and a pixel format",
                ));
            }
            if !info.render_target {
                return Err(Failure::argument(format!(
                    "textures[{i}] isn't a 2D render target of one sample, layer and level \
                     (MTLTextureUsageRenderTarget)"
                )));
            }
            slots.push(SlotInfo {
                texture: UserData(t.texture),
                device: UserData(info.device.as_ptr()),
                pixel_format: info.pixel_format,
                width: info.width,
                height: info.height,
                scale: f64::from(physical_width) / f64::from(logical.width),
            });
        }
        Ok(slots)
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    Err(Failure::unsupported(
        "Metal textures are not supported by this build: Metal is only on Apple platforms",
    ))
}

/// One of the host's textures as a target of the runtime's device.
pub(crate) fn import(runtime: &orb_api::Runtime, slot: &SlotInfo) -> Result<orb_api::Target> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let texture = std::ptr::NonNull::new(slot.texture.get())
            .ok_or_else(|| Failure::argument("the texture is null"))?;
        let texture = runtime
            .with_device(|gpu| {
                // SAFETY: the C API's contract: a live MTLTexture that the
                // host keeps until the session lets go of it.
                unsafe { gpu.import_metal_texture(texture) }
            })
            .map_err(Failure::unsupported)?;
        orb_api::Target::borrowed(runtime, &texture, slot.scale).map_err(Failure::from)
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        let _ = (runtime, slot);
        Err(Failure::unsupported("Metal is only on Apple platforms"))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_attach_metal_borrowed_texture(
    map: mln_map,
    descriptor: *const mln_metal_borrowed_texture_descriptor,
    options: *const mln_render_session_attach_options,
    out_session: *mut mln_render_session,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let d = sized(descriptor, "descriptor")?;
            let slots = metal_ring(
                d.logical_size,
                d.physical_width,
                d.physical_height,
                d.textures,
                d.texture_count,
            )?;
            crate::session::attach(
                map,
                d.logical_size,
                TargetSpec::Ring(slots),
                options,
                out_session,
                completion,
            )
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_set_metal_borrowed_texture_target(
    session: mln_render_session,
    replacement: *const mln_metal_borrowed_texture_replacement,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let r = sized(replacement, "replacement")?;
            let slots = metal_ring(
                r.logical_size,
                r.physical_width,
                r.physical_height,
                r.textures,
                r.texture_count,
            )?;
            let handle = handles::session(session.0)?;
            handle.usable()?;
            let depth = handle.ring_depth();
            if depth == 0 {
                return Err(Failure::unsupported(
                    if crate::session::renders_offscreen(&handle) {
                        "an offscreen session renders into a texture of its own and takes no target"
                    } else {
                        "a native surface session takes a surface, not a caller-owned texture"
                    },
                ));
            }
            if slots.len() != depth as usize {
                return Err(Failure::argument(format!(
                    "a replacement keeps the ring depth: give it {depth} textures, as the ring \
                     attached with"
                )));
            }
            let device = handle
                .ring
                .lock()
                .unwrap()
                .slots
                .first()
                .map(|s| s.info.device.get());
            if device != Some(slots[0].device.get()) {
                return Err(Failure::argument(
                    "the ring keeps its device: give it textures of the same MTLDevice",
                ));
            }
            if handle.lifecycle() == MLN_RENDER_SESSION_STATE_ATTACHING {
                return Err(Failure::state("the session is still attaching"));
            }
            crate::session::no_frames_held(&handle)?;
            let logical = r.logical_size;
            crate::session::command(
                &handle,
                Completion::from_c(completion)?,
                move |state, id| crate::session::retarget_ring(state, id, logical, slots),
            )
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_acquire_frame(
    session: mln_render_session,
    out_frame: *mut mln_acquired_frame,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            if out_frame.is_null() {
                return Err(Failure::argument("out_frame is null"));
            }
            let handle = handles::session(session.0)?;
            handle.usable()?;
            if handle.ring_depth() == 0 {
                return Err(Failure::unsupported(
                    "the session lacks MLN_RENDER_SESSION_CAPABILITY_FRAME_ACQUISITION",
                ));
            }
            let mut ring = handle.ring.lock().unwrap();
            let generation = ring.generation;
            let Some((slot, s)) = (ring.slots.iter_mut().enumerate())
                .find(|(_, s)| matches!(s.state, SlotState::Ready))
            else {
                return put(out_frame, mln_acquired_frame(0), "out_frame");
            };
            let result = s.frame;
            let id = handles::insert(Object::Frame(Arc::new(FrameHandle {
                session: handle.clone(),
                ring_generation: generation,
                slot,
                info: s.info,
                result,
                views: AtomicUsize::new(0),
            })));
            s.state = SlotState::Acquired(id);
            handle.count_acquired(1);
            put(out_frame, mln_acquired_frame(id), "out_frame")
        })
    }
}

/// An accessor of an acquired frame: checks its output record, then the
/// frame, then writes what `read` gives.
///
/// # Safety
/// `out` is null or a writable `T` whose `size` says how much is.
unsafe fn read_frame<T>(
    f: mln_acquired_frame,
    out: *mut T,
    what: &str,
    read: impl FnOnce(&FrameHandle) -> Result<T>,
) -> Result {
    // SAFETY: the caller's contract.
    unsafe { sized(out.cast_const(), what)? };
    let handle = frame(f)?;
    let value = read(&handle)?;
    // SAFETY: as above.
    unsafe { put(out, value, what) }
}

fn other_backend<T>(_: &FrameHandle) -> Result<T> {
    Err(Failure::unsupported(
        "the frame is of a different render backend: Orbis's rings are Metal textures",
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_acquired_frame_get_result(
    frame: mln_acquired_frame,
    out_result: *mut mln_render_frame_result,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            read_frame(frame, out_result, "out_result", |f| Ok(f.result))
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_acquired_frame_get_producer_sync(
    frame: mln_acquired_frame,
    out_sync: *mut mln_gpu_sync,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            read_frame(frame, out_sync, "out_sync", |f| {
                f.backed()?;
                Ok(mln_gpu_sync {
                    size: std::mem::size_of::<mln_gpu_sync>() as u32,
                    kind: MLN_GPU_SYNC_CPU_COMPLETE,
                    handle: 0,
                    value: 0,
                })
            })
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_acquired_frame_get_metal_texture(
    frame: mln_acquired_frame,
    out_frame: *mut mln_metal_texture_frame,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            read_frame(frame, out_frame, "out_frame", |f| {
                f.backed()?;
                Ok(mln_metal_texture_frame {
                    size: std::mem::size_of::<mln_metal_texture_frame>() as u32,
                    width: f.info.width,
                    height: f.info.height,
                    scale_factor: f.info.scale,
                    slot: f.slot as u32,
                    texture: f.info.texture.get(),
                    device: f.info.device.get(),
                    pixel_format: f.info.pixel_format,
                })
            })
        })
    }
}

// The accessors of the backends Orbis's rings aren't of.

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_acquired_frame_get_vulkan_texture(
    frame: mln_acquired_frame,
    out_frame: *mut mln_vulkan_texture_frame,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            read_frame(frame, out_frame, "out_frame", other_backend)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_acquired_frame_get_open_gl_texture(
    frame: mln_acquired_frame,
    out_frame: *mut mln_open_gl_texture_frame,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            read_frame(frame, out_frame, "out_frame", other_backend)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_acquired_frame_get_web_gpu_texture(
    frame: mln_acquired_frame,
    out_frame: *mut mln_web_gpu_texture_frame,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            read_frame(frame, out_frame, "out_frame", other_backend)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_acquired_frame_get_io_surface(
    frame: mln_acquired_frame,
    out_frame: *mut mln_io_surface_frame,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            read_frame(frame, out_frame, "out_frame", other_backend)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_acquired_frame_get_android_hardware_buffer(
    frame: mln_acquired_frame,
    out_frame: *mut mln_android_hardware_buffer_frame,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            read_frame(frame, out_frame, "out_frame", other_backend)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_acquired_frame_get_dma_buf(
    frame: mln_acquired_frame,
    out_frame: *mut mln_dma_buf_frame,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            read_frame(frame, out_frame, "out_frame", other_backend)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_acquired_frame_get_dxgi_shared_texture(
    frame: mln_acquired_frame,
    out_frame: *mut mln_dxgi_shared_texture_frame,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            read_frame(frame, out_frame, "out_frame", other_backend)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_acquired_frame_release(
    frame: mln_acquired_frame,
    consumer_completion: *const mln_gpu_sync,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = self::frame(frame)?;
            let sync = sized(consumer_completion, "consumer_completion")?;
            match sync.kind {
                MLN_GPU_SYNC_CPU_COMPLETE => {}
                kind if kind <= MLN_GPU_SYNC_D3D12_FENCE => {
                    return Err(Failure::unsupported(format!(
                        "the session waits for no gpu sync kind but MLN_GPU_SYNC_CPU_COMPLETE, \
                         not {kind}"
                    )));
                }
                kind => {
                    return Err(Failure::unsupported(format!(
                        "{kind} names no gpu sync kind"
                    )));
                }
            }
            if handle.views.load(Ordering::Acquire) > 0 {
                return Err(Failure::new(
                    MLN_STATUS_BUSY,
                    "a view of the frame is open: end it first",
                ));
            }
            if handles::remove(frame.0).is_some() {
                handle.give_back(false);
            }
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_acquired_frame_discard(
    frame: mln_acquired_frame,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointer.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = self::frame(frame)?;
            if handles::remove(frame.0).is_some() {
                handle.give_back(true);
            }
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_acquired_frame_view_begin(
    frame: mln_acquired_frame,
    out_scope: *mut *mut c_void,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            if out_scope.is_null() {
                return Err(Failure::argument("out_scope must not be null"));
            }
            let handle = self::frame(frame)?;
            handle.backed()?;
            handle.views.fetch_add(1, Ordering::AcqRel);
            handle.session.open_views.fetch_add(1, Ordering::AcqRel);
            out_scope.write(Box::into_raw(Box::new(handle)).cast());
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_acquired_frame_view_end(scope: *mut c_void) {
    if scope.is_null() {
        return;
    }
    // SAFETY: a scope `mln_acquired_frame_view_begin` made, by the C API's
    // contract.
    let handle = unsafe { Box::from_raw(scope.cast::<Arc<FrameHandle>>()) };
    handle.views.fetch_sub(1, Ordering::AcqRel);
    handle.session.open_views.fetch_sub(1, Ordering::AcqRel);
}
