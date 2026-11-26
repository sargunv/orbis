// GPU devices: the GPU and queue that render sessions share, and the
// target kinds they render to.
//
// Orbis's runtime owns its GPU device, made on the platform's own backend
// and default adapter when its first session makes a target, so an
// `mln_gpu_device` is a token: it names that backend and adapter, and the
// sessions attached with it render on their runtime's device. A wrapped
// MTLDevice must be that device, which an attach checks once the runtime
// has made it; a device can't be named by UUID or LUID, wrap the host's
// Vulkan, OpenGL or WebGPU device, or share the host's queue.

use std::sync::Arc;

use crate::base::{Failure, Result, Retained, UserData, boundary, fresh_out, put, sized};
use crate::handles::{self, Object};
use crate::sys::*;

/// A device, as the sessions attached with it see it.
#[derive(Default)]
pub(crate) struct GpuDevice {
    /// The host's MTLDevice, when the device wraps it.
    pub(crate) metal: Option<Retained>,
}

/// The backend every Orbis device renders with: wgpu's on the platform.
pub(crate) const BACKEND: mln_render_backend = if cfg!(target_vendor = "apple") {
    MLN_RENDER_BACKEND_METAL
} else {
    MLN_RENDER_BACKEND_VULKAN
};

/// A backend as a bit of `mln_render_backend_flags`.
pub(crate) const fn backend_flag(backend: mln_render_backend) -> mln_render_backend_flags {
    1 << (backend - 1)
}

/// The kinds of target that sessions render to.
pub(crate) const TARGET_KINDS: mln_render_target_kind_flags = if cfg!(target_vendor = "apple") {
    MLN_RENDER_TARGET_KIND_FLAG_OFFSCREEN
        | MLN_RENDER_TARGET_KIND_FLAG_METAL_LAYER
        | MLN_RENDER_TARGET_KIND_FLAG_METAL_BORROWED_TEXTURES
} else {
    MLN_RENDER_TARGET_KIND_FLAG_OFFSCREEN
};

/// A target kind as a bit of `mln_render_target_kind_flags`.
pub(crate) const fn kind_flag(kind: mln_render_target_kind) -> mln_render_target_kind_flags {
    1 << (kind - 1)
}

/// The adapter options' identifiers: a UUID or a LUID, of these sizes.
const UUID_BYTES: usize = 16;
const LUID_BYTES: usize = 8;

fn check_options(o: &mln_gpu_device_options) -> Result {
    let known = MLN_GPU_DEVICE_OPTION_BACKEND
        | MLN_GPU_DEVICE_OPTION_ADAPTER_INDEX
        | MLN_GPU_DEVICE_OPTION_ADAPTER_UUID
        | MLN_GPU_DEVICE_OPTION_ADAPTER_LUID
        | MLN_GPU_DEVICE_OPTION_OPEN_GL_CLIENT_API
        | MLN_GPU_DEVICE_OPTION_OPEN_GL_SINGLE_CONTEXT;
    if o.fields & !known != 0 {
        return Err(Failure::argument("options.fields sets an unknown field"));
    }
    let has = |field| o.fields & field != 0;
    let backend = has(MLN_GPU_DEVICE_OPTION_BACKEND).then_some(o.backend);
    if backend
        .is_some_and(|b| !(MLN_RENDER_BACKEND_METAL..=MLN_RENDER_BACKEND_WEB_GPU).contains(&b))
    {
        return Err(Failure::argument("options.backend names no render backend"));
    }
    let id = |field, view: mln_buffer_view, bytes, what| {
        if has(field) && (view.data.is_null() || view.size != bytes) {
            return Err(Failure::argument(format!(
                "options.{what} must hold {bytes} bytes"
            )));
        }
        Ok(())
    };
    id(
        MLN_GPU_DEVICE_OPTION_ADAPTER_UUID,
        o.adapter_uuid,
        UUID_BYTES,
        "adapter_uuid",
    )?;
    id(
        MLN_GPU_DEVICE_OPTION_ADAPTER_LUID,
        o.adapter_luid,
        LUID_BYTES,
        "adapter_luid",
    )?;
    let adapters = [
        MLN_GPU_DEVICE_OPTION_ADAPTER_INDEX,
        MLN_GPU_DEVICE_OPTION_ADAPTER_UUID,
        MLN_GPU_DEVICE_OPTION_ADAPTER_LUID,
    ];
    if adapters.into_iter().filter(|&f| has(f)).count() > 1 {
        return Err(Failure::argument(
            "options name the GPU by more than one of adapter_index, adapter_uuid and adapter_luid",
        ));
    }
    let gl_api = has(MLN_GPU_DEVICE_OPTION_OPEN_GL_CLIENT_API).then_some(o.open_gl_client_api);
    if gl_api
        .is_some_and(|api| !matches!(api, MLN_OPEN_GL_CLIENT_API_GL | MLN_OPEN_GL_CLIENT_API_GLES))
    {
        return Err(Failure::argument("options.open_gl_client_api names no API"));
    }
    check_queue_lock(&o.queue_lock, "options.queue_lock")?;

    if let Some(backend) = backend
        && backend != BACKEND
    {
        return Err(Failure::unsupported(
            "this build renders with its platform's own backend alone: Metal on Apple \
             platforms, Vulkan elsewhere",
        ));
    }
    if o.queue_lock.lock.is_some() {
        return Err(Failure::unsupported(
            "Orbis's devices submit to a queue of their own, which no host shares",
        ));
    }
    if gl_api.is_some() {
        return Err(Failure::unsupported(
            "Orbis renders through Metal or Vulkan, with no OpenGL API",
        ));
    }
    if has(MLN_GPU_DEVICE_OPTION_ADAPTER_UUID) || has(MLN_GPU_DEVICE_OPTION_ADAPTER_LUID) {
        return Err(Failure::unsupported(
            "Orbis names its GPU by adapter index alone, not by UUID or LUID",
        ));
    }
    // The runtime's device is on the default adapter, the one GPU its
    // sessions render on.
    if has(MLN_GPU_DEVICE_OPTION_ADAPTER_INDEX) && o.adapter_index != 0 {
        return Err(Failure::new(
            MLN_STATUS_NOT_FOUND,
            format!(
                "adapter {} names no GPU: Orbis renders on its runtime's adapter, 0",
                o.adapter_index
            ),
        ));
    }
    Ok(())
}

/// Checks a host's queue lock: both callbacks or neither, and user data only
/// with them.
fn check_queue_lock(lock: &mln_queue_lock, what: &str) -> Result {
    if lock.lock.is_some() != lock.unlock.is_some() {
        return Err(Failure::argument(format!(
            "{what} sets one of lock and unlock: set both or neither"
        )));
    }
    if lock.lock.is_none() && (lock.release_user_data.is_some() || !lock.user_data.is_null()) {
        return Err(Failure::argument(format!(
            "{what} names user data but no lock and unlock"
        )));
    }
    Ok(())
}

fn required(object: *mut std::ffi::c_void, what: &str) -> Result {
    match object.is_null() {
        true => Err(Failure::argument(format!("{what} must not be null"))),
        false => Ok(()),
    }
}

/// Hands the caller a new device.
///
/// # Safety
/// `out_device` is null or writable.
unsafe fn give(out_device: *mut mln_gpu_device, device: GpuDevice) -> Result {
    // SAFETY: the caller's pointer.
    unsafe {
        put(
            out_device,
            mln_gpu_device(handles::insert(Object::GpuDevice(Arc::new(device)))),
            "out_device",
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_gpu_device_create(
    options: *const mln_gpu_device_options,
    out_device: *mut mln_gpu_device,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            if !options.is_null() {
                check_options(sized(options, "options")?)?;
            }
            fresh_out(out_device, "out_device")?;
            give(out_device, GpuDevice::default())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_gpu_device_wrap_vulkan(
    descriptor: *const mln_vulkan_device_descriptor,
    out_device: *mut mln_gpu_device,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let d = sized(descriptor, "descriptor")?;
            let c = &d.context;
            required(c.instance, "descriptor.context.instance")?;
            required(c.physical_device, "descriptor.context.physical_device")?;
            required(c.device, "descriptor.context.device")?;
            required(c.graphics_queue, "descriptor.context.graphics_queue")?;
            if d.enabled_extension_count > 0 && d.enabled_extensions.is_null() {
                return Err(Failure::argument(
                    "descriptor.enabled_extensions is null but its count isn't zero",
                ));
            }
            check_queue_lock(&d.queue_lock, "descriptor.queue_lock")?;
            fresh_out(out_device, "out_device")?;
            Err(Failure::unsupported(
                "wrapping a Vulkan device is not supported by this build: Orbis renders on its \
                 runtime's own device",
            ))
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_gpu_device_wrap_metal(
    descriptor: *const mln_metal_device_descriptor,
    out_device: *mut mln_gpu_device,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let d = sized(descriptor, "descriptor")?;
            required(d.device, "descriptor.device")?;
            fresh_out(out_device, "out_device")?;
            if BACKEND != MLN_RENDER_BACKEND_METAL {
                return Err(Failure::unsupported(
                    "Metal is not supported by this build: Metal is only on Apple platforms",
                ));
            }
            if !d.command_queue.is_null() {
                return Err(Failure::unsupported(
                    "Orbis's devices submit to a command queue of their own: wrap the MTLDevice \
                     without one",
                ));
            }
            give(
                out_device,
                GpuDevice {
                    metal: Some(Retained::new(UserData(d.device))),
                },
            )
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_gpu_device_wrap_open_gl(
    context: *const mln_open_gl_context_descriptor,
    out_device: *mut mln_gpu_device,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            if context.is_null() {
                return Err(Failure::argument("context is null"));
            }
            let c = &*context;
            // SAFETY: `platform` tags the union's variant.
            match c.platform {
                MLN_OPEN_GL_CONTEXT_PLATFORM_WGL => {
                    let wgl = c.data.wgl;
                    required(wgl.device_context, "context.wgl.device_context")?;
                    required(wgl.share_context, "context.wgl.share_context")?;
                }
                MLN_OPEN_GL_CONTEXT_PLATFORM_EGL => {
                    let egl = c.data.egl;
                    required(egl.display, "context.egl.display")?;
                    required(egl.config, "context.egl.config")?;
                    required(egl.share_context, "context.egl.share_context")?;
                }
                MLN_OPEN_GL_CONTEXT_PLATFORM_WEB_GL => {}
                _ => {
                    return Err(Failure::argument(
                        "context.platform names no OpenGL context platform",
                    ));
                }
            }
            fresh_out(out_device, "out_device")?;
            Err(Failure::unsupported(
                "OpenGL contexts are not supported by this build: Orbis renders through Metal or \
                 Vulkan",
            ))
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_gpu_device_wrap_web_gpu(
    context: *const mln_web_gpu_context_descriptor,
    out_device: *mut mln_gpu_device,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            if context.is_null() {
                return Err(Failure::argument("context is null"));
            }
            if (*context).device.is_null() {
                return Err(Failure::argument("context.device must not be null"));
            }
            fresh_out(out_device, "out_device")?;
            Err(Failure::unsupported(
                "Orbis wraps no WebGPU device: its runtime renders on a device of its own",
            ))
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_gpu_device_get_capabilities(
    device: mln_gpu_device,
    out_capabilities: *mut mln_gpu_device_capabilities,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            handles::gpu_device(device.0)?;
            sized(out_capabilities.cast_const(), "out_capabilities")?;
            put(
                out_capabilities,
                mln_gpu_device_capabilities {
                    size: std::mem::size_of::<mln_gpu_device_capabilities>() as u32,
                    backend: BACKEND,
                    target_kinds: TARGET_KINDS,
                    export_kinds: 0,
                    sync_kinds: MLN_GPU_SYNC_KIND_FLAG_CPU_COMPLETE,
                },
                "out_capabilities",
            )
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_gpu_device_get_vulkan_context(
    device: mln_gpu_device,
    out_result: *mut mln_gpu_device_vulkan_context_result,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            handles::gpu_device(device.0)?;
            sized(out_result.cast_const(), "out_result")?;
            Err(Failure::unsupported(
                "Orbis shares no Vulkan objects: its runtime's device is wgpu's own",
            ))
        })
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mln_gpu_device_release(device: mln_gpu_device) {
    handles::remove(device.0);
}
