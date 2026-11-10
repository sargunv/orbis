//! A device lost in the middle of a frame: the frame's remaining calls
//! fail as part of the loss, not as errors that stop the process.

use orb_hal::*;
use orb_hal_wgpu::WgpuGpu;

#[test]
fn a_frame_goes_on_after_a_loss_in_it() {
    let mut gpu = pollster::block_on(WgpuGpu::offscreen()).expect("a GPU adapter");
    gpu.lose();
    // Adreno X1's Vulkan lost devices before a render bundle's `finish`
    // (#141), which wgpu fails with "Parent device is lost", and panics on.
    let bundle = gpu.create_render_bundle(
        &RenderBundleDesc {
            label: "after the loss",
            color_formats: &[TextureFormat::Rgba8Unorm],
            depth_format: None,
            sample_count: 1,
        },
        &[],
    );
    let texture = gpu.create_texture(&TextureDesc {
        label: "target",
        width: 4,
        height: 4,
        array_layers: 1,
        dimension: TextureViewDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT,
        sample_count: 1,
        mip_levels: 1,
        view_formats: &[],
    });
    gpu.submit(&[Pass::Render(RenderPass {
        label: "after the loss",
        colors: vec![ColorAttachment {
            target: texture,
            resolve: None,
            store: StoreOp::Store,
            load: LoadOp::Clear([0.0; 4]),
        }],
        depth: None,
        commands: vec![RenderCommand::ExecuteBundles(vec![bundle])],
    })]);
    gpu.end_frame(0);
    assert!(gpu.lost());
}

/// Gathered writes on a device the driver lost before Orbis heard: the
/// staging belt's buffers come back invalid, and the upload goes with the
/// loss rather than panicking (Adreno X1's Vulkan, #141), frame after
/// frame.
#[test]
fn gathered_writes_go_quietly_with_the_device() {
    let mut gpu = pollster::block_on(WgpuGpu::offscreen()).expect("a GPU adapter");
    let buffer = gpu.create_buffer(&BufferDesc {
        label: "written",
        size: 1 << 20,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    gpu.write_buffer(buffer, 0, &[1; 16]);
    gpu.submit(&[]);
    gpu.end_frame(0);
    gpu.device().destroy();
    // More than the belt's free chunk holds, so it makes another.
    let run = vec![2; 48 << 10];
    for frame in 1..3 {
        for i in 0..4 {
            gpu.write_buffer(buffer, i * (64 << 10), &run);
        }
        gpu.submit(&[]);
        gpu.end_frame(frame);
    }
    assert!(gpu.lost());
}
