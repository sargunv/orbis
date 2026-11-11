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

/// Host data outlives a lost device: what was registered with it (in the
/// browser, a job worker's packed tile, as its reply arrives) is found,
/// and released, under its id on the device made to replace it, though
/// that device's own ids start again. Before, the map dropping a tile that
/// arrived for the lost device released its bytes on the new one, which
/// had no such id and panicked, the map's wasm with it.
#[test]
fn host_data_moves_to_the_device_that_replaces_a_lost_one() {
    let mut lost = pollster::block_on(WgpuGpu::offscreen()).expect("a GPU adapter");
    let ids: Vec<_> = (0..3u8).map(|i| lost.register_bytes(vec![i; 8])).collect();
    lost.release_host_bytes(ids[0]);
    lost.lose();
    let mut gpu = pollster::block_on(WgpuGpu::offscreen()).expect("a GPU adapter");
    gpu.adopt_host_data(lost.take_host_data());
    assert_eq!(gpu.read_host_bytes(ids[1], 0..8), vec![1; 8]);
    gpu.release_host_bytes(ids[1]);
    let fresh = gpu.register_bytes(vec![9; 4]);
    assert!(
        !ids.contains(&fresh),
        "{fresh:?} reuses an id of the lost device's"
    );
    assert_eq!(gpu.read_host_bytes(ids[2], 0..8), vec![2; 8]);
    assert_eq!(gpu.read_host_bytes(fresh, 0..4), vec![9; 4]);
}
