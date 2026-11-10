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
