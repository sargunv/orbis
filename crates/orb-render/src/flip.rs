//! A map drawn upside down into its target
//! ([`Renderer::set_flip_y`](crate::Renderer::set_flip_y)): the map's
//! passes draw it the right way up into a canvas of the frame's size, as
//! they would draw into the target, and one full-screen draw writes the
//! canvas into the target turned over, a texel a pixel. The picture is
//! the upright one mirrored exactly: nothing the map draws, its culling,
//! labels and screen-space reads included, knows which way up it ends.

use orb_hal::*;
use orb_shaders::COMPOSITE_OVER_SHADER;

use crate::offscreen::{Composites, Mode, Offscreen};
use crate::pipeline::{Attachments, Draw};
use crate::release::Release;

/// The canvas and the draw that turns it over.
#[derive(Debug)]
pub(crate) struct Flip {
    /// The canvas at the frame's size, with its composite; made by the
    /// first frame.
    canvas: Option<Offscreen>,
    /// The turning draw's pipeline, into the host's target.
    pipeline: RenderPipelineId,
}

impl Flip {
    /// A flip into targets of `host`'s format, its pipeline asked for now.
    pub(crate) fn new(gpu: &mut dyn Gpu, composites: &Composites, host: Attachments) -> Self {
        let layouts = [composites.layout()];
        let draw = Draw {
            label: "flip",
            ..Draw::new(&COMPOSITE_OVER_SHADER, &layouts, host, None)
        };
        Self {
            canvas: None,
            pipeline: draw.make(gpu),
        }
    }

    /// The turning draw's pipeline: until it compiles the map shows
    /// nothing, so it compiles ahead of the map's own.
    pub(crate) fn pipeline(&self) -> PipelineId {
        self.pipeline.into()
    }

    /// The canvas the map's passes draw into for a frame of `size`, in the
    /// host's `format`, made again at another size.
    pub(crate) fn canvas(
        &mut self,
        gpu: &mut dyn Gpu,
        composites: &mut Composites,
        format: TextureFormat,
        size: [u32; 2],
    ) -> TextureId {
        Offscreen::fit(
            &mut self.canvas,
            gpu,
            composites,
            ("flip canvas", format),
            (size, 1),
            (Mode::Flipped, None),
            1.0,
        );
        let canvas = self.canvas.as_ref().expect("fit made it");
        canvas.target.texture()
    }

    /// The pass that writes the canvas into `target` upside down, or
    /// clears it while the pipeline compiles; whether it drew.
    pub(crate) fn pass(&self, gpu: &mut dyn Gpu, target: TextureId) -> (Pass, bool) {
        let ready = gpu.pipeline_ready(self.pipeline.into());
        let commands = match (&self.canvas, ready) {
            (Some(canvas), true) => {
                Composites::commands_through(&canvas.composite, self.pipeline).to_vec()
            }
            _ => Vec::new(),
        };
        let drew = !commands.is_empty();
        let pass = Pass::Render(RenderPass {
            label: "flip",
            colors: vec![ColorAttachment {
                target,
                resolve: None,
                load: LoadOp::Clear([0.0; 4]),
                store: StoreOp::Store,
            }],
            depth: None,
            commands,
        });
        (pass, drew)
    }
}

impl Release for Flip {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.canvas, self.pipeline).release(gpu);
    }
}
