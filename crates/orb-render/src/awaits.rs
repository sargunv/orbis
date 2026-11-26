//! What a recording waited for. Draws recorded while a pipeline they need
//! compiles leave it out, or draw with a stand-in that draws only to
//! within rounding of it ([`Variants::drawn`](crate::pipeline::Variants::drawn)),
//! and draws of a row without candidates draw nothing: each recording
//! keeps what it lacked, and records again once one of those pipelines
//! compiled or one of those rows gained candidates, and only then. A frame
//! never depends on the order pipelines compiled in, and a pipeline
//! nothing waited for records nothing again.

use orb_hal::{Gpu, PipelineId};

/// What a draw lacks: pipelines it needs that haven't compiled, or that
/// a stand-in it draws with stands for, `None` for one not made yet.
#[derive(Clone, Debug, Default)]
pub(crate) struct Lacks(Vec<Option<PipelineId>>);

impl Lacks {
    /// Those of `needs` that haven't compiled (`None`: one not made).
    pub(crate) fn of(
        gpu: &mut dyn Gpu,
        needs: impl IntoIterator<Item = Option<PipelineId>>,
    ) -> Self {
        let mut lacks = Self::default();
        lacks.all(gpu, needs);
        lacks
    }

    /// What lacks something it can't name: what makes it records again
    /// on any pipeline's arrival.
    pub(crate) fn unmade() -> Self {
        Self(vec![None])
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Lacks `pipeline` (`None`: one not made yet).
    pub(crate) fn push(&mut self, pipeline: Option<PipelineId>) {
        if !self.0.contains(&pipeline) {
            self.0.push(pipeline);
        }
    }

    pub(crate) fn extend(&mut self, other: &Self) {
        for &p in &other.0 {
            self.push(p);
        }
    }

    /// `pipeline` once compiled; else nothing, and it is lacked.
    pub(crate) fn need<P: Copy + Into<PipelineId>>(
        &mut self,
        gpu: &mut dyn Gpu,
        pipeline: Option<P>,
    ) -> Option<P> {
        let compiled = pipeline.filter(|&p| gpu.pipeline_ready(p.into()));
        if compiled.is_none() {
            self.push(pipeline.map(Into::into));
        }
        compiled
    }

    /// Whether every one of `needs` compiled; those that haven't are
    /// lacked.
    pub(crate) fn all(
        &mut self,
        gpu: &mut dyn Gpu,
        needs: impl IntoIterator<Item = Option<PipelineId>>,
    ) -> bool {
        let mut all = true;
        for need in needs {
            all &= self.need(gpu, need).is_some();
        }
        all
    }
}

/// Whether every one of `needs` compiled (`None`: one not made).
pub(crate) fn compiled(
    gpu: &mut dyn Gpu,
    needs: impl IntoIterator<Item = Option<PipelineId>>,
) -> bool {
    (needs.into_iter()).all(|p| p.is_some_and(|p| gpu.pipeline_ready(p)))
}

/// What one recording waited for ([`Lacks`] of each of its draws), and
/// the rows it left out for want of candidates.
#[derive(Debug, Default)]
pub(crate) struct Awaits {
    pipelines: Vec<PipelineId>,
    /// It lacked a pipeline not made yet, which any arrival may be.
    unmade: bool,
    rows: Vec<u32>,
    /// What drew nothing for what it lacked.
    waiting: u32,
}

impl Awaits {
    /// Something lacked `lacks`, and drew nothing (`drew` false) or a
    /// stand-in.
    pub(crate) fn wait(&mut self, lacks: Lacks, drew: bool) {
        if lacks.is_empty() {
            return;
        }
        self.waiting += u32::from(!drew);
        for lack in lacks.0 {
            match lack {
                Some(p) if !self.pipelines.contains(&p) => self.pipelines.push(p),
                Some(_) => {}
                None => self.unmade = true,
            }
        }
    }

    /// It left out `row`, which had no candidates.
    pub(crate) fn leave_out(&mut self, row: u32) {
        self.rows.push(row);
    }

    /// The rows it left out.
    pub(crate) fn rows(&self) -> &[u32] {
        &self.rows
    }

    /// Takes the rows it left out, leaving the pipelines it lacked.
    pub(crate) fn take_rows(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.rows)
    }

    /// What drew nothing for want of a pipeline.
    pub(crate) fn waiting(&self) -> u32 {
        self.waiting
    }

    /// Whether it lacks any pipeline: what it recorded isn't what the
    /// frame wants yet.
    pub(crate) fn lacking(&self) -> bool {
        self.unmade || !self.pipelines.is_empty()
    }

    /// Whether a pipeline it lacked may have arrived since: one it named
    /// compiled, or it lacked one not made then.
    pub(crate) fn arrived(&self, gpu: &mut dyn Gpu) -> bool {
        self.unmade || (self.pipelines.iter()).any(|&p| gpu.pipeline_ready(p))
    }

    /// Whether a pipeline it named hasn't settled: once nothing is
    /// compiling, one nothing compiles.
    pub(crate) fn stuck(&self, gpu: &mut dyn Gpu) -> bool {
        (self.pipelines.iter()).any(|&p| !gpu.pipeline_ready(p))
    }

    /// Lacks the pipelines `other` lacked too.
    pub(crate) fn absorb(&mut self, other: Self) {
        self.unmade |= other.unmade;
        for p in other.pipelines {
            if !self.pipelines.contains(&p) {
                self.pipelines.push(p);
            }
        }
    }
}
