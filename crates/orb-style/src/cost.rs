//! The GPU cost model (plan §8.4, decision 0017): which per-feature paint
//! values the GPU interprets as bytecode and which the CPU evaluates into
//! property rows. Every binding decision goes through [`CostModel::binding`],
//! for the diff's change classes and for the renderer's plan alike.

use orb_expr::PropertyExpression;
use orb_expr::gpu::Program;

use crate::Binding;

/// Paint properties the renderer can bind as bytecode: the fill, fill
/// outline, line and circle shaders' data-driven numbers and colours.
const BYTECODE_PROPERTIES: &[&str] = &[
    "fill-color",
    "fill-opacity",
    "fill-outline-color",
    "line-color",
    "line-opacity",
    "line-width",
    "line-gap-width",
    "line-offset",
    "line-blur",
    "circle-color",
    "circle-radius",
    "circle-blur",
    "circle-opacity",
    "circle-stroke-color",
    "circle-stroke-width",
    "circle-stroke-opacity",
];

/// How much interpretation a per-feature value may cost the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostModel {
    /// The most instructions a value may take and still be interpreted
    /// when property rows could hold it instead. 0 binds every such value
    /// as rows. Values that read both feature state and feature data have
    /// no row binding and are interpreted up to `orb_expr::gpu::MAX_OPS`.
    pub max_ops: usize,
}

impl CostModel {
    /// Rows wherever rows can hold the value.
    pub const ROWS: Self = Self { max_ops: 0 };
    /// Every value the GPU can evaluate interpreted.
    pub const ALL: Self = Self {
        max_ops: orb_expr::gpu::MAX_OPS,
    };
    /// The measured default (decision 0017): on an Adreno 619 even a
    /// two-instruction `match` costs twice the rows frame, so only values
    /// that rows cannot hold are interpreted.
    pub const DEFAULT: Self = Self::ROWS;
}

impl Default for CostModel {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl CostModel {
    /// How a parsed paint value of property `name` binds.
    pub fn binding(&self, name: &str, value: &PropertyExpression) -> Binding {
        if self.program(name, value).is_some() {
            return Binding::Bytecode;
        }
        if value.reads_feature_state() && !value.reads_feature_properties() {
            return Binding::State;
        }
        match value.kind() {
            orb_expr::Kind::Constant => Binding::Constant,
            orb_expr::Kind::Camera => Binding::Camera,
            orb_expr::Kind::Source => Binding::Source,
            orb_expr::Kind::Composite => Binding::Composite,
        }
    }

    /// The program of a value this model binds as bytecode.
    pub fn program(&self, name: &str, value: &PropertyExpression) -> Option<Program> {
        if !value.reads_feature_properties() || !BYTECODE_PROPERTIES.contains(&name) {
            return None;
        }
        let program = Program::compile(value.expression()).ok()?;
        let mixed = value.reads_feature_state();
        (mixed || program.len() <= self.max_ops).then_some(program)
    }
}
