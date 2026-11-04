//! Paint expressions lowered to bytecode for the GPU interpreter
//! (plan §8.4's `expr` module, §11.3's `GpuBytecode` binding plan).
//!
//! [`Program::compile`] lowers the bounded, GPU-evaluable subset of an
//! expression; anything else stays on the CPU, with a [`Fallback`] saying
//! why. Programs of one render node share a [`Layout`]: the feature inputs
//! packed per feature beside its property row, the raw feature-state values
//! in its state block, and one string dictionary. [`Layout::encode`] emits
//! the words `expr.wgsl` interprets.
//!
//! **Precision.** The CPU evaluates in f64. Every branch the GPU takes
//! (`match`, `case`, `step`, comparisons, `coalesce`, `in`) must equal the
//! VM's, so branch operands carry exact values: numbers as f64 bit patterns
//! compared through a total order, strings as dictionary ids, booleans and
//! nulls as tags. Arithmetic, interpolation and maths run in f32 and may
//! only produce outputs: one that feeds a branch falls back at compile time
//! ([`Fallback::InexactBranch`]). A numeric `feature-state` operand of a
//! branch falls back too ([`Fallback::StateBranch`]), so a binding never
//! depends on state written later.
//!
//! **Values** are four words `[tag, x, y, z]`: a number is its f64 bits in
//! `x`, `y` and its f32 in `z`; a colour its straight RGBA as two unorm16
//! pairs; a string its dictionary id; a boolean 0 or 1.

use crate::Expression;
use crate::color::Color;
use crate::ir::{Call, CmpOp, ColorSpace, Interpolation, Ir, MatchLabels, MathFn, NodeId, Op};
use crate::value::{Type, Value};

/// The most instructions a program may have.
pub const MAX_OPS: usize = 32;
/// The most labels, branches or arguments of one instruction.
pub const MAX_STOPS: usize = 16;
/// Registers of the interpreter (`expr.wgsl`).
pub const REGISTERS: usize = 16;

/// Why an expression stays on the CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Fallback {
    /// An operator, operand type or result type outside the subset.
    Unsupported,
    /// More than [`MAX_OPS`] instructions, [`MAX_STOPS`] stops or
    /// [`REGISTERS`] live values.
    Limit,
    /// f32 arithmetic would decide a branch.
    InexactBranch,
    /// A numeric `feature-state` value would decide a branch.
    StateBranch,
}

/// Where an input value comes from.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Source {
    Property(Box<str>),
    Id,
    GeometryType,
    State(Box<str>),
}

/// A conversion applied when packing, so the GPU never parses strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Coerce {
    None,
    /// `to-number`: null is 0, an unconvertible value fails.
    Number,
    /// `to-color`: a CSS string or an RGBA array; anything else fails.
    Color,
}

/// One value a program reads per feature (or per feature state).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Input {
    pub source: Source,
    pub coerce: Coerce,
}

impl Input {
    /// Words per packed value.
    pub fn words(&self) -> u32 {
        if self.coerce == Coerce::Color { 3 } else { 2 }
    }

    pub fn is_state(&self) -> bool {
        matches!(self.source, Source::State(_))
    }
}

// Value tags, as `expr.wgsl` numbers them.
const NULL: u32 = 0;
const NUMBER: u32 = 1;
const BOOL: u32 = 2;
const STRING: u32 = 3;
const COLOR: u32 = 4;
const ERROR: u32 = 5;
/// An array or object: equal to nothing, of no asserted type.
const OTHER: u32 = 6;
/// A failed pack-time coercion.
const FAIL: u32 = 7;
/// An absent property: `null` that `has` reports missing.
const ABSENT: u32 = 8;
/// High word of a packed non-number: a negative quiet NaN no number packs to.
const BOXED: u32 = 0xfff8_0000;
/// The id of a string outside the dictionary.
pub const UNKNOWN_STRING: u32 = u32::MAX;
/// Operands at or above this address the constant pool.
const CONSTANT: u32 = 0x8000_0000;
/// Input table flags (`expr.wgsl`).
const STATE_INPUT: u32 = 1 << 31;
const COLOR_INPUT: u32 = 1 << 30;

// Opcodes.
const OP_INPUT: u32 = 1;
const OP_ZOOM: u32 = 2;
const OP_NOT: u32 = 3;
const OP_COMPARE: u32 = 4;
const OP_ALL_ANY: u32 = 5;
const OP_CASE: u32 = 6;
const OP_MATCH: u32 = 7;
const OP_COALESCE: u32 = 8;
const OP_STEP: u32 = 9;
const OP_INTERPOLATE: u32 = 10;
const OP_NARY: u32 = 11;
const OP_BINARY: u32 = 12;
const OP_UNARY: u32 = 13;
const OP_ASSERT: u32 = 14;
const OP_COERCE: u32 = 15;
const OP_IN: u32 = 16;
const OP_TYPE: u32 = 17;

/// Words of a program's header: count, result, constants, inputs, default.
const HEADER: u32 = 8;

#[derive(Clone, Debug, PartialEq)]
enum Constant {
    Value([u32; 4]),
    /// A string literal, by index into the program's strings.
    String(u32),
}

#[derive(Clone, Copy, Debug)]
struct Instruction {
    op: u32,
    aux: u32,
    dst: u32,
    w: [u32; 3],
}

impl Instruction {
    /// Whether `w[0]` is a table offset, and which words are operands.
    fn shape(&self) -> (bool, [bool; 3]) {
        match self.op {
            OP_INPUT | OP_ZOOM => (false, [false; 3]),
            OP_NOT | OP_UNARY | OP_TYPE => (false, [true, false, false]),
            OP_COMPARE | OP_BINARY => (false, [true, true, false]),
            OP_CASE | OP_STEP | OP_IN => (true, [false, true, false]),
            OP_MATCH | OP_INTERPOLATE => (true, [false, true, true]),
            _ => (true, [false; 3]),
        }
    }

    fn count(&self) -> usize {
        (self.aux & 0xff) as usize
    }
}

/// A compiled program, before [`Layout::encode`] fixes its inputs and
/// strings to a node's layout.
#[derive(Clone, Debug)]
pub struct Program {
    code: Vec<Instruction>,
    tables: Vec<u32>,
    constants: Vec<Constant>,
    inputs: Vec<Input>,
    strings: Vec<Box<str>>,
    result: u32,
}

impl Program {
    /// Lowers `expression`, or says why it stays on the CPU. Results must
    /// be numbers, colours or booleans.
    pub fn compile(expression: &Expression) -> Result<Self, Fallback> {
        let ir = &expression.ir;
        if !matches!(expression.ty(), Type::Number | Type::Color | Type::Boolean) {
            return Err(Fallback::Unsupported);
        }
        let mut c = Compiler {
            ir,
            operand: vec![None; ir.nodes.len()],
            exact: vec![true; ir.nodes.len()],
            state: vec![false; ir.nodes.len()],
            limit: false,
            coercions: Vec::new(),
            program: Program {
                code: Vec::new(),
                tables: Vec::new(),
                constants: Vec::new(),
                inputs: Vec::new(),
                strings: Vec::new(),
                result: 0,
            },
        };
        for id in 0..ir.nodes.len() as NodeId {
            c.node(id)?;
        }
        if c.limit {
            return Err(Fallback::Limit);
        }
        let result = c.operand(ir.root())?;
        let mut program = c.program;
        program.result = result;
        program.optimize();
        if program.code.len() > MAX_OPS {
            return Err(Fallback::Limit);
        }
        program.allocate()?;
        Ok(program)
    }

    /// Instructions the interpreter runs.
    pub fn len(&self) -> usize {
        self.code.len()
    }

    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }

    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

    /// Calls `f` on every operand word of `ins`, with its table.
    fn operands(ins: &mut Instruction, tables: &mut [u32], mut f: impl FnMut(&mut u32)) {
        let (table, words) = ins.shape();
        for (w, &is) in ins.w.iter_mut().zip(&words) {
            if is {
                f(w);
            }
        }
        if table {
            let start = ins.w[0] as usize;
            let len = table_len(ins);
            tables[start..start + len].iter_mut().for_each(f);
        }
    }

    /// Drops instructions whose results nothing reads (inputs that a
    /// coercion re-reads packed), and the inputs no instruction loads.
    fn optimize(&mut self) {
        let n = self.code.len();
        let mut live = vec![false; n];
        if self.result < CONSTANT {
            live[self.result as usize] = true;
        }
        for i in (0..n).rev() {
            if live[i] {
                let mut ins = self.code[i];
                Self::operands(&mut ins, &mut self.tables, |o| {
                    if *o < CONSTANT {
                        live[*o as usize] = true;
                    }
                });
            }
        }
        let mut renumber = vec![u32::MAX; n];
        let mut code = Vec::new();
        for (i, ins) in self.code.iter().enumerate() {
            if live[i] {
                renumber[i] = code.len() as u32;
                code.push(*ins);
            }
        }
        for ins in &mut code {
            Self::operands(ins, &mut self.tables, |o| {
                if *o < CONSTANT {
                    *o = renumber[*o as usize];
                }
            });
        }
        if self.result < CONSTANT {
            self.result = renumber[self.result as usize];
        }
        self.code = code;
        let mut used = vec![false; self.inputs.len()];
        for ins in &self.code {
            if ins.op == OP_INPUT {
                used[ins.w[0] as usize] = true;
            }
        }
        let mut index = vec![0; self.inputs.len()];
        let mut inputs = Vec::new();
        for (i, input) in std::mem::take(&mut self.inputs).into_iter().enumerate() {
            if used[i] {
                index[i] = inputs.len() as u32;
                inputs.push(input);
            }
        }
        for ins in &mut self.code {
            if ins.op == OP_INPUT {
                ins.w[0] = index[ins.w[0] as usize];
            }
        }
        self.inputs = inputs;
    }

    /// Assigns registers: an instruction's result lives until its last
    /// use. Results are written after operands are read, so a result may
    /// reuse an operand's register.
    fn allocate(&mut self) -> Result<(), Fallback> {
        let n = self.code.len();
        let mut last = vec![0usize; n];
        for i in 0..n {
            let mut ins = self.code[i];
            Self::operands(&mut ins, &mut self.tables, |o| {
                if *o < CONSTANT {
                    last[*o as usize] = i;
                }
            });
        }
        if self.result < CONSTANT {
            last[self.result as usize] = usize::MAX;
        }
        let mut register = vec![0u32; n];
        let mut free: Vec<u32> = (0..REGISTERS as u32).rev().collect();
        for i in 0..n {
            let mut ins = self.code[i];
            let mut read = Vec::new();
            Self::operands(&mut ins, &mut self.tables, |o| {
                if *o < CONSTANT {
                    read.push(*o as usize);
                    *o = register[*o as usize];
                }
            });
            read.sort_unstable();
            read.dedup();
            for j in read {
                if last[j] == i {
                    free.push(register[j]);
                }
            }
            register[i] = free.pop().ok_or(Fallback::Limit)?;
            ins.dst = register[i];
            self.code[i] = ins;
        }
        if self.result < CONSTANT {
            self.result = register[self.result as usize];
        }
        Ok(())
    }
}

fn table_len(ins: &Instruction) -> usize {
    let n = ins.count();
    match ins.op {
        OP_CASE | OP_MATCH | OP_INTERPOLATE => 2 * n,
        OP_STEP => 1 + 2 * n,
        _ => n,
    }
}

struct Compiler<'a> {
    ir: &'a Ir,
    /// Each compiled node's operand: a virtual register (its instruction)
    /// or a constant. Literals get theirs on first use.
    operand: Vec<Option<u32>>,
    /// The node's numbers equal the VM's f64 values.
    exact: Vec<bool>,
    /// The node's value may come from `feature-state`.
    state: Vec<bool>,
    /// A limit was exceeded; compilation goes on to find unsupported
    /// operators, which take precedence.
    limit: bool,
    /// Coerced copies of nodes: node, coercion, operand.
    coercions: Vec<(NodeId, Coerce, u32)>,
    program: Program,
}

/// Whether a value of type `t` may be a number at run time.
fn may_number(t: &Type) -> bool {
    matches!(t, Type::Number | Type::Value)
}

fn may_string(t: &Type) -> bool {
    matches!(t, Type::String | Type::Value)
}

fn f32_bits(n: f64) -> u32 {
    (n as f32).to_bits()
}

/// The f64 bit pattern, with NaN canonical.
fn f64_bits(n: f64) -> [u32; 2] {
    let bits = if n.is_nan() {
        f64::NAN.to_bits()
    } else {
        n.to_bits()
    };
    [bits as u32, (bits >> 32) as u32]
}

fn unorm16(c: f32) -> u32 {
    (c.clamp(0.0, 1.0) * 65535.0).round() as u32
}

/// A colour as two unorm16 pairs of its straight components.
fn color_words(c: &Color) -> [u32; 2] {
    let [r, g, b, a] = c.straight().map(unorm16);
    [r | g << 16, b | a << 16]
}

impl Compiler<'_> {
    fn constant(&mut self, c: Constant) -> u32 {
        let index = match self.program.constants.iter().position(|k| *k == c) {
            Some(i) => i,
            None => {
                self.program.constants.push(c);
                self.program.constants.len() - 1
            }
        };
        CONSTANT | index as u32
    }

    fn string(&mut self, s: &str) -> u32 {
        let index = match self.program.strings.iter().position(|k| &**k == s) {
            Some(i) => i,
            None => {
                self.program.strings.push(s.into());
                self.program.strings.len() - 1
            }
        };
        self.constant(Constant::String(index as u32))
    }

    /// A value's constant, if it has a GPU representation.
    fn value(&mut self, v: &Value<'_>) -> Result<u32, Fallback> {
        Ok(match v {
            Value::String(s) => self.string(s),
            v => self.constant(Constant::Value(pack_value(v).ok_or(Fallback::Unsupported)?)),
        })
    }

    fn operand(&mut self, id: NodeId) -> Result<u32, Fallback> {
        if let Some(o) = self.operand[id as usize] {
            return Ok(o);
        }
        // Nodes compiled only as part of their user (`typeof`) have no
        // operand of their own.
        let Op::Literal(v) = &self.ir.node(id).op else {
            return Err(Fallback::Unsupported);
        };
        let o = self.value(v)?;
        self.operand[id as usize] = Some(o);
        Ok(o)
    }

    fn operands(&mut self, ids: &[NodeId]) -> Result<Vec<u32>, Fallback> {
        ids.iter().map(|&id| self.operand(id)).collect()
    }

    fn input(&mut self, input: Input) -> u32 {
        match self.program.inputs.iter().position(|i| *i == input) {
            Some(i) => i as u32,
            None => {
                self.program.inputs.push(input);
                self.program.inputs.len() as u32 - 1
            }
        }
    }

    fn push(&mut self, op: u32, aux: u32, w: [u32; 3]) -> u32 {
        self.program.code.push(Instruction { op, aux, dst: 0, w });
        self.program.code.len() as u32 - 1
    }

    fn emit(&mut self, id: NodeId, op: u32, aux: u32, w: [u32; 3]) {
        self.operand[id as usize] = Some(self.push(op, aux, w));
    }

    /// An instruction with a table of operands in `w[0]`.
    fn push_table(&mut self, op: u32, aux: u32, count: usize, table: Vec<u32>, w: [u32; 2]) -> u32 {
        self.limit |= count > MAX_STOPS;
        let start = self.program.tables.len() as u32;
        self.program.tables.extend(table);
        self.push(op, aux << 8 | count as u32, [start, w[0], w[1]])
    }

    fn emit_table(
        &mut self,
        id: NodeId,
        op: u32,
        aux: u32,
        count: usize,
        table: Vec<u32>,
        w: [u32; 2],
    ) {
        self.operand[id as usize] = Some(self.push_table(op, aux, count, table, w));
    }

    /// A `case`, `match`, `coalesce` or `step` whose outputs are `out(id)`:
    /// plain operands, or their coerced copies.
    fn selection(
        &mut self,
        id: NodeId,
        out: &mut dyn FnMut(&mut Self, NodeId) -> Result<u32, Fallback>,
    ) -> Result<u32, Fallback> {
        Ok(match &self.ir.node(id).op {
            Op::Case {
                branches,
                otherwise,
            } => {
                let mut table = Vec::new();
                for &(test, output) in branches {
                    table.extend([self.operand(test)?, out(self, output)?]);
                }
                let otherwise = out(self, *otherwise)?;
                self.push_table(OP_CASE, 0, branches.len(), table, [otherwise, 0])
            }
            Op::Match {
                input,
                labels,
                outputs,
                otherwise,
            } => {
                let mut table = Vec::new();
                match labels {
                    MatchLabels::Number(labels) => {
                        for &(label, i) in labels {
                            table.push(self.value(&Value::Number(label))?);
                            table.push(out(self, outputs[i as usize])?);
                        }
                    }
                    MatchLabels::String(labels) => {
                        for (label, i) in labels {
                            table.push(self.string(label));
                            table.push(out(self, outputs[*i as usize])?);
                        }
                    }
                }
                let count = table.len() / 2;
                let w = [self.operand(*input)?, out(self, *otherwise)?];
                self.push_table(OP_MATCH, 0, count, table, w)
            }
            Op::Coalesce(args) => {
                let table = args
                    .iter()
                    .map(|&a| out(self, a))
                    .collect::<Result<Vec<_>, _>>()?;
                self.push_table(OP_COALESCE, 0, args.len(), table, [0; 2])
            }
            Op::Step {
                input,
                labels,
                outputs,
            } => {
                let mut table = vec![out(self, outputs[0])?];
                for (label, &output) in labels.iter().zip(outputs).skip(1) {
                    table.push(self.value(&Value::Number(*label))?);
                    table.push(out(self, output)?);
                }
                let input = self.operand(*input)?;
                self.push_table(OP_STEP, 0, labels.len() - 1, table, [input, 0])
            }
            _ => unreachable!("selections"),
        })
    }

    /// Requires an exact, state-free operand for a numeric branch.
    fn branch(&self, id: NodeId) -> Result<(), Fallback> {
        if self.state[id as usize] && may_number(&self.ir.node(id).ty) {
            Err(Fallback::StateBranch)
        } else if !self.exact[id as usize] {
            Err(Fallback::InexactBranch)
        } else {
            Ok(())
        }
    }

    /// The selection of `values`: exact and state-derived as they are.
    fn select(&mut self, id: NodeId, values: impl IntoIterator<Item = NodeId>) {
        for v in values {
            self.exact[id as usize] &= self.exact[v as usize];
            self.state[id as usize] |= self.state[v as usize];
        }
    }

    fn alias(&mut self, id: NodeId, target: NodeId) -> Result<(), Fallback> {
        self.operand[id as usize] = Some(self.operand(target)?);
        self.exact[id as usize] = self.exact[target as usize];
        self.state[id as usize] = self.state[target as usize];
        Ok(())
    }

    /// `id`'s value converted at pack time, nulls kept: an input packed
    /// converted, a literal converted, or a selection of those. The
    /// `to-number` or `to-color` instruction around it finishes the job.
    fn coerced(&mut self, id: NodeId, coerce: Coerce) -> Result<u32, Fallback> {
        if let Some(&(_, _, o)) = self.coercions.iter().find(|c| c.0 == id && c.1 == coerce) {
            return Ok(o);
        }
        let ir = self.ir;
        let source = match &ir.node(id).op {
            Op::Literal(v) => {
                let packed = match (coerce, v) {
                    (_, Value::Null) => [NULL, 0, 0, 0],
                    (Coerce::Number, v) => match v.to_number() {
                        n if n.is_nan() => [FAIL, 0, 0, 0],
                        n => pack_value(&Value::Number(n)).expect("numbers pack"),
                    },
                    (_, v) => match coerce_color(v) {
                        Some(c) => [COLOR, color_words(&c)[0], color_words(&c)[1], 0],
                        None => [FAIL, 0, 0, 0],
                    },
                };
                return Ok(self.constant(Constant::Value(packed)));
            }
            Op::Get(k) => Some(Source::Property(ir.keys[*k as usize].clone())),
            Op::Id => Some(Source::Id),
            Op::Call(Call::FeatureState, args) => Some(state_key(ir, args)?),
            Op::Call(Call::Let, args) => {
                return self.coerced(*args.last().expect("let result"), coerce);
            }
            Op::Call(Call::Var, args) => return self.coerced(args[0], coerce),
            Op::Case { .. } | Op::Match { .. } | Op::Coalesce(_) | Op::Step { .. } => None,
            _ => return Err(Fallback::Unsupported),
        };
        let o = match source {
            Some(source) => {
                let index = self.input(Input { source, coerce });
                self.push(OP_INPUT, 0, [index, 0, 0])
            }
            None => self.selection(id, &mut |c, out| c.coerced(out, coerce))?,
        };
        self.coercions.push((id, coerce, o));
        Ok(o)
    }

    fn node(&mut self, id: NodeId) -> Result<(), Fallback> {
        let ir = self.ir;
        let node = ir.node(id);
        let ty = |id: NodeId| &ir.node(id).ty;
        match &node.op {
            Op::Literal(_) => {}
            Op::Get(k) | Op::Has(k) => {
                let index = self.input(Input {
                    source: Source::Property(ir.keys[*k as usize].clone()),
                    coerce: Coerce::None,
                });
                let has = u32::from(matches!(node.op, Op::Has(_)));
                self.emit(id, OP_INPUT, has, [index, 0, 0]);
            }
            Op::Id | Op::GeometryType => {
                let source = match node.op {
                    Op::Id => Source::Id,
                    _ => Source::GeometryType,
                };
                let index = self.input(Input {
                    source,
                    coerce: Coerce::None,
                });
                self.emit(id, OP_INPUT, 0, [index, 0, 0]);
            }
            Op::Call(Call::FeatureState, args) => {
                let source = state_key(ir, args)?;
                let index = self.input(Input {
                    source,
                    coerce: Coerce::None,
                });
                self.emit(id, OP_INPUT, 0, [index, 0, 0]);
                self.state[id as usize] = true;
            }
            Op::Call(Call::Let, args) => self.alias(id, *args.last().expect("let result"))?,
            // An error only where it is selected, as the VM evaluates lazily.
            Op::Call(Call::Error, _) => {
                self.operand[id as usize] = Some(self.constant(Constant::Value([ERROR, 0, 0, 0])));
            }
            // Only `["==", ["typeof", x], "number"]` and the like, below.
            Op::TypeOf(_) => {}
            Op::Call(Call::Var, args) => self.alias(id, args[0])?,
            Op::Zoom => self.emit(id, OP_ZOOM, 0, [0; 3]),
            Op::Not(a) => {
                let a = self.operand(*a)?;
                self.emit(id, OP_NOT, 0, [a, 0, 0]);
            }
            Op::Compare {
                op: op @ (CmpOp::Eq | CmpOp::Ne),
                lhs,
                rhs,
                collator: None,
                ..
            } if let Some((x, name)) = type_test(ir, *lhs, *rhs) => {
                // Legacy functions' defaults test `typeof` against a name.
                let tag = match name {
                    "null" => NULL,
                    "number" => NUMBER,
                    "boolean" => BOOL,
                    "string" => STRING,
                    "color" => COLOR,
                    _ => return Err(Fallback::Unsupported),
                };
                let x = self.operand(x)?;
                let ne = u32::from(*op == CmpOp::Ne);
                self.emit(id, OP_TYPE, tag | ne << 8, [x, 0, 0]);
            }
            Op::Compare {
                op,
                lhs,
                rhs,
                untyped,
                collator: None,
            } => {
                let (lt, rt) = (ty(*lhs), ty(*rhs));
                let literal = |id: NodeId| matches!(ir.node(id).op, Op::Literal(_));
                let strings = may_string(lt) && may_string(rt);
                // Unknown strings share one id, and the GPU has no
                // collation: strings compare only for equality with a
                // literal.
                if strings && (op.is_order() || !(literal(*lhs) || literal(*rhs))) {
                    return Err(Fallback::Unsupported);
                }
                if may_number(lt) && may_number(rt) {
                    self.branch(*lhs)?;
                    self.branch(*rhs)?;
                }
                let (a, b) = (self.operand(*lhs)?, self.operand(*rhs)?);
                let cmp = match op {
                    CmpOp::Eq => 0,
                    CmpOp::Ne => 1,
                    CmpOp::Lt => 2,
                    CmpOp::Le => 3,
                    CmpOp::Gt => 4,
                    CmpOp::Ge => 5,
                };
                self.emit(id, OP_COMPARE, cmp | u32::from(*untyped) << 4, [a, b, 0]);
            }
            Op::All(args) | Op::Any(args) => {
                let table = self.operands(args)?;
                let any = u32::from(matches!(node.op, Op::Any(_)));
                self.emit_table(id, OP_ALL_ANY, any, args.len(), table, [0; 2]);
            }
            Op::Case { .. } | Op::Match { .. } | Op::Coalesce(_) | Op::Step { .. } => {
                let values: Vec<NodeId> = match &node.op {
                    Op::Case {
                        branches,
                        otherwise,
                    } => branches.iter().map(|b| b.1).chain([*otherwise]).collect(),
                    Op::Match {
                        input,
                        labels,
                        outputs,
                        otherwise,
                    } => {
                        if matches!(labels, MatchLabels::Number(_)) {
                            self.branch(*input)?;
                        }
                        outputs.iter().copied().chain([*otherwise]).collect()
                    }
                    Op::Step { input, outputs, .. } => {
                        self.branch(*input)?;
                        outputs.clone()
                    }
                    Op::Coalesce(args) => args.clone(),
                    _ => unreachable!(),
                };
                self.select(id, values);
                let o = self.selection(id, &mut |c, out| c.operand(out))?;
                self.operand[id as usize] = Some(o);
            }
            Op::Interpolate {
                color_space: ColorSpace::Rgb,
                interpolation,
                input,
                labels,
                outputs,
            } if matches!(node.ty, Type::Number | Type::Color) => {
                let mut table = Vec::new();
                for (label, &out) in labels.iter().zip(outputs) {
                    table.push(self.value(&Value::Number(*label))?);
                    table.push(self.operand(out)?);
                }
                let (kind, params) = match *interpolation {
                    Interpolation::Linear => (0, [1.0, 0.0, 0.0, 0.0]),
                    Interpolation::Exponential(base) => (0, [base, 0.0, 0.0, 0.0]),
                    Interpolation::CubicBezier(p) => (1, p),
                };
                let params = self.constant(Constant::Value(params.map(f32_bits)));
                let input = self.operand(*input)?;
                self.exact[id as usize] = false;
                self.select(id, outputs.iter().copied());
                self.emit_table(
                    id,
                    OP_INTERPOLATE,
                    kind,
                    labels.len(),
                    table,
                    [input, params],
                );
            }
            Op::Sum(args) | Op::Product(args) | Op::Min(args) | Op::Max(args) => {
                let f = match node.op {
                    Op::Sum(_) => 0,
                    Op::Product(_) => 1,
                    Op::Min(_) => 2,
                    _ => 3,
                };
                let table = self.operands(args)?;
                self.exact[id as usize] = false;
                self.select(id, args.iter().copied());
                self.emit_table(id, OP_NARY, f, args.len(), table, [0; 2]);
            }
            Op::Sub(a, b) | Op::Div(a, b) | Op::Rem(a, b) | Op::Pow(a, b) => {
                let f = match node.op {
                    Op::Sub(..) => 0,
                    Op::Div(..) => 1,
                    Op::Rem(..) => 2,
                    _ => 3,
                };
                let w = [self.operand(*a)?, self.operand(*b)?, 0];
                self.exact[id as usize] = false;
                self.select(id, [*a, *b]);
                self.emit(id, OP_BINARY, f, w);
            }
            Op::Neg(a) | Op::Math(_, a) => {
                let f = match node.op {
                    Op::Neg(_) => 0,
                    Op::Math(f, _) => match f {
                        MathFn::Abs => 1,
                        MathFn::Floor => 2,
                        MathFn::Ceil => 3,
                        MathFn::Round => 4,
                        MathFn::Sqrt => 5,
                        MathFn::Ln => 6,
                        MathFn::Log2 => 7,
                        MathFn::Log10 => 8,
                        MathFn::Sin => 9,
                        MathFn::Cos => 10,
                        MathFn::Tan => 11,
                        MathFn::Asin => 12,
                        MathFn::Acos => 13,
                        MathFn::Atan => 14,
                    },
                    _ => unreachable!(),
                };
                let a_operand = self.operand(*a)?;
                self.exact[id as usize] = false;
                self.select(id, [*a]);
                self.emit(id, OP_UNARY, f, [a_operand, 0, 0]);
            }
            Op::Assert(args) => {
                let tag = match node.ty {
                    Type::Number => NUMBER,
                    Type::Boolean => BOOL,
                    Type::String => STRING,
                    Type::Color => COLOR,
                    _ => return Err(Fallback::Unsupported),
                };
                let table = self.operands(args)?;
                self.select(id, args.iter().copied());
                self.emit_table(id, OP_ASSERT, tag, args.len(), table, [0; 2]);
            }
            Op::Coerce(args) if matches!(node.ty, Type::Number | Type::Color) => {
                let coerce = if node.ty == Type::Number {
                    Coerce::Number
                } else {
                    Coerce::Color
                };
                let table = args
                    .iter()
                    .map(|&a| self.coerced(a, coerce))
                    .collect::<Result<Vec<_>, _>>()?;
                let target = if coerce == Coerce::Number {
                    NUMBER
                } else {
                    COLOR
                };
                self.select(id, args.iter().copied());
                self.emit_table(id, OP_COERCE, target, args.len(), table, [0; 2]);
            }
            Op::In { needle, haystack } => {
                let Op::Literal(Value::Array(items)) = &ir.node(*haystack).op else {
                    return Err(Fallback::Unsupported);
                };
                if may_number(ty(*needle)) {
                    self.branch(*needle)?;
                }
                let mut table = Vec::new();
                for item in items.iter() {
                    // Arrays and objects equal no needle.
                    if matches!(
                        item,
                        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
                    ) {
                        let c = self.value(item)?;
                        if !table.contains(&c) {
                            table.push(c);
                        }
                    }
                }
                let needle = self.operand(*needle)?;
                self.emit_table(id, OP_IN, 0, table.len(), table, [needle, 0]);
            }
            _ => return Err(Fallback::Unsupported),
        }
        Ok(())
    }
}

/// `typeof`'s operand and the type name it is compared with.
fn type_test(ir: &Ir, lhs: NodeId, rhs: NodeId) -> Option<(NodeId, &str)> {
    match (&ir.node(lhs).op, &ir.node(rhs).op) {
        (Op::TypeOf(x), Op::Literal(Value::String(s)))
        | (Op::Literal(Value::String(s)), Op::TypeOf(x)) => Some((*x, s)),
        _ => None,
    }
}

/// The key of `["feature-state", key]` with a literal key.
fn state_key(ir: &Ir, args: &[NodeId]) -> Result<Source, Fallback> {
    match &ir.node(args[0]).op {
        Op::Literal(Value::String(key)) => Ok(Source::State(key.as_ref().into())),
        _ => Err(Fallback::Unsupported),
    }
}

/// `to-color` of a value, as the VM converts it.
fn coerce_color(v: &Value<'_>) -> Option<Color> {
    match v {
        Value::Color(c) => Some(*c),
        Value::String(s) => Color::parse(s),
        Value::Array(items) => crate::eval::rgba_array(items, &mut None),
        _ => None,
    }
}

/// A value in the interpreter's representation, if it has one. Strings
/// need a dictionary and pack through [`Layout`].
fn pack_value(v: &Value<'_>) -> Option<[u32; 4]> {
    Some(match v {
        Value::Null => [NULL, 0, 0, 0],
        Value::Bool(b) => [BOOL, u32::from(*b), 0, 0],
        Value::Number(n) => {
            let [lo, hi] = f64_bits(*n);
            [NUMBER, lo, hi, f32_bits(*n)]
        }
        Value::Color(c) => {
            let [rg, ba] = color_words(c);
            [COLOR, rg, ba, 0]
        }
        _ => return None,
    })
}

/// The inputs and strings a render node's programs share, and how they
/// pack: input values per feature beside the property row, state values
/// in the feature's state block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    /// Per-feature inputs, in packing order.
    columns: Vec<Input>,
    /// Feature-state inputs, in packing order.
    state: Vec<Input>,
    /// Sorted; a string's id is its index.
    dictionary: Vec<Box<str>>,
}

impl Layout {
    pub fn new<'a>(programs: impl IntoIterator<Item = &'a Program>) -> Self {
        let mut layout = Self::default();
        for program in programs {
            for input in &program.inputs {
                let list = if input.is_state() {
                    &mut layout.state
                } else {
                    &mut layout.columns
                };
                if !list.contains(input) {
                    list.push(input.clone());
                }
            }
            layout.dictionary.extend(program.strings.iter().cloned());
        }
        layout.columns.sort();
        layout.state.sort();
        layout.dictionary.sort();
        layout.dictionary.dedup();
        layout
    }

    /// Packed words per feature beside its property row.
    pub fn column_words(&self) -> u32 {
        self.columns.iter().map(Input::words).sum()
    }

    /// Packed words in a feature's state block.
    pub fn state_words(&self) -> u32 {
        self.state.iter().map(Input::words).sum()
    }

    pub fn columns(&self) -> &[Input] {
        &self.columns
    }

    fn string_id(&self, s: &str) -> u32 {
        self.dictionary
            .binary_search_by(|k| (**k).cmp(s))
            .map_or(UNKNOWN_STRING, |i| i as u32)
    }

    fn offset(list: &[Input], input: &Input) -> u32 {
        list.iter()
            .take_while(|i| *i != input)
            .map(Input::words)
            .sum()
    }

    /// The words `expr.wgsl` interprets for `program`, one of the
    /// layout's, falling back to `default` (a number or colour) where the
    /// expression evaluates to an error, null, NaN or the wrong type.
    pub fn encode(&self, program: &Program, default: &Value<'_>) -> Vec<u32> {
        let n = program.code.len() as u32;
        let tables = HEADER + 4 * n;
        let constants = tables + program.tables.len() as u32;
        let inputs = constants + 4 * program.constants.len() as u32;
        let mut words = vec![n, program.result, constants, inputs];
        words.extend(pack_value(default).unwrap_or([NULL, 0, 0, 0]));
        for ins in &program.code {
            let (table, _) = ins.shape();
            let mut w = ins.w;
            if table {
                w[0] += tables;
            }
            words.extend([ins.op | ins.dst << 8 | ins.aux << 16, w[0], w[1], w[2]]);
        }
        words.extend(&program.tables);
        for c in &program.constants {
            words.extend(match c {
                Constant::Value(v) => *v,
                Constant::String(i) => {
                    [STRING, self.string_id(&program.strings[*i as usize]), 0, 0]
                }
            });
        }
        for input in &program.inputs {
            let color = if input.coerce == Coerce::Color {
                COLOR_INPUT
            } else {
                0
            };
            words.push(if input.is_state() {
                STATE_INPUT | color | Self::offset(&self.state, input)
            } else {
                color | Self::offset(&self.columns, input)
            });
        }
        words
    }

    /// Packs one value of `input`; `None` is an absent property or state.
    fn pack(&self, input: &Input, value: Option<Value<'_>>, out: &mut [u32]) {
        match input.coerce {
            Coerce::Color => {
                let packed = match value {
                    None | Some(Value::Null) => [NULL, 0, 0],
                    Some(v) => match coerce_color(&v) {
                        Some(c) => {
                            let [rg, ba] = color_words(&c);
                            [COLOR, rg, ba]
                        }
                        None => [FAIL, 0, 0],
                    },
                };
                out[..3].copy_from_slice(&packed);
            }
            Coerce::Number => {
                let packed = match value {
                    None | Some(Value::Null) => [0, BOXED | NULL],
                    Some(v) => match v.to_number() {
                        n if n.is_nan() => [0, BOXED | FAIL],
                        n => f64_bits(n),
                    },
                };
                out[..2].copy_from_slice(&packed);
            }
            Coerce::None => {
                let packed = match value {
                    None => [0, BOXED | ABSENT],
                    Some(Value::Number(n)) => f64_bits(n),
                    Some(Value::Null) => [0, BOXED | NULL],
                    Some(Value::Bool(b)) => [u32::from(b), BOXED | BOOL],
                    Some(Value::String(s)) => [self.string_id(&s), BOXED | STRING],
                    Some(_) => [0, BOXED | OTHER],
                };
                out[..2].copy_from_slice(&packed);
            }
        }
    }

    /// Packs one feature's inputs into `out` (`column_words` long), its
    /// values given by `value` (`None`: an absent property).
    pub fn pack_feature<'v>(
        &self,
        value: &mut dyn FnMut(&Source) -> Option<Value<'v>>,
        out: &mut [u32],
    ) {
        let mut at = 0;
        for input in &self.columns {
            self.pack(input, value(&input.source), &mut out[at..]);
            at += input.words() as usize;
        }
    }

    /// Packs each feature's inputs: feature `features[i]`'s at
    /// `out[i * stride + offset..]`.
    pub fn pack_columns(
        &self,
        layer: &orb_tile::Layer,
        features: &[u32],
        stride: usize,
        offset: usize,
        out: &mut [u32],
    ) {
        let columns: Vec<_> = self
            .columns
            .iter()
            .map(|input| match &input.source {
                Source::Property(key) => layer.property(key),
                _ => None,
            })
            .collect();
        for (i, &f) in features.iter().enumerate() {
            let row = f as usize;
            let mut column = columns.iter();
            let mut value = |source: &Source| {
                let property = column.next().expect("one column per input");
                match source {
                    Source::Property(_) => {
                        property.and_then(|c| c.get(row)).map(crate::tile::value)
                    }
                    Source::Id => {
                        Some(layer.ids().get(row).map_or(Value::Null, crate::tile::value))
                    }
                    Source::GeometryType => Some(Value::String(
                        crate::tile::geometry_type(layer.geometry_type(row)).into(),
                    )),
                    Source::State(_) => unreachable!("state inputs pack per state"),
                }
            };
            let start = i * stride + offset;
            self.pack_feature(&mut value, &mut out[start..]);
        }
    }

    /// Packs a feature's state values into `out` (`state_words` long).
    pub fn pack_state(&self, state: &[(String, Value<'static>)], out: &mut [u32]) {
        let mut at = 0;
        for input in &self.state {
            let Source::State(key) = &input.source else {
                unreachable!("state inputs")
            };
            let value = state
                .iter()
                .rev()
                .find(|(k, _)| **k == **key)
                .map(|(_, v)| v.reborrow());
            self.pack(input, value, &mut out[at..]);
            at += input.words() as usize;
        }
    }

    /// Per column, the smallest and largest number among `features`'
    /// values (`[inf, -inf]` without numbers), for [`Expression::bound`].
    pub fn ranges(&self, layer: &orb_tile::Layer, features: &[u32]) -> Vec<[f64; 2]> {
        let stride = self.column_words() as usize;
        let mut words = vec![0; stride * features.len()];
        self.pack_columns(layer, features, stride, 0, &mut words);
        let mut at = 0;
        self.columns
            .iter()
            .map(|input| {
                let mut range = [f64::INFINITY, f64::NEG_INFINITY];
                if input.coerce != Coerce::Color {
                    for row in words.chunks_exact(stride.max(1)) {
                        let (lo, hi) = (row[at], row[at + 1]);
                        if hi & BOXED != BOXED {
                            let n = f64::from_bits(u64::from(hi) << 32 | u64::from(lo));
                            if !n.is_nan() {
                                range = [range[0].min(n), range[1].max(n)];
                            }
                        }
                    }
                }
                at += input.words() as usize;
                range
            })
            .collect()
    }
}

/// An interval of numbers; NaN-free, possibly unbounded.
type Interval = [f64; 2];
const ANY: Interval = [f64::NEG_INFINITY, f64::INFINITY];

fn hull(a: Interval, b: Interval) -> Interval {
    [a[0].min(b[0]), a[1].max(b[1])]
}

fn sane(i: Interval) -> Interval {
    if i[0].is_nan() || i[1].is_nan() {
        ANY
    } else {
        i
    }
}

impl Expression {
    /// An upper bound of the result's magnitude at `zoom` over features
    /// whose numeric inputs lie in `range(input)` (`None`: unknown), for
    /// culling margins of GPU-evaluated widths. Feature state is unknown.
    pub fn bound(&self, zoom: f64, range: &dyn Fn(&Input) -> Option<Interval>) -> f64 {
        let ir = &self.ir;
        let mut out: Vec<Interval> = Vec::with_capacity(ir.nodes.len());
        let empty = [f64::INFINITY, f64::NEG_INFINITY];
        for node in &ir.nodes {
            let at = |id: NodeId| out[id as usize];
            let property = |k: u32, coerce| {
                range(&Input {
                    source: Source::Property(ir.keys[k as usize].clone()),
                    coerce,
                })
                .unwrap_or(ANY)
            };
            let all =
                |ids: &mut dyn Iterator<Item = NodeId>| ids.fold(empty, |a, id| hull(a, at(id)));
            let coerced = |id: NodeId| match &ir.node(id).op {
                Op::Get(k) => property(*k, Coerce::Number),
                Op::Literal(v) => match match v {
                    Value::Null => 0.0,
                    v => v.to_number(),
                } {
                    n if n.is_nan() => empty,
                    n => [n, n],
                },
                _ => ANY,
            };
            let value = match &node.op {
                Op::Literal(v) => match v.as_f64() {
                    Some(n) if !n.is_nan() => [n, n],
                    _ => empty,
                },
                Op::Zoom => [zoom, zoom],
                Op::Get(k) => property(*k, Coerce::None),
                Op::Call(Call::Let, args) => at(*args.last().expect("let result")),
                Op::Call(Call::Var, args) => at(args[0]),
                Op::Case {
                    branches,
                    otherwise,
                } => all(&mut branches.iter().map(|b| b.1).chain([*otherwise])),
                Op::Match {
                    outputs, otherwise, ..
                } => all(&mut outputs.iter().copied().chain([*otherwise])),
                Op::Coalesce(args) | Op::Assert(args) => all(&mut args.iter().copied()),
                Op::Coerce(args) if node.ty == Type::Number => {
                    args.iter().fold(empty, |a, &id| hull(a, coerced(id)))
                }
                Op::Step {
                    input: x,
                    labels,
                    outputs,
                }
                | Op::Interpolate {
                    input: x,
                    labels,
                    outputs,
                    ..
                } => {
                    let [lo, hi] = at(*x);
                    // Only the stops the input can reach.
                    let first = labels.iter().rposition(|&l| l <= lo).unwrap_or(0);
                    let last = labels.iter().rposition(|&l| l <= hi).unwrap_or(0);
                    // Between stops, an interpolation reaches the next.
                    let last = if matches!(node.op, Op::Interpolate { .. }) && hi > labels[last] {
                        (last + 1).min(labels.len() - 1)
                    } else {
                        last
                    };
                    all(&mut outputs[first..=last.max(first)].iter().copied())
                }
                Op::Sum(args) => args.iter().fold([0.0, 0.0], |a, &id| {
                    let b = at(id);
                    [a[0] + b[0], a[1] + b[1]]
                }),
                Op::Product(args) => args.iter().fold([1.0, 1.0], |a, &id| product(a, at(id))),
                Op::Min(args) => args.iter().fold(empty, |a, &id| {
                    let b = at(id);
                    if a[0] > a[1] {
                        b
                    } else {
                        [a[0].min(b[0]), a[1].min(b[1])]
                    }
                }),
                Op::Max(args) => args.iter().fold(empty, |a, &id| {
                    let b = at(id);
                    if a[0] > a[1] {
                        b
                    } else {
                        [a[0].max(b[0]), a[1].max(b[1])]
                    }
                }),
                Op::Sub(a, b) => {
                    let (a, b) = (at(*a), at(*b));
                    [a[0] - b[1], a[1] - b[0]]
                }
                Op::Neg(a) => {
                    let a = at(*a);
                    [-a[1], -a[0]]
                }
                Op::Math(f, a) => {
                    let [lo, hi] = at(*a);
                    match f {
                        MathFn::Abs if lo >= 0.0 => [lo, hi],
                        MathFn::Abs => [0.0, (-lo).max(hi)],
                        MathFn::Floor => [lo.floor(), hi.floor()],
                        MathFn::Ceil => [lo.ceil(), hi.ceil()],
                        MathFn::Round => [lo.round(), hi.round()],
                        MathFn::Sqrt => [lo.max(0.0).sqrt(), hi.sqrt()],
                        MathFn::Sin | MathFn::Cos => [-1.0, 1.0],
                        _ => ANY,
                    }
                }
                _ => ANY,
            };
            out.push(sane(value));
        }
        let [lo, hi] = out[ir.root() as usize];
        if lo > hi { 0.0 } else { lo.abs().max(hi.abs()) }
    }
}

fn product(a: Interval, b: Interval) -> Interval {
    let p = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]];
    if p.iter().any(|v| v.is_nan()) {
        return ANY;
    }
    [
        p.iter().copied().fold(f64::INFINITY, f64::min),
        p.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PropertyExpression, PropertySpec, PropertyType};

    fn parse(source: &str) -> PropertyExpression {
        let spec = PropertySpec {
            ty: PropertyType::Number,
            default: None,
            zoom: true,
            feature: true,
            interpolated: true,
            tokens: false,
        };
        PropertyExpression::parse(&crate::json::parse(source).unwrap(), &spec).unwrap()
    }

    fn compile(source: &str) -> Result<Program, Fallback> {
        Program::compile(parse(source).expression())
    }

    #[test]
    fn branches_need_exact_state_free_operands() {
        assert!(compile(r#"["step", ["get", "n"], 0, 0.1, 1]"#).is_ok());
        assert!(compile(r#"["case", ["<", ["get", "n"], 16777217], 1, 2]"#).is_ok());
        assert_eq!(
            compile(r#"["case", ["<", ["+", ["get", "n"], 1], 2], 1, 2]"#).unwrap_err(),
            Fallback::InexactBranch
        );
        assert_eq!(
            compile(r#"["match", ["*", ["get", "n"], 2], 4, 1, 2]"#).unwrap_err(),
            Fallback::InexactBranch
        );
        assert_eq!(
            compile(r#"["step", ["number", ["feature-state", "n"], 0], 0, 1, 2]"#).unwrap_err(),
            Fallback::StateBranch
        );
        // Boolean state and interpolation inputs are no numeric branches.
        assert!(
            compile(r#"["case", ["boolean", ["feature-state", "hover"], false], ["get", "w"], 1]"#)
                .is_ok()
        );
        assert!(
            compile(
                r#"["interpolate", ["linear"], ["number", ["feature-state", "t"], 0], 0, 1, 1, 2]"#
            )
            .is_ok()
        );
        // Unknown strings share an id: two inputs never compare as strings.
        assert_eq!(
            compile(r#"["case", ["==", ["get", "a"], ["get", "b"]], 1, 2]"#).unwrap_err(),
            Fallback::Unsupported
        );
    }

    #[test]
    fn limits_and_registers() {
        let labels: Vec<String> = (0..17).map(|i| format!("\"l{i}\", {i}")).collect();
        let wide = format!(r#"["match", ["get", "c"], {}, 0]"#, labels.join(", "));
        assert_eq!(compile(&wide).unwrap_err(), Fallback::Limit);
        let mut deep = r#"["get", "n"]"#.to_string();
        for _ in 0..40 {
            deep = format!(r#"["+", {deep}, 1]"#);
        }
        assert_eq!(compile(&deep).unwrap_err(), Fallback::Limit);
        // Inputs a coercion re-reads are dropped; literals take no
        // instructions.
        let p = compile(r#"["to-number", ["get", "n"], 3]"#).unwrap();
        assert_eq!((p.len(), p.inputs().len()), (2, 1));
        let p = compile(r#"["match", ["get", "c"], "a", 1, "b", 2, 3]"#).unwrap();
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn layouts_share_inputs_and_strings() {
        let a = compile(r#"["match", ["get", "class"], "a", 1, 2]"#).unwrap();
        let b = compile(r#"["match", ["get", "class"], "b", 3, ["get", "w"]]"#).unwrap();
        let layout = Layout::new([&a, &b]);
        assert_eq!(layout.columns().len(), 2);
        assert_eq!(layout.column_words(), 4);
        assert_eq!(layout.dictionary.len(), 2);
    }

    #[test]
    fn bounds_cover_the_zoom_and_inputs() {
        let e =
            parse(r#"["interpolate", ["linear"], ["zoom"], 10, ["*", ["get", "w"], 2], 20, 40]"#);
        let range = |_: &Input| Some([1.0, 3.0]);
        assert_eq!(e.expression().bound(10.0, &range), 6.0);
        assert_eq!(e.expression().bound(20.0, &range), 40.0);
        let e = parse(r#"["step", ["zoom"], 1, 14, ["coalesce", ["get", "w"], 2]]"#);
        assert_eq!(e.expression().bound(12.0, &|_| None), 1.0);
        assert_eq!(e.expression().bound(15.0, &|_| None), f64::INFINITY);
        assert_eq!(e.expression().bound(15.0, &|_| Some([-5.0, 3.0])), 5.0);
    }
}
