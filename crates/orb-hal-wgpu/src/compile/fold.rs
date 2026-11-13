//! A stage's naga module folded to the values its pipeline gives its
//! boolean override constants, as `orb_shaders::specialise` folds the
//! browser's WGSL: each use of one is its literal, every `if` the literals
//! decide is taken or dropped, with what follows a taken `return` in its
//! block, and what only the dropped code used goes. Apple's shader
//! compiler takes as long over code no branch reaches as over code that
//! runs, and about 30 ms over a stage with any texture in it, which
//! override constants alone leave to it. The build's modules are folded
//! here, with no WGSL parsed.

use rustc_hash::FxHashMap as HashMap;

use naga::{
    Arena, BinaryOperator, Block, Expression, Handle, Literal, LocalVariable, Module, Override,
    Scalar, Statement, TypeInner, UnaryOperator,
};

/// Folds `constants`' boolean overrides into `module`: false if none of
/// them is one.
pub(super) fn fold(module: &mut Module, constants: &[(String, f64)]) -> bool {
    let values: Vec<(Handle<Override>, bool)> = (module.overrides.iter())
        .filter(|(_, o)| module.types[o.ty].inner == TypeInner::Scalar(Scalar::BOOL))
        .filter_map(|(handle, o)| {
            let name = o.name.as_deref()?;
            let (_, value) = constants.iter().find(|(n, _)| n == name)?;
            Some((handle, *value != 0.0))
        })
        .collect();
    if values.is_empty() {
        return false;
    }
    let functions = (module.functions.iter_mut()).map(|(_, f)| f);
    for function in functions.chain(module.entry_points.iter_mut().map(|e| &mut e.function)) {
        for (_, expression) in function.expressions.iter_mut() {
            if let Expression::Override(o) = *expression
                && let Some(&(_, value)) = values.iter().find(|(h, _)| *h == o)
            {
                *expression = Expression::Literal(Literal::Bool(value));
            }
        }
        let body = std::mem::take(&mut function.body);
        let mut folder = Folder {
            expressions: &function.expressions,
            known: HashMap::default(),
            loaded: HashMap::default(),
        };
        function.body = folder.block(body).0;
        // The `let`s' names, which keep what they name (some of it the
        // dropped code's) through compaction.
        function.named_expressions.clear();
    }
    // What only the dropped code used: an expression its `Emit` went with
    // is invalid left over, and a global only it read (a texture) would
    // stay in the stage.
    naga::compact::compact(module, naga::compact::KeepUnused::No);
    true
}

/// Folds a function's body: its expressions, the values the folded code
/// has stored in its locals so far, and what each load of a local read
/// where it was emitted (a later store doesn't change it). WGSL's `&&` and
/// `||` come to naga as an `if` that stores either side into a local the
/// condition then loads, so `with_patterns && x` folds through that local.
struct Folder<'a> {
    expressions: &'a Arena<Expression>,
    known: HashMap<Handle<LocalVariable>, bool>,
    loaded: HashMap<Handle<Expression>, bool>,
}

impl Folder<'_> {
    /// `block` with the `if`s its literals decide folded, and whether it
    /// returns (what follows it in its own block is then never reached).
    fn block(&mut self, block: Block) -> (Block, bool) {
        let mut out = Block::with_capacity(block.len());
        for (statement, span) in block.span_into_iter() {
            let returns = match statement {
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => match self.decided(condition) {
                    Some(taken) => {
                        let (kept, returns) = self.block(if taken { accept } else { reject });
                        if !kept.is_empty() {
                            out.push(Statement::Block(kept), span);
                        }
                        returns
                    }
                    None => {
                        let accept = self.branch(accept);
                        let reject = self.branch(reject);
                        out.push(
                            Statement::If {
                                condition,
                                accept,
                                reject,
                            },
                            span,
                        );
                        false
                    }
                },
                Statement::Block(inner) => {
                    let (inner, returns) = self.block(inner);
                    out.push(Statement::Block(inner), span);
                    returns
                }
                Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => {
                    // A later turn sees what an earlier one stored.
                    for block in [&body, &continuing] {
                        self.forget_stored(block);
                    }
                    let body = self.branch(body);
                    let continuing = self.branch(continuing);
                    out.push(
                        Statement::Loop {
                            body,
                            continuing,
                            break_if,
                        },
                        span,
                    );
                    false
                }
                Statement::Switch {
                    selector,
                    mut cases,
                } => {
                    for case in &mut cases {
                        case.body = self.branch(std::mem::take(&mut case.body));
                    }
                    out.push(Statement::Switch { selector, cases }, span);
                    false
                }
                Statement::Store { pointer, value } => {
                    if let Expression::LocalVariable(local) = self.expressions[pointer] {
                        match self.decided(value) {
                            Some(b) => self.known.insert(local, b),
                            None => self.known.remove(&local),
                        };
                    }
                    out.push(Statement::Store { pointer, value }, span);
                    false
                }
                Statement::Emit(range) => {
                    for at in range.clone() {
                        if let Expression::Load { pointer } = self.expressions[at]
                            && let Expression::LocalVariable(local) = self.expressions[pointer]
                            && let Some(&value) = self.known.get(&local)
                        {
                            self.loaded.insert(at, value);
                        }
                    }
                    out.push(Statement::Emit(range), span);
                    false
                }
                returning @ Statement::Return { .. } => {
                    out.push(returning, span);
                    true
                }
                other => {
                    // A local passed by pointer may be written there.
                    if let Statement::Call { arguments, .. } = &other {
                        for &argument in arguments {
                            if let Expression::LocalVariable(local) = self.expressions[argument] {
                                self.known.remove(&local);
                            }
                        }
                    }
                    out.push(other, span);
                    false
                }
            };
            if returns {
                return (out, true);
            }
        }
        (out, false)
    }

    /// Forgets the values of the locals `block` stores into, at any depth
    /// (or passes to a call).
    fn forget_stored(&mut self, block: &Block) {
        for statement in block.iter() {
            match statement {
                Statement::Store { pointer, .. } => {
                    if let Expression::LocalVariable(local) = self.expressions[*pointer] {
                        self.known.remove(&local);
                    }
                }
                Statement::Call { arguments, .. } => {
                    for &argument in arguments {
                        if let Expression::LocalVariable(local) = self.expressions[argument] {
                            self.known.remove(&local);
                        }
                    }
                }
                Statement::Block(inner) => self.forget_stored(inner),
                Statement::If { accept, reject, .. } => {
                    self.forget_stored(accept);
                    self.forget_stored(reject);
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    self.forget_stored(body);
                    self.forget_stored(continuing);
                }
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        self.forget_stored(&case.body);
                    }
                }
                _ => {}
            }
        }
    }

    /// A block that may or may not run: folded, and what it stores is
    /// known after it only where it stored what was known before.
    fn branch(&mut self, block: Block) -> Block {
        let before = self.known.clone();
        let (block, _) = self.block(block);
        let after = std::mem::replace(&mut self.known, before);
        self.known
            .retain(|local, value| after.get(local) == Some(value));
        block
    }

    /// What condition `at` comes to, if its literals decide it.
    fn decided(&self, at: Handle<Expression>) -> Option<bool> {
        match self.expressions[at] {
            Expression::Literal(Literal::Bool(b)) => Some(b),
            Expression::Load { .. } => self.loaded.get(&at).copied(),
            Expression::Unary {
                op: UnaryOperator::LogicalNot,
                expr,
            } => self.decided(expr).map(|b| !b),
            Expression::Binary { op, left, right } => {
                let short = match op {
                    BinaryOperator::LogicalAnd => false,
                    BinaryOperator::LogicalOr => true,
                    _ => return None,
                };
                match (self.decided(left), self.decided(right)) {
                    (Some(b), _) | (_, Some(b)) if b == short => Some(short),
                    (Some(_), Some(_)) => Some(!short),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "
override plane: bool = false;
override rows: bool = true;
@group(0) @binding(0) var dem: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> table: array<f32>;

fn height(at: vec2<u32>) -> f32 {
    if plane { return 0.0; }
    let texel = textureLoad(dem, at, 0);
    return texel.r;
}

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var z = height(vec2<u32>(i, 0u));
    if rows && !plane {
        z += table[i];
    }
    return vec4<f32>(0.0, 0.0, z, 1.0);
}
";

    fn validated(module: &Module) {
        naga::valid::Validator::new(Default::default(), naga::valid::Capabilities::all())
            .validate(module)
            .unwrap_or_else(|e| panic!("{e:?}"));
    }

    /// The plane variant leaves the DEM and the table out of its module,
    /// and still validates; the full one keeps both.
    #[test]
    fn a_variant_leaves_what_its_constants_decide_out() {
        let parsed = naga::front::wgsl::parse_str(SOURCE).unwrap();
        let mut plane = parsed.clone();
        let constants = [("plane".to_owned(), 1.0), ("rows".to_owned(), 1.0)];
        assert!(fold(&mut plane, &constants));
        validated(&plane);
        assert_eq!(
            plane.global_variables.len(),
            0,
            "{:?}",
            plane.global_variables
        );
        let mut full = parsed.clone();
        let constants = [("plane".to_owned(), 0.0), ("rows".to_owned(), 1.0)];
        assert!(fold(&mut full, &constants));
        validated(&full);
        assert_eq!(full.global_variables.len(), 2);
        let mut none = parsed;
        assert!(!fold(&mut none, &[("other".to_owned(), 1.0)]));
    }

    /// A local a loop stores into isn't known inside the loop from what
    /// it held before: a later turn sees the earlier turn's.
    #[test]
    fn a_loop_forgets_what_it_stores() {
        let source = "
override plane: bool = false;
@group(0) @binding(0) var<storage, read> table: array<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var z = 0.0;
    var first = !plane && i > 0u;
    for (var k = 0u; k < 2u; k++) {
        if first { z += table[k]; }
        first = true;
    }
    return vec4<f32>(0.0, 0.0, z, 1.0);
}
";
        let mut plane = naga::front::wgsl::parse_str(source).unwrap();
        assert!(fold(&mut plane, &[("plane".to_owned(), 1.0)]));
        validated(&plane);
        assert_eq!(
            plane.global_variables.len(),
            1,
            "the second turn reads the table"
        );
    }
    /// A condition reads a local as it was loaded, not as later stores
    /// left it.
    #[test]
    fn a_load_keeps_the_value_it_read() {
        let source = "
override plane: bool = false;
@group(0) @binding(0) var<storage, read> table: array<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var z = 0.0;
    var on = i > 0u;
    on = plane;
    let was = on;
    on = !plane;
    if was { z += table[i]; }
    if on { z += 1.0; }
    return vec4<f32>(0.0, 0.0, z, 1.0);
}
";
        let mut plane = naga::front::wgsl::parse_str(source).unwrap();
        assert!(fold(&mut plane, &[("plane".to_owned(), 1.0)]));
        validated(&plane);
        assert_eq!(plane.global_variables.len(), 1, "the table is read");
    }
}
