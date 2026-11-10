//! The compiled form of an expression.
//!
//! Nodes live in one arena in post order: every operand is defined before
//! the node that uses it, and the root is the last node. That is the shape of
//! an SSA DAG, so a register VM or a column evaluator can be
//! lowered from it by a linear walk. Each node carries its static type
//! and the inputs it depends on.

use std::ops::BitOr;

use crate::value::{Type, Value};

pub(crate) type NodeId = u32;

/// Index into an expression's table of literal property keys.
pub(crate) type KeyId = u32;

/// The evaluation inputs a node depends on, its own and its operands'.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Deps(u16);

impl Deps {
    pub(crate) const NONE: Self = Self(0);
    pub(crate) const ZOOM: Self = Self(1);
    pub(crate) const FEATURE: Self = Self(2);
    pub(crate) const LINE_PROGRESS: Self = Self(4);

    pub(crate) const FEATURE_STATE: Self = Self(8 | 2);
    /// The feature's own data: properties, id and geometry, as opposed to
    /// its state. Implies [`FEATURE`](Self::FEATURE).
    pub(crate) const PROPERTIES: Self = Self(2048 | 2);
    pub(crate) const GLOBAL_STATE: Self = Self(16);
    pub(crate) const HEATMAP: Self = Self(32);
    pub(crate) const ELEVATION: Self = Self(64);
    pub(crate) const ACCUMULATED: Self = Self(128);
    pub(crate) const LOCALE: Self = Self(256);
    pub(crate) const IMAGE: Self = Self(512);
    pub(crate) const SCRIPT: Self = Self(1024);
    /// The feature's geometry (`within`, `distance`): evaluation needs it
    /// in the context. Implies [`PROPERTIES`](Self::PROPERTIES).
    pub(crate) const GEOMETRY: Self = Self(4096 | 2048 | 2);
    /// The camera beyond its zoom: `pitch` and `bearing` (Orbis's own).
    pub(crate) const CAMERA: Self = Self(8192);

    pub(crate) fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0 && other.0 != 0
    }

    /// Whether every input this depends on is among `other`'s.
    pub(crate) fn within(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }

    pub(crate) fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for Deps {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub op: Op,
    pub ty: Type,
    pub deps: Deps,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
        }
    }

    pub(crate) fn is_order(self) -> bool {
        !matches!(self, Self::Eq | Self::Ne)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MathFn {
    Abs,
    Floor,
    Ceil,
    Round,
    Sqrt,
    Ln,
    Log2,
    Log10,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
}

/// An `interpolate` curve's interpolation type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Interpolation {
    Linear,
    Exponential(f64),
    /// Control points `x1, y1, x2, y2`, each in `0..=1`.
    CubicBezier([f64; 4]),
}

/// `match` labels, sorted for binary search, each with its output index.
#[derive(Clone, Debug)]
pub(crate) enum MatchLabels {
    Number(Vec<(f64, u32)>),
    String(Vec<(Box<str>, u32)>),
}

/// What a legacy filter tests: a property, `$id` or `$type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Subject {
    Property(KeyId),
    Id,
    Type,
}

/// Legacy filters, lowered to GL JS's internal `filter-*` operators, which
/// keep the legacy semantics: a missing property or a type mismatch is
/// `false`, never an error.
#[derive(Clone, Debug)]
pub(crate) enum LegacyFilter {
    /// `filter-==`, `filter-<`, …; `!=` is lowered to `!` around `==`.
    Compare {
        subject: Subject,
        op: CmpOp,
        value: Value<'static>,
    },
    /// `filter-in-small` / `filter-in-large`.
    In {
        subject: Subject,
        values: Vec<Value<'static>>,
    },
    /// `filter-has` / `filter-has-id`.
    Has(Subject),
}

#[derive(Clone, Debug)]
pub(crate) enum Call {
    Let,
    Var,
    Error,
    FeatureState,
    GlobalState(KeyId),
    Heatmap,
    Elevation,
    Accumulated,
    Split,
    Join,
    IndexOf,
    Slice,
    Array,
    SupportedScript,
    Collator,
    ResolvedLocale,
    NumberFormat,
    Format,
    Image,
    Geometry {
        within: bool,
        geometries: Vec<crate::geometry::Geometry>,
    },
    /// The camera's pitch, degrees.
    Pitch,
    /// The camera's bearing, degrees.
    Bearing,
    /// Ground metres per logical pixel at the camera's target, unpitched
    /// (style-spec#1881).
    MetersPerPixel,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ColorSpace {
    #[default]
    Rgb,
    Lab,
    Hcl,
}

#[derive(Clone, Debug)]
pub(crate) enum Op {
    Call(Call, Vec<NodeId>),
    Literal(Value<'static>),
    Zoom,
    LineProgress,
    Id,
    GeometryType,
    Properties,
    /// A feature property under a literal key.
    Get(KeyId),
    /// A feature property under a computed key.
    GetDynamic(NodeId),
    GetFrom {
        key: NodeId,
        object: NodeId,
    },
    Has(KeyId),
    HasDynamic(NodeId),
    HasIn {
        key: NodeId,
        object: NodeId,
    },
    Match {
        input: NodeId,
        labels: MatchLabels,
        outputs: Vec<NodeId>,
        otherwise: NodeId,
    },
    Case {
        branches: Vec<(NodeId, NodeId)>,
        otherwise: NodeId,
    },
    Coalesce(Vec<NodeId>),
    Step {
        input: NodeId,
        labels: Vec<f64>,
        outputs: Vec<NodeId>,
    },
    Interpolate {
        color_space: ColorSpace,
        interpolation: Interpolation,
        input: NodeId,
        labels: Vec<f64>,
        outputs: Vec<NodeId>,
    },
    Compare {
        op: CmpOp,
        lhs: NodeId,
        rhs: NodeId,
        /// Either side is typed `value`, so ordering checks types at run time.
        untyped: bool,
        collator: Option<NodeId>,
    },
    All(Vec<NodeId>),
    Any(Vec<NodeId>),
    Not(NodeId),
    In {
        needle: NodeId,
        haystack: NodeId,
    },
    Sum(Vec<NodeId>),
    Product(Vec<NodeId>),
    Sub(NodeId, NodeId),
    Neg(NodeId),
    Div(NodeId, NodeId),
    Rem(NodeId, NodeId),
    Pow(NodeId, NodeId),
    Min(Vec<NodeId>),
    Max(Vec<NodeId>),
    Math(MathFn, NodeId),
    /// A coercion to the node's type (`to-number`, `to-color`, …, and the
    /// implicit ones): the first argument that converts wins.
    Coerce(Vec<NodeId>),
    /// An assertion of the node's type (`number`, `string`, …, and the
    /// implicit ones): the first argument of that type wins.
    Assert(Vec<NodeId>),
    Rgba {
        r: NodeId,
        g: NodeId,
        b: NodeId,
        a: Option<NodeId>,
    },
    ToRgba(NodeId),
    Concat(Vec<NodeId>),
    Upcase(NodeId),
    Downcase(NodeId),
    Length(NodeId),
    At {
        index: NodeId,
        array: NodeId,
    },
    TypeOf(NodeId),
    Legacy(LegacyFilter),
}

/// Runs `$body` with `$id` bound to each operand of `$op`, in evaluation
/// order: the one walk behind [`Op::for_each_child`] (`&Op`, so `$id` is a
/// `&NodeId`) and [`Op::map_children`] (`&mut Op`, a `&mut NodeId`).
macro_rules! each_child {
    ($op:expr, |$id:ident| $body:expr) => {
        match $op {
            Op::Literal(_)
            | Op::Zoom
            | Op::LineProgress
            | Op::Id
            | Op::GeometryType
            | Op::Properties
            | Op::Get(_)
            | Op::Has(_)
            | Op::Legacy(_) => {}
            Op::GetDynamic($id)
            | Op::HasDynamic($id)
            | Op::Not($id)
            | Op::Neg($id)
            | Op::Math(_, $id)
            | Op::ToRgba($id)
            | Op::Upcase($id)
            | Op::Downcase($id)
            | Op::Length($id)
            | Op::TypeOf($id) => $body,
            Op::GetFrom { key: a, object: b }
            | Op::HasIn { key: a, object: b }
            | Op::In {
                needle: a,
                haystack: b,
            }
            | Op::Sub(a, b)
            | Op::Div(a, b)
            | Op::Rem(a, b)
            | Op::Pow(a, b)
            | Op::At { index: a, array: b } => {
                for $id in [a, b] {
                    $body;
                }
            }
            Op::Compare {
                lhs, rhs, collator, ..
            } => {
                for $id in [lhs, rhs].into_iter().chain(collator) {
                    $body;
                }
            }
            Op::Match {
                input,
                outputs,
                otherwise,
                ..
            } => {
                for $id in std::iter::once(input).chain(outputs).chain([otherwise]) {
                    $body;
                }
            }
            Op::Case {
                branches,
                otherwise,
            } => {
                for (test, out) in branches {
                    for $id in [test, out] {
                        $body;
                    }
                }
                let $id = otherwise;
                $body;
            }
            Op::Step { input, outputs, .. } | Op::Interpolate { input, outputs, .. } => {
                for $id in std::iter::once(input).chain(outputs) {
                    $body;
                }
            }
            Op::Rgba { r, g, b, a } => {
                for $id in [r, g, b].into_iter().chain(a) {
                    $body;
                }
            }
            Op::Call(_, ids)
            | Op::Coalesce(ids)
            | Op::All(ids)
            | Op::Any(ids)
            | Op::Sum(ids)
            | Op::Product(ids)
            | Op::Min(ids)
            | Op::Max(ids)
            | Op::Coerce(ids)
            | Op::Assert(ids)
            | Op::Concat(ids) => {
                for $id in ids {
                    $body;
                }
            }
        }
    };
}

impl Op {
    /// Calls `f` on each operand, in evaluation order.
    pub(crate) fn for_each_child(&self, mut f: impl FnMut(NodeId)) {
        each_child!(self, |id| f(*id));
    }

    /// Replaces each operand with `f(operand)`, in evaluation order.
    pub(crate) fn map_children(&mut self, mut f: impl FnMut(NodeId) -> NodeId) {
        each_child!(self, |id| *id = f(*id));
    }

    /// The node's own inputs, not counting its operands'.
    pub(crate) fn own_deps(&self) -> Deps {
        match self {
            Op::Call(call, _) => match call {
                Call::FeatureState => Deps::FEATURE_STATE,
                Call::GlobalState(_) => Deps::GLOBAL_STATE,
                Call::Heatmap => Deps::HEATMAP,
                Call::Elevation => Deps::ELEVATION,
                Call::Accumulated => Deps::ACCUMULATED,
                Call::Collator => Deps::LOCALE,
                Call::Image => Deps::IMAGE,
                Call::SupportedScript => Deps::SCRIPT,
                Call::Geometry { .. } => Deps::GEOMETRY,
                Call::Pitch | Call::Bearing | Call::MetersPerPixel => Deps::CAMERA,
                _ => Deps::NONE,
            },
            Op::Zoom => Deps::ZOOM,
            Op::LineProgress => Deps::LINE_PROGRESS,
            Op::Id
            | Op::GeometryType
            | Op::Properties
            | Op::Get(_)
            | Op::GetDynamic(_)
            | Op::Has(_)
            | Op::HasDynamic(_)
            | Op::Legacy(_) => Deps::PROPERTIES,
            _ => Deps::NONE,
        }
    }
}

/// An expression's node arena and side tables.
#[derive(Clone, Debug, Default)]
pub(crate) struct Ir {
    pub nodes: Vec<Node>,
    /// Each node's index path in the source expression (`[2][1]`), for
    /// the conformance runner's error messages.
    #[cfg(any(test, feature = "conformance"))]
    pub paths: Vec<Box<str>>,
    /// Literal property keys, by `KeyId`.
    pub keys: Vec<Box<str>>,
}

impl Ir {
    pub(crate) fn push(&mut self, op: Op, ty: Type, path: &str) -> NodeId {
        let deps = self.deps_of(&op);
        self.nodes.push(Node { op, ty, deps });
        #[cfg(any(test, feature = "conformance"))]
        self.paths.push(path.into());
        #[cfg(not(any(test, feature = "conformance")))]
        let _ = path;
        (self.nodes.len() - 1) as NodeId
    }

    /// The inputs of a node with `op`: its own and its operands'.
    fn deps_of(&self, op: &Op) -> Deps {
        let mut deps = op.own_deps();
        op.for_each_child(|c| deps = deps | self.nodes[c as usize].deps);
        deps
    }

    /// Counts every node's inputs again, after nodes were replaced.
    pub(crate) fn recount_deps(&mut self) {
        // Post order: operands come before the nodes that read them.
        for i in 0..self.nodes.len() {
            self.nodes[i].deps = self.deps_of(&self.nodes[i].op);
        }
    }

    pub(crate) fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
    }

    pub(crate) fn intern_key(&mut self, key: &str) -> KeyId {
        match self.keys.iter().position(|k| &**k == key) {
            Some(i) => i as KeyId,
            None => {
                self.keys.push(key.into());
                (self.keys.len() - 1) as KeyId
            }
        }
    }

    /// A new arena holding only the nodes reachable from `root`, in post
    /// order, with `root` last. Parsing leaves behind the nodes of failed
    /// overloads and of subtrees folded into literals.
    pub(crate) fn compact(&self, root: NodeId) -> Ir {
        let mut out = Ir {
            keys: self.keys.clone(),
            ..Ir::default()
        };
        let mut map = vec![NodeId::MAX; self.nodes.len()];
        self.copy_into(root, &mut out, &mut map);
        out
    }

    fn copy_into(&self, id: NodeId, out: &mut Ir, map: &mut [NodeId]) -> NodeId {
        if map[id as usize] != NodeId::MAX {
            return map[id as usize];
        }
        let node = &self.nodes[id as usize];
        let mut op = node.op.clone();
        op.map_children(|c| self.copy_into(c, out, map));
        out.nodes.push(Node {
            op,
            ty: node.ty.clone(),
            deps: node.deps,
        });
        #[cfg(any(test, feature = "conformance"))]
        out.paths.push(self.paths[id as usize].clone());
        let new = (out.nodes.len() - 1) as NodeId;
        map[id as usize] = new;
        new
    }

    pub(crate) fn root(&self) -> NodeId {
        (self.nodes.len() - 1) as NodeId
    }
}
