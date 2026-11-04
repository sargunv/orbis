//! The bridge as the emitters see it: Diplomat's HIR, lowered for one
//! backend, narrowed to what the emitters support, with the contracts read
//! from the AST.

use std::collections::BTreeMap;
use std::path::Path;

use diplomat_core::ast;
use diplomat_core::hir::{
    self, FloatType, IntSizeType, IntType, MaybeOwn, Mutability, PrimitiveType, ReturnType,
    SelfType, SpecialMethod, SuccessType, Type, TypeDef, TypeId,
};

/// A scalar that crosses the boundary by value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Prim {
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    Isize,
    Usize,
    F32,
    F64,
}

impl Prim {
    /// The C type Diplomat's header uses.
    pub(crate) fn c(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8 => "int8_t",
            Self::U8 => "uint8_t",
            Self::I16 => "int16_t",
            Self::U16 => "uint16_t",
            Self::I32 => "int32_t",
            Self::U32 => "uint32_t",
            Self::I64 => "int64_t",
            Self::U64 => "uint64_t",
            Self::Isize => "intptr_t",
            Self::Usize => "size_t",
            Self::F32 => "float",
            Self::F64 => "double",
        }
    }

    /// The name in Diplomat's derived C types (`DiplomatU8View`).
    pub(crate) fn derived(self) -> &'static str {
        match self {
            Self::Bool => "Bool",
            Self::I8 => "I8",
            Self::U8 => "U8",
            Self::I16 => "I16",
            Self::U16 => "U16",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::I64 => "I64",
            Self::U64 => "U64",
            Self::Isize => "Isize",
            Self::Usize => "Usize",
            Self::F32 => "F32",
            Self::F64 => "F64",
        }
    }
}

/// A value that crosses the boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Ty {
    Prim(Prim),
    Enum(String),
    Struct(String),
    /// An opaque object: owned (`Box`) or borrowed, possibly null.
    Opaque {
        name: String,
        owned: bool,
        optional: bool,
        mutable: bool,
    },
    /// UTF-8 text in (`&DiplomatStr`, `&str`).
    Str,
    /// A borrowed slice of scalars: copied in as a parameter, copied out
    /// as a return.
    Slice(Prim),
    /// A no-argument callback the callee may keep and call from any
    /// thread (`impl Fn() + 'static`).
    Callback,
}

/// What a method returns on success.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Ret {
    Unit,
    Value(Ty),
    /// Text written to a `DiplomatWrite` the binding passes last.
    Write,
}

/// How a method fails.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Fail {
    Never,
    /// `Result<T, ()>`.
    Unit,
    /// `Result<T, E>` for an enum `E`.
    Enum(String),
}

/// The execution categories of plan-style contracts.
pub(crate) const CATEGORIES: &[&str] = &[
    "command",
    "query",
    "snapshot",
    "event_batch",
    "render_driver",
    "lifecycle",
    "immediate",
];

/// Which threads may use a type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Thread {
    /// Only the thread that created the object.
    Owner,
    /// Any thread.
    Any,
}

#[derive(Clone, Debug)]
pub(crate) struct Param {
    pub(crate) name: String,
    pub(crate) ty: Ty,
}

#[derive(Clone, Debug)]
pub(crate) struct Method {
    pub(crate) name: String,
    pub(crate) abi: String,
    pub(crate) docs: String,
    pub(crate) category: Option<String>,
    /// `Some(mutable)` for `&self` or `&mut self`.
    pub(crate) receiver: Option<bool>,
    pub(crate) params: Vec<Param>,
    pub(crate) ret: Ret,
    pub(crate) fail: Fail,
    /// The method writes the message of the calling thread's last error
    /// (`#[diplomat::attr(contract, error_message)]`).
    pub(crate) error_message: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct Field {
    pub(crate) name: String,
    pub(crate) ty: Ty,
    pub(crate) docs: String,
}

#[derive(Clone, Debug)]
pub(crate) enum Kind {
    Opaque {
        destroy: String,
        thread: Thread,
    },
    Struct {
        fields: Vec<Field>,
    },
    Enum {
        variants: Vec<(String, i64, String)>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct Def {
    pub(crate) name: String,
    pub(crate) docs: String,
    pub(crate) kind: Kind,
    pub(crate) methods: Vec<Method>,
}

/// The whole bridge for one backend.
#[derive(Clone, Debug, Default)]
pub(crate) struct Api {
    pub(crate) defs: Vec<Def>,
    /// For each enum used as an error: the static method writing the last
    /// error's message, if any.
    pub(crate) error_messages: BTreeMap<String, Option<String>>,
}

impl Api {
    pub(crate) fn def(&self, name: &str) -> &Def {
        self.defs
            .iter()
            .find(|d| d.name == name)
            .expect("lowering resolves every type")
    }

    pub(crate) fn fields(&self, name: &str) -> &[Field] {
        match &self.def(name).kind {
            Kind::Struct { fields } => fields,
            _ => unreachable!("{name} is a struct"),
        }
    }

    pub(crate) fn errors(&self) -> impl Iterator<Item = &str> {
        self.error_messages.keys().map(String::as_str)
    }
}

/// Contracts by type name, and by type and method name.
#[derive(Debug, Default)]
struct Contracts {
    threads: BTreeMap<String, Thread>,
    categories: BTreeMap<(String, String), String>,
    error_messages: BTreeMap<(String, String), ()>,
}

/// Lowers the bridge at `entry` (a crate root with `#[diplomat::bridge]`
/// modules) for `backend`, whose `#[diplomat::attr(backend, …)]`
/// attributes apply.
pub(crate) fn lower(entry: &Path, backend: &str) -> Result<Api, Vec<String>> {
    if !entry.is_file() {
        return Err(vec![format!("{}: not a file", entry.display())]);
    }
    let module = syn_inline_mod::parse_and_inline_modules(entry);
    let location = ast::SpanLocation::FilePath(entry.to_string_lossy().into_owned());
    let contracts = contracts(&module, &location)?;

    let mut validator = hir::BasicAttributeValidator::new(backend);
    let support = &mut validator.support;
    support.utf8_strings = true;
    support.callbacks = true;
    support.custom_errors = true;
    support.static_accessors = false;
    let tcx = hir::TypeContext::from_syn(&module, Default::default(), validator, None, &location)
        .map_err(|errors| errors.iter().map(ToString::to_string).collect::<Vec<_>>())?;

    let mut errors = Vec::new();
    let mut api = Api::default();
    for (id, def) in tcx.all_types() {
        if def.attrs().disable {
            continue;
        }
        let name = def.name().as_str().to_owned();
        let methods = match def {
            TypeDef::Opaque(o) => &o.methods,
            TypeDef::Struct(s) => &s.methods,
            TypeDef::OutStruct(s) => &s.methods,
            TypeDef::Enum(e) => &e.methods,
            _ => {
                errors.push(format!("{name}: unsupported kind of type"));
                continue;
            }
        };
        let methods = methods
            .iter()
            .filter(|m| !m.attrs.disable)
            .filter_map(|m| {
                method(&tcx, &name, m, &contracts)
                    .map_err(|e| errors.push(format!("{name}::{}: {e}", m.name.as_str())))
                    .ok()
            })
            .collect();
        let kind = match (id, def) {
            (_, TypeDef::Opaque(o)) => Kind::Opaque {
                destroy: o.dtor_abi_name.as_str().to_owned(),
                thread: contracts
                    .threads
                    .get(&name)
                    .copied()
                    .unwrap_or(Thread::Owner),
            },
            (_, TypeDef::Struct(s)) => Kind::Struct {
                fields: s
                    .fields
                    .iter()
                    .filter_map(|f| {
                        field_ty(&tcx, &f.ty)
                            .map(|ty| Field {
                                name: f.name.as_str().to_owned(),
                                ty,
                                docs: docs(&f.docs),
                            })
                            .map_err(|e| errors.push(format!("{name}.{}: {e}", f.name.as_str())))
                            .ok()
                    })
                    .collect(),
            },
            (_, TypeDef::OutStruct(s)) => Kind::Struct {
                fields: s
                    .fields
                    .iter()
                    .filter_map(|f| {
                        field_ty(&tcx, &f.ty)
                            .map(|ty| Field {
                                name: f.name.as_str().to_owned(),
                                ty,
                                docs: docs(&f.docs),
                            })
                            .map_err(|e| errors.push(format!("{name}.{}: {e}", f.name.as_str())))
                            .ok()
                    })
                    .collect(),
            },
            (TypeId::Enum(_), TypeDef::Enum(e)) => Kind::Enum {
                variants: e
                    .variants
                    .iter()
                    .map(|v| {
                        (
                            v.name.as_str().to_owned(),
                            v.discriminant as i64,
                            docs(&v.docs),
                        )
                    })
                    .collect(),
            },
            _ => unreachable!("enum ids name enums"),
        };
        api.defs.push(Def {
            name,
            docs: docs(def.docs()),
            kind,
            methods,
        });
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    // Enums used as errors, with their message methods.
    let mut error_types: Vec<String> = Vec::new();
    for def in &api.defs {
        for m in &def.methods {
            if let Fail::Enum(e) = &m.fail
                && !error_types.contains(e)
            {
                error_types.push(e.clone());
            }
        }
    }
    for e in error_types {
        let message = api
            .def(&e)
            .methods
            .iter()
            .find(|m| m.error_message)
            .map(|m| m.name.clone());
        api.error_messages.insert(e, message);
    }
    for ((ty, method), ()) in &contracts.error_messages {
        let def = api.defs.iter().find(|d| &d.name == ty);
        let ok = def
            .and_then(|d| d.methods.iter().find(|m| &m.name == method))
            .is_some_and(|m| {
                m.receiver.is_none()
                    && m.params.is_empty()
                    && m.ret == Ret::Write
                    && m.fail == Fail::Never
            });
        if !ok {
            errors.push(format!(
                "{ty}::{method}: `error_message` needs a static method that only writes text"
            ));
        }
    }
    if errors.is_empty() {
        Ok(api)
    } else {
        Err(errors)
    }
}

/// Docs as Markdown, with Rust links left as written.
fn docs(docs: &hir::Docs) -> String {
    let urls = hir::DocsUrlGenerator::with_base_urls(None, Default::default());
    docs.to_markdown(hir::DocsTypeReferenceSyntax::SquareBrackets, &urls)
}

fn method(
    tcx: &hir::TypeContext,
    owner: &str,
    m: &hir::Method,
    contracts: &Contracts,
) -> Result<Method, String> {
    if let Some(special) = &m.attrs.special_method
        && !matches!(special, SpecialMethod::Constructor)
    {
        return Err(format!("special method {special:?} isn't supported"));
    }
    let receiver = match &m.param_self {
        None => None,
        Some(p) => match &p.ty {
            SelfType::Opaque(path) => Some(path.owner.mutability == Mutability::Mutable),
            _ => return Err("only opaque types may take self".into()),
        },
    };
    let params = m
        .params
        .iter()
        .map(|p| {
            Ok(Param {
                name: p.name.as_str().to_owned(),
                ty: param_ty(tcx, &p.ty)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let success = |s: &SuccessType| -> Result<Ret, String> {
        Ok(match s {
            SuccessType::Unit => Ret::Unit,
            SuccessType::Write(hir::WriteType::Str) => Ret::Write,
            SuccessType::OutType(ty) => Ret::Value(out_ty(tcx, ty)?),
            other => return Err(format!("returning {other:?} isn't supported")),
        })
    };
    let (ret, fail) = match &m.output {
        ReturnType::Infallible(s) => (success(s)?, Fail::Never),
        ReturnType::Fallible(s, None) => (success(s)?, Fail::Unit),
        ReturnType::Fallible(s, Some(Type::Enum(e))) => (
            success(s)?,
            Fail::Enum(tcx.resolve_enum(e.tcx_id).name.as_str().to_owned()),
        ),
        ReturnType::Fallible(_, Some(other)) => {
            return Err(format!("only enums can be errors, not {other:?}"));
        }
        ReturnType::Nullable(_) => return Err("Option<T> of values isn't supported".into()),
    };
    if matches!(ret, Ret::Value(Ty::Opaque { owned: false, .. })) {
        return Err("returning a borrowed opaque isn't supported".into());
    }
    let key = (owner.to_owned(), m.name.as_str().to_owned());
    Ok(Method {
        name: m.name.as_str().to_owned(),
        abi: m.abi_name.as_str().to_owned(),
        docs: docs(&m.docs),
        category: contracts.categories.get(&key).cloned(),
        receiver,
        params,
        ret,
        fail,
        error_message: contracts.error_messages.contains_key(&key),
    })
}

fn prim(p: PrimitiveType) -> Result<Prim, String> {
    Ok(match p {
        PrimitiveType::Bool => Prim::Bool,
        PrimitiveType::Int(IntType::I8) => Prim::I8,
        PrimitiveType::Int(IntType::U8) | PrimitiveType::Byte => Prim::U8,
        PrimitiveType::Int(IntType::I16) => Prim::I16,
        PrimitiveType::Int(IntType::U16) => Prim::U16,
        PrimitiveType::Int(IntType::I32) => Prim::I32,
        PrimitiveType::Int(IntType::U32) => Prim::U32,
        PrimitiveType::Int(IntType::I64) => Prim::I64,
        PrimitiveType::Int(IntType::U64) => Prim::U64,
        PrimitiveType::IntSize(IntSizeType::Isize) => Prim::Isize,
        PrimitiveType::IntSize(IntSizeType::Usize) => Prim::Usize,
        PrimitiveType::Float(FloatType::F32) => Prim::F32,
        PrimitiveType::Float(FloatType::F64) => Prim::F64,
        other => return Err(format!("{other:?} isn't supported")),
    })
}

fn param_ty(tcx: &hir::TypeContext, ty: &Type<hir::InputOnly>) -> Result<Ty, String> {
    Ok(match ty {
        Type::Primitive(p) => Ty::Prim(prim(*p)?),
        Type::Enum(e) => Ty::Enum(tcx.resolve_enum(e.tcx_id).name.as_str().to_owned()),
        Type::Struct(s) => Ty::Struct(tcx.resolve_struct(s.tcx_id).name.as_str().to_owned()),
        Type::Opaque(o) => Ty::Opaque {
            name: tcx.resolve_opaque(o.tcx_id).name.as_str().to_owned(),
            owned: false,
            optional: o.is_optional(),
            mutable: o.owner.mutability == Mutability::Mutable,
        },
        Type::Slice(hir::Slice::Str(_, hir::StringEncoding::UnvalidatedUtf8))
        | Type::Slice(hir::Slice::Str(_, hir::StringEncoding::Utf8)) => Ty::Str,
        Type::Slice(hir::Slice::Primitive(MaybeOwn::Borrow(b), p))
            if b.mutability == Mutability::Immutable =>
        {
            Ty::Slice(prim(*p)?)
        }
        Type::Callback(cb) => {
            let inputs = hir::CallbackInstantiationFunctionality::get_inputs(cb)
                .map_err(|()| "unresolved callback")?;
            let output = hir::CallbackInstantiationFunctionality::get_output_type(cb)
                .map_err(|()| "unresolved callback")?;
            if !inputs.is_empty() || !matches!(output, ReturnType::Infallible(SuccessType::Unit)) {
                return Err("only `impl Fn()` callbacks are supported".into());
            }
            Ty::Callback
        }
        other => return Err(format!("parameter type {other:?} isn't supported")),
    })
}

fn out_ty(tcx: &hir::TypeContext, ty: &Type<hir::OutputOnly>) -> Result<Ty, String> {
    Ok(match ty {
        Type::Primitive(p) => Ty::Prim(prim(*p)?),
        Type::Enum(e) => Ty::Enum(tcx.resolve_enum(e.tcx_id).name.as_str().to_owned()),
        Type::Struct(s) => Ty::Struct(
            tcx.resolve_type(hir::StructPathLike::id(s))
                .name()
                .as_str()
                .to_owned(),
        ),
        Type::Opaque(o) => Ty::Opaque {
            name: tcx.resolve_opaque(o.tcx_id).name.as_str().to_owned(),
            owned: o.is_owned(),
            optional: o.is_optional(),
            mutable: false,
        },
        Type::Slice(hir::Slice::Primitive(MaybeOwn::Borrow(_), p)) => Ty::Slice(prim(*p)?),
        other => return Err(format!("return type {other:?} isn't supported")),
    })
}

/// Struct fields: scalars, enums and structs.
fn field_ty<P: hir::TyPosition>(tcx: &hir::TypeContext, ty: &Type<P>) -> Result<Ty, String> {
    Ok(match ty {
        Type::Primitive(p) => Ty::Prim(prim(*p)?),
        Type::Enum(e) => Ty::Enum(tcx.resolve_enum(e.tcx_id).name.as_str().to_owned()),
        Type::Struct(s) => Ty::Struct(
            tcx.resolve_type(hir::StructPathLike::id(s))
                .name()
                .as_str()
                .to_owned(),
        ),
        other => return Err(format!("field type {other:?} isn't supported")),
    })
}

/// Reads `#[diplomat::attr(contract, …)]` from the AST: Diplomat's lowering
/// skips attributes for a backend named `contract`.
fn contracts(module: &syn::File, location: &ast::SpanLocation) -> Result<Contracts, Vec<String>> {
    let file = ast::File::from_syn(module, None, location);
    let mut out = Contracts::default();
    let mut errors = Vec::new();
    for module in file.modules.values() {
        for ty in module.declared_types.values() {
            let (name, attrs, methods) = match ty {
                ast::CustomType::Opaque(o) => (&o.name, &o.attrs, &o.methods),
                ast::CustomType::Struct(s) => (&s.name, &s.attrs, &s.methods),
                ast::CustomType::Enum(e) => (&e.name, &e.attrs, &e.methods),
                _ => continue,
            };
            let name = name.as_str().to_owned();
            for meta in contract_metas(attrs) {
                match thread_rule(meta) {
                    Ok(thread) => {
                        out.threads.insert(name.clone(), thread);
                    }
                    Err(e) => errors.push(format!("{name}: {e}")),
                }
            }
            for m in methods {
                let key = (name.clone(), m.name.as_str().to_owned());
                for meta in contract_metas(&m.attrs) {
                    match meta.path().get_ident().map(ToString::to_string) {
                        Some(c) if CATEGORIES.contains(&c.as_str()) => {
                            if let Some(old) = out.categories.insert(key.clone(), c) {
                                errors
                                    .push(format!("{}::{}: two categories ({old})", key.0, key.1));
                            }
                        }
                        Some(c) if c == "error_message" => {
                            out.error_messages.insert(key.clone(), ());
                        }
                        _ => errors.push(format!(
                            "{}::{}: unknown contract `{}`; expected one of {} or error_message",
                            key.0,
                            key.1,
                            quote_meta(meta),
                            CATEGORIES.join(", ")
                        )),
                    }
                }
            }
        }
    }
    if errors.is_empty() {
        Ok(out)
    } else {
        Err(errors)
    }
}

fn contract_metas(attrs: &ast::Attrs) -> impl Iterator<Item = &syn::Meta> {
    attrs.attrs.iter().filter_map(|a| match &a.cfg {
        ast::attrs::DiplomatBackendAttrCfg::BackendName(n) if n == "contract" => Some(&a.meta),
        _ => None,
    })
}

fn thread_rule(meta: &syn::Meta) -> Result<Thread, String> {
    if let syn::Meta::NameValue(nv) = meta
        && nv.path.is_ident("thread")
        && let syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(s),
            ..
        }) = &nv.value
    {
        return match s.value().as_str() {
            "owner" => Ok(Thread::Owner),
            "any" => Ok(Thread::Any),
            other => Err(format!(
                "thread = \"{other}\": expected \"owner\" or \"any\""
            )),
        };
    }
    Err(format!(
        "unknown type contract `{}`; expected thread = \"owner\" or \"any\"",
        quote_meta(meta)
    ))
}

fn quote_meta(meta: &syn::Meta) -> String {
    match meta {
        syn::Meta::Path(p) => path_text(p),
        syn::Meta::List(l) => path_text(&l.path) + "(…)",
        syn::Meta::NameValue(nv) => path_text(&nv.path) + " = …",
    }
}

fn path_text(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}
