//! Special forms from the pinned TypeScript ParsingContext implementations.
use crate::ir::{Call, NodeId, Op};
use crate::json::{Json, Span};
use crate::parse::{Parser, child_key, js_typeof};
use crate::{Type, Value};

impl Parser<'_> {
    fn call(&mut self, call: Call, args: Vec<NodeId>, ty: Type, key: &str) -> Option<NodeId> {
        Some(self.ir.push(Op::Call(call, args), ty, key))
    }
    pub(super) fn parse_extra(
        &mut self,
        op: &str,
        items: &[Json],
        span: Span,
        key: &str,
        expected: Option<&Type>,
    ) -> Option<NodeId> {
        let n = items.len() - 1;
        match op {
            "var" => {
                if n != 1 || items[1].as_str().is_none() {
                    self.error(
                        key,
                        span,
                        "'var' expression requires exactly one string literal argument.",
                    );
                    return None;
                }
                let name = items[1].as_str().unwrap();
                let Some((_, value)) = self.scope.iter().rev().find(|(k, _)| k == name) else {
                    self.error(child_key(key,1),items[1].span,format!("Unknown variable \"{name}\". Make sure \"{name}\" has been bound in an enclosing \"let\" expression before using it."));
                    return None;
                };
                let value = *value;
                self.call(Call::Var, vec![value], self.ty(value).clone(), key)
            }
            "let" => {
                if n < 3 {
                    self.error(
                        key,
                        span,
                        format!("Expected at least 3 arguments, but found {n} instead."),
                    );
                    return None;
                }
                let mut bindings = Vec::new();
                for i in (1..items.len() - 1).step_by(2) {
                    let Some(name) = items[i].as_str() else {
                        self.error(
                            child_key(key, i),
                            items[i].span,
                            format!(
                                "Expected string, but found {} instead.",
                                js_typeof(&items[i])
                            ),
                        );
                        return None;
                    };
                    if name
                        .chars()
                        .any(|c| !(c.is_ascii_alphanumeric() || c == '_'))
                    {
                        self.error(
                            child_key(key, i),
                            items[i].span,
                            "Variable names must contain only alphanumeric characters or '_'.",
                        );
                        return None;
                    }
                    let value = self.parse(&items[i + 1], &child_key(key, i + 1), None, None)?;
                    bindings.push((name.to_owned(), value));
                }
                let old = self.scope.len();
                let mut args = bindings.iter().map(|(_, v)| *v).collect::<Vec<_>>();
                self.scope.extend(bindings);
                let result = self.parse(items.last().unwrap(), &child_key(key, n), expected, None);
                self.scope.truncate(old);
                let result = result?;
                args.push(result);
                self.call(Call::Let, args, self.ty(result).clone(), key)
            }
            "semiliteral" => {
                if n != 1 {
                    self.error(key,span,format!("'semiliteral' expression requires exactly one argument, but found {n} instead."));
                    return None;
                }
                let Some(items) = items[1].as_array() else {
                    return Some(self.literal(Value::from_json(&items[1]), key, expected));
                };
                let mut args = Vec::new();
                for (i, item) in items.iter().enumerate() {
                    args.push(self.parse(
                        item,
                        &child_key(&child_key(key, 1), i),
                        Some(&Type::Value),
                        None,
                    )?);
                }
                let ty = args.first().map_or(Type::Value, |&a| self.ty(a).clone());
                let ty = if args
                    .iter()
                    .all(|&a| *self.ty(a) == ty && !matches!(ty, Type::Array(..)))
                {
                    ty
                } else {
                    Type::Value
                };
                self.call(
                    Call::Array,
                    args,
                    Type::array(ty, Some(items.len() as u32)),
                    key,
                )
            }
            "global-state" => {
                if n != 1 {
                    self.error(
                        key,
                        span,
                        format!("Expected 1 argument, but found {n} instead."),
                    );
                    return None;
                }
                if items[1].is_null() {
                    self.error(key, span, "Global state property must be defined.");
                    return None;
                }
                let Some(name) = items[1].as_str() else {
                    self.error(
                        key,
                        span,
                        format!(
                            "Global state property must be string, but found {} instead.",
                            js_typeof(&items[1])
                        ),
                    );
                    return None;
                };
                if let Some(state) = self.global_state {
                    let value = state
                        .iter()
                        .find(|(k, _)| k == name)
                        .map_or(Value::Null, |(_, v)| v.clone());
                    let id = self.ir.push(Op::Literal(value), Type::Value, key);
                    self.global_literals.push(id);
                    return Some(id);
                }
                let k = self.ir.intern_key(name);
                self.call(Call::GlobalState(k), vec![], Type::Value, key)
            }
            "index-of" | "slice" => {
                if !(2..=3).contains(&n) {
                    self.error(
                        key,
                        span,
                        format!("Expected 2 or 3 arguments, but found {n} instead."),
                    );
                    return None;
                }
                let first = self.parse(&items[1], &child_key(key, 1), Some(&Type::Value), None);
                let second = self.parse(
                    &items[2],
                    &child_key(key, 2),
                    Some(if op == "slice" {
                        &Type::Number
                    } else {
                        &Type::Value
                    }),
                    None,
                );
                let (first, second) = (first?, second?);
                let ty = self.ty(first).clone();
                let valid = if op == "slice" {
                    matches!(ty, Type::Array(..) | Type::String | Type::Value)
                } else {
                    matches!(
                        ty,
                        Type::Boolean | Type::String | Type::Number | Type::Null | Type::Value
                    )
                };
                if !valid {
                    self.error(
                        key,
                        span,
                        format!(
                            "Expected first argument to be of type {}, but found {ty} instead",
                            if op == "slice" {
                                "array or string"
                            } else {
                                "boolean, string, number or null"
                            }
                        ),
                    );
                    return None;
                }
                let mut args = vec![first, second];
                if n == 3 {
                    args.push(self.parse(
                        &items[3],
                        &child_key(key, 3),
                        Some(&Type::Number),
                        None,
                    )?);
                }
                if op == "slice" {
                    self.call(Call::Slice, args, ty, key)
                } else {
                    self.call(Call::IndexOf, args, Type::Number, key)
                }
            }
            "image" => {
                if n != 1 {
                    self.error(key, span, "Expected two arguments.");
                    return None;
                }
                let name = self.parse(&items[1], &child_key(key, 1), Some(&Type::String), None);
                let Some(name) = name else {
                    self.error(key, span, "No image name provided.");
                    return None;
                };
                self.call(Call::Image, vec![name], Type::ResolvedImage, key)
            }
            "collator" | "number-format" => self.parse_locale(op, items, span, key),
            "format" => self.parse_format(items, span, key),
            "within" | "distance" => {
                if n != 1 {
                    self.error(key,span,format!("'{op}' expression requires exactly one argument, but found {n} instead."));
                    return None;
                }
                let mut geometries = crate::geometry::parse_geojson(&items[1]).unwrap_or_default();
                if op == "within" {
                    geometries.retain(|g| matches!(g, crate::geometry::Geometry::Polygon(_)));
                }
                if geometries.is_empty() {
                    self.error(key,span,format!("'{op}' expression requires valid geojson object that contains polygon geometry type."));
                    return None;
                }
                self.call(
                    Call::Geometry {
                        within: op == "within",
                        geometries,
                    },
                    vec![],
                    if op == "within" {
                        Type::Boolean
                    } else {
                        Type::Number
                    },
                    key,
                )
            }
            _ => unreachable!("special form dispatch"),
        }
    }

    fn option(
        &mut self,
        options: &Json,
        name: &str,
        ty: &Type,
        key: &str,
        default: Value<'static>,
    ) -> Option<NodeId> {
        match options.get(name).filter(|j| Value::from_json(j).truthy()) {
            Some(v) => self.parse(v, &child_key(key, 1), Some(ty), None),
            None => Some(self.literal(default, key, None)),
        }
    }
    fn parse_locale(&mut self, op: &str, items: &[Json], span: Span, key: &str) -> Option<NodeId> {
        let collator = op == "collator";
        if items.len() != if collator { 2 } else { 3 } {
            self.error(
                key,
                span,
                if collator {
                    "Expected one argument."
                } else {
                    "Expected two arguments."
                },
            );
            return None;
        }
        let mut args = Vec::new();
        if !collator {
            args.push(self.parse(&items[1], &child_key(key, 1), Some(&Type::Number), None)?);
        }
        let options = items.last().unwrap();
        if options.as_object().is_none() {
            self.error(
                key,
                span,
                format!(
                    "{} options argument must be an object.",
                    if collator { "Collator" } else { "NumberFormat" }
                ),
            );
            return None;
        }
        if collator {
            args.push(self.option(
                options,
                "case-sensitive",
                &Type::Boolean,
                key,
                Value::Bool(false),
            )?);
            args.push(self.option(
                options,
                "diacritic-sensitive",
                &Type::Boolean,
                key,
                Value::Bool(false),
            )?);
            args.push(self.option(options, "locale", &Type::String, key, Value::Null)?);
            self.call(Call::Collator, args, Type::Collator, key)
        } else {
            for name in [
                "locale",
                "currency",
                "unit",
                "min-fraction-digits",
                "max-fraction-digits",
            ] {
                args.push(self.option(
                    options,
                    name,
                    if name.ends_with("digits") {
                        &Type::Number
                    } else {
                        &Type::String
                    },
                    key,
                    Value::Null,
                )?);
            }
            if options
                .get("currency")
                .is_some_and(|j| Value::from_json(j).truthy())
                && options
                    .get("unit")
                    .is_some_and(|j| Value::from_json(j).truthy())
            {
                self.error(
                    key,
                    span,
                    "NumberFormat options `currency` and `unit` are mutually exclusive",
                );
                return None;
            }
            self.call(Call::NumberFormat, args, Type::String, key)
        }
    }
    fn parse_format(&mut self, items: &[Json], span: Span, key: &str) -> Option<NodeId> {
        if items.len() < 2 {
            self.error(key, span, "Expected at least one argument.");
            return None;
        }
        if items[1].as_object().is_some() || items[1].is_null() {
            self.error(
                key,
                span,
                "First argument must be an image or text section.",
            );
            return None;
        }
        let mut args = Vec::new();
        let mut i = 1;
        while i < items.len() {
            let content = self.parse(&items[i], &child_key(key, 1), Some(&Type::Value), None)?;
            if !matches!(
                self.ty(content),
                Type::String | Type::Value | Type::Null | Type::ResolvedImage
            ) {
                self.error(
                    key,
                    span,
                    "Formatted text type must be 'string', 'value', 'image' or 'null'.",
                );
                return None;
            }
            args.push(content);
            i += 1;
            let options = items.get(i).filter(|j| j.as_object().is_some());
            for (name, ty) in [
                ("font-scale", Type::Number),
                ("text-font", Type::array(Type::String, None)),
                ("text-color", Type::Color),
                ("vertical-align", Type::String),
            ] {
                let id = if let Some(options) = options {
                    if name == "vertical-align"
                        && let Some(s) = options.get(name).and_then(Json::as_str)
                        && !["bottom", "center", "top"].contains(&s)
                    {
                        self.error(key,span,format!("'vertical-align' must be one of: 'bottom', 'center', 'top' but found '{s}' instead."));
                        return None;
                    }
                    self.option(options, name, &ty, key, Value::Null)?
                } else {
                    self.literal(Value::Null, key, None)
                };
                args.push(id);
            }
            if options.is_some() {
                i += 1;
            }
        }
        self.call(Call::Format, args, Type::Formatted, key)
    }
}
