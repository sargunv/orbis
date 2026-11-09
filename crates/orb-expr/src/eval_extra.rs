//! Evaluation of the special forms and host-service operators.
use crate::eval::{Evaluator, Result, error};
use crate::ir::{Call, NodeId};
use crate::{Collator, FormattedSection, NumberFormat, Value};

impl<'a> Evaluator<'_, 'a> {
    pub(super) fn eval_call(&self, id: NodeId, call: &Call, args: &[NodeId]) -> Result<Value<'a>> {
        Ok(match call {
            Call::Let => self.eval(*args.last().expect("let result"))?,
            Call::Var => self.eval(args[0])?,
            Call::Error => return Err(error(id, self.string(args[0])?.into_owned())),
            Call::FeatureState => {
                let key = self.string(args[0])?;
                self.context
                    .feature_state
                    .iter()
                    .rev()
                    .find(|(k, _)| *k == key)
                    .map_or(Value::Null, |(_, v)| v.reborrow())
            }
            Call::GlobalState(key) => self
                .context
                .global_state
                .iter()
                .rev()
                .find(|(k, _)| k == &*self.ir.keys[*key as usize])
                .map_or(Value::Null, |(_, v)| v.reborrow()),
            Call::Heatmap => Value::Number(self.context.heatmap_density),
            Call::Pitch => Value::Number(self.context.pitch),
            Call::Bearing => Value::Number(self.context.bearing),
            Call::Elevation => Value::Number(self.context.elevation),
            Call::Accumulated => self.context.accumulated.reborrow().into_owned(),
            Call::SupportedScript => Value::Bool(
                self.context
                    .is_supported_script
                    .is_none_or(|f| self.string(args[0]).is_ok_and(|s| f(&s))),
            ),
            Call::Array => Value::Array(
                args.iter()
                    .map(|&a| self.eval(a).map(Value::into_owned))
                    .collect::<Result<Vec<_>>>()?
                    .into(),
            ),
            Call::Split => {
                let text = self.string(args[0])?;
                let separator = self.string(args[1])?;
                let pieces = if separator.is_empty() {
                    text.chars().map(|c| c.to_string()).collect::<Vec<_>>()
                } else {
                    text.split(&*separator).map(str::to_owned).collect()
                };
                Value::Array(
                    pieces
                        .into_iter()
                        .map(|s| Value::String(s.into()))
                        .collect::<Vec<_>>()
                        .into(),
                )
            }
            Call::Join => {
                let array = self.eval(args[0])?;
                let separator = self.string(args[1])?;
                Value::String(
                    array
                        .as_array()
                        .expect("typed array")
                        .iter()
                        .map(Value::to_js_string)
                        .collect::<Vec<_>>()
                        .join(&separator)
                        .into(),
                )
            }
            Call::IndexOf => self.index_of(id, args)?,
            Call::Slice => self.slice(id, args)?,
            Call::Collator => {
                let locale = self.eval(args[2])?;
                Value::Collator(Collator {
                    case_sensitive: self.boolean(args[0])?,
                    diacritic_sensitive: self.boolean(args[1])?,
                    locale: self.context.locale.resolved_locale(locale.as_str()),
                })
            }
            Call::ResolvedLocale => {
                let Value::Collator(c) = self.eval(args[0])? else {
                    unreachable!("typed collator")
                };
                Value::String(c.locale.into())
            }
            Call::NumberFormat => {
                let number = self.number(args[0])?;
                let values = args[1..]
                    .iter()
                    .map(|&a| self.eval(a))
                    .collect::<Result<Vec<_>>>()?;
                Value::String(
                    self.context
                        .locale
                        .format_number(
                            number,
                            NumberFormat {
                                locale: values[0].as_str(),
                                currency: values[1].as_str(),
                                unit: values[2].as_str(),
                                min_fraction_digits: values[3].as_f64(),
                                max_fraction_digits: values[4].as_f64(),
                            },
                        )
                        .map_err(|e| error(id, e))?
                        .into(),
                )
            }
            Call::Image => {
                let name = self.string(args[0])?;
                if name.is_empty() {
                    Value::Null
                } else {
                    let available = self.context.available_images.iter().any(|s| s == &*name);
                    Value::Image { name, available }
                }
            }
            Call::Format => {
                let mut sections = Vec::new();
                for section in args.as_chunks::<5>().0 {
                    let content = self.eval(section[0])?;
                    let mut out = FormattedSection::text(String::new());
                    if let Value::Image { name, available } = content {
                        out.image = Some((name.into_owned(), available));
                    } else {
                        out.text = content.to_string();
                        out.scale = self.eval(section[1])?.as_f64();
                        out.font_stack = self.eval(section[2])?.as_array().map(|a| {
                            a.iter()
                                .map(Value::to_js_string)
                                .collect::<Vec<_>>()
                                .join(",")
                        });
                        out.text_color = self.eval(section[3])?.as_color();
                    }
                    out.vertical_align = self.eval(section[4])?.as_str().map(str::to_owned);
                    sections.push(out);
                }
                Value::Formatted(sections.into())
            }
            Call::Geometry { within, geometries } => {
                if *within {
                    Value::Bool(
                        self.context
                            .geometry
                            .is_some_and(|g| crate::geometry::within(g, geometries)),
                    )
                } else {
                    Value::Number(
                        self.context
                            .geometry
                            .map_or(f64::NAN, |g| crate::geometry::distance(g, geometries)),
                    )
                }
            }
        })
    }
    fn index_of(&self, id: NodeId, args: &[NodeId]) -> Result<Value<'a>> {
        let needle = self.eval(args[0])?;
        let haystack = self.eval(args[1])?;
        if !matches!(
            needle,
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
        ) {
            return Err(error(
                id,
                format!(
                    "Expected first argument to be of type boolean, string, number or null, but found {} instead.",
                    needle.type_of()
                ),
            ));
        }
        let start = if args.len() == 3 {
            self.number(args[2])?
        } else {
            0.0
        };
        let index = match haystack {
            Value::Array(a) => {
                let start = slice_index(start, a.len());
                a.iter()
                    .enumerate()
                    .skip(start)
                    .find(|(_, v)| needle.strict_eq(v))
                    .map(|(i, _)| i)
            }
            Value::String(s) => {
                // JS applies fromIndex in UTF-16 units, then the reference
                // translates its result back to a Unicode scalar index.
                let units = s.encode_utf16().collect::<Vec<_>>();
                let start = start.max(0.0).min(units.len() as f64) as usize;
                let n = needle.to_js_string().encode_utf16().collect::<Vec<_>>();
                if n.is_empty() {
                    Some(String::from_utf16_lossy(&units[..start]).chars().count())
                } else {
                    units[start..]
                        .windows(n.len())
                        .position(|w| w == n)
                        .map(|i| {
                            String::from_utf16_lossy(&units[..start + i])
                                .chars()
                                .count()
                        })
                }
            }
            other => {
                return Err(error(
                    id,
                    format!(
                        "Expected second argument to be of type array or string, but found {} instead.",
                        other.type_of()
                    ),
                ));
            }
        };
        Ok(Value::Number(index.map_or(-1.0, |i| i as f64)))
    }
    fn slice(&self, id: NodeId, args: &[NodeId]) -> Result<Value<'a>> {
        let value = self.eval(args[0])?;
        let begin = self.number(args[1])?;
        let end = if args.len() == 3 {
            self.number(args[2])?
        } else {
            f64::INFINITY
        };
        Ok(match value {
            Value::Array(a) => {
                let start = slice_index(begin, a.len());
                let end = slice_index(end, a.len()).max(start);
                Value::Array(a[start..end].to_vec().into())
            }
            Value::String(s) => {
                let chars = s.chars().collect::<Vec<_>>();
                let start = slice_index(begin, chars.len());
                let end = slice_index(end, chars.len()).max(start);
                Value::String(chars[start..end].iter().collect::<String>().into())
            }
            other => {
                return Err(error(
                    id,
                    format!(
                        "Expected first argument to be of type array or string, but found {} instead.",
                        other.type_of()
                    ),
                ));
            }
        })
    }
}
fn slice_index(index: f64, len: usize) -> usize {
    let index = if index.is_nan() { 0.0 } else { index.trunc() };
    (if index < 0.0 {
        (len as f64 + index).max(0.0)
    } else {
        index.min(len as f64)
    }) as usize
}

impl<'a> Evaluator<'_, 'a> {
    pub(super) fn coerce_array(
        &self,
        id: NodeId,
        ty: &crate::Type,
        args: &[NodeId],
    ) -> Result<Value<'a>> {
        use crate::Type;
        let mut input = Value::Null;
        for &arg in args {
            input = self.eval(arg)?;
            if let Value::TypedArray(t, _) = &input
                && t == ty
            {
                return Ok(input);
            }
            let values = match (&input, ty) {
                (Value::Number(n), Type::Padding | Type::NumberArray) => {
                    Some(vec![Value::Number(*n)])
                }
                (Value::String(s), Type::ColorArray) => {
                    crate::Color::parse(s).map(|c| vec![Value::Color(c)])
                }
                (Value::Array(a), Type::ColorArray) => a
                    .iter()
                    .map(|v| v.as_str().and_then(crate::Color::parse).map(Value::Color))
                    .collect(),
                (Value::Array(a), Type::Padding | Type::NumberArray)
                    if a.iter().all(|v| v.as_f64().is_some()) =>
                {
                    Some(a.to_vec())
                }
                (Value::Array(a), Type::VariableAnchorOffsetCollection)
                    if !a.is_empty()
                        && a.len().is_multiple_of(2)
                        && a.as_chunks::<2>().0.iter().all(|p| {
                            p[0].as_str().is_some_and(|s| {
                                [
                                    "center",
                                    "left",
                                    "right",
                                    "top",
                                    "bottom",
                                    "top-left",
                                    "top-right",
                                    "bottom-left",
                                    "bottom-right",
                                ]
                                .contains(&s)
                            }) && p[1].as_array().is_some_and(|a| {
                                a.len() == 2 && a.iter().all(|v| v.as_f64().is_some())
                            })
                        }) =>
                {
                    Some(a.to_vec())
                }
                _ => None,
            };
            if let Some(mut values) = values {
                if *ty == Type::Padding {
                    values = match values.as_slice() {
                        [a] => vec![a.clone(); 4],
                        [a, b] => vec![a.clone(), b.clone(), a.clone(), b.clone()],
                        [a, b, c] => vec![a.clone(), b.clone(), c.clone(), b.clone()],
                        [_, _, _, _] => values,
                        _ => continue,
                    };
                }
                return Ok(Value::TypedArray(ty.clone(), values.into()));
            }
        }
        Err(error(
            id,
            format!(
                "Could not parse {ty} from value '{}'",
                input
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| input.to_json_string())
            ),
        ))
    }
    pub(super) fn interpolate_typed(
        &self,
        id: NodeId,
        a: Value<'a>,
        b: Value<'a>,
        t: f64,
        ty: &crate::Type,
        space: crate::ir::ColorSpace,
    ) -> Result<Value<'a>> {
        use crate::{Type, ir::ColorSpace};
        if *ty == Type::ProjectionDefinition {
            return Ok(Value::Object(
                vec![
                    ("from".into(), a.into_owned()),
                    ("to".into(), b.into_owned()),
                    ("transition".into(), Value::Number(t)),
                ]
                .into(),
            ));
        }
        if let (Value::TypedArray(_, a), Value::TypedArray(_, b)) = (&a, &b) {
            if a.len() != b.len() && *ty != Type::NumberArray {
                return Err(error(
                    id,
                    if *ty == Type::ColorArray {
                        format!(
                            "colorArray: Arrays have mismatched length ({} vs. {}), cannot interpolate.",
                            a.len(),
                            b.len()
                        )
                    } else {
                        format!(
                            "Cannot interpolate values of different length. from: {}, to: {}",
                            Value::Array(a.clone()),
                            Value::Array(b.clone())
                        )
                    },
                ));
            }
            if *ty == Type::NumberArray {
                return Ok(Value::TypedArray(
                    ty.clone(),
                    a.iter()
                        .enumerate()
                        .map(|(i, a)| {
                            Value::Number(
                                a.as_f64().expect("typed number")
                                    + t * (b.get(i).and_then(Value::as_f64).unwrap_or(f64::NAN)
                                        - a.as_f64().expect("typed number")),
                            )
                        })
                        .collect::<Vec<_>>()
                        .into(),
                ));
            }
            let mut out = Vec::new();
            for (i, (a, b)) in a.iter().zip(b.iter()).enumerate() {
                if *ty == Type::VariableAnchorOffsetCollection && i.is_multiple_of(2) {
                    if !a.strict_eq(b) {
                        return Err(error(
                            id,
                            format!(
                                "Cannot interpolate values containing mismatched anchors. from[{i}]: {a}, to[{i}]: {b}"
                            ),
                        ));
                    }
                    out.push(a.clone());
                } else {
                    let value = if *ty == Type::ColorArray {
                        let (Value::Color(a), Value::Color(b)) = (a, b) else {
                            unreachable!("typed color array")
                        };
                        Value::Color(if space == ColorSpace::Rgb {
                            crate::Color::interpolate(*a, *b, t)
                        } else {
                            crate::color_spaces::interpolate(*a, *b, t, space)
                        })
                    } else {
                        self.interpolate(id, a.reborrow(), b.reborrow(), t)?
                            .into_owned()
                    };
                    out.push(value);
                }
            }
            return Ok(Value::TypedArray(ty.clone(), out.into()));
        }
        if let (Value::Color(a), Value::Color(b)) = (&a, &b)
            && space != ColorSpace::Rgb
        {
            return Ok(Value::Color(crate::color_spaces::interpolate(
                *a, *b, t, space,
            )));
        }
        self.interpolate(id, a, b, t)
    }
}
