//! A small, strict JSON parser (RFC 8259) that keeps byte offsets for
//! diagnostics, and a compact writer matching JavaScript's `JSON.stringify`.
//!
//! Core crates don't use `serde_json` (wasm size); `orb-style`'s v8 front end reads
//! whole style documents with this parser.

use std::fmt;

use crate::js;

/// Nesting deeper than this is rejected, which also bounds the recursion of
/// everything that walks a parsed document.
pub(crate) const MAX_DEPTH: usize = 128;

/// A byte range in the source text. Synthetic nodes (built by lowering) carry
/// the span of the source node they came from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

/// A JSON value with its span.
#[derive(Clone, Debug, PartialEq)]
pub struct Json {
    pub value: JsonValue,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    /// Members in document order. Duplicate keys are kept; lookups take the
    /// last one, as `JSON.parse` does.
    Object(Vec<Member>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    pub key: String,
    pub key_span: Span,
    pub value: Json,
}

/// A syntax error at a byte offset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonError {
    pub offset: u32,
    pub message: &'static str,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at byte {}", self.message, self.offset)
    }
}

impl std::error::Error for JsonError {}

/// Parses one JSON document. A leading byte-order mark is skipped, as a
/// browser's `Response.json()` does.
pub fn parse(text: &str) -> Result<Json, JsonError> {
    if text.len() > u32::MAX as usize {
        return Err(JsonError {
            offset: 0,
            message: "document too large",
        });
    }
    let start = if text.starts_with('\u{feff}') { 3 } else { 0 };
    let mut parser = Parser {
        text: text.as_bytes(),
        pos: start,
    };
    parser.skip_whitespace();
    let value = parser.value(0)?;
    parser.skip_whitespace();
    if parser.pos != parser.text.len() {
        return Err(parser.error("unexpected trailing characters"));
    }
    Ok(value)
}

impl Json {
    pub fn new(value: JsonValue, span: Span) -> Self {
        Self { value, span }
    }

    pub fn is_null(&self) -> bool {
        matches!(self.value, JsonValue::Null)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self.value {
            JsonValue::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self.value {
            JsonValue::Number(n) => Some(n),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match &self.value {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match &self.value {
            JsonValue::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[Member]> {
        match &self.value {
            JsonValue::Object(o) => Some(o),
            _ => None,
        }
    }

    /// The member `key` of an object; the last one if the key repeats.
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?
            .iter()
            .rev()
            .find(|m| m.key == key)
            .map(|m| &m.value)
    }

    /// [`get`](Self::get), to change the member in place.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Json> {
        match &mut self.value {
            JsonValue::Object(members) => members
                .iter_mut()
                .rev()
                .find(|m| m.key == key)
                .map(|m| &mut m.value),
            _ => None,
        }
    }

    /// The JavaScript `typeof`-style name used in diagnostics: `null`,
    /// `boolean`, `number`, `string`, `array` or `object`.
    pub fn type_name(&self) -> &'static str {
        match self.value {
            JsonValue::Null => "null",
            JsonValue::Bool(_) => "boolean",
            JsonValue::Number(_) => "number",
            JsonValue::String(_) => "string",
            JsonValue::Array(_) => "array",
            JsonValue::Object(_) => "object",
        }
    }
}

/// Compact output, byte for byte what `JSON.stringify` writes.
impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.value {
            JsonValue::Null => f.write_str("null"),
            JsonValue::Bool(b) => write!(f, "{b}"),
            JsonValue::Number(n) => js::write_json_number(f, *n),
            JsonValue::String(s) => js::write_json_string(f, s),
            JsonValue::Array(items) => {
                f.write_str("[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(",")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_str("]")
            }
            JsonValue::Object(members) => {
                f.write_str("{")?;
                for (i, m) in members.iter().enumerate() {
                    if i > 0 {
                        f.write_str(",")?;
                    }
                    js::write_json_string(f, &m.key)?;
                    write!(f, ":{}", m.value)?;
                }
                f.write_str("}")
            }
        }
    }
}

struct Parser<'a> {
    text: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn error(&self, message: &'static str) -> JsonError {
        JsonError {
            offset: self.pos as u32,
            message,
        }
    }

    fn span(&self, start: usize) -> Span {
        Span {
            start: start as u32,
            end: self.pos as u32,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.text.get(self.pos).copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.pos += 1;
        }
    }

    fn expect_literal(&mut self, literal: &[u8]) -> Result<(), JsonError> {
        if self.text[self.pos..].starts_with(literal) {
            self.pos += literal.len();
            Ok(())
        } else {
            Err(self.error("invalid literal"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, JsonError> {
        if depth >= MAX_DEPTH {
            return Err(self.error("nesting too deep"));
        }
        let start = self.pos;
        let value = match self.peek() {
            None => return Err(self.error("unexpected end of input")),
            Some(b'n') => {
                self.expect_literal(b"null")?;
                JsonValue::Null
            }
            Some(b't') => {
                self.expect_literal(b"true")?;
                JsonValue::Bool(true)
            }
            Some(b'f') => {
                self.expect_literal(b"false")?;
                JsonValue::Bool(false)
            }
            Some(b'"') => JsonValue::String(self.string()?),
            Some(b'[') => {
                self.pos += 1;
                let mut items = Vec::new();
                self.skip_whitespace();
                if self.peek() == Some(b']') {
                    self.pos += 1;
                } else {
                    loop {
                        self.skip_whitespace();
                        items.push(self.value(depth + 1)?);
                        self.skip_whitespace();
                        match self.peek() {
                            Some(b',') => self.pos += 1,
                            Some(b']') => {
                                self.pos += 1;
                                break;
                            }
                            _ => return Err(self.error("expected ',' or ']'")),
                        }
                    }
                }
                JsonValue::Array(items)
            }
            Some(b'{') => {
                self.pos += 1;
                let mut members = Vec::new();
                self.skip_whitespace();
                if self.peek() == Some(b'}') {
                    self.pos += 1;
                } else {
                    loop {
                        self.skip_whitespace();
                        if self.peek() != Some(b'"') {
                            return Err(self.error("expected a string key"));
                        }
                        let key_start = self.pos;
                        let key = self.string()?;
                        let key_span = self.span(key_start);
                        self.skip_whitespace();
                        if self.peek() != Some(b':') {
                            return Err(self.error("expected ':'"));
                        }
                        self.pos += 1;
                        self.skip_whitespace();
                        let value = self.value(depth + 1)?;
                        members.push(Member {
                            key,
                            key_span,
                            value,
                        });
                        self.skip_whitespace();
                        match self.peek() {
                            Some(b',') => self.pos += 1,
                            Some(b'}') => {
                                self.pos += 1;
                                break;
                            }
                            _ => return Err(self.error("expected ',' or '}'")),
                        }
                    }
                }
                JsonValue::Object(members)
            }
            Some(b'-' | b'0'..=b'9') => JsonValue::Number(self.number()?),
            Some(_) => return Err(self.error("unexpected character")),
        };
        Ok(Json {
            value,
            span: self.span(start),
        })
    }

    fn digits(&mut self) -> usize {
        let start = self.pos;
        while let Some(b'0'..=b'9') = self.peek() {
            self.pos += 1;
        }
        self.pos - start
    }

    fn number(&mut self) -> Result<f64, JsonError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(self.error("invalid number")),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if self.digits() == 0 {
                return Err(self.error("invalid number"));
            }
        }
        if let Some(b'e' | b'E') = self.peek() {
            self.pos += 1;
            if let Some(b'+' | b'-') = self.peek() {
                self.pos += 1;
            }
            if self.digits() == 0 {
                return Err(self.error("invalid number"));
            }
        }
        // The slice is ASCII and a decimal literal, rounded correctly, as
        // `JSON.parse` rounds it.
        let text = std::str::from_utf8(&self.text[start..self.pos]).expect("ASCII");
        crate::decimal::parse(text).ok_or(JsonError {
            offset: start as u32,
            message: "invalid number",
        })
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let Some(digits) = self.text.get(self.pos..self.pos + 4) else {
            return Err(self.error("invalid unicode escape"));
        };
        let mut value = 0;
        for &d in digits {
            let v = match d {
                b'0'..=b'9' => d - b'0',
                b'a'..=b'f' => d - b'a' + 10,
                b'A'..=b'F' => d - b'A' + 10,
                _ => return Err(self.error("invalid unicode escape")),
            };
            value = value * 16 + u32::from(v);
        }
        self.pos += 4;
        Ok(value)
    }

    fn string(&mut self) -> Result<String, JsonError> {
        debug_assert_eq!(self.peek(), Some(b'"'));
        self.pos += 1;
        let mut out = String::new();
        loop {
            let run = self.pos;
            while let Some(b) = self.peek() {
                if b == b'"' || b == b'\\' || b < 0x20 {
                    break;
                }
                self.pos += 1;
            }
            // The input is a `str` and the run stops only at ASCII bytes, so
            // it is valid UTF-8.
            out.push_str(std::str::from_utf8(&self.text[run..self.pos]).expect("UTF-8 run"));
            match self.peek() {
                None => return Err(self.error("unterminated string")),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    let Some(escape) = self.peek() else {
                        return Err(self.error("unterminated string"));
                    };
                    self.pos += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let unit = self.hex4()?;
                            let code = if (0xd800..0xdc00).contains(&unit) {
                                if !self.text[self.pos..].starts_with(b"\\u") {
                                    return Err(self.error("lone surrogate in string"));
                                }
                                self.pos += 2;
                                let low = self.hex4()?;
                                if !(0xdc00..0xe000).contains(&low) {
                                    return Err(self.error("lone surrogate in string"));
                                }
                                0x10000 + ((unit - 0xd800) << 10) + (low - 0xdc00)
                            } else {
                                unit
                            };
                            match char::from_u32(code) {
                                Some(c) => out.push(c),
                                None => return Err(self.error("lone surrogate in string")),
                            }
                        }
                        _ => return Err(self.error("invalid escape")),
                    }
                }
                Some(_) => return Err(self.error("control character in string")),
            }
        }
    }
}

/// A document crosses between workers as its text.
impl serde::Serialize for Json {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for Json {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        parse(&text).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(text: &str) -> String {
        parse(text).unwrap().to_string()
    }

    #[test]
    fn parses_and_writes_compact() {
        assert_eq!(
            roundtrip(r#" { "a" : [1, 2.5, -0.0, 1e21, true, null], "b": "x\"\né" } "#),
            r#"{"a":[1,2.5,0,1e+21,true,null],"b":"x\"\né"}"#
        );
        assert_eq!(roundtrip("\u{feff}[]"), "[]");
        assert_eq!(roundtrip(r#""😀""#), "\"😀\"");
    }

    #[test]
    fn spans_are_byte_ranges() {
        let doc = parse(r#"{"k": [10, "é"]}"#).unwrap();
        let JsonValue::Object(members) = &doc.value else {
            panic!()
        };
        assert_eq!(members[0].key_span, Span { start: 1, end: 4 });
        let items = members[0].value.as_array().unwrap();
        assert_eq!(items[0].span, Span { start: 7, end: 9 });
        assert_eq!(items[1].span, Span { start: 11, end: 15 });
    }

    #[test]
    fn rejects_invalid_documents() {
        for bad in [
            "",
            "[1,]",
            "{\"a\":1,}",
            "01",
            "1.",
            ".5",
            "+1",
            "[1 2]",
            "'a'",
            "\"\t\"",
            r#""\ud800""#,
            r#""\x""#,
            "nul",
            "{1:2}",
            "[] []",
            "NaN",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} should not parse");
        }
        let deep = "[".repeat(MAX_DEPTH + 1) + &"]".repeat(MAX_DEPTH + 1);
        assert_eq!(parse(&deep).unwrap_err().message, "nesting too deep");
    }

    #[test]
    fn duplicate_keys_resolve_to_the_last() {
        let doc = parse(r#"{"a": 1, "a": 2}"#).unwrap();
        assert_eq!(doc.get("a").and_then(Json::as_f64), Some(2.0));
    }
}
