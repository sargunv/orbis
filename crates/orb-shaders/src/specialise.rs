//! A stage's WGSL specialised to the values its pipeline gives its boolean
//! override constants: each written in as a literal, its declaration
//! dropped, and every `if` whose condition that decides taken or dropped,
//! with what follows a taken `return` in its block.
//!
//! The source has to leave that code out: Apple's shader compiler takes as
//! long over code no branch reaches as over code that runs, and a stage
//! with any texture in its source pays it about 30 ms on an M6, whether a
//! branch reaches it or not. Override constants alone leave both to the
//! driver (naga and Tint write the branch, its condition constant), so a
//! plane variant compiled its terrain's DEM reads all the same.
//!
//! It reads the build's stripped WGSL (`strip` in build.rs): no comments
//! but the `//planes ` lines of a shader without clip planes, which hold
//! no braces (the test of every variant checks), so no braces but the
//! code's. It folds an `if C {` whose `C` is a literal, its negation, or
//! `||` or `&&` of terms of which the literals decide it; other uses of a
//! constant keep its literal for the driver to fold.

/// `source` with each of `constants` that is a boolean override written
/// in, and the code that decides dropped; `None` if none is one. Those
/// constants are no longer the stage's to set.
pub fn specialise(source: &str, constants: &[(&str, f64)]) -> Option<String> {
    let mut out = None;
    for &(name, value) in constants {
        let text = out.as_deref().unwrap_or(source);
        let declaration = format!("override {name}: bool");
        let Some(at) = find_word(text, &declaration, 0) else {
            continue;
        };
        let end = at + text[at..].find(';').expect("a declaration ends") + 1;
        let literal = if value != 0.0 { "true" } else { "false" };
        let mut written = String::with_capacity(text.len());
        written.push_str(&replace_word(&text[..at], name, literal));
        written.push_str(&replace_word(&text[end..], name, literal));
        out = Some(written);
    }
    out.map(|text| fold(&text))
}

fn is_word(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Where `word` next stands alone in `text` from `from`.
fn find_word(text: &str, word: &str, from: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut at = from;
    while let Some(i) = text[at..].find(word) {
        let i = at + i;
        let before = i == 0 || !is_word(bytes[i - 1]);
        let after = bytes.get(i + word.len()).is_none_or(|&c| !is_word(c));
        if before && after {
            return Some(i);
        }
        at = i + word.len();
    }
    None
}

/// `text` with `word` standing alone as `with`, but where it names a
/// member (`a.word`).
fn replace_word(text: &str, word: &str, with: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    while let Some(i) = find_word(text, word, at) {
        out.push_str(&text[at..i]);
        let member = text[..i].trim_end().ends_with('.');
        out.push_str(if member { word } else { with });
        at = i + word.len();
    }
    out.push_str(&text[at..]);
    out
}

/// The index past the `}` closing the block whose `{` is at `open`.
fn block_end(text: &str, open: usize) -> usize {
    let mut depth = 0;
    for (i, c) in text[open..].bytes().enumerate() {
        match c {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return open + i + 1;
                }
            }
            _ => {}
        }
    }
    panic!("an unclosed block")
}

/// The `}` closing the block that holds `at`.
fn enclosing_close(text: &str, at: usize) -> usize {
    let mut depth = 0;
    for (i, c) in text[at..].bytes().enumerate() {
        match c {
            b'{' => depth += 1,
            b'}' if depth == 0 => return at + i,
            b'}' => depth -= 1,
            _ => {}
        }
    }
    panic!("a statement outside a block")
}

/// `text` with every `if` whose condition its literals decide folded,
/// until none is left.
fn fold(text: &str) -> String {
    let mut text = text.to_owned();
    let mut from = 0;
    while let Some(at) = find_word(&text, "if", from) {
        match fold_at(&text, at) {
            Some(folded) => text = folded,
            None => from = at + 2,
        }
    }
    text
}

/// `text` with the `if` statement at `at` folded, if its condition
/// decides it, or its condition without the terms that don't count.
fn fold_at(text: &str, at: usize) -> Option<String> {
    let open = at + text[at..].find('{')?;
    let condition = text[at + 2..open].trim();
    let taken = match simplify(condition) {
        Condition::Known(taken) => taken,
        Condition::Open(simpler) if simpler != condition => {
            return Some(format!("{}if {simpler} {}", &text[..at], &text[open..]));
        }
        Condition::Open(_) => return None,
    };
    let then_end = block_end(text, open);
    // `else { … }` or `else if …` after it.
    let (otherwise, end) = match else_at(text, then_end) {
        Some(start) => {
            let end = if text[start..].starts_with('{') {
                block_end(text, start)
            } else {
                chain_end(text, start)
            };
            (Some(&text[start..end]), end)
        }
        None => (None, then_end),
    };
    let kept = match (taken, otherwise) {
        (true, _) => &text[open..then_end],
        (false, Some(otherwise)) => otherwise,
        (false, None) => "",
    };
    // An `else if` stands for the block after its `else`.
    let before = text[..at].trim_end();
    let in_else =
        (before.strip_suffix("else")).is_some_and(|b| b.bytes().last().is_none_or(|c| !is_word(c)));
    let mut out = String::with_capacity(text.len());
    if in_else {
        if kept.is_empty() {
            // `… else if false { … }`: the `else` goes too.
            out.push_str(&text[..before.len() - 4]);
        } else if kept.starts_with('{') {
            out.push_str(&text[..at]);
            out.push_str(kept);
        } else {
            out.push_str(&text[..at]);
            out.push('{');
            out.push_str(kept);
            out.push('}');
        }
        out.push_str(&text[end..]);
    } else if returns(kept) {
        // It returns: the rest of its block is never reached.
        out.push_str(&text[..at]);
        out.push_str(kept);
        out.push_str(&text[enclosing_close(text, end)..]);
    } else {
        out.push_str(&text[..at]);
        out.push_str(kept);
        out.push_str(&text[end..]);
    }
    Some(out)
}

/// Where the statement after an `else` at `at` starts, if one is there.
fn else_at(text: &str, at: usize) -> Option<usize> {
    let rest = text[at..].trim_start();
    let tail = rest.strip_prefix("else")?;
    if tail.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    let start = text.len() - tail.len();
    Some(start + (tail.len() - tail.trim_start().len()))
}

/// The end of the `if` chain starting at `at`.
fn chain_end(text: &str, at: usize) -> usize {
    let open = at + text[at..].find('{').expect("an if's block");
    let end = block_end(text, open);
    match else_at(text, end) {
        Some(start) if text[start..].starts_with('{') => block_end(text, start),
        Some(start) => chain_end(text, start),
        None => end,
    }
}

/// Whether a block's last statement at its own depth is a `return`.
fn returns(block: &str) -> bool {
    let Some(inner) = (block.trim().strip_prefix('{')).and_then(|b| b.strip_suffix('}')) else {
        return false;
    };
    let Some(last) = inner.trim_end().strip_suffix(';') else {
        return false;
    };
    // The last statement starts after the last `;` or block at depth 0.
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in last.bytes().enumerate() {
        match c {
            b'{' | b'(' => depth += 1,
            b'}' | b')' => {
                depth -= 1;
                if depth == 0 && c == b'}' {
                    start = i + 1;
                }
            }
            b';' if depth == 0 => start = i + 1,
            _ => {}
        }
    }
    let statement = last[start..].trim_start();
    find_word(statement, "return", 0) == Some(0)
}

/// What a condition comes to.
enum Condition {
    /// Its literals decide it.
    Known(bool),
    /// They don't: the terms of it that count.
    Open(String),
}

/// What condition `c` comes to, given its literals.
fn simplify(c: &str) -> Condition {
    let c = strip_parens(c.trim());
    match c {
        "true" => return Condition::Known(true),
        "false" => return Condition::Known(false),
        _ => {}
    }
    if let Some(inner) = c.strip_prefix('!')
        && let inner = inner.trim()
        && (matches!(inner, "true" | "false") || is_parenthesised(inner))
        && let Condition::Known(b) = simplify(inner)
    {
        return Condition::Known(!b);
    }
    // WGSL takes no `||` beside an `&&` without parentheses.
    for (op, short) in [("||", true), ("&&", false)] {
        let terms = split_top(c, op);
        if terms.len() < 2 {
            continue;
        }
        let mut open = Vec::new();
        for term in terms {
            match simplify(term) {
                Condition::Known(b) if b == short => return Condition::Known(short),
                Condition::Known(_) => {}
                Condition::Open(_) => open.push(term.trim()),
            }
        }
        return match open.as_slice() {
            [] => Condition::Known(!short),
            [one] => Condition::Open((*one).to_owned()),
            _ => Condition::Open(open.join(&format!(" {op} "))),
        };
    }
    Condition::Open(c.to_owned())
}

/// Whether `c` is one parenthesised expression.
fn is_parenthesised(c: &str) -> bool {
    if !c.starts_with('(') || !c.ends_with(')') {
        return false;
    }
    let mut depth = 0;
    for (i, b) in c.bytes().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => depth -= 1,
            _ => {}
        }
        if depth == 0 && i + 1 < c.len() {
            return false;
        }
    }
    true
}

fn strip_parens(mut c: &str) -> &str {
    while is_parenthesised(c) {
        c = c[1..c.len() - 1].trim();
    }
    c
}

/// `c`'s terms around `op` outside parentheses and brackets.
fn split_top<'a>(c: &'a str, op: &str) -> Vec<&'a str> {
    let mut terms = Vec::new();
    let (mut depth, mut start, mut i) = (0i32, 0, 0);
    let bytes = c.as_bytes();
    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            _ if depth == 0 && c[i..].starts_with(op) => {
                terms.push(&c[start..i]);
                i += op.len();
                start = i;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    terms.push(&c[start..]);
    terms
}

#[cfg(test)]
mod tests {
    use super::specialise;

    const SOURCE: &str = "override plane: bool = false;
fn ground(p: f32) -> f32 {
if plane {
return 0.0;
}
let t = textureLoad(dem, vec2<i32>(0), 0, 0);
return t.r * p;
}
fn on_globe() -> bool {
return !plane && frame.globe > 0.0;
}
fn visible(c: f32) -> f32 {
if plane || c <= 0.0 {
return 1.0;
}
if c > 1.0 && !plane {
return 2.0;
} else if plane {
return 3.0;
} else {
return 4.0;
}
}
";

    #[test]
    fn a_plane_variant_leaves_out_what_plane_decides() {
        let plane = specialise(SOURCE, &[("plane", 1.0)]).expect("plane is a bool");
        assert!(!plane.contains("override"), "{plane}");
        assert!(!plane.contains("textureLoad"), "{plane}");
        assert!(
            plane.contains("return !true && frame.globe > 0.0;"),
            "{plane}"
        );
        assert!(!plane.contains("return 2.0"), "{plane}");
        assert!(!plane.contains("c <= 0.0"), "{plane}");
        assert!(!plane.contains("return 4.0"), "{plane}");
    }

    #[test]
    fn a_full_variant_keeps_what_runs() {
        let full = specialise(SOURCE, &[("plane", 0.0)]).expect("plane is a bool");
        assert!(full.contains("textureLoad"), "{full}");
        assert!(!full.contains("return 0.0"), "{full}");
        assert!(full.contains("if c <= 0.0 {"), "{full}");
        assert!(full.contains("if c > 1.0 {"), "{full}");
        assert!(!full.contains("return 3.0"), "{full}");
        assert!(full.contains("return 4.0"), "{full}");
    }

    #[test]
    fn other_constants_leave_it_as_it_is() {
        assert!(specialise(SOURCE, &[("elevation_stops", 2.0)]).is_none());
    }
}
