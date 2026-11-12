//! What the reader needs of a schema's SQL: a table's columns, which one
//! aliases the rowid and the indexes its constraints make, and an index's
//! columns. Anything else in the SQL is skipped.

use crate::{Error, Result};

/// A table with a rowid.
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    pub name: String,
    /// Its B-tree's root page.
    pub root: u32,
    /// Its columns' names, in order.
    pub columns: Vec<String>,
    /// The column that is the rowid (`INTEGER PRIMARY KEY`): the row
    /// stores null there.
    pub rowid_alias: Option<usize>,
    /// The columns of each index its constraints make
    /// (`sqlite_autoindex_<table>_<n>`), in order.
    pub(crate) constraint_indexes: Vec<Vec<String>>,
}

impl Table {
    /// The index of the column named `name` (ASCII case-insensitively).
    pub fn column(&self, name: &str) -> Option<usize> {
        self.columns
            .iter()
            .position(|c| c.eq_ignore_ascii_case(name))
    }
}

/// An index: its entries are its columns' values, then the row's rowid.
#[derive(Clone, Debug, PartialEq)]
pub struct Index {
    pub name: String,
    pub table: String,
    pub root: u32,
    pub columns: Vec<String>,
    /// Whether each column sorts descending.
    pub(crate) descending: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    /// A keyword or an identifier, quoted or not.
    Word(String),
    Quoted(String),
    Open,
    Close,
    Comma,
    Other,
}

fn tokens(sql: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut chars = sql.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        match c {
            c if c.is_whitespace() => {}
            '(' => out.push(Token::Open),
            ')' => out.push(Token::Close),
            ',' => out.push(Token::Comma),
            '-' if chars.peek().is_some_and(|&(_, n)| n == '-') => {
                for (_, n) in chars.by_ref() {
                    if n == '\n' {
                        break;
                    }
                }
            }
            '/' if chars.peek().is_some_and(|&(_, n)| n == '*') => {
                chars.next();
                let mut star = false;
                for (_, n) in chars.by_ref() {
                    if star && n == '/' {
                        break;
                    }
                    star = n == '*';
                }
            }
            '"' | '`' | '[' | '\'' => {
                let close = if c == '[' { ']' } else { c };
                let mut text = String::new();
                while let Some((_, n)) = chars.next() {
                    if n == close {
                        // A doubled quote is one quote in the name.
                        if close != ']' && chars.peek().is_some_and(|&(_, m)| m == close) {
                            chars.next();
                            text.push(close);
                            continue;
                        }
                        break;
                    }
                    text.push(n);
                }
                out.push(if c == '\'' {
                    Token::Other
                } else {
                    Token::Quoted(text)
                });
            }
            c if c.is_alphanumeric() || c == '_' || c == '$' => {
                let mut end = at + c.len_utf8();
                while let Some(&(next, n)) = chars.peek() {
                    if !(n.is_alphanumeric() || n == '_' || n == '$') {
                        break;
                    }
                    end = next + n.len_utf8();
                    chars.next();
                }
                out.push(Token::Word(sql[at..end].to_string()));
            }
            _ => out.push(Token::Other),
        }
    }
    out
}

impl Token {
    fn name(&self) -> Option<&str> {
        match self {
            Token::Word(w) | Token::Quoted(w) => Some(w),
            _ => None,
        }
    }

    fn is(&self, keyword: &str) -> bool {
        matches!(self, Token::Word(w) if w.eq_ignore_ascii_case(keyword))
    }
}

/// The top-level, comma-separated items of the parenthesized list that
/// starts at `tokens[0]`, and the tokens after it.
fn list(tokens: &[Token]) -> Result<(Vec<&[Token]>, &[Token])> {
    if tokens.first() != Some(&Token::Open) {
        return Err(Error::unsupported("schema SQL without a column list"));
    }
    let (mut depth, mut start, mut items) = (0usize, 1, Vec::new());
    for (i, token) in tokens.iter().enumerate() {
        match token {
            Token::Open => depth += 1,
            Token::Close => {
                depth -= 1;
                if depth == 0 {
                    items.push(&tokens[start..i]);
                    return Ok((items, &tokens[i + 1..]));
                }
            }
            Token::Comma if depth == 1 => {
                items.push(&tokens[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    Err(Error::corrupt("schema SQL with an unclosed list"))
}

/// The names in an item list such as `(a, b DESC)`, with whether each
/// sorts descending.
fn names(items: &[&[Token]]) -> Vec<(String, bool)> {
    items
        .iter()
        .filter_map(|item| {
            let name = item.first()?.name()?.to_string();
            Some((name, item.iter().any(|t| t.is("DESC"))))
        })
        .collect()
}

/// The columns and constraint indexes of `CREATE TABLE … (…)`.
pub(crate) fn table(name: &str, root: u32, sql: &str) -> Result<Table> {
    let tokens = tokens(sql);
    let open = tokens
        .iter()
        .position(|t| *t == Token::Open)
        .ok_or_else(|| Error::unsupported(format!("table {name} isn't made by a column list")))?;
    let (items, after) = list(&tokens[open..])?;
    if after.iter().any(|t| t.is("WITHOUT")) {
        return Err(Error::unsupported(format!("table {name} is WITHOUT ROWID")));
    }
    let mut table = Table {
        name: name.to_string(),
        root,
        columns: Vec::new(),
        rowid_alias: None,
        constraint_indexes: Vec::new(),
    };
    // Each column's declared type.
    let mut types: Vec<String> = Vec::new();
    for item in items {
        let Some(first) = item.first() else { continue };
        let constraint = ["CONSTRAINT", "PRIMARY", "UNIQUE", "CHECK", "FOREIGN"]
            .iter()
            .any(|k| first.is(k));
        if constraint {
            // `[CONSTRAINT name] PRIMARY KEY (…)` or `UNIQUE (…)`.
            let Some(at) = item.iter().position(|t| t.is("PRIMARY") || t.is("UNIQUE")) else {
                continue;
            };
            let Some(open) = item[at..].iter().position(|t| *t == Token::Open) else {
                continue;
            };
            let (columns, _) = list(&item[at + open..])?;
            let named = names(&columns);
            // A one-column `INTEGER` primary key is the rowid, not an index.
            let alias = match &named[..] {
                [(column, false)] if item[at].is("PRIMARY") => table
                    .column(column)
                    .filter(|&c| types[c].eq_ignore_ascii_case("INTEGER")),
                _ => None,
            };
            match alias {
                Some(c) => table.rowid_alias = Some(c),
                None => table
                    .constraint_indexes
                    .push(named.into_iter().map(|(n, _)| n).collect()),
            }
            continue;
        }
        let Some(column) = first.name() else {
            return Err(Error::corrupt(format!(
                "table {name} has an unnamed column"
            )));
        };
        let column = column.to_string();
        let index = table.columns.len();
        table.columns.push(column.clone());
        let words: Vec<&Token> = item[1..].iter().collect();
        let declared: Vec<&str> = words
            .iter()
            .take_while(|t| {
                ![
                    "CONSTRAINT",
                    "PRIMARY",
                    "UNIQUE",
                    "NOT",
                    "NULL",
                    "DEFAULT",
                    "CHECK",
                ]
                .iter()
                .any(|k| t.is(k))
                    && !t.is("REFERENCES")
                    && !t.is("COLLATE")
                    && !t.is("GENERATED")
                    && !t.is("AS")
            })
            .filter_map(|t| t.name())
            .collect();
        types.push(declared.join(" "));
        let primary = words.iter().position(|t| t.is("PRIMARY"));
        if let Some(at) = primary {
            let descending = words.get(at + 2).is_some_and(|t| t.is("DESC"));
            if declared.len() == 1 && declared[0].eq_ignore_ascii_case("INTEGER") && !descending {
                table.rowid_alias = Some(index);
            } else {
                table.constraint_indexes.push(vec![column.clone()]);
            }
        }
        if words.iter().any(|t| t.is("UNIQUE")) {
            table.constraint_indexes.push(vec![column]);
        }
    }
    Ok(table)
}

/// The columns of `CREATE [UNIQUE] INDEX … ON table (…)`.
pub(crate) fn index(name: &str, table: &str, root: u32, sql: &str) -> Result<Index> {
    let tokens = tokens(sql);
    let on = tokens
        .iter()
        .position(|t| t.is("ON"))
        .ok_or_else(|| Error::corrupt(format!("index {name} names no table")))?;
    let open = tokens[on..]
        .iter()
        .position(|t| *t == Token::Open)
        .ok_or_else(|| Error::corrupt(format!("index {name} has no columns")))?;
    let (items, _) = list(&tokens[on + open..])?;
    let (columns, descending) = names(&items).into_iter().unzip();
    Ok(Index {
        name: name.to_string(),
        table: table.to_string(),
        root,
        columns,
        descending,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_table_names_its_columns_rowid_and_constraint_indexes() {
        let t = table(
            "tiles",
            2,
            "CREATE TABLE tiles (id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, \
             url_template TEXT NOT NULL, \"pixel ratio\" INTEGER, z INTEGER DEFAULT (1 + 2), \
             -- a comment, (with a paren\n data BLOB UNIQUE, UNIQUE (url_template, z))",
        )
        .unwrap();
        assert_eq!(
            t.columns,
            ["id", "url_template", "pixel ratio", "z", "data"]
        );
        assert_eq!(t.rowid_alias, Some(0));
        assert_eq!(
            t.constraint_indexes,
            [
                vec!["data".to_string()],
                vec!["url_template".into(), "z".into()]
            ]
        );
        let keyed = table("m", 2, "CREATE TABLE m (name TEXT PRIMARY KEY, value)").unwrap();
        assert_eq!(keyed.rowid_alias, None);
        assert_eq!(keyed.constraint_indexes, [vec!["name".to_string()]]);
        let int_key = table(
            "k",
            2,
            "CREATE TABLE k (a INT PRIMARY KEY, b, PRIMARY KEY (a))",
        )
        .unwrap();
        assert_eq!(int_key.rowid_alias, None, "INT isn't INTEGER");
        assert_eq!(int_key.constraint_indexes.len(), 2);
        let table_key = table("k", 2, "CREATE TABLE k (a INTEGER, b, PRIMARY KEY(a))").unwrap();
        assert_eq!(table_key.rowid_alias, Some(0));
        assert!(table_key.constraint_indexes.is_empty());
        assert!(table("w", 2, "CREATE TABLE w (a PRIMARY KEY) WITHOUT ROWID").is_err());
    }

    #[test]
    fn an_index_names_its_columns() {
        let i = index(
            "tile_index",
            "tiles",
            3,
            "CREATE UNIQUE INDEX tile_index on tiles (zoom_level, [tile_column], tile_row DESC)",
        )
        .unwrap();
        assert_eq!(i.columns, ["zoom_level", "tile_column", "tile_row"]);
        assert_eq!(i.descending, [false, false, true]);
    }
}
