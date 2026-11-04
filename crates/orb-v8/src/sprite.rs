use std::collections::HashSet;

use orb_expr::json::Json;

use crate::Cx;

/// One sprite sheet. The default namespace exposes unprefixed image names;
/// other namespaces expose `id:name`, following GL JS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sprite {
    pub id: String,
    pub url: String,
}

pub(crate) fn parse(cx: &mut Cx, value: Option<&Json>) -> Vec<Sprite> {
    let Some(value) = value else {
        return Vec::new();
    };
    if let Some(url) = value.as_str() {
        return vec![Sprite {
            id: "default".into(),
            url: url.into(),
        }];
    }
    let Some(items) = cx.array(value, "sprite") else {
        return Vec::new();
    };
    let mut ids = HashSet::new();
    items
        .iter()
        .enumerate()
        .filter_map(|(i, item)| {
            let path = format!("sprite[{i}]");
            cx.object(item, &path)?;
            let id = item
                .get("id")
                .and_then(|v| cx.string(v, &format!("{path}.id")));
            let url = item
                .get("url")
                .and_then(|v| cx.string(v, &format!("{path}.url")));
            let (Some(id), Some(url)) = (id, url) else {
                cx.error(&path, item.span, "sprite requires id and url");
                return None;
            };
            if !ids.insert(id.clone()) {
                cx.error(&path, item.span, "duplicate sprite id");
            }
            Some(Sprite { id, url })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn parses_both_sprite_forms_and_rejects_duplicate_ids() {
        let style = |sprite: &str| {
            format!(r#"{{"version":8,"sprite":{sprite},"sources":{{}},"layers":[]}}"#)
        };
        let single = crate::parse(&style(r#""https://example.com/sprite""#)).unwrap();
        assert_eq!(single.sprites[0].id, "default");
        let array = r#"[{"id":"one","url":"test://one"},{"id":"two","url":"test://two"}]"#;
        assert_eq!(crate::parse(&style(array)).unwrap().sprites.len(), 2);
        assert!(crate::parse(&style(&array.replace("two", "one"))).is_err());
    }
}
