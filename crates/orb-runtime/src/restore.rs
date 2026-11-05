//! What the app set on a map at run time, beside its style document:
//! feature state and the images it added. A map made afresh takes it back:
//! the same map on a new GPU, or a new one in another wasm instance (the
//! web's switch to WebGL2).

use std::fmt::Write as _;

use crate::{Map, StyleImage, Transaction};

/// A map's feature state and added images.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct AppState {
    /// The feature state, as GL JS `setFeatureState` calls (JSON).
    feature_state: String,
    images: Vec<(String, StyleImage)>,
}

impl Map {
    /// The feature state and added images now.
    pub fn app_state(&self) -> AppState {
        let Some(style) = &self.style else {
            return AppState::default();
        };
        let mut calls = Vec::new();
        for op in style.states.carried() {
            let orb_style::FeatureStateOp::Set {
                source,
                source_layer,
                id,
                state,
            } = op
            else {
                continue;
            };
            let Some(index) = style.doc.source_index(source) else {
                continue;
            };
            let string = |s: &str| orb_expr::Value::String(s.into()).to_json_string();
            let mut target = format!(
                r#"{{"source":{}"#,
                string(&style.doc.style().sources[index].id)
            );
            if let Some(layer) = &source_layer {
                let _ = write!(target, r#","sourceLayer":{}"#, string(layer));
            }
            let _ = write!(target, r#","id":{}}}"#, string(&id));
            let state: Vec<String> = state
                .iter()
                .map(|(k, v)| format!("{}:{}", string(k), v.to_json_string()))
                .collect();
            calls.push(format!(
                r#"["setFeatureState",{target},{{{}}}]"#,
                state.join(",")
            ));
        }
        AppState {
            feature_state: format!("[{}]", calls.join(",")),
            images: (style.added_images.iter())
                .map(|(name, image)| (name.clone(), (**image).clone()))
                .collect(),
        }
    }

    /// Sets `state` again: now, or once the style loads.
    pub fn restore(&mut self, state: AppState) {
        if self.style.is_none() {
            self.restoring = Some(state);
            return;
        }
        for (name, image) in state.images {
            let _ = self.add_image(&name, image);
        }
        if let Ok(tx) = Transaction::from_commands(&state.feature_state)
            && !tx.is_empty()
        {
            let _ = self.edit(&tx);
        }
    }
}
