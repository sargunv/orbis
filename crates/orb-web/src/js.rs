//! Reading and building plain JS objects: the options, messages and
//! events orb-web trades with the page and its workers.

use js_sys::{Object, Reflect};
use wasm_bindgen::JsValue;

/// `object[key]`: `undefined` where it has none, or isn't an object.
pub(crate) fn get(object: &JsValue, key: &str) -> JsValue {
    Reflect::get(object, &key.into()).unwrap_or(JsValue::UNDEFINED)
}

/// Sets `object[key]`; a frozen object keeps what it had.
pub(crate) fn set(object: &Object, key: &str, value: &JsValue) {
    let _ = Reflect::set(object, &key.into(), value);
}

/// An object of `fields`.
pub(crate) fn object(fields: &[(&str, JsValue)]) -> Object {
    let object = Object::new();
    for (key, value) in fields {
        set(&object, key, value);
    }
    object
}
