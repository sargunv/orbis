//! [`Storage`] in the origin private file system: keys are paths of
//! directories and files under its root.

use js_sys::{Reflect, Uint8Array};
use orb_host::{Storage, StorageDone};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{FileSystemDirectoryHandle as Dir, FileSystemFileHandle};

/// The origin private file system (`navigator.storage.getDirectory()`),
/// on the page or in a worker.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OpfsStorage;

impl Storage for OpfsStorage {
    fn read(&self, key: &str, range: Option<(u64, u64)>, done: StorageDone<Option<Vec<u8>>>) {
        let key = key.to_string();
        wasm_bindgen_futures::spawn_local(
            async move { done(read(&key, range).await.map_err(text)) },
        );
    }
    fn write(&self, key: &str, bytes: Vec<u8>, append: bool, done: StorageDone<()>) {
        let key = key.to_string();
        wasm_bindgen_futures::spawn_local(async move {
            done(write(&key, &bytes, append).await.map_err(text))
        });
    }
    fn remove(&self, key: &str, done: StorageDone<()>) {
        let key = key.to_string();
        wasm_bindgen_futures::spawn_local(async move { done(remove(&key).await.map_err(text)) });
    }
    fn list(&self, prefix: &str, done: StorageDone<Vec<String>>) {
        let prefix = prefix.to_string();
        wasm_bindgen_futures::spawn_local(async move { done(list(&prefix).await.map_err(text)) });
    }
}

fn text(e: JsValue) -> String {
    format!("storage: {e:?}")
}

fn not_found(e: &JsValue) -> bool {
    Reflect::get(e, &"name".into())
        .ok()
        .and_then(|n| n.as_string())
        .is_some_and(|n| n == "NotFoundError" || n == "TypeMismatchError")
}

async fn root() -> Result<Dir, JsValue> {
    let navigator = Reflect::get(&js_sys::global(), &"navigator".into())?;
    let storage: web_sys::StorageManager =
        Reflect::get(&navigator, &"storage".into())?.dyn_into()?;
    JsFuture::from(storage.get_directory()).await?.dyn_into()
}

/// The directory `path` names, made if `create`; `None` if absent.
async fn dir(path: &str, create: bool) -> Result<Option<Dir>, JsValue> {
    let mut dir = root().await?;
    for name in path.split('/').filter(|n| !n.is_empty()) {
        let options = web_sys::FileSystemGetDirectoryOptions::new();
        options.set_create(create);
        match JsFuture::from(dir.get_directory_handle_with_options(name, &options)).await {
            Ok(next) => dir = next.dyn_into()?,
            Err(e) if not_found(&e) => return Ok(None),
            Err(e) => return Err(e),
        }
    }
    Ok(Some(dir))
}

/// A key's directory and file name.
fn split(key: &str) -> (&str, &str) {
    key.rsplit_once('/').unwrap_or(("", key))
}

async fn file(key: &str, create: bool) -> Result<Option<FileSystemFileHandle>, JsValue> {
    let (parent, name) = split(key);
    let Some(dir) = dir(parent, create).await? else {
        return Ok(None);
    };
    let options = web_sys::FileSystemGetFileOptions::new();
    options.set_create(create);
    match JsFuture::from(dir.get_file_handle_with_options(name, &options)).await {
        Ok(handle) => Ok(Some(handle.dyn_into()?)),
        Err(e) if not_found(&e) => Ok(None),
        Err(e) => Err(e),
    }
}

async fn read(key: &str, range: Option<(u64, u64)>) -> Result<Option<Vec<u8>>, JsValue> {
    let Some(handle) = file(key, false).await? else {
        return Ok(None);
    };
    let blob: web_sys::Blob = JsFuture::from(handle.get_file()).await?.dyn_into()?;
    let blob = match range {
        None => blob,
        Some((offset, length)) => {
            let end = offset + length;
            if end as f64 > blob.size() {
                return Err(JsValue::from_str("range past the end"));
            }
            blob.slice_with_f64_and_f64(offset as f64, end as f64)?
        }
    };
    let buffer = JsFuture::from(blob.array_buffer()).await?;
    Ok(Some(Uint8Array::new(&buffer).to_vec()))
}

async fn write(key: &str, bytes: &[u8], append: bool) -> Result<(), JsValue> {
    let handle = file(key, true).await?.ok_or("storage: no file")?;
    let options = web_sys::FileSystemCreateWritableOptions::new();
    options.set_keep_existing_data(append);
    let stream: web_sys::FileSystemWritableFileStream =
        JsFuture::from(handle.create_writable_with_options(&options))
            .await?
            .dyn_into()?;
    if append {
        let file: web_sys::Blob = JsFuture::from(handle.get_file()).await?.dyn_into()?;
        JsFuture::from(stream.seek_with_f64(file.size())?).await?;
    }
    // From an array of its own: WebKit writes a view's whole buffer, here
    // all of the wasm memory, rather than the view's bytes.
    let bytes = Uint8Array::from(bytes);
    JsFuture::from(stream.write_with_js_u8_array(&bytes)?).await?;
    // Closing publishes the writes: until then readers see the old file.
    JsFuture::from(stream.close()).await?;
    Ok(())
}

async fn remove(key: &str) -> Result<(), JsValue> {
    let (parent, name) = split(key);
    let Some(dir) = dir(parent, false).await? else {
        return Ok(());
    };
    let options = web_sys::FileSystemRemoveOptions::new();
    options.set_recursive(true);
    match JsFuture::from(dir.remove_entry_with_options(name, &options)).await {
        Ok(_) => Ok(()),
        Err(e) if not_found(&e) => Ok(()),
        Err(e) => Err(e),
    }
}

async fn list(prefix: &str) -> Result<Vec<String>, JsValue> {
    let Some(dir) = dir(prefix, false).await? else {
        return Ok(Vec::new());
    };
    let names = dir.keys();
    let mut out = Vec::new();
    loop {
        let next = JsFuture::from(names.next()?).await?;
        if Reflect::get(&next, &"done".into())?.as_bool() == Some(true) {
            return Ok(out);
        }
        if let Some(name) = Reflect::get(&next, &"value".into())?.as_string() {
            out.push(name);
        }
    }
}
