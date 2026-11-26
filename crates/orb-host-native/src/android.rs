//! What fetching needs of the Android app it runs in, which [`init`] hands
//! over: HTTP goes through the platform's `HttpURLConnection` in the app's
//! JVM ([`url_connection`](crate::url_connection)), and
//! `file:///android_asset/` URLs read the app's assets.

use std::ffi::CString;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::OnceLock;

use jni::objects::{Global, JObject};
use jni::{Env, JavaVM, jni_sig, jni_str};
use orb_host::FetchError;

/// The asset directory `file://` URLs name: `file:///android_asset/a.json`
/// is the app's `assets/a.json`, as in Android's WebView.
const ASSET_ROOT: &str = "/android_asset/";

/// What [`init`] keeps.
struct App {
    vm: JavaVM,
    /// The app's `AssetManager`, held so its native twin stays valid.
    _assets: Global<JObject<'static>>,
    assets: ndk::asset::AssetManager,
}

static APP: OnceLock<App> = OnceLock::new();

/// The app's JVM, once [`init`] gave it.
pub(crate) fn java_vm() -> Option<&'static JavaVM> {
    APP.get().map(|app| &app.vm)
}

/// Gives fetching the app's JVM and `Context`: `env` is the calling
/// thread's `JNIEnv*`, `context` a `jobject` reference to a `Context`, whose
/// application it keeps. From then on HTTP goes through
/// `HttpURLConnection`, with Android's TLS, trust store, proxy settings and
/// the app's network security config, and `file:///android_asset/` URLs
/// read the app's assets. Later calls change nothing. Before it, HTTP fails
/// (but in a build with the `standalone` feature, which fetches through
/// ureq until then).
///
/// # Safety
/// `env` is the calling thread's live `JNIEnv`, and `context` a live
/// reference to a `Context` in it.
#[allow(unsafe_code)]
pub unsafe fn init(
    env: *mut std::ffi::c_void,
    context: *mut std::ffi::c_void,
) -> Result<(), String> {
    if env.is_null() || context.is_null() {
        return Err("android init needs a JNIEnv and a Context".into());
    }
    // SAFETY: the caller's contract.
    let mut unowned = unsafe { jni::EnvUnowned::from_raw(env.cast()) };
    let initialized = unowned
        .with_env(|env| -> Result<(), jni::errors::Error> {
            // SAFETY: the caller's contract; the reference outlives the call.
            let context = unsafe { JObject::from_raw(env, context.cast()) };
            // What's kept is kept for the process's life: the application,
            // since an activity would leak.
            let application = env
                .call_method(
                    &context,
                    jni_str!("getApplicationContext"),
                    jni_sig!("()Landroid/content/Context;"),
                    &[],
                )?
                .l()?;
            let context = if application.is_null() {
                context
            } else {
                application
            };
            let result = keep_app(env, &context);
            if env.exception_check() {
                env.exception_clear();
            }
            result
        })
        .into_outcome();
    match initialized {
        jni::Outcome::Ok(()) => Ok(()),
        jni::Outcome::Err(e) => Err(format!("android init: {e}")),
        jni::Outcome::Panic(_) => Err("android init panicked".into()),
    }
}

/// Keeps the JVM and `context`'s asset manager, unless they're kept.
#[allow(unsafe_code)]
fn keep_app(env: &mut Env<'_>, context: &JObject<'_>) -> Result<(), jni::errors::Error> {
    if APP.get().is_some() {
        return Ok(());
    }
    let manager = env
        .call_method(
            context,
            jni_str!("getAssets"),
            jni_sig!("()Landroid/content/res/AssetManager;"),
            &[],
        )?
        .l()?;
    let manager = env.new_global_ref(manager)?;
    // SAFETY: a live `AssetManager` reference in this thread's env; the
    // global reference kept beside it keeps the native one valid.
    let native =
        unsafe { ndk_sys::AAssetManager_fromJava(env.get_raw().cast(), manager.as_raw().cast()) };
    let native = NonNull::new(native).ok_or(jni::errors::Error::NullPtr("AAssetManager"))?;
    let _ = APP.set(App {
        vm: env.get_java_vm()?,
        _assets: manager,
        // SAFETY: valid while `_assets` holds its Java object.
        assets: unsafe { ndk::asset::AssetManager::from_ptr(native) },
    });
    Ok(())
}

/// The asset a `file://` URL's decoded path names, when it's under
/// `/android_asset/`.
pub(crate) fn asset_name(path: &Path) -> Option<&str> {
    let name = path.to_str()?.strip_prefix(ASSET_ROOT)?;
    Some(name.trim_start_matches('/'))
}

/// Reads the app's asset `name`, `range` of it when given: `None` before
/// [`init`] gave the app's asset manager.
pub(crate) fn read_asset(
    name: &str,
    range: Option<(u64, u64)>,
) -> Option<Result<Vec<u8>, FetchError>> {
    let app = APP.get()?;
    let asset = CString::new(name)
        .ok()
        .and_then(|name| app.assets.open(&name))
        .ok_or(FetchError::NotFound);
    Some(asset.and_then(|asset| crate::read_from(asset, range)))
}
