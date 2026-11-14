//! What fetching needs of the Android app it runs in, which [`init`] hands
//! over: HTTPS verifies servers with Android's own verifier through
//! rustls-platform-verifier, which runs in the app's JVM (its Kotlin half,
//! `org.rustls:rustls-platform-verifier`, must be in the app); plain HTTP
//! asks the app's network security config whether a host may be reached
//! in cleartext; and `file:///android_asset/` URLs read the app's assets.
//! Until then, or in an app without the verifier's Kotlin half, HTTPS
//! trusts the certificates in Android's system store files.

use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::{Mutex, OnceLock};

use jni::objects::{Global, JObject, JValue};
use jni::{Env, JavaVM, jni_sig, jni_str};
use orb_host::FetchError;
use ureq::tls::RootCerts;

/// The asset directory `file://` URLs name: `file:///android_asset/a.json`
/// is the app's `assets/a.json`, as in Android's WebView.
const ASSET_ROOT: &str = "/android_asset/";

/// The class of rustls-platform-verifier's Kotlin half that verifies
/// certificates: without it the verifier fails every one.
const VERIFIER_CLASS: &str = "org.rustls.platformverifier.CertificateVerifier";

/// Android's system certificate store: Conscrypt's, which Android 14 and
/// later update, then the system image's. Neither has the certificates a
/// user installed.
const SYSTEM_STORES: [&str; 2] = [
    "/apex/com.android.conscrypt/cacerts",
    "/system/etc/security/cacerts",
];

/// What [`init`] keeps.
struct App {
    vm: JavaVM,
    /// The app's `AssetManager`, held so its native twin stays valid.
    _assets: Global<JObject<'static>>,
    assets: ndk::asset::AssetManager,
    /// Whether plain HTTP may reach each host asked about so far: the
    /// network security config doesn't change while the app runs.
    cleartext: Mutex<HashMap<String, bool>>,
}

static APP: OnceLock<App> = OnceLock::new();

/// Set once the verifier has the app's JVM and its Kotlin half is there.
static VERIFIER: OnceLock<()> = OnceLock::new();

/// Whether HTTPS verifies with Android's verifier yet.
pub(crate) fn ready() -> bool {
    VERIFIER.get().is_some()
}

/// Gives fetching the app's JVM and `Context`: `env` is the calling
/// thread's `JNIEnv*`, `context` a `jobject` reference to a `Context`, whose
/// application it keeps. From then on HTTPS verifies servers with Android's
/// verifier and the app's network security config, plain HTTP goes only to
/// hosts that config allows in cleartext, and `file:///android_asset/` URLs
/// read the app's assets. Later calls change nothing.
///
/// In an app without the verifier's Kotlin half
/// (`org.rustls:rustls-platform-verifier`), which would fail every
/// certificate, HTTPS keeps trusting the system store's files, and this
/// returns a warning saying so for the caller to show.
///
/// # Safety
/// `env` is the calling thread's live `JNIEnv`, and `context` a live
/// reference to a `Context` in it.
#[allow(unsafe_code)]
pub unsafe fn init(
    env: *mut std::ffi::c_void,
    context: *mut std::ffi::c_void,
) -> Result<Option<String>, String> {
    if env.is_null() || context.is_null() {
        return Err("android init needs a JNIEnv and a Context".into());
    }
    // SAFETY: the caller's contract.
    let mut unowned = unsafe { jni::EnvUnowned::from_raw(env.cast()) };
    let initialized = unowned
        .with_env(|env| -> Result<bool, jni::errors::Error> {
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
            let result = (|| {
                keep_app(env, &context)?;
                let verifier = has_class(env, &context, VERIFIER_CLASS);
                rustls_platform_verifier::android::init_with_env(env, context)?;
                Ok(verifier)
            })();
            if env.exception_check() {
                env.exception_clear();
            }
            result
        })
        .into_outcome();
    match initialized {
        jni::Outcome::Ok(true) => {
            let _ = VERIFIER.set(());
            Ok(None)
        }
        jni::Outcome::Ok(false) => Ok(Some(format!(
            "the app has no {VERIFIER_CLASS} (org.rustls:rustls-platform-verifier), so HTTPS \
             trusts the system store's certificates alone"
        ))),
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
        cleartext: Mutex::default(),
    });
    Ok(())
}

/// Whether `context`'s class loader finds the class `name`. Leaves no Java
/// exception pending.
fn has_class(env: &mut Env<'_>, context: &JObject<'_>, name: &str) -> bool {
    let loaded = (|| -> Result<(), jni::errors::Error> {
        let loader = env
            .call_method(
                context,
                jni_str!("getClassLoader"),
                jni_sig!("()Ljava/lang/ClassLoader;"),
                &[],
            )?
            .l()?;
        let name = env.new_string(name)?;
        env.call_method(
            &loader,
            jni_str!("loadClass"),
            jni_sig!("(Ljava/lang/String;)Ljava/lang/Class;"),
            &[JValue::from(&name)],
        )?;
        Ok(())
    })();
    if env.exception_check() {
        env.exception_clear();
    }
    loaded.is_ok()
}

/// The error of plain HTTP to `url` that the app's network security config
/// refuses, or `None` when it may go (and for HTTPS). Before [`init`], or
/// when Java can't tell, any host may.
pub(crate) fn cleartext_refusal(url: &str) -> Option<FetchError> {
    crate::connect::cleartext_refusal(url, |host| {
        let Some(app) = APP.get() else {
            return true;
        };
        let mut known = app.cleartext.lock().expect("cleartext poisoned");
        *known
            .entry(host.to_string())
            .or_insert_with(|| cleartext_permitted(&app.vm, host))
    })
}

/// `NetworkSecurityPolicy.isCleartextTrafficPermitted(host)`, on a thread
/// that stays attached to the JVM; true when Java fails to answer.
fn cleartext_permitted(vm: &JavaVM, host: &str) -> bool {
    vm.attach_current_thread(|env| -> Result<bool, jni::errors::Error> {
        let class = env.find_class(jni_str!("android/security/NetworkSecurityPolicy"))?;
        let policy = env
            .call_static_method(
                &class,
                jni_str!("getInstance"),
                jni_sig!("()Landroid/security/NetworkSecurityPolicy;"),
                &[],
            )?
            .l()?;
        let host = env.new_string(host)?;
        let permitted = env.call_method(
            &policy,
            jni_str!("isCleartextTrafficPermitted"),
            jni_sig!("(Ljava/lang/String;)Z"),
            &[JValue::from(&host)],
        );
        if env.exception_check() {
            env.exception_clear();
        }
        permitted?.z()
    })
    .unwrap_or(true)
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

/// The system store's certificates: the PEM files of the first of
/// [`SYSTEM_STORES`] that has any.
pub(crate) fn system_roots() -> RootCerts {
    crate::connect::certificates_in(&SYSTEM_STORES.map(Path::new))
}
