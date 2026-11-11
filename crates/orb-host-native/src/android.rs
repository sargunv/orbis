//! HTTPS on Android: rustls-platform-verifier asks Android's trust manager
//! through the app's JVM (its Kotlin half must be in the app,
//! `org.rustls:rustls-platform-verifier`), so it needs the app's `Context`
//! first. Until [`init`] has it, HTTPS trusts Mozilla's roots instead.

use std::sync::atomic::{AtomicBool, Ordering};

static READY: AtomicBool = AtomicBool::new(false);

/// Whether HTTPS verifies with Android's trust store yet.
pub(crate) fn ready() -> bool {
    READY.load(Ordering::Acquire)
}

/// Gives HTTPS the app's JVM and `Context`, from then on verifying servers
/// with Android's trust store and the app's network security config: `env`
/// is the calling thread's `JNIEnv*`, `context` a `jobject` reference to a
/// `Context` (the application, say). Later calls change nothing.
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
            let context = unsafe { jni::objects::JObject::from_raw(env, context.cast()) };
            rustls_platform_verifier::android::init_with_env(env, context)
        })
        .into_outcome();
    match initialized {
        jni::Outcome::Ok(()) => {
            READY.store(true, Ordering::Release);
            Ok(())
        }
        jni::Outcome::Err(e) => Err(format!("android init: {e}")),
        jni::Outcome::Panic(_) => Err("android init panicked".into()),
    }
}
