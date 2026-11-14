//! The entry point from android-activity's `NativeActivity` glue.
//! `cargo xtask android` packages the library with the web demo's data as
//! assets, which maps read in place (`file:///android_asset/`).
//!
//! Intent extras stand in for arguments: `--es style tokyo` draws the
//! committed Tokyo extract instead of live OpenFreeMap Liberty over Tokyo
//! Station, any other `style` is a URL; `--es at
//! lon,lat,zoom[,bearing,pitch]`;
//! `--es backend gl|vulkan` makes the device on that backend
//! (`ORB_BACKEND`); `--el lose <ms>` loses the GPU that long after loading;
//! `--ez leakcheck true` first makes, draws and drops 20 maps. For example
//! `adb -s "$(scripts/android-serial)" shell am start -n
//! dev.orbis.demo/.MainActivity --es style tokyo`. Everything is logged
//! under the tag `Orbis`.

use std::ffi::CString;
use std::sync::OnceLock;
use std::time::Instant;

use jni::objects::{JObject, JString, JValue};
use jni::refs::Global;
use jni::{jni_sig, jni_str};
use winit::event_loop::EventLoop;
use winit::platform::android::EventLoopBuilderExtAndroid;
use winit::platform::android::activity::{AndroidApp, ndk_sys};

use orb_api::Geodetic;

use crate::{Options, Wake};

/// Live Liberty over Tokyo Station.
const LIBERTY: &str = "https://tiles.openfreemap.org/styles/liberty";
const TOKYO: (f64, f64, f64) = (139.767, 35.681, 13.0);

/// The running activity.
static APP: OnceLock<AndroidApp> = OnceLock::new();

/// Writes `line` to logcat with the tag `Orbis`.
pub(crate) fn log(line: &str) {
    let text = CString::new(line.replace('\0', " ")).expect("no NULs");
    // SAFETY: both strings are NUL-terminated and live through the call.
    #[allow(unsafe_code)]
    unsafe {
        ndk_sys::__android_log_write(
            ndk_sys::android_LogPriority::ANDROID_LOG_INFO.0 as i32,
            c"Orbis".as_ptr(),
            text.as_ptr(),
        );
    }
}

/// The activity's entry point, from android-activity's `NativeActivity`
/// glue.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
fn android_main(app: AndroidApp) {
    let started = Instant::now();
    std::panic::set_hook(Box::new(|info| log(&format!("panic: {info}"))));
    let _ = APP.set(app.clone());
    // HTTPS verifies servers with Android's trust store from here on, and
    // `file:///android_asset/` reads the app's assets.
    let init = with_activity(&app, |env, activity| {
        // SAFETY: the thread's env and a live reference to the activity.
        let init = unsafe { orb_api::init_android(env.get_raw().cast(), activity.as_raw().cast()) };
        Ok(init)
    });
    match init {
        Ok(Ok(())) => {}
        Ok(Err(e)) => log(&format!("android init: {e}")),
        Err(e) => log(&format!("android init: {e}")),
    }
    let extras = match extras(&app) {
        Ok(extras) => extras,
        Err(e) => {
            log(&format!("reading the intent's extras: {e}"));
            Extras::default()
        }
    };
    if let Some(backend) = &extras.backend {
        // SAFETY: nothing else reads or writes the environment yet: the
        // runtime that reads it is made later on this thread.
        unsafe { std::env::set_var("ORB_BACKEND", backend) };
        log(&format!("ORB_BACKEND={backend}"));
    }
    let files = app
        .internal_data_path()
        .expect("the activity has a files directory");
    let tokyo = "file:///android_asset/tokyo-mvt.json".to_owned();
    let mut at = match extras.at.as_deref().map(parse_at).transpose() {
        Ok(at) => at,
        Err(e) => {
            log(&e);
            return;
        }
    };
    let style = match extras.style.as_deref() {
        Some("tokyo") => tokyo.clone(),
        Some(url) => url.to_owned(),
        None => {
            let (lon, lat, zoom) = TOKYO;
            at = at.or(Some((Geodetic::new(lon, lat), zoom, 0.0, 0.0)));
            LIBERTY.to_owned()
        }
    };
    let options = Options {
        style,
        at,
        cache: Some(files.join("orbis-state")),
        lose: extras.lose.map(std::time::Duration::from_millis),
        leakcheck: extras.leakcheck.then_some(tokyo),
    };
    let event_loop = EventLoop::<Wake>::with_user_event()
        .with_android_app(app)
        .build()
        .expect("event loop");
    crate::run(event_loop, options, started);
    // The activity is finishing ([`finishing`]). winit makes one event loop
    // a process, so the next activity starts in a new one.
    std::process::exit(0);
}

/// `lon,lat,zoom[,bearing,pitch]`.
fn parse_at(at: &str) -> Result<(Geodetic, f64, f64, f64), String> {
    let parts: Vec<f64> = at
        .split(',')
        .map(|p| p.trim().parse::<f64>().map_err(|e| format!("at: {e}")))
        .collect::<Result<_, _>>()?;
    match parts[..] {
        [lon, lat, zoom] => Ok((Geodetic::new(lon, lat), zoom, 0.0, 0.0)),
        [lon, lat, zoom, bearing, pitch] => Ok((Geodetic::new(lon, lat), zoom, bearing, pitch)),
        _ => Err("at needs lon,lat,zoom[,bearing,pitch]".into()),
    }
}

/// Whether the activity is going for good, rather than to the background.
pub(crate) fn finishing() -> bool {
    let Some(app) = APP.get() else {
        return false;
    };
    let finishing = with_activity(app, |env, activity| {
        env.call_method(activity, jni_str!("isFinishing"), jni_sig!("()Z"), &[])?
            .z()
    });
    finishing.unwrap_or_else(|e| {
        log(&format!("isFinishing: {e}"));
        false
    })
}

/// The platform's gesture thresholds, from the activity's
/// `ViewConfiguration`, or `None` when the platform won't say.
pub(crate) fn view_configuration() -> Option<crate::ViewConfiguration> {
    let app = APP.get()?;
    let read = with_activity(app, |env, activity| {
        let config = env
            .call_static_method(
                jni_str!("android/view/ViewConfiguration"),
                jni_str!("get"),
                jni_sig!("(Landroid/content/Context;)Landroid/view/ViewConfiguration;"),
                &[JValue::Object(activity)],
            )?
            .l()?;
        let mut int = |name: &'static jni::strings::JNIStr| -> jni::errors::Result<f64> {
            Ok(f64::from(
                env.call_method(&config, name, jni_sig!("()I"), &[])?.i()?,
            ))
        };
        let touch_slop = int(jni_str!("getScaledTouchSlop"))?;
        let double_tap_slop = int(jni_str!("getScaledDoubleTapSlop"))?;
        let min_fling = int(jni_str!("getScaledMinimumFlingVelocity"))?;
        let max_fling = int(jni_str!("getScaledMaximumFlingVelocity"))?;
        let mut timeout = |name: &'static jni::strings::JNIStr| -> jni::errors::Result<f64> {
            Ok(f64::from(
                env.call_static_method(
                    jni_str!("android/view/ViewConfiguration"),
                    name,
                    jni_sig!("()I"),
                    &[],
                )?
                .i()?,
            ))
        };
        Ok(crate::ViewConfiguration {
            touch_slop,
            double_tap_slop,
            min_fling,
            max_fling,
            double_tap_timeout: timeout(jni_str!("getDoubleTapTimeout"))?,
            long_press_timeout: timeout(jni_str!("getLongPressTimeout"))?,
        })
    });
    read.inspect_err(|e| log(&format!("ViewConfiguration: {e}")))
        .ok()
}

/// Calls `f` with the activity, on this thread attached to the VM.
#[allow(unsafe_code)]
fn with_activity<T>(
    app: &AndroidApp,
    f: impl FnOnce(&mut jni::Env, &JObject) -> jni::errors::Result<T>,
) -> jni::errors::Result<T> {
    // SAFETY: android-activity's VM pointer is the process's live VM.
    let vm = unsafe { jni::JavaVM::from_raw(app.vm_as_ptr().cast()) };
    let activity = app.activity_as_ptr() as jni::sys::jobject;
    vm.attach_current_thread(|env| {
        // SAFETY: the activity's global reference lives as long as `app`.
        let activity = unsafe { env.as_cast_raw::<Global<JObject>>(&activity)? };
        f(env, activity.as_ref())
    })
}

/// The intent's extras the demo reads.
#[derive(Debug, Default)]
struct Extras {
    style: Option<String>,
    at: Option<String>,
    backend: Option<String>,
    lose: Option<u64>,
    leakcheck: bool,
}

/// Reads the extras from the activity's intent.
fn extras(app: &AndroidApp) -> jni::errors::Result<Extras> {
    with_activity(app, |env, activity| {
        let intent = env
            .call_method(
                activity,
                jni_str!("getIntent"),
                jni_sig!("()Landroid/content/Intent;"),
                &[],
            )?
            .l()?;
        let mut string = |key: &str| -> jni::errors::Result<Option<String>> {
            let key = env.new_string(key)?;
            let value = env
                .call_method(
                    &intent,
                    jni_str!("getStringExtra"),
                    jni_sig!("(Ljava/lang/String;)Ljava/lang/String;"),
                    &[JValue::Object(&key)],
                )?
                .l()?;
            if value.is_null() {
                return Ok(None);
            }
            let value = env.cast_local::<JString>(value)?;
            value.try_to_string(env).map(Some)
        };
        let style = string("style")?;
        let at = string("at")?;
        let backend = string("backend")?;
        let key = env.new_string("lose")?;
        let lose = env
            .call_method(
                &intent,
                jni_str!("getLongExtra"),
                jni_sig!("(Ljava/lang/String;J)J"),
                &[JValue::Object(&key), JValue::Long(-1)],
            )?
            .j()?;
        let key = env.new_string("leakcheck")?;
        let leakcheck = env
            .call_method(
                &intent,
                jni_str!("getBooleanExtra"),
                jni_sig!("(Ljava/lang/String;Z)Z"),
                &[JValue::Object(&key), JValue::Bool(false)],
            )?
            .z()?;
        Ok(Extras {
            style,
            at,
            backend,
            lose: u64::try_from(lose).ok(),
            leakcheck,
        })
    })
}
