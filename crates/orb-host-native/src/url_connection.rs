//! HTTP on Android through the platform's `java.net.HttpURLConnection`, in
//! the app's JVM ([`init`](crate::android::init)): Android's TLS and trust
//! store (the user's certificates too where the app allows them), its proxy
//! settings and the app's network security config, cleartext policy
//! included, as maps in Android apps fetch with MapLibre Native's SDK.
//! Each GET blocks the fetch thread it runs on, which attaches to the JVM
//! once and stays attached.
//!
//! [`http`](crate::http) sends the headers and follows the redirects, as
//! for ureq: a connection follows none itself and keeps no cache of its own
//! ([`HttpCache`](crate::HttpCache) is in front). Without a `Range`,
//! `HttpURLConnection` asks for gzip and decodes it. Connecting, or a read
//! that waits longer than [`TIMEOUT_MILLISECONDS`], fails rather than hold
//! a fetch thread. An abort disconnects the connection, which ends a
//! connect or read in flight. A failure is a network error, but a plain
//! HTTP URL the network security config refuses fails as unsupported.
//!
//! Android's connection pool speaks HTTP/1.1 alone, so each GET in flight
//! holds a connection, and keeps 5 idle across all hosts: of a burst of
//! more fetches than that, the rest connect afresh in the next (TCP and
//! TLS, about two round trips, side by side), where ureq kept one for each
//! fetch thread. Its size is the process's `http.maxConnections`, read
//! once before the first connection, so it is the app's to raise, not ours.

use std::sync::Arc;

use jni::errors::Error as JniError;
use jni::objects::{Global, JByteArray, JObject, JString, JThrowable, JValue, JValueOwned};
use jni::{Env, jni_sig, jni_str};
use orb_host::abort::Listening;
use orb_host::{Abort, FetchError};

use crate::Reply;

/// How long connecting, and each read, may wait.
const TIMEOUT_MILLISECONDS: i32 = 30_000;

/// The size of the Java array a body is read through.
const BUFFER_SIZE: i32 = 64 << 10;

/// A connection, global so that an abort on another thread can reach it.
type Connection = Arc<Global<JObject<'static>>>;

/// GETs `url` with `headers`, following no redirect. Fails before
/// [`init`](crate::android::init) gave the app's JVM.
pub(crate) fn get(
    url: &str,
    headers: &[(&str, &str)],
    abort: &Abort,
) -> Result<Box<dyn Reply>, FetchError> {
    let Some(vm) = crate::android::java_vm() else {
        return Err(FetchError::UnsupportedUrl(format!(
            "{url}: HTTP on Android needs the app's JVM (mln_android_init or orb_api::init_android)"
        )));
    };
    vm.attach_current_thread(|env| Ok::<_, JniError>(exchange(env, url, headers, abort)))
        .map_err(|e| FetchError::Network(format!("JNI failed: {e}")))?
}

/// One GET, up to its response's headers. Leaves no Java exception
/// pending.
fn exchange(
    env: &mut Env<'_>,
    url: &str,
    headers: &[(&str, &str)],
    abort: &Abort,
) -> Result<Box<dyn Reply>, FetchError> {
    let connection = open(env, url, headers)
        .and_then(|connection| env.new_global_ref(connection))
        .map_err(|e| {
            let error = failure(env, &e);
            refused(env, url, error)
        })?;
    let connection: Connection = Arc::new(connection);
    let listening = abort.on_abort({
        let connection = Arc::clone(&connection);
        move || disconnect(&connection)
    });
    let mut reply = UrlReply {
        connection,
        status: 0,
        headers: Vec::new(),
        abort: abort.clone(),
        listening,
        body_read: true,
    };
    // A connection that hasn't connected yet ignores `disconnect`. An abort
    // after this check and before `getResponseCode` connects waits for the
    // headers, or the timeout, and then the body read sees it.
    if abort.is_aborted() {
        return Err(FetchError::Aborted);
    }
    let responded = (|| -> Result<_, JniError> {
        let status = env
            .call_method(
                &*reply.connection,
                jni_str!("getResponseCode"),
                jni_sig!("()I"),
                &[],
            )?
            .i()?;
        Ok((status, response_headers(env, &reply.connection)?))
    })();
    let (status, headers) = responded.map_err(|e| {
        let error = reply.failure(env, &e);
        refused(env, url, error)
    })?;
    // A status out of range is no response.
    reply.status = u16::try_from(status)
        .map_err(|_| FetchError::Network(format!("{url}: no HTTP response")))?;
    reply.headers = headers;
    reply.body_read = false;
    Ok(Box::new(reply))
}

/// A connection for `url` with `headers` that hasn't connected.
fn open<'local>(
    env: &mut Env<'local>,
    url: &str,
    headers: &[(&str, &str)],
) -> Result<JObject<'local>, JniError> {
    let url = env.new_string(url)?;
    let url = env.new_object(
        jni_str!("java/net/URL"),
        jni_sig!("(Ljava/lang/String;)V"),
        &[JValue::from(&url)],
    )?;
    let connection = env
        .call_method(
            &url,
            jni_str!("openConnection"),
            jni_sig!("()Ljava/net/URLConnection;"),
            &[],
        )?
        .l()?;
    env.call_method(
        &connection,
        jni_str!("setInstanceFollowRedirects"),
        jni_sig!("(Z)V"),
        &[JValue::Bool(false)],
    )?;
    env.call_method(
        &connection,
        jni_str!("setUseCaches"),
        jni_sig!("(Z)V"),
        &[JValue::Bool(false)],
    )?;
    for setter in [jni_str!("setConnectTimeout"), jni_str!("setReadTimeout")] {
        env.call_method(
            &connection,
            setter,
            jni_sig!("(I)V"),
            &[JValue::Int(TIMEOUT_MILLISECONDS)],
        )?;
    }
    let user_agent = [("User-Agent", crate::USER_AGENT)];
    for &(name, value) in user_agent.iter().chain(headers) {
        let (name, value) = (env.new_string(name)?, env.new_string(value)?);
        env.call_method(
            &connection,
            jni_str!("setRequestProperty"),
            jni_sig!("(Ljava/lang/String;Ljava/lang/String;)V"),
            &[JValue::from(&name), JValue::from(&value)],
        )?;
        env.delete_local_ref(name);
        env.delete_local_ref(value);
    }
    Ok(connection)
}

/// The response headers of `connection`, in order.
fn response_headers(
    env: &mut Env<'_>,
    connection: &JObject<'_>,
) -> Result<Vec<(String, String)>, JniError> {
    let mut headers = Vec::new();
    for index in 0.. {
        let header = env.with_local_frame(4, |env| -> Result<_, JniError> {
            let value = env
                .call_method(
                    connection,
                    jni_str!("getHeaderField"),
                    jni_sig!("(I)Ljava/lang/String;"),
                    &[JValue::Int(index)],
                )?
                .l()?;
            if value.is_null() {
                return Ok(None);
            }
            let name = env
                .call_method(
                    connection,
                    jni_str!("getHeaderFieldKey"),
                    jni_sig!("(I)Ljava/lang/String;"),
                    &[JValue::Int(index)],
                )?
                .l()?;
            // The status line is a field without a name.
            Ok(Some((string(env, name)?, string(env, value)?)))
        })?;
        match header {
            None => break,
            Some((Some(name), Some(value))) => headers.push((name, value)),
            Some(_) => {}
        }
    }
    Ok(headers)
}

/// A response whose headers have come.
struct UrlReply {
    connection: Connection,
    status: u16,
    headers: Vec<(String, String)>,
    abort: Abort,
    /// The abort's listener that disconnects, taken off when this drops.
    listening: Listening,
    /// Its body was read, or there's none to close.
    body_read: bool,
}

impl UrlReply {
    /// The error for a JNI call's failure: an abort's, when it fired.
    fn failure(&self, env: &mut Env<'_>, error: &JniError) -> FetchError {
        let failure = failure(env, error);
        match self.abort.is_aborted() {
            true => FetchError::Aborted,
            false => failure,
        }
    }
}

impl Reply for UrlReply {
    fn status(&self) -> u16 {
        self.status
    }

    fn header(&self, name: &str) -> Option<String> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    }

    fn body(mut self: Box<Self>, limit: u64, abort: &Abort) -> Result<Vec<u8>, FetchError> {
        self.body_read = true;
        let Some(vm) = crate::android::java_vm() else {
            return Err(FetchError::Network("no JVM".into()));
        };
        vm.attach_current_thread(|env| -> Result<_, JniError> {
            let opened = (|| -> Result<_, JniError> {
                let connection = &*self.connection;
                let length = env
                    .call_method(
                        connection,
                        jni_str!("getContentLengthLong"),
                        jni_sig!("()J"),
                        &[],
                    )?
                    .j()?;
                let stream = env
                    .call_method(
                        connection,
                        jni_str!("getInputStream"),
                        jni_sig!("()Ljava/io/InputStream;"),
                        &[],
                    )?
                    .l()?;
                let buffer = JByteArray::new(env, BUFFER_SIZE as usize)?;
                Ok((length, stream, buffer))
            })();
            let (length, stream, buffer) = match opened {
                Ok(opened) => opened,
                Err(e) => return Ok(Err(self.failure(env, &e))),
            };
            let mut reader = Stream {
                env,
                stream: &stream,
                buffer: &buffer,
                chunk: vec![0; BUFFER_SIZE as usize],
            };
            let body = crate::read_body(&mut reader, u64::try_from(length).ok(), limit, abort);
            close(reader.env, &stream);
            Ok(body)
        })
        .map_err(|e| FetchError::Network(format!("JNI failed: {e}")))?
    }
}

impl Drop for UrlReply {
    fn drop(&mut self) {
        self.abort.forget(self.listening);
        if self.body_read {
            return;
        }
        // Closed unread, so that the connection serves another request.
        let Some(vm) = crate::android::java_vm() else {
            return;
        };
        let _ = vm.attach_current_thread(|env| -> Result<(), JniError> {
            // Java throws for `getInputStream` of an error status.
            let method = match self.status >= 400 {
                true => jni_str!("getErrorStream"),
                false => jni_str!("getInputStream"),
            };
            let stream = env
                .call_method(
                    &*self.connection,
                    method,
                    jni_sig!("()Ljava/io/InputStream;"),
                    &[],
                )
                .and_then(JValueOwned::l);
            match stream {
                Ok(stream) if !stream.is_null() => close(env, &stream),
                Ok(_) => {}
                Err(_) => env.exception_clear(),
            }
            Ok(())
        });
    }
}

/// A response body's `InputStream`, read through a Java array.
struct Stream<'a, 'local> {
    env: &'a mut Env<'local>,
    stream: &'a JObject<'local>,
    buffer: &'a JByteArray<'local>,
    chunk: Vec<i8>,
}

impl std::io::Read for Stream<'_, '_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let wanted = out.len().min(self.chunk.len()) as i32;
        let read = self
            .env
            .call_method(
                self.stream,
                jni_str!("read"),
                jni_sig!("([BII)I"),
                &[
                    JValue::from(self.buffer),
                    JValue::Int(0),
                    JValue::Int(wanted),
                ],
            )
            .and_then(JValueOwned::i)
            .and_then(|read| {
                let read = usize::try_from(read).unwrap_or(0);
                let chunk = &mut self.chunk[..read];
                self.buffer.get_region(self.env, 0, chunk)?;
                Ok(chunk)
            });
        match read {
            Ok(chunk) => {
                for (out, byte) in out.iter_mut().zip(chunk.iter()) {
                    *out = byte.cast_unsigned();
                }
                Ok(chunk.len())
            }
            Err(e) => Err(std::io::Error::other(failure(self.env, &e).to_string())),
        }
    }
}

/// The text of `string`, or `None` for null.
fn string(env: &Env<'_>, string: JObject<'_>) -> Result<Option<String>, JniError> {
    if string.is_null() {
        return Ok(None);
    }
    env.cast_local::<JString>(string)?
        .try_to_string(env)
        .map(Some)
}

/// Closes `stream`, ignoring any failure. Leaves no Java exception pending.
fn close(env: &mut Env<'_>, stream: &JObject<'_>) {
    if env
        .call_method(stream, jni_str!("close"), jni_sig!("()V"), &[])
        .is_err()
    {
        env.exception_clear();
    }
}

/// Disconnects `connection`, from any thread, which ends a connect or read
/// in flight with an exception.
fn disconnect(connection: &Global<JObject<'static>>) {
    let Some(vm) = crate::android::java_vm() else {
        return;
    };
    let _ = vm.attach_current_thread(|env| -> Result<(), JniError> {
        if env
            .call_method(connection, jni_str!("disconnect"), jni_sig!("()V"), &[])
            .is_err()
        {
            env.exception_clear();
        }
        Ok(())
    });
}

/// The error for a JNI call's failure: a network error with the pending
/// Java exception's message, which it clears, or else `error`'s.
fn failure(env: &mut Env<'_>, error: &JniError) -> FetchError {
    let Some(exception) = env.exception_occurred() else {
        return FetchError::Network(format!("JNI failed: {error}"));
    };
    env.exception_clear();
    FetchError::Network(message(env, &exception).unwrap_or_else(|| error.to_string()))
}

/// `error`, or an unsupported URL when it is a plain HTTP URL's that the
/// app's network security config refuses: no retry changes the policy.
/// Android words the refusal differently across versions ("Cleartext HTTP
/// traffic to <host> not permitted", "CLEARTEXT communication to <host>
/// not permitted by network security policy"), so the policy is asked.
fn refused(env: &mut Env<'_>, url: &str, error: FetchError) -> FetchError {
    let FetchError::Network(message) = &error else {
        return error;
    };
    match crate::origin(url) {
        Some((scheme, host, _)) if scheme == "http" && !cleartext_permitted(env, &host) => {
            FetchError::UnsupportedUrl(message.clone())
        }
        _ => error,
    }
}

/// `NetworkSecurityPolicy.isCleartextTrafficPermitted(host)`; true when
/// Java fails to answer. Leaves no Java exception pending.
fn cleartext_permitted(env: &mut Env<'_>, host: &str) -> bool {
    let permitted = (|| -> Result<bool, JniError> {
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
        env.call_method(
            &policy,
            jni_str!("isCleartextTrafficPermitted"),
            jni_sig!("(Ljava/lang/String;)Z"),
            &[JValue::from(&host)],
        )?
        .z()
    })();
    if env.exception_check() {
        env.exception_clear();
    }
    permitted.unwrap_or(true)
}

/// The message of a Java exception, if it has one. Leaves no Java exception
/// pending.
fn message(env: &mut Env<'_>, exception: &JThrowable<'_>) -> Option<String> {
    let message = env
        .call_method(
            exception,
            jni_str!("toString"),
            jni_sig!("()Ljava/lang/String;"),
            &[],
        )
        .and_then(JValueOwned::l)
        .and_then(|message| string(env, message));
    if env.exception_check() {
        env.exception_clear();
    }
    message.ok().flatten()
}
