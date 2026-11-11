// The runtime's resource provider, resource transform and HTTP header
// transform: one `orb_api::NetworkHook` per runtime, where Orbis's fetch
// would go to the network once its HTTP cache has no fresh copy.
//
// A request whose URL and kind a provider route matches goes to the
// provider, which answers it or passes it through. A request the library
// loads itself has its URL rewritten by the resource transform, then its
// headers set by the header transform, and goes to Orbis's network.

use std::sync::{Arc, Mutex};

use orb_api::{FetchError, NetworkAnswer, NetworkDone, NetworkHook, NetworkRequest, ResourceKind};

use crate::base::{Failure, Result, UserData, boundary, bytes, put, sized, text};
use crate::delivery::{self, Completion, Value};
use crate::handles::{self, Kind, Object};
use crate::sys::*;

/// A runtime's hook and what it holds.
#[derive(Default)]
pub(crate) struct Hook {
    config: Mutex<Config>,
}

#[derive(Default)]
struct Config {
    provider: Option<Arc<Provider>>,
    transform: Option<Arc<Transform>>,
    headers: Option<Arc<Headers>>,
}

impl Hook {
    fn config(&self) -> std::sync::MutexGuard<'_, Config> {
        self.config
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Releases every registration, as the runtime's teardown does.
    pub(crate) fn clear(&self) {
        let mut config = self.config();
        config.provider.take();
        config.transform.take();
        config.headers.take();
    }
}

fn kind_of(kind: ResourceKind) -> mln_resource_kind {
    match kind {
        ResourceKind::Unknown => MLN_RESOURCE_KIND_UNKNOWN,
        ResourceKind::Style => MLN_RESOURCE_KIND_STYLE,
        ResourceKind::Source => MLN_RESOURCE_KIND_SOURCE,
        ResourceKind::Tile => MLN_RESOURCE_KIND_TILE,
        ResourceKind::Glyphs => MLN_RESOURCE_KIND_GLYPHS,
        ResourceKind::SpriteImage => MLN_RESOURCE_KIND_SPRITE_IMAGE,
        ResourceKind::SpriteJson => MLN_RESOURCE_KIND_SPRITE_JSON,
        ResourceKind::Image => MLN_RESOURCE_KIND_IMAGE,
    }
}

impl NetworkHook for Hook {
    fn request(&self, request: NetworkRequest, done: NetworkDone) {
        let (provider, chain) = {
            let config = self.config();
            (
                config.provider.clone(),
                Chain {
                    transform: config.transform.clone(),
                    headers: config.headers.clone(),
                    declined: config.provider.is_some(),
                },
            )
        };
        match provider.filter(|p| p.serves(&request)) {
            Some(provider) => provider.call(request, chain, done),
            None => chain.load(request, done),
        }
    }
}

// MARK: - URL patterns

/// Whether `text` matches the glob `pattern`: `*` any run without `/`,
/// `**` any run, `?` one character but `/`, `\` the next character itself.
fn glob(pattern: &[char], text: &[char]) -> bool {
    match pattern.first() {
        None => text.is_empty(),
        Some('*') if pattern.get(1) == Some(&'*') => {
            (0..=text.len()).any(|i| glob(&pattern[2..], &text[i..]))
        }
        Some('*') => {
            let run = text.iter().take_while(|c| **c != '/').count();
            (0..=run).any(|i| glob(&pattern[1..], &text[i..]))
        }
        Some('?') => text.first().is_some_and(|c| *c != '/') && glob(&pattern[1..], &text[1..]),
        Some('\\') if pattern.len() > 1 => {
            text.first() == Some(&pattern[1]) && glob(&pattern[2..], &text[1..])
        }
        Some(c) => text.first() == Some(c) && glob(&pattern[1..], &text[1..]),
    }
}

/// Rules: which requests each selects, and the pairs (query parameters
/// or headers) it adds to them.
type Rules = Vec<(Pattern, Vec<(String, String)>)>;

/// The most tokens a glob may hold, as mln-ffi bounds its matching.
const MAX_GLOB_TOKENS: usize = 4096;

/// A glob's tokens: each character, escaped character and run of `*`.
fn glob_tokens(pattern: &str) -> usize {
    let mut chars = pattern.chars().peekable();
    let mut tokens = 0;
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                chars.next();
            }
            '*' => while chars.next_if_eq(&'*').is_some() {},
            _ => {}
        }
        tokens += 1;
    }
    tokens
}

/// A route's or rule's selection of requests.
#[derive(Clone, Debug)]
struct Pattern {
    kinds: u32,
    glob: bool,
    url: String,
}

impl Pattern {
    unsafe fn from_c(kinds: u32, flags: u32, url: mln_buffer_view, what: &str) -> Result<Self> {
        if kinds & !MLN_RESOURCE_KIND_FLAG_ALL != 0 {
            return Err(Failure::argument(format!("{what}.kinds has unknown bits")));
        }
        if flags & !MLN_URL_MATCH_GLOB != 0 {
            return Err(Failure::argument(format!("{what}.flags has unknown bits")));
        }
        if url.data.is_null() && url.size != 0 {
            return Err(Failure::argument(format!("{what}.url is null")));
        }
        // SAFETY: the caller's view.
        let url = unsafe { text(url, &format!("{what}.url")) }?;
        if url.contains('\0') {
            return Err(Failure::argument(format!("{what}.url contains a NUL")));
        }
        let glob = flags & MLN_URL_MATCH_GLOB != 0;
        if glob && glob_tokens(&url) > MAX_GLOB_TOKENS {
            return Err(Failure::argument(format!(
                "{what}.url is a glob of more than {MAX_GLOB_TOKENS} tokens"
            )));
        }
        Ok(Self { kinds, glob, url })
    }

    fn matches(&self, kind: ResourceKind, url: &str) -> bool {
        if self.kinds != 0 && self.kinds & (1 << kind_of(kind)) == 0 {
            return false;
        }
        if self.glob {
            let pattern: Vec<char> = self.url.chars().collect();
            let text: Vec<char> = url.chars().collect();
            glob(&pattern, &text)
        } else {
            self.url == url
        }
    }
}

// MARK: - The provider

struct Provider {
    callback: unsafe extern "C" fn(
        *mut std::ffi::c_void,
        *const mln_resource_request_info,
        mln_resource_request,
    ),
    user_data: UserData,
    release_user_data: mln_user_data_release,
    queue: u64,
    routes: Vec<Pattern>,
}

impl Drop for Provider {
    fn drop(&mut self) {
        delivery::release_later(self.queue, self.release_user_data, self.user_data);
    }
}

impl Provider {
    fn serves(&self, request: &NetworkRequest) -> bool {
        self.routes.is_empty()
            || self
                .routes
                .iter()
                .any(|r| r.matches(request.kind, &request.url))
    }

    /// Hands `request` to the provider's callback with a handle that
    /// answers it.
    fn call(self: Arc<Self>, request: NetworkRequest, chain: Chain, done: NetworkDone) {
        let handle = Arc::new(ResourceRequest {
            state: Mutex::new(Some(Box::new((request.clone(), chain, done)))),
        });
        let id = handles::insert(Object::ResourceRequest(handle));
        let provider = self.clone();
        delivery::deliver(
            self.queue,
            Box::new(move || {
                let mut url = request.url.clone();
                let mut resolved = request.url.clone();
                let mut etag = request
                    .prior
                    .as_ref()
                    .and_then(|p| p.etag.clone())
                    .unwrap_or_default();
                let prior = request.prior.as_ref();
                // SAFETY: plain C data.
                let mut info: mln_resource_request_info = unsafe { std::mem::zeroed() };
                info.size = std::mem::size_of::<mln_resource_request_info>() as u32;
                info.requested_url = crate::base::lend(&mut url);
                info.resolved_url = crate::base::lend(&mut resolved);
                info.kind = kind_of(request.kind);
                info.prior_etag = crate::base::lend(&mut etag);
                if let Some((offset, length)) = request.range {
                    info.fields |= MLN_RESOURCE_REQUEST_RANGE;
                    info.range = mln_resource_range {
                        start: offset,
                        end: offset + length - 1,
                    };
                }
                if let Some(modified) = prior.and_then(|p| p.modified) {
                    info.fields |= MLN_RESOURCE_REQUEST_PRIOR_MODIFIED;
                    info.prior_modified_unix_ms = (modified as i64).saturating_mul(1000);
                }
                if let Some(prior) = prior {
                    info.prior_data = prior.bytes.as_ptr();
                    info.prior_data_size = prior.bytes.len();
                }
                // SAFETY: the provider's registration; `info` lives until it
                // returns.
                unsafe {
                    (provider.callback)(provider.user_data.get(), &info, mln_resource_request(id))
                };
            }),
        );
    }
}

/// What a request does after the provider: the transforms, and whether a
/// provider declined it.
struct Chain {
    transform: Option<Arc<Transform>>,
    headers: Option<Arc<Headers>>,
    declined: bool,
}

/// `url` without its credentials, query and fragment, for messages.
fn redacted(url: &str) -> String {
    let url = url.split(['?', '#']).next().unwrap_or_default();
    match url.split_once("://") {
        Some((scheme, rest)) => {
            let (authority, path) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
            let host = authority.rsplit('@').next().unwrap_or_default();
            format!("{scheme}://{host}{path}")
        }
        None => url.to_string(),
    }
}

impl Chain {
    /// Loads `request` as the library does: transformed, then through
    /// Orbis's network.
    fn load(self, request: NetworkRequest, done: NetworkDone) {
        let scheme = request
            .url
            .split(':')
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !matches!(scheme.as_str(), "http" | "https" | "file") {
            let url = redacted(&request.url);
            let message = if self.declined {
                format!(
                    "the registered resource provider declined {url}, and the library can't load \
                     its \"{scheme}\" scheme"
                )
            } else {
                format!(
                    "the library can't load {url}: its \"{scheme}\" scheme needs a resource \
                     provider (mln_runtime_set_resource_provider)"
                )
            };
            return done(NetworkAnswer::Failed(FetchError::UnsupportedUrl(message)));
        }
        let headers = self.headers;
        let (kind, original) = (request.kind, request.request.url.clone());
        let then = move |url: std::result::Result<String, FetchError>| {
            let url = match url {
                Ok(url) => url,
                Err(e) => return done(NetworkAnswer::Failed(e)),
            };
            let mut out = request.request.clone();
            out.url = url;
            if let Some(h) = headers
                && (out.url.starts_with("http://") || out.url.starts_with("https://"))
            {
                match h.headers(request.kind, &out.url) {
                    Ok(set) => {
                        for (name, value) in set {
                            out.headers.retain(|(n, _)| !n.eq_ignore_ascii_case(&name));
                            out.headers.push((name, value));
                        }
                    }
                    Err(e) => return done(NetworkAnswer::Failed(e)),
                }
            }
            done(NetworkAnswer::Network(out));
        };
        match self.transform {
            Some(t) => t.rewrite(kind, original, Box::new(then)),
            None => then(Ok(original)),
        }
    }
}

/// A request a provider holds: unanswered while it has its parts.
pub(crate) struct ResourceRequest {
    state: Mutex<Option<Box<(NetworkRequest, Chain, NetworkDone)>>>,
}

impl ResourceRequest {
    fn take(&self) -> Option<(NetworkRequest, Chain, NetworkDone)> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .map(|parts| *parts)
    }

    /// Fails the request as a release without an answer does.
    fn fail(&self) {
        if let Some((_, _, done)) = self.take() {
            done(NetworkAnswer::Failed(FetchError::Network(
                "the resource request was released without a response".into(),
            )));
        }
    }
}

impl Drop for ResourceRequest {
    fn drop(&mut self) {
        self.fail();
    }
}

/// Takes the request a handle names out of the table: every answer ends
/// it.
fn request(handle: mln_resource_request) -> Result<Arc<ResourceRequest>> {
    match handles::lookup(handle.0, Kind::ResourceRequest)? {
        Object::ResourceRequest(r) => {
            handles::remove(handle.0);
            Ok(r)
        }
        _ => unreachable!(),
    }
}

/// Unix milliseconds as seconds, not before 1970.
fn seconds(ms: i64) -> u64 {
    (ms.max(0) / 1000) as u64
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The answer a provider's response gives.
unsafe fn answer(response: *const mln_resource_response) -> Result<NetworkAnswer> {
    // SAFETY: the caller's pointer.
    let r = unsafe { sized(response, "response") }?;
    let known = MLN_RESOURCE_RESPONSE_MODIFIED
        | MLN_RESOURCE_RESPONSE_EXPIRES
        | MLN_RESOURCE_RESPONSE_RETRY_AFTER;
    if r.fields & !known != 0 {
        return Err(Failure::argument(
            "mln_resource_response.fields contains unknown bits",
        ));
    }
    if r.bytes.is_null() && r.byte_count != 0 {
        return Err(Failure::argument("response.bytes is null"));
    }
    // SAFETY: the caller's view.
    let etag = unsafe { bytes(r.etag) };
    if etag.contains(&0) {
        return Err(Failure::argument("response.etag contains a NUL"));
    }
    let etag =
        std::str::from_utf8(etag).map_err(|_| Failure::argument("response.etag isn't UTF-8"))?;
    let etag = (!etag.is_empty()).then(|| etag.to_string());
    let has = |bit| r.fields & bit != 0;
    let fresh_until = if r.must_revalidate {
        Some(now())
    } else if has(MLN_RESOURCE_RESPONSE_EXPIRES) {
        Some(seconds(r.expires_unix_ms))
    } else {
        Some(now())
    };
    let modified = has(MLN_RESOURCE_RESPONSE_MODIFIED).then(|| seconds(r.modified_unix_ms));
    Ok(match r.status {
        MLN_RESOURCE_RESPONSE_STATUS_OK => NetworkAnswer::Body {
            bytes: if r.byte_count == 0 {
                Vec::new()
            } else {
                // SAFETY: the caller's `byte_count` bytes.
                unsafe { std::slice::from_raw_parts(r.bytes, r.byte_count) }.to_vec()
            },
            fresh_until,
            etag,
            modified,
        },
        MLN_RESOURCE_RESPONSE_STATUS_NO_CONTENT => NetworkAnswer::Body {
            bytes: Vec::new(),
            fresh_until,
            etag,
            modified,
        },
        MLN_RESOURCE_RESPONSE_STATUS_NOT_MODIFIED => NetworkAnswer::NotModified { fresh_until },
        MLN_RESOURCE_RESPONSE_STATUS_ERROR => {
            // SAFETY: the caller's view.
            let message = unsafe { text(r.error_message, "response.error_message") }?;
            let message = if message.is_empty() {
                "resource provider failed".to_string()
            } else {
                message
            };
            NetworkAnswer::Failed(match r.error_reason {
                MLN_RESOURCE_ERROR_REASON_NOT_FOUND => FetchError::NotFound,
                MLN_RESOURCE_ERROR_REASON_OTHER
                | MLN_RESOURCE_ERROR_REASON_SERVER
                | MLN_RESOURCE_ERROR_REASON_CONNECTION
                | MLN_RESOURCE_ERROR_REASON_RATE_LIMIT => FetchError::Network(message),
                _ => {
                    return Err(Failure::argument("response.error_reason names no reason"));
                }
            })
        }
        _ => return Err(Failure::argument("response.status names no status")),
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_resource_request_complete(
    handle: mln_resource_request,
    response: *const mln_resource_response,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let request = request(handle)?;
            // A refused answer fails the request, as a release does.
            let answer = answer(response)?;
            let (_, _, done) = request
                .take()
                .ok_or_else(|| Failure::state("the request was already answered"))?;
            done(answer);
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_resource_request_pass_through(
    handle: mln_resource_request,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointer.
    unsafe {
        boundary(out_diagnostic, || {
            let (request, chain, done) = request(handle)?
                .take()
                .ok_or_else(|| Failure::state("the request was already answered"))?;
            chain.load(request, done);
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_resource_request_is_cancelled(
    handle: mln_resource_request,
    out_cancelled: *mut bool,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            handles::lookup(handle.0, Kind::ResourceRequest)?;
            // Orbis finishes every request it starts, so none is cancelled.
            put(out_cancelled, false, "out_cancelled")
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_resource_request_set_cancel_callback(
    handle: mln_resource_request,
    handler: *const mln_resource_request_cancel_handler,
    out_cancelled: *mut bool,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            handles::lookup(handle.0, Kind::ResourceRequest)?;
            let h = sized(handler, "handler")?;
            if h.callback.is_none() {
                return Err(Failure::argument("handler.callback is null"));
            }
            delivery::check_queue(h.queue.0, "handler.queue")?;
            if out_cancelled.is_null() {
                return Err(Failure::argument("out_cancelled is null"));
            }
            // Orbis never cancels a request, so the callback never runs; its
            // user data goes back when the request ends, as Native's does.
            put(out_cancelled, false, "out_cancelled")?;
            delivery::release_later(h.queue.0, h.release_user_data, UserData(h.user_data));
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mln_resource_request_release(handle: mln_resource_request) {
    if let Ok(Object::ResourceRequest(r)) = handles::lookup(handle.0, Kind::ResourceRequest) {
        handles::remove(handle.0);
        r.fail();
    }
}

// MARK: - The resource transform

enum Transform {
    Rules(Rules),
    Callback {
        callback: unsafe extern "C" fn(
            *mut std::ffi::c_void,
            mln_resource_kind,
            mln_buffer_view,
            mln_resource_transform_request,
        ),
        user_data: UserData,
        release_user_data: mln_user_data_release,
        queue: u64,
    },
}

impl Drop for Transform {
    fn drop(&mut self) {
        if let Transform::Callback {
            user_data,
            release_user_data,
            queue,
            ..
        } = self
        {
            delivery::release_later(*queue, *release_user_data, *user_data);
        }
    }
}

type Rewritten = Box<dyn FnOnce(std::result::Result<String, FetchError>) + Send>;

/// Percent-encodes all but the unreserved characters.
fn encode(text: &str) -> String {
    let mut out = String::new();
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// `url` with `parameters` after its query, before its fragment.
fn with_parameters(url: &str, parameters: &[(String, String)]) -> String {
    if parameters.is_empty() {
        return url.to_string();
    }
    let (base, fragment) = match url.find('#') {
        Some(i) => url.split_at(i),
        None => (url, ""),
    };
    let mut out = base.to_string();
    for (name, value) in parameters {
        if !out.contains('?') {
            out.push('?');
        } else if !out.ends_with('?') && !out.ends_with('&') {
            out.push('&');
        }
        out.push_str(&encode(name));
        out.push('=');
        out.push_str(&encode(value));
    }
    out.push_str(fragment);
    out
}

impl Transform {
    fn rewrite(self: Arc<Self>, kind: ResourceKind, url: String, then: Rewritten) {
        match &*self {
            Transform::Rules(rules) => {
                let rule = rules.iter().find(|(p, _)| p.matches(kind, &url));
                then(Ok(match rule {
                    Some((_, parameters)) => with_parameters(&url, parameters),
                    None => url,
                }));
            }
            Transform::Callback {
                callback,
                user_data,
                queue,
                ..
            } => {
                let pending = Arc::new(TransformRequest {
                    then: Mutex::new(Some(then)),
                });
                let id = handles::insert(Object::TransformRequest(pending));
                let (callback, user_data) = (*callback, *user_data);
                let keep = self.clone();
                delivery::deliver(
                    *queue,
                    Box::new(move || {
                        let _keep = keep;
                        let mut url = url;
                        let view = crate::base::lend(&mut url);
                        // SAFETY: the transform's registration; `url` lives
                        // until it returns.
                        unsafe {
                            callback(
                                user_data.get(),
                                kind_of(kind),
                                view,
                                mln_resource_transform_request(id),
                            );
                        };
                    }),
                );
            }
        }
    }
}

/// A rewrite a transform callback holds.
pub(crate) struct TransformRequest {
    then: Mutex<Option<Rewritten>>,
}

impl TransformRequest {
    fn finish(&self, result: std::result::Result<String, FetchError>) {
        let then = self
            .then
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(then) = then {
            then(result);
        }
    }

    fn fail(&self) {
        self.finish(Err(FetchError::Network(
            "the resource transform request was released without an answer".into(),
        )));
    }
}

impl Drop for TransformRequest {
    fn drop(&mut self) {
        self.fail();
    }
}

fn transform_request(handle: mln_resource_transform_request) -> Result<Arc<TransformRequest>> {
    match handles::lookup(handle.0, Kind::TransformRequest)? {
        Object::TransformRequest(r) => {
            handles::remove(handle.0);
            Ok(r)
        }
        _ => unreachable!(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_resource_transform_request_complete(
    request: mln_resource_transform_request,
    url: mln_buffer_view,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let pending = transform_request(request)?;
            if url.data.is_null() || url.size == 0 {
                return Err(Failure::argument("url must not be empty"));
            }
            let url = text(url, "url")?;
            if url.contains('\0') {
                return Err(Failure::argument("url contains a NUL"));
            }
            pending.finish(Ok(url));
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_resource_transform_request_fail(
    request: mln_resource_transform_request,
    error: *const mln_resource_transform_error,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let pending = transform_request(request)?;
            let e = sized(error, "error")?;
            if e.fields & !MLN_RESOURCE_TRANSFORM_ERROR_RETRY_AFTER != 0 {
                return Err(Failure::argument("error.fields has unknown bits"));
            }
            if e.reason > MLN_RESOURCE_ERROR_REASON_RATE_LIMIT {
                return Err(Failure::argument("error.reason names no reason"));
            }
            let message = text(e.message, "error.message")?;
            let message = if message.is_empty() {
                "the resource transform failed the request".to_string()
            } else {
                message
            };
            pending.finish(Err(match e.reason {
                MLN_RESOURCE_ERROR_REASON_NOT_FOUND => FetchError::NotFound,
                _ => FetchError::Network(message),
            }));
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mln_resource_transform_request_release(request: mln_resource_transform_request) {
    if let Ok(Object::TransformRequest(r)) = handles::lookup(request.0, Kind::TransformRequest) {
        handles::remove(request.0);
        r.fail();
    }
}

// MARK: - The HTTP header transform

/// The headers the library and the platform manage.
const RESERVED: [&str; 15] = [
    "accept-encoding",
    "connection",
    "content-length",
    "host",
    "if-modified-since",
    "if-none-match",
    "keep-alive",
    "proxy-authorization",
    "proxy-connection",
    "range",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "user-agent",
];

fn check_header(name: &str, value: &str) -> Result {
    let token = |c: char| c.is_ascii_alphanumeric() || "!#$%&'*+-.^_`|~".contains(c);
    if name.is_empty() || !name.chars().all(token) {
        return Err(Failure::argument(format!(
            "the header name \"{name}\" isn't a token"
        )));
    }
    if RESERVED.contains(&name.to_ascii_lowercase().as_str()) {
        return Err(Failure::argument(format!(
            "the header \"{name}\" is one the library manages"
        )));
    }
    if value.contains(['\r', '\n', '\0']) {
        return Err(Failure::argument(format!(
            "the value of the header \"{name}\" contains a line break or a NUL"
        )));
    }
    Ok(())
}

enum Headers {
    Rules(Rules),
    Callback {
        callback: unsafe extern "C" fn(
            *mut std::ffi::c_void,
            mln_resource_kind,
            mln_buffer_view,
            *mut mln_http_header_transform_response,
        ) -> mln_status,
        user_data: UserData,
        release_user_data: mln_user_data_release,
    },
}

impl Drop for Headers {
    fn drop(&mut self) {
        if let Headers::Callback {
            user_data,
            release_user_data,
            ..
        } = self
        {
            delivery::release_later(0, *release_user_data, *user_data);
        }
    }
}

std::thread_local! {
    /// The header lists the callbacks running on this thread fill.
    static ACTIVE: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl Headers {
    fn headers(
        &self,
        kind: ResourceKind,
        url: &str,
    ) -> std::result::Result<Vec<(String, String)>, FetchError> {
        match self {
            Headers::Rules(rules) => Ok(rules
                .iter()
                .find(|(p, _)| p.matches(kind, url))
                .map(|(_, h)| h.clone())
                .unwrap_or_default()),
            Headers::Callback {
                callback,
                user_data,
                ..
            } => {
                let mut set: Vec<(String, String)> = Vec::new();
                let context = std::ptr::from_mut(&mut set).cast::<std::ffi::c_void>();
                let mut response = mln_http_header_transform_response {
                    size: std::mem::size_of::<mln_http_header_transform_response>() as u32,
                    context,
                };
                let mut url = url.to_string();
                let view = crate::base::lend(&mut url);
                ACTIVE.with(|a| a.borrow_mut().push(context as usize));
                // SAFETY: the transform's registration; the response and URL
                // live until it returns.
                let status =
                    unsafe { callback(user_data.get(), kind_of(kind), view, &mut response) };
                ACTIVE.with(|a| a.borrow_mut().pop());
                if status == MLN_STATUS_OK {
                    Ok(set)
                } else {
                    Err(FetchError::Network(
                        "the HTTP header transform failed the request".into(),
                    ))
                }
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_http_header_transform_response_set_header(
    response: *mut mln_http_header_transform_response,
    name: mln_buffer_view,
    value: mln_buffer_view,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let r = sized(response.cast_const(), "response")?;
            // The header is checked first: a bad one is refused wherever
            // it's set.
            if name.data.is_null() && name.size != 0 {
                return Err(Failure::argument("name is null"));
            }
            let name = text(name, "name")?;
            let value = text(value, "value")?;
            check_header(&name, &value)?;
            let context = r.context as usize;
            if context == 0 || !ACTIVE.with(|a| a.borrow().contains(&context)) {
                return Err(Failure::state(
                    "the response belongs to no running header transform",
                ));
            }
            // SAFETY: the active callback's header list.
            let set = &mut *(r.context.cast::<Vec<(String, String)>>());
            set.retain(|(n, _)| !n.eq_ignore_ascii_case(&name));
            set.push((name, value));
            Ok(())
        })
    }
}

// MARK: - Registration

/// Runs `apply` on the runtime's hook on its thread, then completes.
fn configure(
    runtime: mln_runtime,
    completion: *const mln_completion,
    apply: impl FnOnce(&mut Config) + Send + 'static,
) -> Result {
    let handle = handles::runtime(runtime.0)?;
    // SAFETY: the caller's pointer.
    let completion = unsafe { Completion::from_c(completion) }?;
    let hook = handle.hook.clone();
    handle.post(move |state| {
        // The old registration is released outside the lock.
        let old = {
            let mut config = hook.config();
            let before = Config {
                provider: config.provider.clone(),
                transform: config.transform.clone(),
                headers: config.headers.clone(),
            };
            apply(&mut config);
            before
        };
        drop(old);
        state.complete(completion, Ok(Value::None), 0);
    })
}

/// Reads `count` routes or rules from `items`, named `noun i` in
/// messages. Each steps to the next by its own size, so a caller built
/// against a later header passes larger ones, whose members past this
/// header's struct are skipped.
unsafe fn patterns<T>(
    items: *const T,
    count: usize,
    what: &str,
    noun: &str,
    each: impl Fn(&T, &str) -> Result<(Pattern, Vec<(String, String)>)>,
) -> Result<Rules> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if items.is_null() {
        return Err(Failure::argument(format!("{what} is null")));
    }
    let mut at = items.cast::<u8>();
    (0..count)
        .map(|i| {
            let label = format!("{noun} {i}");
            // SAFETY: the caller's array; every sized item starts with its
            // u32 size, and the one before it said where this one starts.
            let size = unsafe { at.cast::<u32>().read_unaligned() } as usize;
            if size < std::mem::size_of::<T>() {
                return Err(Failure::argument(format!("{label} has an invalid size")));
            }
            // SAFETY: as above; `size` bytes are the item's.
            let item = unsafe { &*at.cast::<T>() };
            at = at.wrapping_add(size);
            each(item, &label)
        })
        .collect()
}

unsafe fn pairs(
    names: impl Iterator<Item = (mln_buffer_view, mln_buffer_view)>,
    what: &str,
) -> Result<Vec<(String, String)>> {
    names
        .enumerate()
        .map(|(i, (name, value))| {
            if name.data.is_null() || name.size == 0 {
                return Err(Failure::argument(format!(
                    "{what}[{i}].name must not be empty"
                )));
            }
            // SAFETY: the caller's views.
            let name = unsafe { text(name, &format!("{what}[{i}].name")) }?;
            if name.contains('\0') {
                return Err(Failure::argument(format!(
                    "{what}[{i}].name contains a NUL"
                )));
            }
            if value.data.is_null() && value.size != 0 {
                return Err(Failure::argument(format!("{what}[{i}].value is null")));
            }
            // SAFETY: as above.
            let value = unsafe { text(value, &format!("{what}[{i}].value")) }?;
            Ok((name, value))
        })
        .collect()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_set_resource_provider(
    runtime: mln_runtime,
    provider: *const mln_resource_provider,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            handles::runtime(runtime.0)?;
            let p = sized(provider, "provider")?;
            let callback = p
                .callback
                .ok_or_else(|| Failure::argument("provider.callback is null"))?;
            delivery::check_queue(p.queue.0, "provider.queue")?;
            Completion::from_c(completion)?;
            let routes = patterns(
                p.routes,
                p.route_count,
                "provider.routes",
                "route",
                |r: &mln_resource_route, what| {
                    Ok((Pattern::from_c(r.kinds, r.flags, r.url, what)?, Vec::new()))
                },
            )?;
            let provider = Arc::new(Provider {
                callback,
                user_data: UserData(p.user_data),
                release_user_data: p.release_user_data,
                queue: p.queue.0,
                routes: routes.into_iter().map(|(p, _)| p).collect(),
            });
            configure(runtime, completion, move |c| c.provider = Some(provider))
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_clear_resource_provider(
    runtime: mln_runtime,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            configure(runtime, completion, |c| c.provider = None)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_set_resource_transform(
    runtime: mln_runtime,
    transform: *const mln_resource_transform,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            handles::runtime(runtime.0)?;
            let t = sized(transform, "transform")?;
            Completion::from_c(completion)?;
            let transform = match (t.callback, t.rule_count) {
                (Some(_), n) if n > 0 => {
                    return Err(Failure::argument(
                        "transform sets both a callback and rules",
                    ));
                }
                (None, 0) => {
                    return Err(Failure::argument(
                        "transform sets neither a callback nor rules",
                    ));
                }
                (Some(callback), _) => {
                    delivery::check_queue(t.queue.0, "transform.queue")?;
                    Transform::Callback {
                        callback,
                        user_data: UserData(t.user_data),
                        release_user_data: t.release_user_data,
                        queue: t.queue.0,
                    }
                }
                (None, _) => {
                    if !t.user_data.is_null() || t.release_user_data.is_some() || t.queue.0 != 0 {
                        return Err(Failure::argument(
                            "a transform's rules take no user_data, release or queue",
                        ));
                    }
                    Transform::Rules(patterns(
                        t.rules,
                        t.rule_count,
                        "transform.rules",
                        "rule",
                        |r: &mln_resource_transform_rule, what| {
                            let pattern = Pattern::from_c(r.kinds, r.flags, r.url, what)?;
                            if r.parameter_count > 0 && r.parameters.is_null() {
                                return Err(Failure::argument(format!(
                                    "{what}.parameters is null"
                                )));
                            }
                            let parameters = if r.parameter_count == 0 {
                                &[][..]
                            } else {
                                std::slice::from_raw_parts(r.parameters, r.parameter_count)
                            };
                            let parameters = pairs(
                                parameters.iter().map(|p| (p.name, p.value)),
                                &format!("{what}.parameters"),
                            )?;
                            Ok((pattern, parameters))
                        },
                    )?)
                }
            };
            let transform = Arc::new(transform);
            configure(runtime, completion, move |c| c.transform = Some(transform))
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_clear_resource_transform(
    runtime: mln_runtime,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            configure(runtime, completion, |c| c.transform = None)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_set_http_header_transform(
    runtime: mln_runtime,
    transform: *const mln_http_header_transform,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            handles::runtime(runtime.0)?;
            let t = sized(transform, "transform")?;
            Completion::from_c(completion)?;
            let headers = match (t.callback, t.rule_count) {
                (Some(_), n) if n > 0 => {
                    return Err(Failure::argument(
                        "transform sets both a callback and rules",
                    ));
                }
                (None, 0) => {
                    return Err(Failure::argument(
                        "transform sets neither a callback nor rules",
                    ));
                }
                (Some(callback), _) => Headers::Callback {
                    callback,
                    user_data: UserData(t.user_data),
                    release_user_data: t.release_user_data,
                },
                (None, _) => {
                    if t.release_user_data.is_some() {
                        return Err(Failure::argument("transform's rules take no release"));
                    }
                    Headers::Rules(patterns(
                        t.rules,
                        t.rule_count,
                        "transform.rules",
                        "header rule",
                        |r: &mln_http_header_rule, what| {
                            let pattern = Pattern::from_c(r.kinds, r.flags, r.url, what)?;
                            if r.header_count > 0 && r.headers.is_null() {
                                return Err(Failure::argument(format!("{what}.headers is null")));
                            }
                            let headers = if r.header_count == 0 {
                                &[][..]
                            } else {
                                std::slice::from_raw_parts(r.headers, r.header_count)
                            };
                            let headers = pairs(
                                headers.iter().map(|h| (h.name, h.value)),
                                &format!("{what}.headers"),
                            )?;
                            for (name, value) in &headers {
                                check_header(name, value).map_err(|e| {
                                    Failure::argument(format!("{what}: {}", e.message))
                                })?;
                            }
                            Ok((pattern, headers))
                        },
                    )?)
                }
            };
            let headers = Arc::new(headers);
            configure(runtime, completion, move |c| c.headers = Some(headers))
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_clear_http_header_transform(
    runtime: mln_runtime,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            configure(runtime, completion, |c| c.headers = None)
        })
    }
}

/// A runtime's hook, made with it and handed to its Orbis runtime.
pub(crate) fn hook() -> Arc<Hook> {
    Arc::new(Hook::default())
}

// MARK: - The ambient cache

/// Runs `job` with the runtime's Orbis runtime on its thread, then
/// completes; an operation needs its completion.
fn operation(
    runtime: mln_runtime,
    completion: *const mln_completion,
    job: impl FnOnce(&orb_api::Runtime) -> Result + Send + 'static,
) -> Result {
    let handle = handles::runtime(runtime.0)?;
    // SAFETY: the caller's pointer.
    let completion = unsafe { Completion::required(completion) }?;
    handle.post(move |state| {
        let result = job(&state.runtime).map(|()| Value::None);
        state.complete(Some(completion), result, 0);
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_run_cache_database_operation(
    runtime: mln_runtime,
    operation: mln_cache_database_operation,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            handles::runtime(runtime.0)?;
            if !(MLN_CACHE_DATABASE_OPERATION_CLEAR..=MLN_CACHE_DATABASE_OPERATION_RESET)
                .contains(&operation)
            {
                return Err(Failure::argument(format!(
                    "{operation} names no cache database operation"
                )));
            }
            self::operation(runtime, completion, move |r| {
                match operation {
                    MLN_CACHE_DATABASE_OPERATION_CLEAR => r.clear_http_cache(),
                    MLN_CACHE_DATABASE_OPERATION_INVALIDATE => r.invalidate_http_cache(),
                    // Orbis keeps each body in a file of its own, which
                    // leaves nothing to compact.
                    MLN_CACHE_DATABASE_OPERATION_PACK => {}
                    _ => {
                        r.clear_http_cache();
                        // A runtime without a cache file has no regions.
                        for region in r.regions().unwrap_or_default() {
                            r.delete_region(region.id).map_err(Failure::from)?;
                        }
                    }
                }
                Ok(())
            })
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_set_maximum_ambient_cache_size(
    runtime: mln_runtime,
    size: u64,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            operation(runtime, completion, move |r| {
                r.set_http_cache_size(size);
                Ok(())
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matches(pattern: &str, url: &str) -> bool {
        let p: Vec<char> = pattern.chars().collect();
        let t: Vec<char> = url.chars().collect();
        glob(&p, &t)
    }

    #[test]
    fn globs_match_as_mln_ffi_documents() {
        assert!(matches(
            "https://*.example.com/tiles/**",
            "https://a.example.com/tiles/1/2/3.pbf"
        ));
        assert!(!matches(
            "https://*.example.com/tiles/**",
            "https://a.b/c.example.com/tiles/1"
        ));
        assert!(matches("custom://?.json", "custom://a.json"));
        assert!(!matches("custom://?.json", "custom:///.json"));
        assert!(matches("a\\*b", "a*b"));
        assert!(!matches("a\\*b", "axb"));
    }

    #[test]
    fn parameters_go_after_the_query_before_the_fragment() {
        let p = [("key".to_string(), "a b".to_string())];
        assert_eq!(with_parameters("http://x/y", &p), "http://x/y?key=a%20b");
        assert_eq!(
            with_parameters("http://x/y?k=1#f", &p),
            "http://x/y?k=1&key=a%20b#f"
        );
    }

    #[test]
    fn redaction_keeps_scheme_host_and_path() {
        assert_eq!(
            redacted("jar://user:password@archive/packaged/style.json?access_token=secret#token"),
            "jar://archive/packaged/style.json"
        );
    }
}
