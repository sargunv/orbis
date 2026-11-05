//! The native HTTP cache: each response body in a file of its own under
//! `blobs/`, and one index of keys, freshness, validators and use times,
//! loaded once when the cache opens and rewritten whole, at most every
//! [`FLUSH_EVERY`], by a thread of its own. Nothing is fsynced: a crash
//! loses at most the last batch of index changes, and the blobs those
//! changes named are swept on the next open (plan §18 row 15: Native's
//! cold open spends 1,011 ms of 1,327 ms in fsync).
//!
//! Entries beyond the size limit go least recently used first.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

/// How often index changes are written.
const FLUSH_EVERY: Duration = Duration::from_secs(2);
const INDEX_VERSION: &str = "orbis-cache 1";

/// Seconds since the Unix epoch.
pub(crate) fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// An HTTP cache in a directory, shared by the fetches of a process.
#[derive(Debug)]
pub struct HttpCache {
    inner: Arc<Inner>,
    flusher: Option<std::thread::JoinHandle<()>>,
}

#[derive(Debug)]
struct Inner {
    dir: PathBuf,
    max_bytes: u64,
    state: Mutex<State>,
    changed: Condvar,
}

#[derive(Debug, Default)]
struct State {
    entries: HashMap<String, Entry>,
    next_id: u64,
    bytes: u64,
    dirty: bool,
    closing: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct Entry {
    id: u64,
    size: u64,
    fresh_until: u64,
    last_used: u64,
    etag: Option<String>,
    last_modified: Option<String>,
}

/// A cached response.
#[derive(Debug)]
pub(crate) struct Cached {
    pub(crate) bytes: Vec<u8>,
    pub(crate) fresh: bool,
    pub(crate) etag: Option<String>,
    pub(crate) last_modified: Option<String>,
}

/// What a cache holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub entries: usize,
    pub bytes: u64,
}

impl HttpCache {
    /// Opens (or creates) the cache in `dir`, keeping at most `max_bytes`
    /// of response bodies.
    pub fn open(dir: &Path, max_bytes: u64) -> std::io::Result<Arc<Self>> {
        std::fs::create_dir_all(dir.join("blobs"))?;
        let mut state = std::fs::read_to_string(dir.join("index"))
            .ok()
            .and_then(|text| parse_index(&text))
            .unwrap_or_default();
        state.bytes = state.entries.values().map(|e| e.size).sum();
        let swept_below = state.next_id;
        let inner = Arc::new(Inner {
            dir: dir.to_owned(),
            max_bytes,
            state: Mutex::new(state),
            changed: Condvar::new(),
        });
        let flusher = {
            let inner = inner.clone();
            std::thread::Builder::new()
                .name("cache".into())
                .spawn(move || {
                    inner.sweep(swept_below);
                    inner.flush_loop();
                })?
        };
        Ok(Arc::new(Self {
            inner,
            flusher: Some(flusher),
        }))
    }

    /// What the cache holds now.
    pub fn stats(&self) -> CacheStats {
        let state = self.inner.lock();
        CacheStats {
            entries: state.entries.len(),
            bytes: state.bytes,
        }
    }

    /// Writes pending index changes now.
    pub fn flush(&self) -> std::io::Result<()> {
        self.inner.write_index()
    }

    /// Removes every entry.
    pub fn clear(&self) {
        let ids: Vec<_> = {
            let mut state = self.inner.lock();
            state.bytes = 0;
            state.dirty = true;
            state.entries.drain().map(|(_, e)| e.id).collect()
        };
        self.inner.changed.notify_one();
        self.inner.remove_blobs(ids);
    }

    pub(crate) fn lookup(&self, key: &str) -> Option<Cached> {
        let now = now();
        let entry = {
            let mut state = self.inner.lock();
            let entry = state.entries.get_mut(key)?;
            entry.last_used = now;
            entry.clone()
        };
        match std::fs::read(self.inner.blob(entry.id)) {
            Ok(bytes) if bytes.len() as u64 == entry.size => Some(Cached {
                bytes,
                fresh: entry.fresh_until > now,
                etag: entry.etag,
                last_modified: entry.last_modified,
            }),
            _ => {
                self.remove_if(key, entry.id);
                None
            }
        }
    }

    pub(crate) fn store(
        &self,
        key: &str,
        bytes: &[u8],
        fresh_until: u64,
        etag: Option<&str>,
        last_modified: Option<&str>,
    ) {
        if key.contains(['\t', '\n']) || bytes.len() as u64 > self.inner.max_bytes / 4 {
            return;
        }
        let id = {
            let mut state = self.inner.lock();
            state.next_id += 1;
            state.next_id
        };
        let path = self.inner.blob(id);
        if crate::write_atomically(&path, bytes).is_err() {
            return;
        }
        let header = |v: Option<&str>| v.filter(|v| !v.contains(['\t', '\n'])).map(str::to_owned);
        let entry = Entry {
            id,
            size: bytes.len() as u64,
            fresh_until,
            last_used: now(),
            etag: header(etag),
            last_modified: header(last_modified),
        };
        let mut removed = Vec::new();
        {
            let mut state = self.inner.lock();
            state.bytes += entry.size;
            if let Some(old) = state.entries.insert(key.to_owned(), entry) {
                state.bytes -= old.size;
                removed.push(old.id);
            }
            if state.bytes > self.inner.max_bytes {
                removed.extend(evict(&mut state, self.inner.max_bytes / 10 * 9));
            }
            state.dirty = true;
        }
        self.inner.changed.notify_one();
        self.inner.remove_blobs(removed);
    }

    /// Marks an entry fresh again after the server answered 304.
    pub(crate) fn refresh(&self, key: &str, fresh_until: u64) {
        let mut state = self.inner.lock();
        if let Some(entry) = state.entries.get_mut(key) {
            entry.fresh_until = fresh_until;
            state.dirty = true;
            self.inner.changed.notify_one();
        }
    }

    pub(crate) fn remove(&self, key: &str) {
        let removed = {
            let mut state = self.inner.lock();
            let removed = state.entries.remove(key);
            if let Some(entry) = &removed {
                state.bytes -= entry.size;
                state.dirty = true;
            }
            removed
        };
        if let Some(entry) = removed {
            self.inner.changed.notify_one();
            self.inner.remove_blobs([entry.id]);
        }
    }

    /// Removes `key` if it is still the entry `id`, whose blob is gone.
    fn remove_if(&self, key: &str, id: u64) {
        let mut state = self.inner.lock();
        if state.entries.get(key).is_some_and(|e| e.id == id) {
            let entry = state.entries.remove(key).expect("entry present");
            state.bytes -= entry.size;
            state.dirty = true;
        }
    }
}

impl Drop for HttpCache {
    fn drop(&mut self) {
        self.inner.lock().closing = true;
        self.inner.changed.notify_one();
        if let Some(flusher) = self.flusher.take() {
            let _ = flusher.join();
        }
    }
}

/// Drops least recently used entries until `target` bytes remain, and
/// returns their blob ids.
fn evict(state: &mut State, target: u64) -> Vec<u64> {
    let mut by_use: Vec<_> = state
        .entries
        .iter()
        .map(|(key, e)| (e.last_used, e.id, key.clone()))
        .collect();
    by_use.sort_unstable();
    let mut removed = Vec::new();
    for (_, id, key) in by_use {
        if state.bytes <= target {
            break;
        }
        let entry = state.entries.remove(&key).expect("entry present");
        state.bytes -= entry.size;
        removed.push(id);
    }
    removed
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("cache poisoned")
    }

    fn blob(&self, id: u64) -> PathBuf {
        self.dir.join("blobs").join(id.to_string())
    }

    fn remove_blobs(&self, ids: impl IntoIterator<Item = u64>) {
        for id in ids {
            let _ = std::fs::remove_file(self.blob(id));
        }
    }

    /// Deletes blobs older than this open that the index doesn't name: a
    /// crash's lost index changes, and partial writes.
    fn sweep(&self, below: u64) {
        let Ok(files) = std::fs::read_dir(self.dir.join("blobs")) else {
            return;
        };
        let live: std::collections::HashSet<u64> =
            self.lock().entries.values().map(|e| e.id).collect();
        for file in files.flatten() {
            let name = file.file_name();
            let name = name.to_string_lossy();
            // A blob `<id>`, or `write_atomically`'s `<id>.<pid>.<n>.partial`;
            // newer ids are this open's own writes.
            let (id, partial) = name
                .split_once('.')
                .map_or((&*name, false), |(id, _)| (id, true));
            if id
                .parse::<u64>()
                .is_ok_and(|id| id <= below && (partial || !live.contains(&id)))
            {
                let _ = std::fs::remove_file(file.path());
            }
        }
    }

    fn flush_loop(&self) {
        loop {
            {
                let mut state = self.lock();
                while !state.dirty && !state.closing {
                    state = self.changed.wait(state).expect("cache poisoned");
                }
                if !state.closing {
                    let (state, _) = self
                        .changed
                        .wait_timeout_while(state, FLUSH_EVERY, |s| !s.closing)
                        .expect("cache poisoned");
                    drop(state);
                }
            }
            let _ = self.write_index();
            if self.lock().closing {
                return;
            }
        }
    }

    fn write_index(&self) -> std::io::Result<()> {
        let text = {
            let mut state = self.lock();
            if !state.dirty {
                return Ok(());
            }
            state.dirty = false;
            index_text(&state)
        };
        crate::write_atomically(&self.dir.join("index"), text.as_bytes())
    }
}

/// The index: a version line, the next blob id, then one line per entry of
/// tab-separated id, size, freshness, last use, ETag, Last-Modified and
/// key (keys and headers with tabs or newlines aren't cached).
fn index_text(state: &State) -> String {
    let mut text = format!("{INDEX_VERSION}\n{}\n", state.next_id);
    for (key, e) in &state.entries {
        let _ = writeln!(
            text,
            "{}\t{}\t{}\t{}\t{}\t{}\t{key}",
            e.id,
            e.size,
            e.fresh_until,
            e.last_used,
            e.etag.as_deref().unwrap_or(""),
            e.last_modified.as_deref().unwrap_or(""),
        );
    }
    text
}

fn parse_index(text: &str) -> Option<State> {
    let mut lines = text.lines();
    if lines.next()? != INDEX_VERSION {
        return None;
    }
    let mut state = State {
        next_id: lines.next()?.parse().ok()?,
        ..State::default()
    };
    let text = |s: &str| (!s.is_empty()).then(|| s.to_owned());
    for line in lines {
        let mut f = line.splitn(7, '\t');
        let mut number = || f.next()?.parse::<u64>().ok();
        let (id, size, fresh_until, last_used) = (number()?, number()?, number()?, number()?);
        let (etag, last_modified, key) = (f.next()?, f.next()?, f.next()?);
        state.entries.insert(
            key.to_owned(),
            Entry {
                id,
                size,
                fresh_until,
                last_used,
                etag: text(etag),
                last_modified: text(last_modified),
            },
        );
    }
    Some(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("orb-cache-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn entries_survive_reopening_and_stale_ones_say_so() {
        let dir = dir("reopen");
        let now = now();
        {
            let cache = HttpCache::open(&dir, 1 << 20).unwrap();
            cache.store("https://a/fresh", b"one", now + 60, Some("\"e1\""), None);
            cache.store(
                "https://a/stale",
                b"two",
                now - 1,
                None,
                Some("Mon, 12 Jan 1970 13:46:40 GMT"),
            );
            assert_eq!(
                cache.stats(),
                CacheStats {
                    entries: 2,
                    bytes: 6
                }
            );
        }
        let cache = HttpCache::open(&dir, 1 << 20).unwrap();
        let fresh = cache.lookup("https://a/fresh").unwrap();
        assert_eq!(
            (fresh.bytes.as_slice(), fresh.fresh, fresh.etag.as_deref()),
            (&b"one"[..], true, Some("\"e1\""))
        );
        let stale = cache.lookup("https://a/stale").unwrap();
        assert!(!stale.fresh);
        assert!(stale.last_modified.is_some());
        assert!(cache.lookup("https://a/missing").is_none());
        drop(cache);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn evicts_least_recently_used_and_sweeps_orphans() {
        let dir = dir("evict");
        let now = now();
        let cache = HttpCache::open(&dir, 400).unwrap();
        for i in 0..4 {
            cache.store(&format!("https://a/{i}"), &[i; 100], now + 60, None, None);
            // Use the first entry every time: it stays.
            cache
                .inner
                .lock()
                .entries
                .get_mut("https://a/0")
                .unwrap()
                .last_used = now + 10;
        }
        cache.store("https://a/4", &[4; 100], now + 60, None, None);
        let stats = cache.stats();
        assert!(stats.bytes <= 360, "{stats:?}");
        assert!(cache.lookup("https://a/0").is_some());
        assert!(cache.lookup("https://a/1").is_none());
        assert!(cache.lookup("https://a/4").is_some());
        let blobs = std::fs::read_dir(dir.join("blobs")).unwrap().count();
        assert_eq!(blobs, cache.stats().entries);
        // A blob the index lost in a crash goes on the next open.
        drop(cache);
        std::fs::write(dir.join("blobs/2"), b"orphan").unwrap();
        let cache = HttpCache::open(&dir, 400).unwrap();
        cache.flush().unwrap();
        drop(cache);
        // The sweep runs on the cache thread, before the drop's join.
        assert!(!dir.join("blobs/2").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
