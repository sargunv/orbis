//! How long a response stays fresh in the native host's
//! [`HttpCache`](crate::HttpCache).
//!
//! The cache is a private one (RFC 9111 §3): it serves one user, so it
//! stores `private` responses and ignores what only shared caches obey.

/// The longest heuristic freshness (RFC 9111 §4.2.2) given a response with
/// a `Last-Modified` but no explicit lifetime.
const HEURISTIC_MAX: u64 = 7 * 24 * 3600;

use orb_host::http::{CacheControl, MaxAge, parse_date};

/// Whether a URL is fetched over HTTP, and so cached.
pub(crate) fn cacheable(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

/// The response headers freshness reads.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Headers<'a> {
    pub(crate) cache_control: Option<&'a str>,
    pub(crate) expires: Option<&'a str>,
    pub(crate) date: Option<&'a str>,
    pub(crate) last_modified: Option<&'a str>,
}

/// How long a response stays fresh in the cache, as its headers say.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Freshness {
    /// Until when (Unix seconds).
    pub(crate) until: u64,
    /// The headers gave `until` (`max-age`, `no-cache` or `Expires`),
    /// rather than the cache guessing it from `Last-Modified`: only a
    /// given lifetime has a map load what it draws from the response
    /// again ([`orb_host::Fetched::fresh_until`]), as the web host, GL JS
    /// and MapLibre Native read only those.
    pub(crate) explicit: bool,
    /// Once stale, it answers nothing before the server revalidates it
    /// (`must-revalidate`, `no-cache`), not even while a map revalidates
    /// it ([`orb_host::Fetch::get_or_stale`]).
    pub(crate) must_revalidate: bool,
}

/// How long a response received at `now` stays fresh, or `None` when it
/// may not be stored (`no-store`).
pub(crate) fn freshness(headers: Headers<'_>, now: u64) -> Option<Freshness> {
    let cache_control = CacheControl::of(headers.cache_control.unwrap_or(""));
    let fresh = |until, explicit| Freshness {
        until,
        explicit,
        must_revalidate: cache_control.must_revalidate,
    };
    match cache_control.max_age {
        MaxAge::NoStore => return None,
        MaxAge::Seconds(age) => return Some(fresh(now.saturating_add(age), true)),
        MaxAge::Unsaid => {}
    }
    let date = headers.date.and_then(|d| parse_date(d, now)).unwrap_or(now);
    if let Some(expires) = headers.expires {
        // An invalid date, such as "0", means already expired.
        let lifetime = parse_date(expires, now).map_or(0, |e| e.saturating_sub(date));
        return Some(fresh(now.saturating_add(lifetime), true));
    }
    let heuristic = headers
        .last_modified
        .and_then(|m| parse_date(m, now))
        .map_or(0, |modified| {
            (date.saturating_sub(modified) / 10).min(HEURISTIC_MAX)
        });
    Some(fresh(now.saturating_add(heuristic), false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freshness_follows_rfc_9111() {
        let now = 1_000_000;
        let freshness = |cache_control, expires, last_modified| {
            super::freshness(
                Headers {
                    cache_control,
                    expires,
                    date: Some("Mon, 12 Jan 1970 13:46:40 GMT"),
                    last_modified,
                },
                now,
            )
        };
        let fresh = |cache_control, expires, last_modified| {
            freshness(cache_control, expires, last_modified).map(|f| f.until)
        };
        assert_eq!(
            fresh(Some("public, max-age=600"), None, None),
            Some(now + 600)
        );
        assert_eq!(fresh(Some("max-age=600, no-cache"), None, None), Some(now));
        assert_eq!(fresh(Some("no-store"), None, None), None);
        // Only a lifetime the headers give is explicit.
        let explicit = |cache_control, expires, last_modified| {
            freshness(cache_control, expires, last_modified).is_some_and(|f| f.explicit)
        };
        assert!(explicit(Some("max-age=60"), None, None));
        assert!(explicit(None, Some("Mon, 12 Jan 1970 14:46:40 GMT"), None));
        assert!(!explicit(None, None, Some("Mon, 12 Jan 1970 13:46:40 GMT")));
        assert!(
            freshness(Some("max-age=60, must-revalidate"), None, None)
                .unwrap()
                .must_revalidate
        );
        // A private cache stores private responses and ignores s-maxage.
        assert_eq!(
            fresh(Some("private, max-age=60"), None, None),
            Some(now + 60)
        );
        assert_eq!(
            fresh(Some("max-age=60, s-maxage=600"), None, None),
            Some(now + 60)
        );
        // Expires counts from the server's Date, in any HTTP date form.
        assert_eq!(
            fresh(None, Some("Mon, 12 Jan 1970 14:46:40 GMT"), None),
            Some(now + 3600)
        );
        assert_eq!(
            fresh(None, Some("Monday, 12-Jan-70 14:46:40 GMT"), None),
            Some(now + 3600)
        );
        assert_eq!(fresh(None, Some("0"), None), Some(now));
        // A tenth of the age since Last-Modified, capped at a week.
        assert_eq!(
            fresh(None, None, Some("Mon, 12 Jan 1970 11:00:00 GMT")),
            Some(now + 1000)
        );
        assert_eq!(
            fresh(None, None, Some("Mon Jan 12 11:00:00 1970")),
            Some(now + 1000)
        );
        assert_eq!(
            fresh(None, None, Some("Thu, 01 Jan 1970 00:00:00 GMT")),
            Some(now + 100_000)
        );
        assert_eq!(fresh(None, None, None), Some(now));
    }
}
