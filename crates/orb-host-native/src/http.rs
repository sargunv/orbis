//! What the native host reads from HTTP: how long a response stays fresh
//! in its [`HttpCache`](crate::HttpCache), and HTTP dates.
//!
//! The cache is a private one (RFC 9111 §3): it serves one user, so it
//! stores `private` responses and ignores what only shared caches obey.

/// The longest heuristic freshness (RFC 9111 §4.2.2) given a response with
/// a `Last-Modified` but no explicit lifetime.
const HEURISTIC_MAX: u64 = 7 * 24 * 3600;

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

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

/// Until when (Unix seconds) a response received at `now` is fresh, or
/// `None` when it may not be stored (`no-store`).
pub(crate) fn fresh_until(headers: Headers<'_>, now: u64) -> Option<u64> {
    match orb_host::MaxAge::of(headers.cache_control.unwrap_or("")) {
        orb_host::MaxAge::NoStore => return None,
        orb_host::MaxAge::Seconds(age) => return Some(now.saturating_add(age)),
        orb_host::MaxAge::Unsaid => {}
    }
    let date = headers.date.and_then(|d| http_date(d, now)).unwrap_or(now);
    if let Some(expires) = headers.expires {
        // An invalid date, such as "0", means already expired.
        let lifetime = http_date(expires, now).map_or(0, |e| e.saturating_sub(date));
        return Some(now.saturating_add(lifetime));
    }
    let heuristic = headers
        .last_modified
        .and_then(|m| http_date(m, now))
        .map_or(0, |modified| {
            (date.saturating_sub(modified) / 10).min(HEURISTIC_MAX)
        });
    Some(now.saturating_add(heuristic))
}

/// Whether the response's headers give its lifetime (`max-age`,
/// `no-cache` or `Expires`), rather than the cache guessing it from its
/// `Last-Modified`: only a given lifetime has a map load what it draws
/// from the response again ([`orb_host::Fetched::fresh_until`]), as the
/// web host, GL JS and MapLibre Native read only those.
pub(crate) fn explicit(headers: Headers<'_>) -> bool {
    match orb_host::MaxAge::of(headers.cache_control.unwrap_or("")) {
        orb_host::MaxAge::Seconds(_) => true,
        orb_host::MaxAge::NoStore => false,
        orb_host::MaxAge::Unsaid => headers.expires.is_some(),
    }
}

/// Parses an HTTP date to Unix seconds, in any of the forms RFC 9110
/// §5.6.7 has recipients read: an IMF-fixdate (`Sun, 06 Nov 1994 08:49:37
/// GMT`), the form servers send; the obsolete RFC 850 form (`Sunday,
/// 06-Nov-94 08:49:37 GMT`), whose two-digit year is the latest one no
/// more than 50 years after `now`; and C's asctime (`Sun Nov  6 08:49:37
/// 1994`).
pub(crate) fn http_date(text: &str, now: u64) -> Option<u64> {
    let mut parts = text.split_ascii_whitespace();
    let weekday = parts.next()?;
    let (day, month, year, clock) = if weekday.ends_with(',') {
        let first = parts.next()?;
        let (day, month, year) = match first.split_once('-') {
            Some((day, rest)) => {
                let (month, year) = rest.split_once('-')?;
                (day, month, year)
            }
            None => (first, parts.next()?, parts.next()?),
        };
        let clock = parts.next()?;
        if parts.next()? != "GMT" {
            return None;
        }
        (day, month, year, clock)
    } else {
        let (month, day, clock) = (parts.next()?, parts.next()?, parts.next()?);
        (day, month, parts.next()?, clock)
    };
    if parts.next().is_some() {
        return None;
    }
    let day: u64 = day.parse().ok()?;
    let month = MONTHS.iter().position(|m| *m == month)? as u64 + 1;
    let mut year: u64 = year.parse().ok()?;
    if year < 100 {
        let latest = civil(now / 86_400).0 + 50;
        year += (latest - year) / 100 * 100;
    }
    let mut clock = clock.split(':').map(|n| n.parse::<u64>().ok());
    let (h, m, s) = (clock.next()??, clock.next()??, clock.next()??);
    if clock.next().is_some()
        || !(1970..=9999).contains(&year)
        || !(1..=31).contains(&day)
        || h > 23
        || m > 59
        || s > 60
    {
        return None;
    }
    Some(days(year, month, day) * 86_400 + h * 3600 + m * 60 + s)
}

/// Unix seconds as an IMF-fixdate, [`http_date`]'s inverse.
pub(crate) fn format_http_date(secs: u64) -> String {
    const DAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
    let (days, rest) = (secs / 86_400, secs % 86_400);
    let (year, month, day) = civil(days);
    format!(
        "{}, {day:02} {} {year} {:02}:{:02}:{:02} GMT",
        DAYS[(days % 7) as usize],
        MONTHS[month as usize - 1],
        rest / 3600,
        rest / 60 % 60,
        rest % 60
    )
}

/// Days since the Unix epoch of a civil date from 1970 on (Howard
/// Hinnant's algorithm).
fn days(year: u64, month: u64, day: u64) -> u64 {
    let (y, mp) = if month > 2 {
        (year, month - 3)
    } else {
        (year - 1, month + 9)
    };
    let era = y / 400;
    let yoe = y - era * 400;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The civil date (year, month, day) of a day count, [`days`]' inverse.
fn civil(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + u64::from(month <= 2), month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-11-25.
    const NOW: u64 = 1_795_564_800;

    #[test]
    fn parses_http_dates() {
        let date = |text: &str| http_date(text, NOW);
        assert_eq!(date("Sun, 06 Nov 1994 08:49:37 GMT"), Some(784_111_777));
        assert_eq!(date("Thu, 01 Jan 1970 00:00:00 GMT"), Some(0));
        assert_eq!(date("Tue, 29 Feb 2028 12:00:00 GMT"), Some(1_835_438_400));
        assert_eq!(date("0"), None);
        assert_eq!(date("Sun, 06 Nov 1994 08:49:37 PST"), None);
        assert_eq!(date("Sun, 06 Nov 1994 08:49:37 GMT junk"), None);
        assert_eq!(date("Sun, 06 Nov 999999999999 08:49:37 GMT"), None);
        assert_eq!(
            date("Sun, 06 Nov 1994 08:49:18446744073709551615 GMT"),
            None
        );
        for secs in [0, 784_111_777, 1_835_438_400] {
            assert_eq!(date(&format_http_date(secs)), Some(secs));
        }
        assert_eq!(
            format_http_date(784_111_777),
            "Sun, 06 Nov 1994 08:49:37 GMT"
        );
    }

    #[test]
    fn parses_obsolete_http_dates() {
        let date = |text: &str| http_date(text, NOW);
        assert_eq!(date("Sunday, 06-Nov-94 08:49:37 GMT"), Some(784_111_777));
        assert_eq!(date("Sun Nov  6 08:49:37 1994"), Some(784_111_777));
        assert_eq!(date("Tuesday, 29-Feb-28 12:00:00 GMT"), Some(1_835_438_400));
        // A two-digit year is the latest no more than 50 years ahead.
        assert_eq!(
            date("Monday, 01-Jan-76 00:00:00 GMT"),
            date("Wed, 01 Jan 2076 00:00:00 GMT")
        );
        assert_eq!(
            date("Monday, 01-Jan-77 00:00:00 GMT"),
            date("Sat, 01 Jan 1977 00:00:00 GMT")
        );
    }

    #[test]
    fn freshness_follows_rfc_9111() {
        let now = 1_000_000;
        let fresh = |cache_control, expires, last_modified| {
            fresh_until(
                Headers {
                    cache_control,
                    expires,
                    date: Some("Mon, 12 Jan 1970 13:46:40 GMT"),
                    last_modified,
                },
                now,
            )
        };
        assert_eq!(
            fresh(Some("public, max-age=600"), None, None),
            Some(now + 600)
        );
        assert_eq!(fresh(Some("max-age=600, no-cache"), None, None), Some(now));
        assert_eq!(fresh(Some("no-store"), None, None), None);
        // Only a lifetime the headers give is explicit.
        let explicit = |cache_control, expires, last_modified| {
            super::explicit(Headers {
                cache_control,
                expires,
                date: None,
                last_modified,
            })
        };
        assert!(explicit(Some("max-age=60"), None, None));
        assert!(explicit(None, Some("Mon, 12 Jan 1970 14:46:40 GMT"), None));
        assert!(!explicit(None, None, Some("Mon, 12 Jan 1970 13:46:40 GMT")));
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
