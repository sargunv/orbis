//! What the HTTP caches of `orb-host-native` and `orb-web` share: how
//! long a response stays fresh, and the key a request is stored under.
//!
//! Both caches answer fresh entries without the network. A stale entry is
//! fetched again (revalidated where the host can send validators) and is
//! answered as it is when the network fails, so a cached map opens offline.

/// The longest heuristic freshness (RFC 9111 §4.2.2) given a response with
/// a `Last-Modified` but no explicit lifetime.
const HEURISTIC_MAX: u64 = 7 * 24 * 3600;

/// The response headers freshness reads.
#[derive(Clone, Copy, Debug, Default)]
pub struct Headers<'a> {
    pub cache_control: Option<&'a str>,
    pub expires: Option<&'a str>,
    pub date: Option<&'a str>,
    pub last_modified: Option<&'a str>,
}

/// Until when (Unix seconds) a response received at `now` is fresh, or
/// `None` when it may not be stored (`no-store`, `private`).
pub fn fresh_until(headers: Headers<'_>, now: u64) -> Option<u64> {
    let directives = headers.cache_control.unwrap_or("");
    let mut max_age = None;
    for directive in directives.split(',').map(str::trim) {
        let (name, value) = directive.split_once('=').unwrap_or((directive, ""));
        match name.to_ascii_lowercase().as_str() {
            "no-store" | "private" => return None,
            "no-cache" => max_age = Some(0),
            "max-age" | "s-maxage" if max_age != Some(0) => {
                max_age = value.trim_matches('"').parse::<u64>().ok().or(max_age);
            }
            _ => {}
        }
    }
    if let Some(age) = max_age {
        return Some(now.saturating_add(age));
    }
    let date = headers.date.and_then(http_date).unwrap_or(now);
    if let Some(expires) = headers.expires {
        // An invalid date, such as "0", means already expired.
        let lifetime = http_date(expires).map_or(0, |e| e.saturating_sub(date));
        return Some(now.saturating_add(lifetime));
    }
    let heuristic = headers
        .last_modified
        .and_then(http_date)
        .map_or(0, |modified| {
            (date.saturating_sub(modified) / 10).min(HEURISTIC_MAX)
        });
    Some(now.saturating_add(heuristic))
}

/// Parses an IMF-fixdate (`Sun, 06 Nov 1994 08:49:37 GMT`), the form
/// servers must send, to Unix seconds.
pub fn http_date(text: &str) -> Option<u64> {
    let mut parts = text.split_ascii_whitespace();
    let _weekday = parts.next()?;
    let day: u64 = parts.next()?.parse().ok()?;
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let month = parts.next()?;
    let month = MONTHS.iter().position(|m| *m == month)? as u64 + 1;
    let year: u64 = parts.next()?.parse().ok()?;
    let mut clock = parts.next()?.split(':').map(|n| n.parse::<u64>().ok());
    let (h, m, s) = (clock.next()??, clock.next()??, clock.next()??);
    if parts.next()? != "GMT" || year < 1970 || !(1..=31).contains(&day) || h > 23 || m > 59 {
        return None;
    }
    // Days from the civil date (Howard Hinnant's algorithm).
    let (y, mp) = if month > 2 {
        (year, month - 3)
    } else {
        (year - 1, month + 9)
    };
    let era = y / 400;
    let yoe = y - era * 400;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + h * 3600 + m * 60 + s)
}

/// Unix seconds as an IMF-fixdate, [`http_date`]'s inverse.
pub fn format_http_date(secs: u64) -> String {
    const DAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let (days, rest) = (secs / 86_400, secs % 86_400);
    // The civil date of a day count (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    format!(
        "{}, {day:02} {} {year} {:02}:{:02}:{:02} GMT",
        DAYS[(days % 7) as usize],
        MONTHS[month as usize - 1],
        rest / 3600,
        rest / 60 % 60,
        rest % 60
    )
}

/// The key a request is cached under: its URL, or for a byte range the URL
/// behind `bytes=<first>-<last> ` (no URL starts with a digit-led token).
pub fn key(url: &str, range: Option<(u64, u64)>) -> String {
    match range {
        Some((offset, length)) => format!("bytes={offset}-{} {url}", offset + length - 1),
        None => url.to_string(),
    }
}

/// Whether a URL is fetched over the network, and so cached.
pub fn cacheable(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_http_dates() {
        assert_eq!(
            http_date("Sun, 06 Nov 1994 08:49:37 GMT"),
            Some(784_111_777)
        );
        assert_eq!(http_date("Thu, 01 Jan 1970 00:00:00 GMT"), Some(0));
        assert_eq!(
            http_date("Tue, 29 Feb 2028 12:00:00 GMT"),
            Some(1_835_438_400)
        );
        assert_eq!(http_date("0"), None);
        for secs in [0, 784_111_777, 1_835_438_400] {
            assert_eq!(http_date(&format_http_date(secs)), Some(secs));
        }
        assert_eq!(
            format_http_date(784_111_777),
            "Sun, 06 Nov 1994 08:49:37 GMT"
        );
        assert_eq!(http_date("Sunday, 06-Nov-94 08:49:37 GMT"), None);
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
        assert_eq!(fresh(Some("private, max-age=60"), None, None), None);
        // Expires counts from the server's Date.
        assert_eq!(
            fresh(None, Some("Mon, 12 Jan 1970 14:46:40 GMT"), None),
            Some(now + 3600)
        );
        assert_eq!(fresh(None, Some("0"), None), Some(now));
        // A tenth of the age since Last-Modified, capped at a week.
        assert_eq!(
            fresh(None, None, Some("Mon, 12 Jan 1970 11:00:00 GMT")),
            Some(now + 1000)
        );
        assert_eq!(
            fresh(None, None, Some("Thu, 01 Jan 1970 00:00:00 GMT")),
            Some(now + 100_000)
        );
        assert_eq!(fresh(None, None, None), Some(now));
    }
}
