//! What both hosts read from HTTP responses: `Cache-Control`, HTTP dates
//! and `Retry-After`.

use std::time::Duration;

/// What a response's `Cache-Control` says to a private cache (RFC 9111
/// §5.2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheControl {
    pub max_age: MaxAge,
    /// Once stale, it answers nothing before the server revalidates it
    /// (`must-revalidate`, `no-cache`).
    pub must_revalidate: bool,
}

/// How long a response stays fresh, as its `Cache-Control` says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaxAge {
    /// `no-store`: it has no lifetime.
    NoStore,
    /// Fresh this many seconds from its arrival: its `max-age`, 0 for
    /// `no-cache`.
    Seconds(u64),
    /// Neither: `Expires`, or a heuristic, decides.
    Unsaid,
}

impl CacheControl {
    /// What the `Cache-Control` header `directives` says.
    pub fn of(directives: &str) -> Self {
        let (mut max_age, mut must_revalidate, mut no_store) = (MaxAge::Unsaid, false, false);
        for directive in directives.split(',').map(str::trim) {
            let (name, value) = directive.split_once('=').unwrap_or((directive, ""));
            match name.trim().to_ascii_lowercase().as_str() {
                "no-store" => no_store = true,
                "no-cache" => {
                    max_age = MaxAge::Seconds(0);
                    must_revalidate = true;
                }
                "must-revalidate" => must_revalidate = true,
                "max-age" if max_age != MaxAge::Seconds(0) => {
                    if let Ok(seconds) = value.trim_matches('"').parse() {
                        max_age = MaxAge::Seconds(seconds);
                    }
                }
                _ => {}
            }
        }
        if no_store {
            max_age = MaxAge::NoStore;
        }
        Self {
            max_age,
            must_revalidate,
        }
    }
}

/// The wait an HTTP `Retry-After` header asks for at `now` (Unix
/// seconds), RFC 9110 §10.2.3: whole seconds, or until an HTTP date, zero
/// once past. `None` when it reads as neither.
pub fn retry_after(header: &str, now: u64) -> Option<Duration> {
    let header = header.trim();
    match header.parse::<u64>() {
        Ok(seconds) => Some(Duration::from_secs(seconds)),
        Err(_) => parse_date(header, now).map(|at| Duration::from_secs(at.saturating_sub(now))),
    }
}

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// Parses an HTTP date to Unix seconds, in any of the forms RFC 9110
/// §5.6.7 has recipients read: an IMF-fixdate (`Sun, 06 Nov 1994 08:49:37
/// GMT`), the form servers send; the obsolete RFC 850 form (`Sunday,
/// 06-Nov-94 08:49:37 GMT`), whose two-digit year is the latest one no
/// more than 50 years after `now`; and C's asctime (`Sun Nov  6 08:49:37
/// 1994`).
pub fn parse_date(text: &str, now: u64) -> Option<u64> {
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

/// Unix seconds as an IMF-fixdate, [`parse_date`]'s inverse.
pub fn format_date(secs: u64) -> String {
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
    fn cache_control_says_how_long_a_response_stays_fresh() {
        let max_age = |directives| CacheControl::of(directives).max_age;
        assert_eq!(max_age("public, max-age=300"), MaxAge::Seconds(300));
        assert_eq!(max_age("Max-Age=\"60\""), MaxAge::Seconds(60));
        assert_eq!(max_age("no-cache, max-age=300"), MaxAge::Seconds(0));
        assert_eq!(max_age("max-age=300, no-store"), MaxAge::NoStore);
        assert_eq!(max_age("max-age=soon"), MaxAge::Unsaid);
        assert_eq!(max_age("private"), MaxAge::Unsaid);
        // A stale copy may stand in while it revalidates unless the
        // server forbids it.
        let must_revalidate = |directives| CacheControl::of(directives).must_revalidate;
        assert!(must_revalidate("max-age=60, must-revalidate"));
        assert!(must_revalidate("No-Cache"));
        assert!(must_revalidate("no-cache=\"Set-Cookie\", max-age=60"));
        assert!(!must_revalidate("max-age=60, proxy-revalidate"));
        assert!(!must_revalidate("max-age=0"));
    }

    #[test]
    fn retry_after_reads_seconds_or_a_date() {
        assert_eq!(retry_after(" 120 ", NOW), Some(Duration::from_secs(120)));
        assert_eq!(retry_after("-1", NOW), None);
        assert_eq!(retry_after("soon", NOW), None);
        let past = "Sun, 06 Nov 1994 08:49:37 GMT";
        assert_eq!(retry_after(past, NOW), Some(Duration::ZERO));
        let soon = format_date(NOW + 30);
        assert_eq!(retry_after(&soon, NOW), Some(Duration::from_secs(30)));
    }

    #[test]
    fn parses_http_dates() {
        let date = |text: &str| parse_date(text, NOW);
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
            assert_eq!(date(&format_date(secs)), Some(secs));
        }
        assert_eq!(format_date(784_111_777), "Sun, 06 Nov 1994 08:49:37 GMT");
    }

    #[test]
    fn parses_obsolete_http_dates() {
        let date = |text: &str| parse_date(text, NOW);
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
}
