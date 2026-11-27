//! When to try again what failed: the one backoff failed tiles, reloads
//! of expired tiles and glyph ranges share.

/// The first retry's delay after a failure (ms); each further failure in
/// a row doubles it, up to [`RETRY_MAX_MS`].
const RETRY_MS: f64 = 1000.0;
const RETRY_MAX_MS: f64 = 60_000.0;
/// The longest wait a server's `Retry-After` is taken at: a day. One
/// asking for longer is asked again then rather than at a time that
/// never comes (or past what the clock can hold).
const RETRY_AFTER_MAX_MS: f64 = 86_400_000.0;

/// Failures in a row of something fetched, and when it's due to be tried
/// again: a second after the first failure, doubling with each further
/// one up to a minute, and never sooner than its server asked
/// (`Retry-After`), up to a day. Times are milliseconds on one monotonic
/// clock.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Backoff {
    failures: u32,
    /// When the next try is due, while one waits.
    due: Option<f64>,
}

impl Backoff {
    /// Counts a failure at `now`, or a try starting then that counts as
    /// one until it succeeds: the next is due after the backoff, or after
    /// the `wait` its server asked for when that's longer.
    pub fn fail(&mut self, now: f64, wait: f64) {
        let delay = (RETRY_MS * 2f64.powi(self.failures.min(16) as i32)).min(RETRY_MAX_MS);
        self.due = Some(now + delay.max(wait.min(RETRY_AFTER_MAX_MS)));
        self.failures = self.failures.saturating_add(1);
    }

    /// The try under way, counted already ([`fail`](Self::fail)), was
    /// answered at `now` with a `wait` (`Retry-After`): the next is due no
    /// sooner, up to a day. With none under way (another try succeeded
    /// since), nothing changes.
    pub fn defer(&mut self, now: f64, wait: f64) {
        if self.failures == 0 {
            return;
        }
        let at = now + wait.clamp(0.0, RETRY_AFTER_MAX_MS);
        self.due = Some(self.due.map_or(at, |due| due.max(at)));
    }

    /// The next try, if one waits, is due at once.
    pub fn hurry(&mut self) {
        if let Some(due) = &mut self.due {
            *due = f64::NEG_INFINITY;
        }
    }

    /// When the next try is due, while one waits.
    pub fn due(&self) -> Option<f64> {
        self.due
    }

    /// Whether a try is due at `now`, which then starts: it waits no more,
    /// and its failures stay counted for the next one's backoff.
    pub fn take(&mut self, now: f64) -> bool {
        let due = self.due.is_some_and(|due| due <= now);
        if due {
            self.due = None;
        }
        due
    }

    /// Whether the last try failed.
    pub fn failed(&self) -> bool {
        self.failures > 0
    }

    /// The last try succeeded: nothing waits, and the next failure backs
    /// off from the start.
    pub fn succeeded(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failures_in_a_row_double_the_wait_up_to_a_minute() {
        let mut b = Backoff::default();
        assert!(!b.take(f64::INFINITY), "nothing failed");
        let waits: Vec<f64> = (0..9)
            .map(|_| {
                b.fail(0.0, 0.0);
                let due = b.due().unwrap();
                assert!(!b.take(due - 1.0) && b.take(due));
                due
            })
            .collect();
        assert_eq!(
            waits,
            [
                1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0, 60000.0, 60000.0, 60000.0
            ]
        );
        assert!(b.failed());
        b.succeeded();
        assert!(!b.failed());
        b.fail(0.0, 0.0);
        assert_eq!(b.due(), Some(1000.0), "a success starts it over");
    }

    #[test]
    fn a_server_s_wait_outlasts_the_backoff_up_to_a_day() {
        let mut b = Backoff::default();
        b.fail(0.0, 30_000.0);
        assert_eq!(b.due(), Some(30_000.0));
        assert!(b.take(30_000.0));
        b.fail(30_000.0, 500.0);
        assert_eq!(b.due(), Some(32_000.0), "a shorter wait doesn't shorten it");
        b.fail(0.0, f64::MAX);
        assert_eq!(b.due(), Some(86_400_000.0));
    }

    /// A try counted as it starts, which its server then puts off.
    #[test]
    fn a_wait_answering_a_try_under_way_puts_it_off_without_counting() {
        let mut b = Backoff::default();
        b.fail(0.0, 0.0);
        b.defer(200.0, 30_000.0);
        assert_eq!(b.due(), Some(30_200.0));
        b.defer(300.0, 0.0);
        assert_eq!(b.due(), Some(30_200.0), "never sooner");
        assert!(b.take(30_200.0));
        b.fail(30_200.0, 0.0);
        assert_eq!(b.due(), Some(32_200.0), "the second failure's wait");
        b.succeeded();
        b.defer(32_500.0, 30_000.0);
        assert_eq!(b.due(), None, "no try under way to put off");
    }

    #[test]
    fn hurrying_makes_a_waiting_try_due_now() {
        let mut b = Backoff::default();
        b.hurry();
        assert!(!b.take(0.0), "nothing waits");
        b.fail(0.0, 30_000.0);
        b.hurry();
        assert!(b.take(0.0));
        b.fail(0.0, 0.0);
        assert_eq!(b.due(), Some(2000.0), "its failures stay counted");
    }
}
