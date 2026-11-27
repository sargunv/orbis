//! A test's outcome as its report's `status` gives it. xtask's `support`
//! reads this file too (by `#[path]`), so a report it can't read fails
//! there rather than miscounting.

use serde_json::Value;

/// How a test came out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    Pass,
    Fail,
    /// Out of scope, for its report's `skip_reasons`.
    Skip,
}

impl Status {
    /// Its name in a report.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Status::Pass => "pass",
            Status::Fail => "fail",
            Status::Skip => "skip",
        }
    }

    /// Pass or fail as `passed` says.
    pub(crate) fn passed(passed: bool) -> Self {
        if passed { Status::Pass } else { Status::Fail }
    }

    /// The `status` of `test`, a report's entry.
    pub(crate) fn of(test: &Value) -> Result<Self, String> {
        let name = test["status"].as_str();
        ([Status::Pass, Status::Fail, Status::Skip].into_iter())
            .find(|status| Some(status.name()) == name)
            .ok_or_else(|| {
                format!(
                    "test {} has no status a report gives: {}",
                    test["id"], test["status"]
                )
            })
    }
}

impl From<Status> for Value {
    fn from(status: Status) -> Value {
        status.name().into()
    }
}
