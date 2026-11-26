//! The key a fetched resource is kept under, shared by the stores that
//! keep resources by request: `orb-host-native`'s HTTP cache and
//! `orb-runtime`'s offline regions.

/// The key a request is kept under: its URL, or for a byte range the URL
/// behind `bytes=<first>-<last> ` (no URL starts with a digit-led token).
pub fn key(url: &str, range: Option<(u64, u64)>) -> String {
    match range {
        Some((offset, length)) => format!("bytes={offset}-{} {url}", offset + length - 1),
        None => url.to_string(),
    }
}
