//! `ORB_GPU_LEDGER=1`: the buffers and textures a device holds, by label,
//! written to stderr when the device is dropped: as they were at the most
//! the device held at once, and as they were at its last submission. Every byte made
//! through [`orb_hal::Gpu`] is in it, render targets too, which the map's
//! own memory counters leave out; what wgpu makes on its own (the
//! surface's images, staging belts) is not.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

#[derive(Debug, Default)]
pub(crate) struct Ledger {
    /// Each live resource's label and bytes, by kind and slot.
    held: HashMap<(bool, u32), (String, u64)>,
    live: BTreeMap<String, (u32, u64)>,
    total: u64,
    peak: u64,
    at_peak: BTreeMap<String, (u32, u64)>,
    /// As it was at the last submission: the map at rest, where a run
    /// ends settled.
    at_submit: (u64, BTreeMap<String, (u32, u64)>),
}

impl Ledger {
    /// A ledger when `ORB_GPU_LEDGER` asks for one.
    pub(crate) fn from_env() -> Option<Box<Self>> {
        (std::env::var_os("ORB_GPU_LEDGER").is_some_and(|v| !v.is_empty() && v != "0"))
            .then(Box::default)
    }

    pub(crate) fn made(&mut self, texture: bool, slot: u32, label: &str, bytes: u64) {
        let e = self.live.entry(label.to_string()).or_default();
        e.0 += 1;
        e.1 += bytes;
        self.total += bytes;
        self.held
            .insert((texture, slot), (label.to_string(), bytes));
        if self.total > self.peak {
            self.peak = self.total;
            self.at_peak.clone_from(&self.live);
        }
    }

    pub(crate) fn submitted(&mut self) {
        if self.at_submit.0 != self.total || self.at_submit.1.len() != self.live.len() {
            self.at_submit = (self.total, self.live.clone());
        }
    }

    pub(crate) fn freed(&mut self, texture: bool, slot: u32) {
        let Some((label, bytes)) = self.held.remove(&(texture, slot)) else {
            return;
        };
        self.total -= bytes;
        if let Some(e) = self.live.get_mut(&label) {
            e.0 -= 1;
            e.1 -= bytes;
            if e.0 == 0 {
                self.live.remove(&label);
            }
        }
    }

    fn table(out: &mut String, title: &str, total: u64, rows: &BTreeMap<String, (u32, u64)>) {
        let mib = |b: u64| b as f64 / f64::from(1 << 20);
        let _ = writeln!(out, "gpu ledger, {title}: {:.1} MiB", mib(total));
        let mut rows: Vec<_> = rows.iter().collect();
        rows.sort_by_key(|(_, (_, b))| std::cmp::Reverse(*b));
        for (label, (n, b)) in rows {
            let _ = writeln!(out, "  {:>8.2} MiB  {n:>5}  {label}", mib(*b));
        }
    }
}

impl Drop for Ledger {
    fn drop(&mut self) {
        let mut out = String::new();
        Self::table(&mut out, "peak", self.peak, &self.at_peak);
        Self::table(
            &mut out,
            "last submission",
            self.at_submit.0,
            &self.at_submit.1,
        );
        eprint!("{out}");
    }
}
