//! Offline regions on a runtime with a state directory (`regions/` in it):
//! downloads through the HTTP cache, and maps that fetch through
//! [`RegionFetch`], so a downloaded area opens with no network.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use orb_host::Storage;
use orb_host_native::{FileStorage, NativeFetch};
use orb_runtime::offline::{self, Download, RegionFetch};
pub use orb_runtime::offline::{RegionDefinition, RegionEvent, RegionInfo};

use crate::Error;

#[derive(Debug)]
pub(crate) struct Regions {
    storage: Arc<FileStorage>,
    network: Arc<NativeFetch>,
    /// What maps fetch through.
    pub(crate) fetch: Arc<RegionFetch>,
    events: Arc<Mutex<VecDeque<RegionEvent>>>,
    downloads: Mutex<HashMap<u32, Download>>,
}

/// Waits for a storage call (the file storage answers at once).
fn wait<T: Send + 'static>(start: impl FnOnce(orb_host::StorageDone<T>)) -> Result<T, Error> {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    start(Box::new(move |r| {
        let _ = tx.send(r);
    }));
    rx.recv()
        .map_err(|_| Error::Native("storage dropped a call".into()))?
        .map_err(Error::Native)
}

impl Regions {
    pub(crate) fn new(dir: &std::path::Path, network: Arc<NativeFetch>) -> Self {
        let storage = Arc::new(FileStorage::new(dir));
        Self {
            fetch: RegionFetch::new(network.clone(), storage.clone()),
            storage,
            network,
            events: Arc::default(),
            downloads: Mutex::default(),
        }
    }

    /// Starts a download; `wake` is called with each event.
    pub(crate) fn download(
        &self,
        definition: RegionDefinition,
        wake: Arc<dyn Fn() + Send + Sync>,
    ) -> Result<u32, Error> {
        let (events, fetch) = (self.events.clone(), self.fetch.clone());
        let on_event: offline::RegionEvents = Arc::new(move |event| {
            if matches!(event, RegionEvent::Complete { .. }) {
                fetch.reload();
            }
            events
                .lock()
                .expect("region events poisoned")
                .push_back(event);
            wake();
        });
        let storage: Arc<dyn Storage> = self.storage.clone();
        let download = wait(|done| {
            offline::download(self.network.clone(), storage, definition, on_event, done)
        })?;
        let id = download.id();
        self.downloads
            .lock()
            .expect("downloads poisoned")
            .insert(id, download);
        Ok(id)
    }

    pub(crate) fn delete(&self, id: u32) -> Result<(), Error> {
        if let Some(download) = self
            .downloads
            .lock()
            .expect("downloads poisoned")
            .remove(&id)
        {
            download.cancel();
        }
        wait(|done| offline::delete(&*self.storage, id, done))?;
        self.fetch.reload();
        Ok(())
    }

    pub(crate) fn list(&self) -> Result<Vec<RegionInfo>, Error> {
        let storage: Arc<dyn Storage> = self.storage.clone();
        wait(|done| offline::list(storage, done))
    }

    pub(crate) fn poll_event(&self) -> Option<RegionEvent> {
        self.events
            .lock()
            .expect("region events poisoned")
            .pop_front()
    }
}
