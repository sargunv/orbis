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
    /// The state directory.
    dir: std::path::PathBuf,
    storage: Arc<FileStorage>,
    network: Arc<NativeFetch>,
    /// What maps fetch through.
    pub(crate) fetch: Arc<RegionFetch>,
    events: Arc<Mutex<VecDeque<RegionEvent>>>,
    downloads: Mutex<HashMap<u32, Download>>,
}

/// Waits for a storage call (the file storage answers at once).
fn wait<T: Send + 'static>(start: impl FnOnce(orb_host::StorageDone<T>)) -> Result<T, Error> {
    orb_host_native::wait(start).map_err(Error::Native)
}

impl Regions {
    pub(crate) fn new(dir: &std::path::Path, network: Arc<NativeFetch>) -> Self {
        let storage = Arc::new(FileStorage::new(dir));
        Self {
            dir: dir.to_owned(),
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

    pub(crate) fn create(&self, definition: RegionDefinition) -> Result<RegionInfo, Error> {
        let storage: Arc<dyn Storage> = self.storage.clone();
        let events: offline::RegionEvents = Arc::new(|_| {});
        let import = wait(|done| offline::import(storage, definition.clone(), events, done))?;
        Ok(RegionInfo {
            id: import.id(),
            definition,
            complete: false,
            bytes: 0,
        })
    }

    /// Copies `source`'s regions in: an Orbis state directory's, or a
    /// MapLibre Native offline database's.
    pub(crate) fn merge(&self, source: &std::path::Path) -> Result<Vec<RegionInfo>, Error> {
        let have = self.list()?;
        let same = |definition: &RegionDefinition| {
            have.iter().find(|r| r.definition == *definition).cloned()
        };
        let mut merged = Vec::new();
        if source.is_dir() {
            let from: Arc<dyn Storage> = Arc::new(FileStorage::new(source));
            for region in wait(|done| offline::list(from, done))? {
                if let Some(found) = same(&region.definition) {
                    merged.push(found);
                    continue;
                }
                merged.push(self.copy_dir(source, region)?);
            }
        } else {
            merged = self.merge_database(source, &same)?;
        }
        self.fetch.reload();
        Ok(merged)
    }

    /// Copies region `region` of the state directory `source` in, under
    /// a new id.
    fn copy_dir(&self, source: &std::path::Path, region: RegionInfo) -> Result<RegionInfo, Error> {
        let fail = |e: std::io::Error| Error::Native(format!("merging regions: {e}"));
        let regions = self.dir.join("regions");
        std::fs::create_dir_all(&regions).map_err(fail)?;
        let id = std::fs::read_dir(&regions)
            .map_err(fail)?
            .filter_map(|e| e.ok()?.file_name().to_str()?.parse::<u32>().ok())
            .max()
            .map_or(1, |n| n + 1);
        let (from, to) = (
            source.join("regions").join(region.id.to_string()),
            regions.join(id.to_string()),
        );
        // Written beside, then moved in whole: a half-copied region never lists.
        let partial = regions.join(format!(".{id}.merging"));
        let _ = std::fs::remove_dir_all(&partial);
        std::fs::create_dir_all(&partial).map_err(fail)?;
        for entry in std::fs::read_dir(&from).map_err(fail)? {
            let entry = entry.map_err(fail)?;
            if entry.file_type().map_err(fail)?.is_file() {
                std::fs::copy(entry.path(), partial.join(entry.file_name())).map_err(fail)?;
            }
        }
        std::fs::rename(&partial, &to).map_err(fail)?;
        Ok(RegionInfo { id, ..region })
    }

    #[cfg(feature = "sqlite")]
    fn merge_database(
        &self,
        source: &std::path::Path,
        same: &dyn Fn(&RegionDefinition) -> Option<RegionInfo>,
    ) -> Result<Vec<RegionInfo>, Error> {
        let file = std::fs::File::open(source)
            .map_err(|e| Error::InvalidArgument(format!("{}: {e}", source.display())))?;
        let pages = FilePages {
            file: Mutex::new(file),
            page_size: std::sync::OnceLock::new(),
        };
        let native = orb_runtime::native_db::NativeDatabase::open(&pages)
            .map_err(|e| Error::Native(format!("{}: {e}", source.display())))?;
        let mut merged = Vec::new();
        for (native_id, definition) in native.regions().map_err(Error::Native)? {
            if let Some(found) = same(&definition) {
                merged.push(found);
                continue;
            }
            let (tx, rx) = std::sync::mpsc::channel();
            let events: offline::RegionEvents = Arc::new(move |event| {
                if !matches!(event, RegionEvent::Progress { .. }) {
                    let _ = tx.send(event);
                }
            });
            let storage: Arc<dyn Storage> = self.storage.clone();
            let import = wait(|done| offline::import(storage, definition.clone(), events, done))?;
            let id = import.id();
            if let Err(e) = native.copy(native_id, &import) {
                let _ = wait(|done| offline::delete(&*self.storage, id, done));
                return Err(Error::Native(format!("{}: {e}", source.display())));
            }
            import.finish();
            match rx.recv() {
                Ok(RegionEvent::Complete { bytes, .. }) => merged.push(RegionInfo {
                    id,
                    definition,
                    complete: true,
                    bytes,
                }),
                Ok(RegionEvent::Failed { message, .. }) => return Err(Error::Native(message)),
                _ => return Err(Error::Native("the merge stopped".into())),
            }
        }
        Ok(merged)
    }

    #[cfg(not(feature = "sqlite"))]
    fn merge_database(
        &self,
        source: &std::path::Path,
        _: &dyn Fn(&RegionDefinition) -> Option<RegionInfo>,
    ) -> Result<Vec<RegionInfo>, Error> {
        Err(Error::Unsupported(format!(
            "{}: merging a database file needs Orbis built with its `sqlite` feature",
            source.display()
        )))
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

/// A database file's pages, read as the merge wants them.
#[cfg(feature = "sqlite")]
struct FilePages {
    file: Mutex<std::fs::File>,
    page_size: std::sync::OnceLock<u32>,
}

#[cfg(feature = "sqlite")]
impl orb_runtime::orb_sqlite::Pages for FilePages {
    fn page(&self, number: u32, _: u32) -> orb_runtime::orb_sqlite::Result<Arc<[u8]>> {
        use orb_runtime::orb_sqlite::{Error, HEADER_LEN, Header};
        use std::io::{Read, Seek, SeekFrom};
        let mut file = self.file.lock().expect("database file poisoned");
        let io = |e: std::io::Error| Error::Corrupt(e.to_string());
        let size = match self.page_size.get() {
            Some(&size) => size,
            None => {
                let mut head = [0; HEADER_LEN];
                file.seek(SeekFrom::Start(0)).map_err(io)?;
                file.read_exact(&mut head).map_err(io)?;
                let size = Header::parse(&head)?.page_size;
                *self.page_size.get_or_init(|| size)
            }
        };
        let at = u64::from(
            number
                .checked_sub(1)
                .ok_or_else(|| Error::Corrupt("page 0".into()))?,
        ) * u64::from(size);
        let mut page = vec![0; size as usize];
        file.seek(SeekFrom::Start(at)).map_err(io)?;
        file.read_exact(&mut page).map_err(io)?;
        Ok(Arc::from(page))
    }
}

#[cfg(all(test, feature = "sqlite"))]
mod tests;
