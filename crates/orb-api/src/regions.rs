//! Offline regions on a runtime with a state directory (`regions/` in it),
//! merged in from other stores, and maps that fetch through
//! [`RegionFetch`], so a region's area opens with no network.

use std::sync::Arc;

use orb_host::Storage;
use orb_host_native::{FileStorage, NativeFetch};
use orb_runtime::modules;
use orb_runtime::offline::{self, RegionFetch};
pub use orb_runtime::offline::{RegionDefinition, RegionInfo};

use crate::Error;

#[derive(Debug)]
pub(crate) struct Regions {
    /// The state directory.
    dir: std::path::PathBuf,
    storage: Arc<FileStorage>,
    /// What maps fetch through.
    pub(crate) fetch: Arc<RegionFetch>,
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
            fetch: RegionFetch::new(network, storage.clone(), modules::every()),
            storage,
        }
    }

    pub(crate) fn create(&self, definition: RegionDefinition) -> Result<RegionInfo, Error> {
        let storage: Arc<dyn Storage> = self.storage.clone();
        let events: offline::RegionEvents = Arc::new(|_| {});
        let import = wait(|done| {
            offline::import(&modules::every(), storage, definition.clone(), events, done)
        })?;
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
                merged.push(self.copy(source, region)?);
            }
        } else {
            merged = self.merge_database(source, &same)?;
        }
        self.fetch.reload();
        Ok(merged)
    }

    /// Copies region `region` of the state directory `source` in, under an
    /// id the store picks: listed incomplete until its manifest, copied
    /// last, lands, and removed if any of it fails. Files stream across, as
    /// a region's extracts can outgrow a phone's memory.
    fn copy(&self, source: &std::path::Path, region: RegionInfo) -> Result<RegionInfo, Error> {
        let id = self.create(region.definition.clone())?.id;
        let (from, to) = (
            source.join("regions").join(region.id.to_string()),
            self.dir.join("regions").join(id.to_string()),
        );
        let copied = (|| {
            let fail = |e: std::io::Error| Error::Native(format!("merging regions: {e}"));
            for entry in std::fs::read_dir(&from).map_err(fail)? {
                let entry = entry.map_err(fail)?;
                let name = entry.file_name();
                if entry.file_type().map_err(fail)?.is_file() && name != "manifest.json" {
                    std::fs::copy(entry.path(), to.join(name)).map_err(fail)?;
                }
            }
            let manifest = std::fs::read(from.join("manifest.json")).map_err(fail)?;
            let key = format!("regions/{id}/manifest.json");
            wait(|done| self.storage.write(&key, manifest, false, done))
        })();
        if copied.is_err() {
            let _ = wait(|done| offline::delete(&*self.storage, id, done));
        }
        copied.map(|()| RegionInfo { id, ..region })
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
            file: std::sync::Mutex::new(file),
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
                if !matches!(event, offline::RegionEvent::Progress { .. }) {
                    let _ = tx.send(event);
                }
            });
            let storage: Arc<dyn Storage> = self.storage.clone();
            let import = wait(|done| {
                offline::import(&modules::every(), storage, definition.clone(), events, done)
            })?;
            let id = import.id();
            if let Err(e) = native.copy(native_id, &import) {
                let _ = wait(|done| offline::delete(&*self.storage, id, done));
                return Err(Error::Native(format!("{}: {e}", source.display())));
            }
            import.finish();
            match rx.recv() {
                Ok(offline::RegionEvent::Complete { bytes, .. }) => merged.push(RegionInfo {
                    id,
                    definition,
                    complete: true,
                    bytes,
                }),
                Ok(offline::RegionEvent::Failed { message, .. }) => {
                    return Err(Error::Native(message));
                }
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
        wait(|done| offline::delete(&*self.storage, id, done))?;
        self.fetch.reload();
        Ok(())
    }

    pub(crate) fn list(&self) -> Result<Vec<RegionInfo>, Error> {
        let storage: Arc<dyn Storage> = self.storage.clone();
        wait(|done| offline::list(storage, done))
    }
}

/// A database file's pages, read as the merge wants them.
#[cfg(feature = "sqlite")]
struct FilePages {
    file: std::sync::Mutex<std::fs::File>,
    page_size: std::sync::OnceLock<u32>,
}

#[cfg(feature = "sqlite")]
impl orb_runtime::orb_sqlite::Pages for FilePages {
    fn page(&self, number: u32, _: u32) -> orb_runtime::orb_sqlite::Result<Arc<[u8]>> {
        use orb_runtime::orb_sqlite::{Error, HEADER_LEN, Header};
        use std::io::{Read, Seek, SeekFrom};
        let mut file = crate::lock(&self.file);
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
