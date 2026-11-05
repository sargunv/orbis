//! [`Storage`] as files under a directory, on the caller's thread.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use orb_host::{Storage, StorageDone};

/// Keys are paths under a root directory; whole writes replace files
/// atomically ([`crate::write_atomically`]).
#[derive(Debug)]
pub struct FileStorage {
    root: PathBuf,
}

impl FileStorage {
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_owned(),
        }
    }

    fn path(&self, key: &str) -> Result<PathBuf, String> {
        if key.split('/').any(|part| part == ".." || part.is_empty()) && !key.is_empty() {
            return Err(format!("invalid storage key {key:?}"));
        }
        Ok(self.root.join(key))
    }

    fn read_now(&self, key: &str, range: Option<(u64, u64)>) -> Result<Option<Vec<u8>>, String> {
        let path = self.path(key)?;
        let mut file = match std::fs::File::open(&path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        let mut bytes = Vec::new();
        let read = match range {
            None => file.read_to_end(&mut bytes).map(drop),
            Some((offset, length)) => {
                bytes.resize(length as usize, 0);
                file.seek(SeekFrom::Start(offset))
                    .and_then(|_| file.read_exact(&mut bytes))
            }
        };
        read.map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Some(bytes))
    }

    fn write_now(&self, key: &str, bytes: &[u8], append: bool) -> Result<(), String> {
        let path = self.path(key)?;
        let fail = |e: std::io::Error| format!("{}: {e}", path.display());
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(fail)?;
        }
        if append {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .and_then(|mut file| file.write_all(bytes))
                .map_err(fail)
        } else {
            crate::write_atomically(&path, bytes).map_err(fail)
        }
    }

    fn remove_now(&self, key: &str) -> Result<(), String> {
        let path = self.path(key)?;
        let removed = match std::fs::metadata(&path) {
            Ok(m) if m.is_dir() => std::fs::remove_dir_all(&path),
            Ok(_) => std::fs::remove_file(&path),
            Err(_) => Ok(()),
        };
        removed.map_err(|e| format!("{}: {e}", path.display()))
    }

    fn list_now(&self, prefix: &str) -> Result<Vec<String>, String> {
        let path = self.path(prefix)?;
        match std::fs::read_dir(&path) {
            Ok(entries) => Ok(entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }
}

impl Storage for FileStorage {
    fn read(&self, key: &str, range: Option<(u64, u64)>, done: StorageDone<Option<Vec<u8>>>) {
        done(self.read_now(key, range));
    }
    fn write(&self, key: &str, bytes: Vec<u8>, append: bool, done: StorageDone<()>) {
        done(self.write_now(key, &bytes, append));
    }
    fn remove(&self, key: &str, done: StorageDone<()>) {
        done(self.remove_now(key));
    }
    fn list(&self, prefix: &str, done: StorageDone<Vec<String>>) {
        done(self.list_now(prefix));
    }
}

/// Runs a storage operation and waits for its outcome, on a thread that
/// may block.
pub fn wait<T: Send + 'static>(start: impl FnOnce(StorageDone<T>)) -> Result<T, String> {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    start(Box::new(move |result| {
        let _ = tx.send(result);
    }));
    rx.recv()
        .map_err(|_| "storage dropped an operation".to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_writes_appends_lists_and_removes() {
        let root = std::env::temp_dir().join(format!("orb-storage-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let storage = FileStorage::new(&root);
        storage.write_now("a/b", b"hello", false).unwrap();
        storage.write_now("a/b", b" world", true).unwrap();
        assert_eq!(
            storage.read_now("a/b", None).unwrap().unwrap(),
            b"hello world"
        );
        assert_eq!(
            storage.read_now("a/b", Some((6, 5))).unwrap().unwrap(),
            b"world"
        );
        assert_eq!(storage.read_now("a/c", None).unwrap(), None);
        assert!(storage.read_now("a/b", Some((8, 5))).is_err());
        assert_eq!(storage.list_now("a").unwrap(), ["b"]);
        assert_eq!(storage.list_now("").unwrap(), ["a"]);
        assert!(storage.read_now("../x", None).is_err());
        storage.remove_now("a").unwrap();
        assert!(storage.list_now("a").unwrap().is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }
}
