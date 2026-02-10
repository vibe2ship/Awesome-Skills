//! LSM storage engine implementation.
//!
//! This is the main entry point for the storage engine.
//!
//! # Write Path
//!
//! 1. Write to WAL (if enabled)
//! 2. Write to MemTable
//! 3. When MemTable is full, make it immutable
//! 4. Flush immutable MemTable to SSTable
//!
//! # Read Path
//!
//! 1. Check MemTable
//! 2. Check immutable MemTables (newest first)
//! 3. Check L0 SSTables (all of them, newest first)
//! 4. Check L1+ SSTables (use index to find correct SSTable)

use crate::compaction::{CompactionStrategy, CompactionTask, LeveledCompaction};
use crate::iterator::{LiveIterator, MergeIterator};
use crate::manifest::Manifest;
use crate::memtable::MemTable;
use crate::sstable::builder::SSTableBuilder;
use crate::sstable::SSTable;
use crate::wal::{Wal, WalReader, WalRecord};
use crate::{Entry, Key, LsmConfig, LsmError, Result, Value};
use bytes::Bytes;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// The LSM storage engine.
pub struct LsmDb {
    /// Configuration
    config: LsmConfig,

    /// Current mutable MemTable
    memtable: RwLock<Arc<MemTable>>,

    /// Immutable MemTables waiting to be flushed
    imm_memtables: RwLock<Vec<Arc<MemTable>>>,

    /// Open SSTable handles
    sstables: RwLock<HashMap<u64, Arc<Mutex<SSTable>>>>,

    /// Manifest for metadata
    manifest: Mutex<Manifest>,

    /// Write-ahead log
    wal: Option<Mutex<Wal>>,

    /// Next MemTable ID
    next_mem_id: AtomicU64,

    /// Compaction strategy
    compaction_strategy: Box<dyn CompactionStrategy>,

    /// Whether the database is closed
    closed: AtomicBool,
}

impl LsmDb {
    /// Opens or creates a database at the given path.
    ///
    /// TODO: Implement this function
    /// - Create directory if needed
    /// - Open or create manifest
    /// - Recover WAL if exists
    /// - Open existing SSTables
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        todo!("implement LsmDb::open")
    }

    /// Opens with custom configuration.
    ///
    /// TODO: Implement this function
    pub fn open_with_config(path: impl AsRef<Path>, config: LsmConfig) -> Result<Self> {
        todo!("implement LsmDb::open_with_config")
    }

    /// Puts a key-value pair.
    ///
    /// TODO: Implement this function
    /// - Write to WAL
    /// - Write to MemTable
    /// - Check if flush needed
    pub fn put(&self, key: impl Into<Key>, value: impl Into<Value>) -> Result<()> {
        todo!("implement LsmDb::put")
    }

    /// Deletes a key.
    ///
    /// TODO: Implement this function
    /// - Write tombstone to WAL
    /// - Write tombstone to MemTable
    pub fn delete(&self, key: impl Into<Key>) -> Result<()> {
        todo!("implement LsmDb::delete")
    }

    /// Gets the value for a key.
    ///
    /// TODO: Implement this function
    /// Search order:
    /// 1. MemTable
    /// 2. Immutable MemTables
    /// 3. L0 SSTables
    /// 4. L1+ SSTables
    pub fn get(&self, key: &[u8]) -> Result<Option<Value>> {
        todo!("implement LsmDb::get")
    }

    /// Scans keys in the given range.
    ///
    /// TODO: Implement this function
    pub fn scan(&self, start: &[u8], end: &[u8]) -> Result<impl Iterator<Item = Entry>> {
        todo!("implement LsmDb::scan")
    }

    /// Forces a flush of the current MemTable.
    ///
    /// TODO: Implement this function
    pub fn flush(&self) -> Result<()> {
        todo!("implement LsmDb::flush")
    }

    /// Triggers a manual compaction.
    ///
    /// TODO: Implement this function
    pub fn compact(&self) -> Result<()> {
        todo!("implement LsmDb::compact")
    }

    /// Runs the compaction strategy and performs compaction if needed.
    ///
    /// TODO: Implement this function
    fn maybe_compact(&self) -> Result<()> {
        todo!("implement LsmDb::maybe_compact")
    }

    /// Executes a compaction task.
    ///
    /// TODO: Implement this function
    fn do_compact(&self, task: CompactionTask) -> Result<()> {
        todo!("implement LsmDb::do_compact")
    }

    /// Flushes an immutable MemTable to an SSTable.
    ///
    /// TODO: Implement this function
    fn flush_memtable(&self, memtable: Arc<MemTable>) -> Result<()> {
        todo!("implement LsmDb::flush_memtable")
    }

    /// Switches the current MemTable to immutable.
    ///
    /// TODO: Implement this function
    fn switch_memtable(&self) -> Result<()> {
        todo!("implement LsmDb::switch_memtable")
    }

    /// Checks if a flush is needed.
    fn needs_flush(&self) -> bool {
        self.memtable.read().is_full()
    }

    /// Gets the path for an SSTable file.
    fn sstable_path(&self, id: u64) -> String {
        format!("{}/{:06}.sst", self.config.data_dir, id)
    }

    /// Gets the path for a WAL file.
    fn wal_path(&self, id: u64) -> String {
        format!("{}/{:06}.wal", self.config.data_dir, id)
    }

    /// Closes the database.
    ///
    /// TODO: Implement this function
    pub fn close(&self) -> Result<()> {
        todo!("implement LsmDb::close")
    }
}

impl Drop for LsmDb {
    fn drop(&mut self) {
        if !self.closed.load(Ordering::Relaxed) {
            let _ = self.close();
        }
    }
}

/// Builder for opening a database with custom options.
pub struct LsmDbBuilder {
    path: String,
    config: LsmConfig,
}

impl LsmDbBuilder {
    /// Creates a new builder.
    pub fn new(path: impl Into<String>) -> Self {
        LsmDbBuilder {
            path: path.into(),
            config: LsmConfig::default(),
        }
    }

    /// Sets the MemTable size.
    pub fn memtable_size(mut self, size: usize) -> Self {
        self.config.memtable_size = size;
        self
    }

    /// Sets the block size.
    pub fn block_size(mut self, size: usize) -> Self {
        self.config.block_size = size;
        self
    }

    /// Enables or disables WAL.
    pub fn enable_wal(mut self, enable: bool) -> Self {
        self.config.enable_wal = enable;
        self
    }

    /// Enables or disables WAL sync.
    pub fn sync_wal(mut self, sync: bool) -> Self {
        self.config.sync_wal = sync;
        self
    }

    /// Builds and opens the database.
    pub fn open(self) -> Result<LsmDb> {
        let mut config = self.config;
        config.data_dir = self.path;
        LsmDb::open_with_config(&config.data_dir, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_lsm_open() {
        let dir = tempdir().unwrap();
        let db = LsmDb::open(dir.path()).unwrap();
        db.close().unwrap();
    }

    #[test]
    fn test_lsm_put_get() {
        let dir = tempdir().unwrap();
        let db = LsmDb::open(dir.path()).unwrap();

        db.put(b"key1".as_slice(), b"value1".as_slice()).unwrap();
        db.put(b"key2".as_slice(), b"value2".as_slice()).unwrap();

        assert_eq!(db.get(b"key1").unwrap(), Some(Bytes::from_static(b"value1")));
        assert_eq!(db.get(b"key2").unwrap(), Some(Bytes::from_static(b"value2")));
        assert!(db.get(b"key3").unwrap().is_none());
    }

    #[test]
    fn test_lsm_delete() {
        let dir = tempdir().unwrap();
        let db = LsmDb::open(dir.path()).unwrap();

        db.put(b"key1".as_slice(), b"value1".as_slice()).unwrap();
        db.delete(b"key1".as_slice()).unwrap();

        assert!(db.get(b"key1").unwrap().is_none());
    }

    #[test]
    fn test_lsm_overwrite() {
        let dir = tempdir().unwrap();
        let db = LsmDb::open(dir.path()).unwrap();

        db.put(b"key1".as_slice(), b"value1".as_slice()).unwrap();
        db.put(b"key1".as_slice(), b"value2".as_slice()).unwrap();

        assert_eq!(db.get(b"key1").unwrap(), Some(Bytes::from_static(b"value2")));
    }

    #[test]
    fn test_lsm_flush() {
        let dir = tempdir().unwrap();
        let db = LsmDbBuilder::new(dir.path().to_str().unwrap())
            .memtable_size(100) // Very small to force flush
            .enable_wal(false)
            .open()
            .unwrap();

        // Write enough data to trigger flush
        for i in 0..100 {
            let key = format!("key{:03}", i);
            let value = format!("value{:03}", i);
            db.put(key.as_bytes(), value.as_bytes()).unwrap();
        }

        // Force flush
        db.flush().unwrap();

        // Data should still be readable
        assert_eq!(
            db.get(b"key000").unwrap(),
            Some(Bytes::from("value000"))
        );
    }

    #[test]
    fn test_lsm_recovery() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();

        // Write some data
        {
            let db = LsmDbBuilder::new(path)
                .enable_wal(true)
                .sync_wal(true)
                .open()
                .unwrap();

            db.put(b"key1".as_slice(), b"value1".as_slice()).unwrap();
            db.put(b"key2".as_slice(), b"value2".as_slice()).unwrap();
            // Don't close cleanly - simulates crash
        }

        // Reopen and verify recovery
        {
            let db = LsmDb::open(path).unwrap();

            assert_eq!(db.get(b"key1").unwrap(), Some(Bytes::from_static(b"value1")));
            assert_eq!(db.get(b"key2").unwrap(), Some(Bytes::from_static(b"value2")));
        }
    }

    #[test]
    fn test_lsm_scan() {
        let dir = tempdir().unwrap();
        let db = LsmDb::open(dir.path()).unwrap();

        db.put(b"a".as_slice(), b"1".as_slice()).unwrap();
        db.put(b"b".as_slice(), b"2".as_slice()).unwrap();
        db.put(b"c".as_slice(), b"3".as_slice()).unwrap();
        db.put(b"d".as_slice(), b"4".as_slice()).unwrap();

        let entries: Vec<_> = db.scan(b"b", b"d").unwrap().collect();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].key.as_ref(), b"b");
        assert_eq!(entries[1].key.as_ref(), b"c");
    }

    #[test]
    fn test_lsm_builder() {
        let dir = tempdir().unwrap();

        let db = LsmDbBuilder::new(dir.path().to_str().unwrap())
            .memtable_size(1024 * 1024)
            .block_size(4096)
            .enable_wal(true)
            .sync_wal(false)
            .open()
            .unwrap();

        db.put(b"test".as_slice(), b"value".as_slice()).unwrap();
        assert!(db.get(b"test").unwrap().is_some());
    }
}
