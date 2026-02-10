//! SSTable (Sorted String Table) implementation.
//!
//! An SSTable is an immutable on-disk data structure containing sorted key-value pairs.
//!
//! # File Format
//!
//! ```text
//! +----------------------+
//! |    Data Block 0      |
//! +----------------------+
//! |    Data Block 1      |
//! +----------------------+
//! |        ...           |
//! +----------------------+
//! |    Data Block N      |
//! +----------------------+
//! |    Bloom Filter      |
//! +----------------------+
//! |    Index Block       |
//! +----------------------+
//! |       Footer         |
//! +----------------------+
//! ```

pub mod block;
pub mod builder;
pub mod iterator;

use crate::{checksum, Entry, Key, LsmError, Result, Value};
use bytes::Bytes;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use block::Block;

/// Magic number at the end of SSTable files.
pub const SSTABLE_MAGIC: u64 = 0x4C534D5353540001; // "LSMSST" + version

/// SSTable footer size in bytes.
pub const FOOTER_SIZE: usize = 8 + 8 + 8 + 4; // bloom_offset + index_offset + magic + checksum

/// An SSTable file.
pub struct SSTable {
    /// Unique SSTable ID
    id: u64,

    /// File path
    path: String,

    /// File handle
    file: BufReader<File>,

    /// Index block (loaded in memory)
    index: Vec<IndexEntry>,

    /// Bloom filter (loaded in memory)
    bloom: Option<crate::bloom::BloomFilter>,

    /// First key in the SSTable
    first_key: Key,

    /// Last key in the SSTable
    last_key: Key,

    /// File size in bytes
    file_size: u64,

    /// Block cache (optional)
    block_cache: Option<Arc<BlockCache>>,
}

/// An entry in the index block.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    /// Last key in the block
    pub last_key: Key,

    /// Offset of the block in the file
    pub offset: u64,

    /// Size of the block in bytes
    pub size: u32,
}

/// Block cache using LRU eviction.
pub struct BlockCache {
    // TODO: Implement LRU cache
    capacity: usize,
}

impl BlockCache {
    /// Creates a new block cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        BlockCache { capacity }
    }

    /// Gets a block from the cache.
    ///
    /// TODO: Implement this function
    pub fn get(&self, _sstable_id: u64, _block_offset: u64) -> Option<Arc<Block>> {
        todo!("implement BlockCache::get")
    }

    /// Inserts a block into the cache.
    ///
    /// TODO: Implement this function
    pub fn insert(&self, _sstable_id: u64, _block_offset: u64, _block: Arc<Block>) {
        todo!("implement BlockCache::insert")
    }
}

impl SSTable {
    /// Opens an existing SSTable file.
    ///
    /// TODO: Implement this function
    /// - Read and verify footer
    /// - Load index block
    /// - Load bloom filter
    pub fn open(id: u64, path: impl AsRef<Path>, cache: Option<Arc<BlockCache>>) -> Result<Self> {
        todo!("implement SSTable::open")
    }

    /// Returns the SSTable ID.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns the file path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the file size.
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Returns the first key.
    pub fn first_key(&self) -> &Key {
        &self.first_key
    }

    /// Returns the last key.
    pub fn last_key(&self) -> &Key {
        &self.last_key
    }

    /// Checks if a key might exist in this SSTable using the bloom filter.
    ///
    /// TODO: Implement this function
    pub fn may_contain(&self, key: &[u8]) -> bool {
        todo!("implement SSTable::may_contain")
    }

    /// Gets the value for a key.
    ///
    /// Returns:
    /// - Ok(Some(value)) if key exists
    /// - Ok(None) if key was deleted (tombstone)
    /// - Err(KeyNotFound) if key doesn't exist
    ///
    /// TODO: Implement this function
    pub fn get(&mut self, key: &[u8]) -> Result<Option<Value>> {
        todo!("implement SSTable::get")
    }

    /// Finds the block that might contain the key.
    ///
    /// TODO: Implement this function
    /// Uses binary search on the index.
    fn find_block(&self, key: &[u8]) -> Option<usize> {
        todo!("implement SSTable::find_block")
    }

    /// Reads a block from the file.
    ///
    /// TODO: Implement this function
    fn read_block(&mut self, index: usize) -> Result<Block> {
        todo!("implement SSTable::read_block")
    }

    /// Creates an iterator over all entries.
    ///
    /// TODO: Implement this function
    pub fn iter(&mut self) -> Result<iterator::SSTableIterator> {
        todo!("implement SSTable::iter")
    }

    /// Creates an iterator starting from the given key.
    ///
    /// TODO: Implement this function
    pub fn range_from(&mut self, key: &[u8]) -> Result<iterator::SSTableIterator> {
        todo!("implement SSTable::range_from")
    }

    /// Returns the number of blocks.
    pub fn block_count(&self) -> usize {
        self.index.len()
    }
}

/// Metadata about an SSTable (stored in manifest).
#[derive(Debug, Clone)]
pub struct SSTableMeta {
    /// SSTable ID
    pub id: u64,

    /// File path
    pub path: String,

    /// First key
    pub first_key: Key,

    /// Last key
    pub last_key: Key,

    /// File size
    pub file_size: u64,

    /// Level in the LSM tree
    pub level: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sstable::builder::SSTableBuilder;
    use tempfile::tempdir;

    fn create_test_sstable(entries: Vec<(&[u8], &[u8])>) -> (String, u64) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");
        let path_str = path.to_str().unwrap().to_string();

        let mut builder = SSTableBuilder::new(4096, 10);
        for (key, value) in entries {
            builder.add(Entry::new(key, value));
        }
        builder.finish(&path_str).unwrap();

        // Keep the dir alive by leaking it (for tests only)
        std::mem::forget(dir);

        (path_str, 1)
    }

    #[test]
    fn test_sstable_open() {
        let (path, id) = create_test_sstable(vec![
            (b"key1", b"value1"),
            (b"key2", b"value2"),
            (b"key3", b"value3"),
        ]);

        let sst = SSTable::open(id, &path, None).unwrap();
        assert_eq!(sst.id(), id);
        assert_eq!(sst.first_key().as_ref(), b"key1");
        assert_eq!(sst.last_key().as_ref(), b"key3");
    }

    #[test]
    fn test_sstable_get() {
        let (path, id) = create_test_sstable(vec![
            (b"key1", b"value1"),
            (b"key2", b"value2"),
            (b"key3", b"value3"),
        ]);

        let mut sst = SSTable::open(id, &path, None).unwrap();

        assert_eq!(sst.get(b"key1").unwrap(), Some(Bytes::from_static(b"value1")));
        assert_eq!(sst.get(b"key2").unwrap(), Some(Bytes::from_static(b"value2")));
        assert!(matches!(sst.get(b"key4"), Err(LsmError::KeyNotFound)));
    }

    #[test]
    fn test_sstable_bloom_filter() {
        let (path, id) = create_test_sstable(vec![
            (b"key1", b"value1"),
            (b"key2", b"value2"),
        ]);

        let sst = SSTable::open(id, &path, None).unwrap();

        // Keys that exist should pass bloom filter
        assert!(sst.may_contain(b"key1"));
        assert!(sst.may_contain(b"key2"));

        // Most keys that don't exist should fail bloom filter
        // (some may pass due to false positives)
    }

    #[test]
    fn test_sstable_iterator() {
        let (path, id) = create_test_sstable(vec![
            (b"a", b"1"),
            (b"b", b"2"),
            (b"c", b"3"),
        ]);

        let mut sst = SSTable::open(id, &path, None).unwrap();
        let iter = sst.iter().unwrap();

        let entries: Vec<Entry> = iter.collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].key.as_ref(), b"a");
        assert_eq!(entries[1].key.as_ref(), b"b");
        assert_eq!(entries[2].key.as_ref(), b"c");
    }
}
