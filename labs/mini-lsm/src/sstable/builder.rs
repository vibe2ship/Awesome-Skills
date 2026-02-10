//! SSTable builder implementation.
//!
//! Builds an SSTable from a sorted sequence of entries.

use crate::bloom::BloomFilter;
use crate::sstable::block::BlockBuilder;
use crate::sstable::{IndexEntry, FOOTER_SIZE, SSTABLE_MAGIC};
use crate::{checksum, Entry, Key, LsmError, Result};
use bytes::{BufMut, Bytes, BytesMut};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Builder for creating SSTables.
pub struct SSTableBuilder {
    /// Current block being built
    block_builder: BlockBuilder,

    /// Completed blocks (offset, size, last_key)
    blocks: Vec<(u64, u32, Key)>,

    /// All keys (for bloom filter)
    keys: Vec<Key>,

    /// Target block size
    block_size: usize,

    /// Bits per key for bloom filter
    bloom_bits_per_key: usize,

    /// First key in the SSTable
    first_key: Option<Key>,

    /// Last key in the SSTable
    last_key: Option<Key>,

    /// Total data written so far
    data_size: u64,
}

impl SSTableBuilder {
    /// Creates a new SSTable builder.
    ///
    /// TODO: Implement this function
    pub fn new(block_size: usize, bloom_bits_per_key: usize) -> Self {
        todo!("implement SSTableBuilder::new")
    }

    /// Adds an entry to the SSTable.
    /// Entries MUST be added in sorted order.
    ///
    /// TODO: Implement this function
    pub fn add(&mut self, entry: Entry) {
        todo!("implement SSTableBuilder::add")
    }

    /// Returns the estimated file size.
    pub fn estimated_size(&self) -> u64 {
        // data + bloom + index + footer
        self.data_size
            + self.keys.len() as u64 * self.bloom_bits_per_key as u64 / 8
            + self.blocks.len() as u64 * 32 // rough index estimate
            + FOOTER_SIZE as u64
    }

    /// Returns true if the builder is empty.
    pub fn is_empty(&self) -> bool {
        self.first_key.is_none()
    }

    /// Finishes building and writes to the given path.
    ///
    /// TODO: Implement this function
    /// - Finish current block
    /// - Write all data blocks
    /// - Write bloom filter
    /// - Write index block
    /// - Write footer
    pub fn finish(self, path: impl AsRef<Path>) -> Result<SSTableMeta> {
        todo!("implement SSTableBuilder::finish")
    }

    /// Finishes the current block and starts a new one.
    ///
    /// TODO: Implement this function
    fn finish_block(&mut self, writer: &mut impl Write) -> Result<()> {
        todo!("implement SSTableBuilder::finish_block")
    }

    /// Builds the bloom filter.
    ///
    /// TODO: Implement this function
    fn build_bloom_filter(&self) -> BloomFilter {
        todo!("implement SSTableBuilder::build_bloom_filter")
    }

    /// Builds the index block.
    ///
    /// TODO: Implement this function
    fn build_index(&self) -> Bytes {
        todo!("implement SSTableBuilder::build_index")
    }

    /// Builds the footer.
    ///
    /// TODO: Implement this function
    fn build_footer(bloom_offset: u64, index_offset: u64) -> Bytes {
        todo!("implement SSTableBuilder::build_footer")
    }
}

/// Metadata returned after building an SSTable.
#[derive(Debug, Clone)]
pub struct SSTableMeta {
    /// First key
    pub first_key: Key,

    /// Last key
    pub last_key: Key,

    /// File size
    pub file_size: u64,

    /// Number of blocks
    pub block_count: usize,

    /// Number of entries
    pub entry_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sstable::SSTable;
    use tempfile::tempdir;

    #[test]
    fn test_builder_empty() {
        let builder = SSTableBuilder::new(4096, 10);
        assert!(builder.is_empty());
    }

    #[test]
    fn test_builder_add_entries() {
        let mut builder = SSTableBuilder::new(4096, 10);

        builder.add(Entry::new(b"key1".as_slice(), b"value1".as_slice()));
        builder.add(Entry::new(b"key2".as_slice(), b"value2".as_slice()));

        assert!(!builder.is_empty());
    }

    #[test]
    fn test_builder_finish() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let mut builder = SSTableBuilder::new(4096, 10);
        builder.add(Entry::new(b"key1".as_slice(), b"value1".as_slice()));
        builder.add(Entry::new(b"key2".as_slice(), b"value2".as_slice()));
        builder.add(Entry::new(b"key3".as_slice(), b"value3".as_slice()));

        let meta = builder.finish(&path).unwrap();

        assert_eq!(meta.first_key.as_ref(), b"key1");
        assert_eq!(meta.last_key.as_ref(), b"key3");
        assert!(meta.file_size > 0);
    }

    #[test]
    fn test_builder_multiple_blocks() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        // Very small block size to force multiple blocks
        let mut builder = SSTableBuilder::new(64, 10);

        for i in 0..100 {
            let key = format!("key{:03}", i);
            let value = format!("value{:03}", i);
            builder.add(Entry::new(key.as_bytes(), value.as_bytes()));
        }

        let meta = builder.finish(&path).unwrap();

        // Should have multiple blocks
        assert!(meta.block_count > 1);
    }

    #[test]
    fn test_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let mut builder = SSTableBuilder::new(4096, 10);
        builder.add(Entry::new(b"a".as_slice(), b"1".as_slice()));
        builder.add(Entry::new(b"b".as_slice(), b"2".as_slice()));
        builder.add(Entry::new(b"c".as_slice(), b"3".as_slice()));

        builder.finish(&path).unwrap();

        // Open and verify
        let mut sst = SSTable::open(1, &path, None).unwrap();

        assert_eq!(sst.get(b"a").unwrap(), Some(Bytes::from_static(b"1")));
        assert_eq!(sst.get(b"b").unwrap(), Some(Bytes::from_static(b"2")));
        assert_eq!(sst.get(b"c").unwrap(), Some(Bytes::from_static(b"3")));
    }
}
