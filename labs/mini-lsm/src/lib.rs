//! Mini-LSM: A pedagogical Log-Structured Merge Tree storage engine.
//!
//! # Overview
//!
//! LSM-trees provide high write throughput by:
//! 1. Buffering writes in memory (MemTable)
//! 2. Flushing to immutable sorted files (SSTables)
//! 3. Background compaction to merge and clean up
//!
//! # Architecture
//!
//! ```text
//! Write: WAL -> MemTable -> (flush) -> SSTable
//! Read: MemTable -> Immutable MemTable -> L0 SSTables -> L1 -> ...
//! ```

pub mod bloom;
pub mod compaction;
pub mod iterator;
pub mod lsm;
pub mod manifest;
pub mod memtable;
pub mod sstable;
pub mod wal;

use bytes::Bytes;
use thiserror::Error;

/// A key in the key-value store.
pub type Key = Bytes;

/// A value in the key-value store.
pub type Value = Bytes;

/// Timestamp for MVCC (optional, for advanced implementation).
pub type Timestamp = u64;

/// Errors that can occur in LSM operations.
#[derive(Debug, Error)]
pub enum LsmError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("key not found")]
    KeyNotFound,

    #[error("invalid data: {0}")]
    InvalidData(String),

    #[error("checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: u32, actual: u32 },

    #[error("block not found at offset {0}")]
    BlockNotFound(u64),

    #[error("sstable corrupted: {0}")]
    Corrupted(String),

    #[error("memtable full")]
    MemTableFull,

    #[error("database closed")]
    Closed,
}

/// Result type for LSM operations.
pub type Result<T> = std::result::Result<T, LsmError>;

/// Configuration for the LSM engine.
#[derive(Debug, Clone)]
pub struct LsmConfig {
    /// Directory for data files
    pub data_dir: String,

    /// Maximum MemTable size in bytes before flushing
    pub memtable_size: usize,

    /// Block size in bytes for SSTables
    pub block_size: usize,

    /// Number of bits per key for Bloom filter
    pub bloom_bits_per_key: usize,

    /// Maximum number of L0 files before triggering compaction
    pub l0_compaction_trigger: usize,

    /// Size ratio between levels (e.g., 10 means L1 is 10x L0)
    pub level_size_ratio: usize,

    /// Maximum number of levels
    pub max_levels: usize,

    /// Whether to enable WAL
    pub enable_wal: bool,

    /// Whether to sync WAL after each write
    pub sync_wal: bool,
}

impl Default for LsmConfig {
    fn default() -> Self {
        LsmConfig {
            data_dir: "./data".to_string(),
            memtable_size: 4 * 1024 * 1024, // 4MB
            block_size: 4 * 1024,            // 4KB
            bloom_bits_per_key: 10,
            l0_compaction_trigger: 4,
            level_size_ratio: 10,
            max_levels: 7,
            enable_wal: true,
            sync_wal: true,
        }
    }
}

/// Key-value entry for storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// The key
    pub key: Key,

    /// The value (None means tombstone/deletion)
    pub value: Option<Value>,

    /// Timestamp for MVCC (0 if not used)
    pub timestamp: Timestamp,
}

impl Entry {
    /// Creates a new entry with a value.
    pub fn new(key: impl Into<Key>, value: impl Into<Value>) -> Self {
        Entry {
            key: key.into(),
            value: Some(value.into()),
            timestamp: 0,
        }
    }

    /// Creates a tombstone (deletion marker).
    pub fn tombstone(key: impl Into<Key>) -> Self {
        Entry {
            key: key.into(),
            value: None,
            timestamp: 0,
        }
    }

    /// Creates an entry with timestamp.
    pub fn with_timestamp(key: impl Into<Key>, value: Option<impl Into<Value>>, ts: Timestamp) -> Self {
        Entry {
            key: key.into(),
            value: value.map(|v| v.into()),
            timestamp: ts,
        }
    }

    /// Returns true if this is a tombstone.
    pub fn is_tombstone(&self) -> bool {
        self.value.is_none()
    }

    /// Returns the encoded size of this entry.
    ///
    /// TODO: Implement this function
    pub fn encoded_size(&self) -> usize {
        todo!("implement Entry::encoded_size")
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare by key first, then by timestamp (descending for MVCC)
        match self.key.cmp(&other.key) {
            std::cmp::Ordering::Equal => other.timestamp.cmp(&self.timestamp),
            ord => ord,
        }
    }
}

/// Encodes a key-value pair to bytes.
///
/// Format: [key_len: u32][value_len: u32][key][value]
/// For tombstone: value_len = u32::MAX
///
/// TODO: Implement this function
pub fn encode_entry(entry: &Entry) -> Vec<u8> {
    todo!("implement encode_entry")
}

/// Decodes a key-value pair from bytes.
///
/// TODO: Implement this function
pub fn decode_entry(data: &[u8]) -> Result<(Entry, usize)> {
    todo!("implement decode_entry")
}

/// Computes CRC32 checksum.
pub fn checksum(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_new() {
        let entry = Entry::new(b"key".as_slice(), b"value".as_slice());
        assert_eq!(entry.key.as_ref(), b"key");
        assert_eq!(entry.value.as_ref().unwrap().as_ref(), b"value");
        assert!(!entry.is_tombstone());
    }

    #[test]
    fn test_entry_tombstone() {
        let entry = Entry::tombstone(b"key".as_slice());
        assert_eq!(entry.key.as_ref(), b"key");
        assert!(entry.is_tombstone());
    }

    #[test]
    fn test_entry_ordering() {
        let e1 = Entry::new(b"a".as_slice(), b"1".as_slice());
        let e2 = Entry::new(b"b".as_slice(), b"2".as_slice());
        let e3 = Entry::with_timestamp(b"a".as_slice(), Some(b"3".as_slice()), 10);

        assert!(e1 < e2);
        // Same key, higher timestamp should come first (for MVCC)
        assert!(e3 < e1);
    }

    #[test]
    fn test_encode_decode_entry() {
        let entry = Entry::new(b"hello".as_slice(), b"world".as_slice());
        let encoded = encode_entry(&entry);
        let (decoded, size) = decode_entry(&encoded).unwrap();

        assert_eq!(decoded.key, entry.key);
        assert_eq!(decoded.value, entry.value);
        assert_eq!(size, encoded.len());
    }

    #[test]
    fn test_encode_decode_tombstone() {
        let entry = Entry::tombstone(b"deleted".as_slice());
        let encoded = encode_entry(&entry);
        let (decoded, _) = decode_entry(&encoded).unwrap();

        assert_eq!(decoded.key, entry.key);
        assert!(decoded.is_tombstone());
    }

    #[test]
    fn test_checksum() {
        let data = b"hello world";
        let sum1 = checksum(data);
        let sum2 = checksum(data);
        assert_eq!(sum1, sum2);

        let different = b"hello worle";
        let sum3 = checksum(different);
        assert_ne!(sum1, sum3);
    }
}
