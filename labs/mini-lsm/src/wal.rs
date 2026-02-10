//! Write-Ahead Log (WAL) implementation.
//!
//! The WAL ensures durability by writing operations to disk before
//! applying them to the MemTable.
//!
//! # WAL Record Format
//!
//! ```text
//! +---------------+---------------+---------------+
//! | length (u32)  | checksum (u32)| data          |
//! +---------------+---------------+---------------+
//! ```
//!
//! # Recovery
//!
//! On startup, replay the WAL to reconstruct the MemTable.

use crate::{checksum, Entry, Key, LsmError, Result, Value};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// WAL record type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum RecordType {
    /// Put operation
    Put = 1,
    /// Delete operation
    Delete = 2,
}

/// A record in the WAL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    /// Record type
    pub record_type: RecordType,
    /// Key
    pub key: Bytes,
    /// Value (None for Delete)
    pub value: Option<Bytes>,
}

impl WalRecord {
    /// Creates a Put record.
    pub fn put(key: impl Into<Bytes>, value: impl Into<Bytes>) -> Self {
        WalRecord {
            record_type: RecordType::Put,
            key: key.into(),
            value: Some(value.into()),
        }
    }

    /// Creates a Delete record.
    pub fn delete(key: impl Into<Bytes>) -> Self {
        WalRecord {
            record_type: RecordType::Delete,
            key: key.into(),
            value: None,
        }
    }

    /// Converts to an Entry.
    pub fn to_entry(&self) -> Entry {
        match self.record_type {
            RecordType::Put => Entry::new(self.key.clone(), self.value.clone().unwrap()),
            RecordType::Delete => Entry::tombstone(self.key.clone()),
        }
    }

    /// Encodes the record to bytes.
    ///
    /// TODO: Implement this function
    /// Format: [type: u8][key_len: u32][value_len: u32][key][value]
    pub fn encode(&self) -> Bytes {
        todo!("implement WalRecord::encode")
    }

    /// Decodes a record from bytes.
    ///
    /// TODO: Implement this function
    pub fn decode(data: &[u8]) -> Result<Self> {
        todo!("implement WalRecord::decode")
    }
}

/// Write-Ahead Log.
pub struct Wal {
    /// WAL file writer
    writer: BufWriter<File>,

    /// WAL file path
    path: String,

    /// Whether to sync after each write
    sync: bool,

    /// Current file size
    size: u64,
}

impl Wal {
    /// Creates or opens a WAL file.
    ///
    /// TODO: Implement this function
    pub fn new(path: impl AsRef<Path>, sync: bool) -> Result<Self> {
        todo!("implement Wal::new")
    }

    /// Appends a record to the WAL.
    ///
    /// TODO: Implement this function
    /// - Encode the record
    /// - Write length + checksum + data
    /// - Optionally sync
    pub fn append(&mut self, record: &WalRecord) -> Result<()> {
        todo!("implement Wal::append")
    }

    /// Syncs the WAL to disk.
    ///
    /// TODO: Implement this function
    pub fn sync(&mut self) -> Result<()> {
        todo!("implement Wal::sync")
    }

    /// Returns the file path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the current size.
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// WAL reader for recovery.
pub struct WalReader {
    reader: BufReader<File>,
    path: String,
}

impl WalReader {
    /// Opens a WAL file for reading.
    ///
    /// TODO: Implement this function
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        todo!("implement WalReader::open")
    }

    /// Returns an iterator over all records.
    ///
    /// TODO: Implement this function
    pub fn iter(&mut self) -> WalIterator<'_> {
        todo!("implement WalReader::iter")
    }

    /// Recovers all entries from the WAL.
    ///
    /// TODO: Implement this function
    pub fn recover(&mut self) -> Result<Vec<Entry>> {
        todo!("implement WalReader::recover")
    }
}

/// Iterator over WAL records.
pub struct WalIterator<'a> {
    reader: &'a mut BufReader<File>,
}

impl<'a> Iterator for WalIterator<'a> {
    type Item = Result<WalRecord>;

    /// Reads the next record.
    ///
    /// TODO: Implement this function
    fn next(&mut self) -> Option<Self::Item> {
        todo!("implement WalIterator::next")
    }
}

/// Reads one record from the file.
///
/// TODO: Implement this function
fn read_record(reader: &mut impl Read) -> Result<Option<WalRecord>> {
    todo!("implement read_record")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_wal_record_encode_decode() {
        let record = WalRecord::put(b"key".as_slice(), b"value".as_slice());
        let encoded = record.encode();
        let decoded = WalRecord::decode(&encoded).unwrap();

        assert_eq!(record.record_type, decoded.record_type);
        assert_eq!(record.key, decoded.key);
        assert_eq!(record.value, decoded.value);
    }

    #[test]
    fn test_wal_record_delete() {
        let record = WalRecord::delete(b"key".as_slice());
        let encoded = record.encode();
        let decoded = WalRecord::decode(&encoded).unwrap();

        assert_eq!(decoded.record_type, RecordType::Delete);
        assert_eq!(decoded.key.as_ref(), b"key");
        assert!(decoded.value.is_none());
    }

    #[test]
    fn test_wal_append() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        {
            let mut wal = Wal::new(&path, false).unwrap();
            wal.append(&WalRecord::put(b"key1".as_slice(), b"value1".as_slice())).unwrap();
            wal.append(&WalRecord::put(b"key2".as_slice(), b"value2".as_slice())).unwrap();
            wal.append(&WalRecord::delete(b"key1".as_slice())).unwrap();
            wal.sync().unwrap();
        }

        // Verify file was written
        assert!(path.exists());
    }

    #[test]
    fn test_wal_recovery() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        // Write some records
        {
            let mut wal = Wal::new(&path, false).unwrap();
            wal.append(&WalRecord::put(b"key1".as_slice(), b"value1".as_slice())).unwrap();
            wal.append(&WalRecord::put(b"key2".as_slice(), b"value2".as_slice())).unwrap();
            wal.append(&WalRecord::delete(b"key1".as_slice())).unwrap();
            wal.sync().unwrap();
        }

        // Recover
        {
            let mut reader = WalReader::open(&path).unwrap();
            let entries = reader.recover().unwrap();

            assert_eq!(entries.len(), 3);
            assert_eq!(entries[0].key.as_ref(), b"key1");
            assert!(!entries[0].is_tombstone());
            assert_eq!(entries[1].key.as_ref(), b"key2");
            assert_eq!(entries[2].key.as_ref(), b"key1");
            assert!(entries[2].is_tombstone());
        }
    }

    #[test]
    fn test_wal_iterator() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        {
            let mut wal = Wal::new(&path, false).unwrap();
            wal.append(&WalRecord::put(b"a".as_slice(), b"1".as_slice())).unwrap();
            wal.append(&WalRecord::put(b"b".as_slice(), b"2".as_slice())).unwrap();
            wal.sync().unwrap();
        }

        {
            let mut reader = WalReader::open(&path).unwrap();
            let records: Vec<_> = reader.iter().map(|r| r.unwrap()).collect();

            assert_eq!(records.len(), 2);
            assert_eq!(records[0].key.as_ref(), b"a");
            assert_eq!(records[1].key.as_ref(), b"b");
        }
    }
}
