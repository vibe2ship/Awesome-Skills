//! Block format for SSTables.
//!
//! A block is the basic unit of storage in an SSTable.
//!
//! # Block Format
//!
//! ```text
//! +--------------------------------------------+
//! |              Entry 0                       |
//! |  [shared_key_len][unshared_key_len][value_len]
//! |  [unshared_key][value]                     |
//! +--------------------------------------------+
//! |              Entry 1                       |
//! +--------------------------------------------+
//! |              ...                           |
//! +--------------------------------------------+
//! |              Entry N                       |
//! +--------------------------------------------+
//! |           Restart Points                   |
//! |  [restart_0][restart_1]...[restart_n]      |
//! +--------------------------------------------+
//! |         Num Restarts (u32)                 |
//! +--------------------------------------------+
//! |          Checksum (u32)                    |
//! +--------------------------------------------+
//! ```
//!
//! # Key Prefix Compression
//!
//! To save space, we use prefix compression:
//! - Store how many bytes are shared with the previous key
//! - Only store the unshared suffix
//!
//! Example:
//! - key1 = "apple"
//! - key2 = "application" -> shared=4 ("appl"), unshared="ication"

use crate::{checksum, Entry, Key, LsmError, Result, Value};
use bytes::{Buf, BufMut, Bytes, BytesMut};

/// Number of entries between restart points.
/// Restart points allow random access within a block.
pub const RESTART_INTERVAL: usize = 16;

/// A data block containing sorted key-value pairs.
#[derive(Debug)]
pub struct Block {
    /// Raw block data
    data: Bytes,

    /// Offsets of restart points
    restarts: Vec<u32>,
}

impl Block {
    /// Decodes a block from raw bytes.
    ///
    /// TODO: Implement this function
    /// - Read restart points from the end
    /// - Verify checksum
    pub fn decode(data: Bytes) -> Result<Self> {
        todo!("implement Block::decode")
    }

    /// Returns the raw data.
    pub fn data(&self) -> &Bytes {
        &self.data
    }

    /// Returns the restart points.
    pub fn restarts(&self) -> &[u32] {
        &self.restarts
    }

    /// Gets the value for a key using binary search on restart points.
    ///
    /// TODO: Implement this function
    pub fn get(&self, key: &[u8]) -> Result<Option<Value>> {
        todo!("implement Block::get")
    }

    /// Creates an iterator over all entries.
    ///
    /// TODO: Implement this function
    pub fn iter(&self) -> BlockIterator {
        todo!("implement Block::iter")
    }

    /// Creates an iterator starting at or after the given key.
    ///
    /// TODO: Implement this function
    pub fn seek(&self, key: &[u8]) -> BlockIterator {
        todo!("implement Block::seek")
    }

    /// Finds the restart point to start searching from.
    ///
    /// TODO: Implement this function
    fn find_restart_point(&self, key: &[u8]) -> usize {
        todo!("implement Block::find_restart_point")
    }

    /// Decodes an entry at the given offset.
    ///
    /// TODO: Implement this function
    /// Returns (key, value, next_offset)
    fn decode_entry(&self, offset: usize, prev_key: &[u8]) -> Result<(Key, Option<Value>, usize)> {
        todo!("implement Block::decode_entry")
    }
}

/// Iterator over entries in a block.
pub struct BlockIterator {
    block: Bytes,
    restarts: Vec<u32>,
    data_end: usize,
    offset: usize,
    prev_key: BytesMut,
    current_entry: Option<Entry>,
}

impl BlockIterator {
    /// Creates a new iterator starting at the given offset.
    fn new(block: Bytes, restarts: Vec<u32>, start_offset: usize) -> Self {
        let data_end = block.len() - restarts.len() * 4 - 4 - 4; // exclude restarts + num_restarts + checksum

        let mut iter = BlockIterator {
            block,
            restarts,
            data_end,
            offset: start_offset,
            prev_key: BytesMut::new(),
            current_entry: None,
        };

        // Load first entry
        iter.load_next();
        iter
    }

    /// Loads the next entry into current_entry.
    ///
    /// TODO: Implement this function
    fn load_next(&mut self) {
        todo!("implement BlockIterator::load_next")
    }

    /// Returns the current entry without advancing.
    pub fn peek(&self) -> Option<&Entry> {
        self.current_entry.as_ref()
    }
}

impl Iterator for BlockIterator {
    type Item = Entry;

    fn next(&mut self) -> Option<Self::Item> {
        let entry = self.current_entry.take()?;
        self.load_next();
        Some(entry)
    }
}

/// Builder for creating a block.
pub struct BlockBuilder {
    /// Buffer for encoded data
    buffer: BytesMut,

    /// Restart points
    restarts: Vec<u32>,

    /// Number of entries since last restart
    entries_since_restart: usize,

    /// Previous key (for prefix compression)
    prev_key: BytesMut,

    /// Number of entries
    entry_count: usize,

    /// Target block size
    target_size: usize,

    /// First key in the block
    first_key: Option<Key>,

    /// Last key in the block
    last_key: Option<Key>,
}

impl BlockBuilder {
    /// Creates a new block builder.
    ///
    /// TODO: Implement this function
    pub fn new(target_size: usize) -> Self {
        todo!("implement BlockBuilder::new")
    }

    /// Adds an entry to the block.
    ///
    /// TODO: Implement this function
    /// - Use prefix compression
    /// - Add restart point if needed
    pub fn add(&mut self, entry: &Entry) {
        todo!("implement BlockBuilder::add")
    }

    /// Returns the estimated size if we add an entry.
    ///
    /// TODO: Implement this function
    pub fn estimated_size_after(&self, entry: &Entry) -> usize {
        todo!("implement BlockBuilder::estimated_size_after")
    }

    /// Returns true if the block is empty.
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Returns the current size estimate.
    pub fn size(&self) -> usize {
        // data + restarts + num_restarts + checksum
        self.buffer.len() + self.restarts.len() * 4 + 4 + 4
    }

    /// Returns true if the block is full (reached target size).
    pub fn is_full(&self) -> bool {
        self.size() >= self.target_size
    }

    /// Returns the first key in the block.
    pub fn first_key(&self) -> Option<&Key> {
        self.first_key.as_ref()
    }

    /// Returns the last key in the block.
    pub fn last_key(&self) -> Option<&Key> {
        self.last_key.as_ref()
    }

    /// Finishes building and returns the encoded block.
    ///
    /// TODO: Implement this function
    pub fn finish(self) -> Bytes {
        todo!("implement BlockBuilder::finish")
    }

    /// Computes shared prefix length between two keys.
    fn shared_prefix_len(a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }
}

/// Encodes a varint (variable-length integer).
fn encode_varint(buf: &mut BytesMut, mut value: u64) {
    while value >= 0x80 {
        buf.put_u8((value as u8) | 0x80);
        value >>= 7;
    }
    buf.put_u8(value as u8);
}

/// Decodes a varint from bytes.
/// Returns (value, bytes_consumed).
fn decode_varint(data: &[u8]) -> Result<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0;

    for (i, &byte) in data.iter().enumerate() {
        if shift > 63 {
            return Err(LsmError::InvalidData("varint too long".to_string()));
        }

        result |= ((byte & 0x7F) as u64) << shift;

        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }

        shift += 7;
    }

    Err(LsmError::InvalidData("unterminated varint".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_encode_decode() {
        let test_values = [0u64, 1, 127, 128, 255, 256, 16383, 16384, u64::MAX];

        for &value in &test_values {
            let mut buf = BytesMut::new();
            encode_varint(&mut buf, value);
            let (decoded, _) = decode_varint(&buf).unwrap();
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn test_block_builder_empty() {
        let builder = BlockBuilder::new(4096);
        assert!(builder.is_empty());
    }

    #[test]
    fn test_block_builder_add_entries() {
        let mut builder = BlockBuilder::new(4096);

        builder.add(&Entry::new(b"apple".as_slice(), b"red".as_slice()));
        builder.add(&Entry::new(b"application".as_slice(), b"app".as_slice()));
        builder.add(&Entry::new(b"banana".as_slice(), b"yellow".as_slice()));

        assert_eq!(builder.first_key().unwrap().as_ref(), b"apple");
        assert_eq!(builder.last_key().unwrap().as_ref(), b"banana");
    }

    #[test]
    fn test_block_encode_decode() {
        let mut builder = BlockBuilder::new(4096);

        builder.add(&Entry::new(b"key1".as_slice(), b"value1".as_slice()));
        builder.add(&Entry::new(b"key2".as_slice(), b"value2".as_slice()));
        builder.add(&Entry::new(b"key3".as_slice(), b"value3".as_slice()));

        let data = builder.finish();
        let block = Block::decode(data).unwrap();

        assert_eq!(block.get(b"key1").unwrap(), Some(Bytes::from_static(b"value1")));
        assert_eq!(block.get(b"key2").unwrap(), Some(Bytes::from_static(b"value2")));
        assert_eq!(block.get(b"key3").unwrap(), Some(Bytes::from_static(b"value3")));
        assert!(block.get(b"key4").unwrap().is_none());
    }

    #[test]
    fn test_block_iterator() {
        let mut builder = BlockBuilder::new(4096);

        builder.add(&Entry::new(b"a".as_slice(), b"1".as_slice()));
        builder.add(&Entry::new(b"b".as_slice(), b"2".as_slice()));
        builder.add(&Entry::new(b"c".as_slice(), b"3".as_slice()));

        let data = builder.finish();
        let block = Block::decode(data).unwrap();

        let entries: Vec<Entry> = block.iter().collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].key.as_ref(), b"a");
        assert_eq!(entries[1].key.as_ref(), b"b");
        assert_eq!(entries[2].key.as_ref(), b"c");
    }

    #[test]
    fn test_block_seek() {
        let mut builder = BlockBuilder::new(4096);

        builder.add(&Entry::new(b"a".as_slice(), b"1".as_slice()));
        builder.add(&Entry::new(b"c".as_slice(), b"3".as_slice()));
        builder.add(&Entry::new(b"e".as_slice(), b"5".as_slice()));

        let data = builder.finish();
        let block = Block::decode(data).unwrap();

        // Seek to "b" should land on "c"
        let mut iter = block.seek(b"b");
        let entry = iter.next().unwrap();
        assert_eq!(entry.key.as_ref(), b"c");

        // Seek to "c" should land on "c"
        let mut iter = block.seek(b"c");
        let entry = iter.next().unwrap();
        assert_eq!(entry.key.as_ref(), b"c");
    }

    #[test]
    fn test_block_tombstone() {
        let mut builder = BlockBuilder::new(4096);

        builder.add(&Entry::new(b"key1".as_slice(), b"value1".as_slice()));
        builder.add(&Entry::tombstone(b"key2".as_slice()));
        builder.add(&Entry::new(b"key3".as_slice(), b"value3".as_slice()));

        let data = builder.finish();
        let block = Block::decode(data).unwrap();

        // Tombstone should return None for value
        assert!(block.get(b"key2").unwrap().is_none());
    }

    #[test]
    fn test_shared_prefix() {
        assert_eq!(BlockBuilder::shared_prefix_len(b"apple", b"application"), 4);
        assert_eq!(BlockBuilder::shared_prefix_len(b"apple", b"banana"), 0);
        assert_eq!(BlockBuilder::shared_prefix_len(b"abc", b"abc"), 3);
        assert_eq!(BlockBuilder::shared_prefix_len(b"", b"abc"), 0);
    }
}
