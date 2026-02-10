//! MemTable implementation using a skip list.
//!
//! The MemTable is an in-memory sorted data structure that buffers writes
//! before they are flushed to disk as SSTables.
//!
//! # Properties
//!
//! - Sorted by key for efficient range scans
//! - O(log n) insert, lookup, and delete
//! - Thread-safe using crossbeam's lock-free skip list
//! - Tracks approximate memory usage

use crate::{Entry, Key, Result, Value};
use bytes::Bytes;
use crossbeam_skiplist::SkipMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// An in-memory table for buffering writes.
pub struct MemTable {
    /// The underlying skip list
    /// Key: user key
    /// Value: (Option<value>, is_tombstone)
    map: SkipMap<Bytes, Option<Bytes>>,

    /// Approximate size of the memtable in bytes
    size: AtomicUsize,

    /// Maximum size before flush
    max_size: usize,

    /// Unique ID for this memtable
    id: u64,
}

impl MemTable {
    /// Creates a new MemTable with the given maximum size.
    ///
    /// TODO: Implement this function
    pub fn new(id: u64, max_size: usize) -> Self {
        todo!("implement MemTable::new")
    }

    /// Returns the memtable ID.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns the approximate size in bytes.
    pub fn size(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Returns true if the memtable is full (size >= max_size).
    ///
    /// TODO: Implement this function
    pub fn is_full(&self) -> bool {
        todo!("implement MemTable::is_full")
    }

    /// Returns true if the memtable is empty.
    ///
    /// TODO: Implement this function
    pub fn is_empty(&self) -> bool {
        todo!("implement MemTable::is_empty")
    }

    /// Inserts a key-value pair.
    ///
    /// TODO: Implement this function
    /// - Update the size estimate
    /// - Handle replacing existing keys
    pub fn put(&self, key: impl Into<Key>, value: impl Into<Value>) {
        todo!("implement MemTable::put")
    }

    /// Deletes a key by inserting a tombstone.
    ///
    /// TODO: Implement this function
    pub fn delete(&self, key: impl Into<Key>) {
        todo!("implement MemTable::delete")
    }

    /// Gets the value for a key.
    ///
    /// Returns:
    /// - Some(Some(value)) if key exists with value
    /// - Some(None) if key was deleted (tombstone)
    /// - None if key not in this memtable
    ///
    /// TODO: Implement this function
    pub fn get(&self, key: &[u8]) -> Option<Option<Bytes>> {
        todo!("implement MemTable::get")
    }

    /// Returns an iterator over all entries.
    ///
    /// TODO: Implement this function
    pub fn iter(&self) -> MemTableIterator<'_> {
        todo!("implement MemTable::iter")
    }

    /// Returns an iterator starting from the given key.
    ///
    /// TODO: Implement this function
    pub fn range_from(&self, key: &[u8]) -> MemTableIterator<'_> {
        todo!("implement MemTable::range_from")
    }

    /// Converts the memtable to a sorted vector of entries.
    /// Used when flushing to SSTable.
    ///
    /// TODO: Implement this function
    pub fn to_entries(&self) -> Vec<Entry> {
        todo!("implement MemTable::to_entries")
    }

    /// Returns the number of entries.
    ///
    /// TODO: Implement this function
    pub fn len(&self) -> usize {
        todo!("implement MemTable::len")
    }
}

/// Iterator over MemTable entries.
pub struct MemTableIterator<'a> {
    inner: crossbeam_skiplist::map::Iter<'a, Bytes, Option<Bytes>>,
}

impl<'a> MemTableIterator<'a> {
    /// Creates a new iterator.
    fn new(inner: crossbeam_skiplist::map::Iter<'a, Bytes, Option<Bytes>>) -> Self {
        MemTableIterator { inner }
    }
}

impl<'a> Iterator for MemTableIterator<'a> {
    type Item = Entry;

    /// Returns the next entry.
    ///
    /// TODO: Implement this function
    fn next(&mut self) -> Option<Self::Item> {
        todo!("implement MemTableIterator::next")
    }
}

/// Builder for creating MemTable from WAL recovery.
pub struct MemTableBuilder {
    entries: Vec<(Key, Option<Value>)>,
    id: u64,
    max_size: usize,
}

impl MemTableBuilder {
    /// Creates a new builder.
    pub fn new(id: u64, max_size: usize) -> Self {
        MemTableBuilder {
            entries: Vec::new(),
            id,
            max_size,
        }
    }

    /// Adds an entry.
    pub fn add(&mut self, key: Key, value: Option<Value>) {
        self.entries.push((key, value));
    }

    /// Builds the MemTable.
    ///
    /// TODO: Implement this function
    pub fn build(self) -> MemTable {
        todo!("implement MemTableBuilder::build")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memtable_new() {
        let mem = MemTable::new(1, 1024);
        assert_eq!(mem.id(), 1);
        assert!(mem.is_empty());
        assert!(!mem.is_full());
    }

    #[test]
    fn test_memtable_put_get() {
        let mem = MemTable::new(1, 1024 * 1024);

        mem.put(b"key1".as_slice(), b"value1".as_slice());
        mem.put(b"key2".as_slice(), b"value2".as_slice());

        assert_eq!(mem.get(b"key1"), Some(Some(Bytes::from_static(b"value1"))));
        assert_eq!(mem.get(b"key2"), Some(Some(Bytes::from_static(b"value2"))));
        assert_eq!(mem.get(b"key3"), None);
    }

    #[test]
    fn test_memtable_delete() {
        let mem = MemTable::new(1, 1024 * 1024);

        mem.put(b"key1".as_slice(), b"value1".as_slice());
        mem.delete(b"key1".as_slice());

        // Tombstone returns Some(None)
        assert_eq!(mem.get(b"key1"), Some(None));
    }

    #[test]
    fn test_memtable_overwrite() {
        let mem = MemTable::new(1, 1024 * 1024);

        mem.put(b"key1".as_slice(), b"value1".as_slice());
        mem.put(b"key1".as_slice(), b"value2".as_slice());

        assert_eq!(mem.get(b"key1"), Some(Some(Bytes::from_static(b"value2"))));
    }

    #[test]
    fn test_memtable_size_tracking() {
        let mem = MemTable::new(1, 100);

        mem.put(b"key1".as_slice(), b"value1".as_slice());
        assert!(mem.size() > 0);

        mem.put(b"key2".as_slice(), b"value2".as_slice());
        let size_after_two = mem.size();
        assert!(size_after_two > mem.size() / 2);
    }

    #[test]
    fn test_memtable_is_full() {
        let mem = MemTable::new(1, 50); // Very small

        mem.put(b"key1".as_slice(), b"a_very_long_value_that_exceeds_the_limit".as_slice());
        assert!(mem.is_full());
    }

    #[test]
    fn test_memtable_iterator() {
        let mem = MemTable::new(1, 1024 * 1024);

        mem.put(b"c".as_slice(), b"3".as_slice());
        mem.put(b"a".as_slice(), b"1".as_slice());
        mem.put(b"b".as_slice(), b"2".as_slice());

        let entries: Vec<Entry> = mem.iter().collect();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].key.as_ref(), b"a");
        assert_eq!(entries[1].key.as_ref(), b"b");
        assert_eq!(entries[2].key.as_ref(), b"c");
    }

    #[test]
    fn test_memtable_range_from() {
        let mem = MemTable::new(1, 1024 * 1024);

        mem.put(b"a".as_slice(), b"1".as_slice());
        mem.put(b"b".as_slice(), b"2".as_slice());
        mem.put(b"c".as_slice(), b"3".as_slice());
        mem.put(b"d".as_slice(), b"4".as_slice());

        let entries: Vec<Entry> = mem.range_from(b"b").collect();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].key.as_ref(), b"b");
        assert_eq!(entries[1].key.as_ref(), b"c");
        assert_eq!(entries[2].key.as_ref(), b"d");
    }

    #[test]
    fn test_memtable_to_entries() {
        let mem = MemTable::new(1, 1024 * 1024);

        mem.put(b"b".as_slice(), b"2".as_slice());
        mem.put(b"a".as_slice(), b"1".as_slice());
        mem.delete(b"c".as_slice());

        let entries = mem.to_entries();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].key.as_ref(), b"a");
        assert!(!entries[0].is_tombstone());
        assert_eq!(entries[2].key.as_ref(), b"c");
        assert!(entries[2].is_tombstone());
    }

    #[test]
    fn test_memtable_builder() {
        let mut builder = MemTableBuilder::new(1, 1024 * 1024);
        builder.add(Bytes::from_static(b"key1"), Some(Bytes::from_static(b"value1")));
        builder.add(Bytes::from_static(b"key2"), None); // Tombstone

        let mem = builder.build();

        assert_eq!(mem.get(b"key1"), Some(Some(Bytes::from_static(b"value1"))));
        assert_eq!(mem.get(b"key2"), Some(None));
    }
}
