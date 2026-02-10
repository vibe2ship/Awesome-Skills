//! Merge iterator implementation.
//!
//! Merges multiple sorted iterators into a single sorted stream.
//! Used for combining MemTable + SSTables during reads.
//!
//! # Semantics
//!
//! - When multiple iterators have the same key, prefer the one from
//!   the "newer" source (lower index in the iterator list)
//! - This implements MVCC-style reads where newer writes win

use crate::Entry;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A merge iterator that combines multiple sorted iterators.
pub struct MergeIterator<I: Iterator<Item = Entry>> {
    /// Heap of iterators with their current entries
    /// Entry with smallest key (and highest priority for same key) is at top
    heap: BinaryHeap<HeapEntry<I>>,

    /// Last key returned (for deduplication)
    last_key: Option<Vec<u8>>,
}

/// Entry in the heap, containing an iterator and its current entry.
struct HeapEntry<I: Iterator<Item = Entry>> {
    /// Current entry from this iterator
    entry: Entry,

    /// Iterator index (for priority - lower index = newer data)
    index: usize,

    /// The iterator
    iter: I,
}

impl<I: Iterator<Item = Entry>> PartialEq for HeapEntry<I> {
    fn eq(&self, other: &Self) -> bool {
        self.entry.key == other.entry.key && self.index == other.index
    }
}

impl<I: Iterator<Item = Entry>> Eq for HeapEntry<I> {}

impl<I: Iterator<Item = Entry>> PartialOrd for HeapEntry<I> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<I: Iterator<Item = Entry>> Ord for HeapEntry<I> {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap, so we reverse the comparison
        // We want smallest key first, then smallest index (newer data)
        match other.entry.key.cmp(&self.entry.key) {
            Ordering::Equal => other.index.cmp(&self.index),
            ord => ord,
        }
    }
}

impl<I: Iterator<Item = Entry>> MergeIterator<I> {
    /// Creates a new merge iterator from multiple iterators.
    /// Iterators should be ordered from newest to oldest.
    ///
    /// TODO: Implement this function
    pub fn new(iters: Vec<I>) -> Self {
        todo!("implement MergeIterator::new")
    }

    /// Returns the current entry without advancing.
    ///
    /// TODO: Implement this function
    pub fn peek(&self) -> Option<&Entry> {
        todo!("implement MergeIterator::peek")
    }

    /// Advances past entries with the same key as the current one.
    /// This implements the "newest wins" semantics.
    ///
    /// TODO: Implement this function
    fn skip_same_key(&mut self) {
        todo!("implement MergeIterator::skip_same_key")
    }
}

impl<I: Iterator<Item = Entry>> Iterator for MergeIterator<I> {
    type Item = Entry;

    /// Returns the next unique entry.
    ///
    /// TODO: Implement this function
    fn next(&mut self) -> Option<Self::Item> {
        todo!("implement MergeIterator::next")
    }
}

/// Two-way merge iterator (simpler case).
pub struct TwoMergeIterator<A: Iterator<Item = Entry>, B: Iterator<Item = Entry>> {
    a: std::iter::Peekable<A>,
    b: std::iter::Peekable<B>,
}

impl<A: Iterator<Item = Entry>, B: Iterator<Item = Entry>> TwoMergeIterator<A, B> {
    /// Creates a new two-way merge iterator.
    /// Iterator `a` has priority for same keys (newer data).
    ///
    /// TODO: Implement this function
    pub fn new(a: A, b: B) -> Self {
        todo!("implement TwoMergeIterator::new")
    }
}

impl<A: Iterator<Item = Entry>, B: Iterator<Item = Entry>> Iterator for TwoMergeIterator<A, B> {
    type Item = Entry;

    /// Returns the next entry.
    ///
    /// TODO: Implement this function
    fn next(&mut self) -> Option<Self::Item> {
        todo!("implement TwoMergeIterator::next")
    }
}

/// Iterator that filters out tombstones.
pub struct LiveIterator<I: Iterator<Item = Entry>> {
    inner: I,
}

impl<I: Iterator<Item = Entry>> LiveIterator<I> {
    /// Creates a new live iterator that skips tombstones.
    pub fn new(inner: I) -> Self {
        LiveIterator { inner }
    }
}

impl<I: Iterator<Item = Entry>> Iterator for LiveIterator<I> {
    type Item = Entry;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next() {
                Some(entry) if !entry.is_tombstone() => return Some(entry),
                Some(_) => continue, // Skip tombstone
                None => return None,
            }
        }
    }
}

/// Iterator adapter that limits to a key range.
pub struct RangeIterator<I: Iterator<Item = Entry>> {
    inner: I,
    end_key: Option<Vec<u8>>,
    done: bool,
}

impl<I: Iterator<Item = Entry>> RangeIterator<I> {
    /// Creates a new range iterator.
    ///
    /// TODO: Implement this function
    pub fn new(inner: I, end_key: Option<Vec<u8>>) -> Self {
        todo!("implement RangeIterator::new")
    }
}

impl<I: Iterator<Item = Entry>> Iterator for RangeIterator<I> {
    type Item = Entry;

    /// Returns the next entry within the range.
    ///
    /// TODO: Implement this function
    fn next(&mut self) -> Option<Self::Item> {
        todo!("implement RangeIterator::next")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    fn entry(key: &str, value: &str) -> Entry {
        Entry::new(Bytes::from(key.to_string()), Bytes::from(value.to_string()))
    }

    fn tombstone(key: &str) -> Entry {
        Entry::tombstone(Bytes::from(key.to_string()))
    }

    #[test]
    fn test_merge_iterator_basic() {
        let iter1 = vec![entry("a", "1"), entry("c", "3")].into_iter();
        let iter2 = vec![entry("b", "2"), entry("d", "4")].into_iter();

        let merged: Vec<_> = MergeIterator::new(vec![iter1, iter2]).collect();

        assert_eq!(merged.len(), 4);
        assert_eq!(merged[0].key.as_ref(), b"a");
        assert_eq!(merged[1].key.as_ref(), b"b");
        assert_eq!(merged[2].key.as_ref(), b"c");
        assert_eq!(merged[3].key.as_ref(), b"d");
    }

    #[test]
    fn test_merge_iterator_duplicate_keys() {
        // iter1 is "newer", so its value should win
        let iter1 = vec![entry("a", "new")].into_iter();
        let iter2 = vec![entry("a", "old")].into_iter();

        let merged: Vec<_> = MergeIterator::new(vec![iter1, iter2]).collect();

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].key.as_ref(), b"a");
        assert_eq!(merged[0].value.as_ref().unwrap().as_ref(), b"new");
    }

    #[test]
    fn test_merge_iterator_multiple_duplicates() {
        let iter1 = vec![entry("a", "1"), entry("b", "1")].into_iter();
        let iter2 = vec![entry("a", "2"), entry("c", "2")].into_iter();
        let iter3 = vec![entry("b", "3"), entry("c", "3")].into_iter();

        let merged: Vec<_> = MergeIterator::new(vec![iter1, iter2, iter3]).collect();

        assert_eq!(merged.len(), 3);
        // "a" -> "1" (from iter1)
        assert_eq!(merged[0].value.as_ref().unwrap().as_ref(), b"1");
        // "b" -> "1" (from iter1)
        assert_eq!(merged[1].value.as_ref().unwrap().as_ref(), b"1");
        // "c" -> "2" (from iter2)
        assert_eq!(merged[2].value.as_ref().unwrap().as_ref(), b"2");
    }

    #[test]
    fn test_two_merge_iterator() {
        let iter1 = vec![entry("a", "1"), entry("c", "3")].into_iter();
        let iter2 = vec![entry("b", "2"), entry("c", "old")].into_iter();

        let merged: Vec<_> = TwoMergeIterator::new(iter1, iter2).collect();

        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].key.as_ref(), b"a");
        assert_eq!(merged[1].key.as_ref(), b"b");
        // "c" should be from iter1 (newer)
        assert_eq!(merged[2].value.as_ref().unwrap().as_ref(), b"3");
    }

    #[test]
    fn test_live_iterator() {
        let entries = vec![
            entry("a", "1"),
            tombstone("b"),
            entry("c", "3"),
            tombstone("d"),
        ];

        let live: Vec<_> = LiveIterator::new(entries.into_iter()).collect();

        assert_eq!(live.len(), 2);
        assert_eq!(live[0].key.as_ref(), b"a");
        assert_eq!(live[1].key.as_ref(), b"c");
    }

    #[test]
    fn test_range_iterator() {
        let entries = vec![
            entry("a", "1"),
            entry("b", "2"),
            entry("c", "3"),
            entry("d", "4"),
        ];

        let range: Vec<_> = RangeIterator::new(
            entries.into_iter(),
            Some(b"c".to_vec()),
        ).collect();

        assert_eq!(range.len(), 2);
        assert_eq!(range[0].key.as_ref(), b"a");
        assert_eq!(range[1].key.as_ref(), b"b");
    }

    #[test]
    fn test_merge_empty() {
        let iters: Vec<std::vec::IntoIter<Entry>> = vec![];
        let merged: Vec<_> = MergeIterator::new(iters).collect();
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_single() {
        let iter = vec![entry("a", "1"), entry("b", "2")].into_iter();
        let merged: Vec<_> = MergeIterator::new(vec![iter]).collect();

        assert_eq!(merged.len(), 2);
    }
}
