//! SSTable iterator implementation.
//!
//! Iterates over all entries in an SSTable in sorted order.

use crate::sstable::block::{Block, BlockIterator};
use crate::sstable::SSTable;
use crate::{Entry, Key, Result};
use bytes::Bytes;

/// Iterator over entries in an SSTable.
pub struct SSTableIterator {
    /// The SSTable being iterated
    blocks: Vec<Bytes>,

    /// Index entries for seeking
    index: Vec<(Key, u64, u32)>,

    /// Current block index
    current_block: usize,

    /// Current block iterator
    block_iter: Option<BlockIterator>,

    /// Current entry (for peek)
    current_entry: Option<Entry>,
}

impl SSTableIterator {
    /// Creates a new iterator over all entries.
    ///
    /// TODO: Implement this function
    pub fn new(blocks: Vec<Bytes>, index: Vec<(Key, u64, u32)>) -> Self {
        todo!("implement SSTableIterator::new")
    }

    /// Creates an iterator starting at or after the given key.
    ///
    /// TODO: Implement this function
    pub fn seek(blocks: Vec<Bytes>, index: Vec<(Key, u64, u32)>, key: &[u8]) -> Self {
        todo!("implement SSTableIterator::seek")
    }

    /// Returns the current entry without advancing.
    pub fn peek(&self) -> Option<&Entry> {
        self.current_entry.as_ref()
    }

    /// Advances to the next block.
    ///
    /// TODO: Implement this function
    fn advance_block(&mut self) {
        todo!("implement SSTableIterator::advance_block")
    }

    /// Loads the next entry.
    ///
    /// TODO: Implement this function
    fn load_next(&mut self) {
        todo!("implement SSTableIterator::load_next")
    }
}

impl Iterator for SSTableIterator {
    type Item = Entry;

    fn next(&mut self) -> Option<Self::Item> {
        let entry = self.current_entry.take()?;
        self.load_next();
        Some(entry)
    }
}

/// Concatenating iterator over multiple SSTables.
pub struct ConcatIterator {
    iters: Vec<SSTableIterator>,
    current: usize,
}

impl ConcatIterator {
    /// Creates a new concatenating iterator.
    ///
    /// TODO: Implement this function
    pub fn new(iters: Vec<SSTableIterator>) -> Self {
        todo!("implement ConcatIterator::new")
    }

    /// Returns the current entry without advancing.
    pub fn peek(&self) -> Option<&Entry> {
        self.iters.get(self.current)?.peek()
    }
}

impl Iterator for ConcatIterator {
    type Item = Entry;

    /// Returns the next entry.
    ///
    /// TODO: Implement this function
    fn next(&mut self) -> Option<Self::Item> {
        todo!("implement ConcatIterator::next")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sstable::block::BlockBuilder;

    fn build_block(entries: Vec<(&[u8], &[u8])>) -> Bytes {
        let mut builder = BlockBuilder::new(4096);
        for (key, value) in entries {
            builder.add(&Entry::new(key, value));
        }
        builder.finish()
    }

    #[test]
    fn test_sstable_iterator() {
        let block1 = build_block(vec![(b"a", b"1"), (b"b", b"2")]);
        let block2 = build_block(vec![(b"c", b"3"), (b"d", b"4")]);

        let blocks = vec![block1, block2];
        let index = vec![
            (Bytes::from_static(b"b"), 0, 0),
            (Bytes::from_static(b"d"), 0, 0),
        ];

        let iter = SSTableIterator::new(blocks, index);
        let entries: Vec<Entry> = iter.collect();

        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].key.as_ref(), b"a");
        assert_eq!(entries[1].key.as_ref(), b"b");
        assert_eq!(entries[2].key.as_ref(), b"c");
        assert_eq!(entries[3].key.as_ref(), b"d");
    }

    #[test]
    fn test_sstable_iterator_seek() {
        let block1 = build_block(vec![(b"a", b"1"), (b"b", b"2")]);
        let block2 = build_block(vec![(b"c", b"3"), (b"d", b"4")]);

        let blocks = vec![block1, block2];
        let index = vec![
            (Bytes::from_static(b"b"), 0, 0),
            (Bytes::from_static(b"d"), 0, 0),
        ];

        let iter = SSTableIterator::seek(blocks, index, b"b");
        let entries: Vec<Entry> = iter.collect();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].key.as_ref(), b"b");
    }

    #[test]
    fn test_concat_iterator() {
        let block1 = build_block(vec![(b"a", b"1"), (b"b", b"2")]);
        let block2 = build_block(vec![(b"c", b"3"), (b"d", b"4")]);

        let iter1 = SSTableIterator::new(
            vec![block1],
            vec![(Bytes::from_static(b"b"), 0, 0)],
        );
        let iter2 = SSTableIterator::new(
            vec![block2],
            vec![(Bytes::from_static(b"d"), 0, 0)],
        );

        let concat = ConcatIterator::new(vec![iter1, iter2]);
        let entries: Vec<Entry> = concat.collect();

        assert_eq!(entries.len(), 4);
    }
}
