//! Compaction strategies for LSM.
//!
//! Compaction merges SSTables to:
//! - Remove deleted keys (tombstones)
//! - Reduce space amplification
//! - Reduce read amplification
//!
//! # Strategies
//!
//! 1. **Leveled Compaction**: Bounds space amplification
//!    - Each level is 10x larger than previous
//!    - SSTables in L1+ are non-overlapping
//!    - Compacts one SSTable at a time to next level
//!
//! 2. **Tiered Compaction**: Bounds write amplification
//!    - Multiple sorted runs per level
//!    - Compacts entire level when full
//!
//! 3. **FIFO Compaction**: Simple time-based
//!    - Just removes old SSTables

use crate::manifest::ManifestState;
use crate::sstable::SSTableMeta;
use crate::{Key, LsmConfig, Result};
use bytes::Bytes;

/// A compaction task to execute.
#[derive(Debug, Clone)]
pub struct CompactionTask {
    /// SSTables from the input level
    pub inputs: Vec<SSTableMeta>,

    /// SSTables from the output level that overlap
    pub outputs: Vec<SSTableMeta>,

    /// Input level
    pub input_level: u32,

    /// Output level
    pub output_level: u32,

    /// Compaction type
    pub compaction_type: CompactionType,
}

/// Type of compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionType {
    /// Flush from L0 to L1
    Flush,
    /// Regular level compaction
    LevelCompaction,
    /// Manual compaction
    Manual,
}

/// Compaction strategy trait.
pub trait CompactionStrategy: Send + Sync {
    /// Picks the next compaction to perform.
    /// Returns None if no compaction is needed.
    fn pick_compaction(&self, state: &ManifestState, config: &LsmConfig) -> Option<CompactionTask>;

    /// Returns the strategy name.
    fn name(&self) -> &'static str;
}

/// Leveled compaction strategy.
pub struct LeveledCompaction;

impl LeveledCompaction {
    /// Creates a new leveled compaction strategy.
    pub fn new() -> Self {
        LeveledCompaction
    }

    /// Computes the target size for a level.
    ///
    /// TODO: Implement this function
    /// L0: unlimited (trigger by count)
    /// L1: base size
    /// L2+: L1 * size_ratio^(level-1)
    fn level_target_size(&self, level: u32, config: &LsmConfig) -> u64 {
        todo!("implement LeveledCompaction::level_target_size")
    }

    /// Picks an SSTable from L0 to compact to L1.
    ///
    /// TODO: Implement this function
    fn pick_l0_compaction(&self, state: &ManifestState) -> Option<CompactionTask> {
        todo!("implement LeveledCompaction::pick_l0_compaction")
    }

    /// Picks an SSTable from level N to compact to level N+1.
    ///
    /// TODO: Implement this function
    fn pick_level_compaction(
        &self,
        state: &ManifestState,
        config: &LsmConfig,
    ) -> Option<CompactionTask> {
        todo!("implement LeveledCompaction::pick_level_compaction")
    }
}

impl Default for LeveledCompaction {
    fn default() -> Self {
        LeveledCompaction::new()
    }
}

impl CompactionStrategy for LeveledCompaction {
    /// Picks the next compaction.
    ///
    /// TODO: Implement this function
    /// Priority:
    /// 1. L0 compaction if too many L0 files
    /// 2. Level compaction if any level exceeds target size
    fn pick_compaction(&self, state: &ManifestState, config: &LsmConfig) -> Option<CompactionTask> {
        todo!("implement LeveledCompaction::pick_compaction")
    }

    fn name(&self) -> &'static str {
        "leveled"
    }
}

/// Tiered compaction strategy.
pub struct TieredCompaction;

impl TieredCompaction {
    /// Creates a new tiered compaction strategy.
    pub fn new() -> Self {
        TieredCompaction
    }
}

impl Default for TieredCompaction {
    fn default() -> Self {
        TieredCompaction::new()
    }
}

impl CompactionStrategy for TieredCompaction {
    /// Picks the next compaction.
    ///
    /// TODO: Implement this function
    fn pick_compaction(&self, state: &ManifestState, config: &LsmConfig) -> Option<CompactionTask> {
        todo!("implement TieredCompaction::pick_compaction")
    }

    fn name(&self) -> &'static str {
        "tiered"
    }
}

/// FIFO compaction strategy.
pub struct FifoCompaction {
    /// Maximum total size before eviction
    max_size: u64,
}

impl FifoCompaction {
    /// Creates a new FIFO compaction strategy.
    pub fn new(max_size: u64) -> Self {
        FifoCompaction { max_size }
    }
}

impl CompactionStrategy for FifoCompaction {
    /// Picks SSTables to remove based on age.
    ///
    /// TODO: Implement this function
    fn pick_compaction(&self, state: &ManifestState, config: &LsmConfig) -> Option<CompactionTask> {
        todo!("implement FifoCompaction::pick_compaction")
    }

    fn name(&self) -> &'static str {
        "fifo"
    }
}

/// Finds SSTables that overlap with a key range.
///
/// TODO: Implement this function
pub fn find_overlapping(
    sstables: &[SSTableMeta],
    start_key: &[u8],
    end_key: &[u8],
) -> Vec<SSTableMeta> {
    todo!("implement find_overlapping")
}

/// Checks if two key ranges overlap.
fn ranges_overlap(a_start: &[u8], a_end: &[u8], b_start: &[u8], b_end: &[u8]) -> bool {
    a_start <= b_end && b_start <= a_end
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_meta(id: u64, level: u32, first: &[u8], last: &[u8], size: u64) -> SSTableMeta {
        SSTableMeta {
            id,
            path: format!("test_{}.sst", id),
            first_key: Bytes::copy_from_slice(first),
            last_key: Bytes::copy_from_slice(last),
            file_size: size,
            level,
        }
    }

    #[test]
    fn test_ranges_overlap() {
        assert!(ranges_overlap(b"a", b"c", b"b", b"d")); // Overlap
        assert!(ranges_overlap(b"a", b"c", b"a", b"c")); // Exact
        assert!(ranges_overlap(b"a", b"d", b"b", b"c")); // Contained
        assert!(!ranges_overlap(b"a", b"b", b"c", b"d")); // No overlap
        assert!(ranges_overlap(b"a", b"b", b"b", b"c")); // Touch at boundary
    }

    #[test]
    fn test_find_overlapping() {
        let sstables = vec![
            make_meta(1, 1, b"a", b"c", 1000),
            make_meta(2, 1, b"d", b"f", 1000),
            make_meta(3, 1, b"g", b"i", 1000),
        ];

        let overlapping = find_overlapping(&sstables, b"b", b"e");
        assert_eq!(overlapping.len(), 2);
        assert!(overlapping.iter().any(|s| s.id == 1));
        assert!(overlapping.iter().any(|s| s.id == 2));
    }

    #[test]
    fn test_leveled_l0_trigger() {
        let strategy = LeveledCompaction::new();
        let config = LsmConfig {
            l0_compaction_trigger: 4,
            ..Default::default()
        };

        let mut state = ManifestState::new(7);

        // Add 3 L0 SSTables - no compaction needed
        for i in 0..3 {
            state.levels[0].push(make_meta(i, 0, b"a", b"z", 1000));
        }
        assert!(strategy.pick_compaction(&state, &config).is_none());

        // Add 4th L0 SSTable - should trigger compaction
        state.levels[0].push(make_meta(3, 0, b"a", b"z", 1000));
        let task = strategy.pick_compaction(&state, &config);
        assert!(task.is_some());
        assert_eq!(task.unwrap().input_level, 0);
    }

    #[test]
    fn test_leveled_level_compaction() {
        let strategy = LeveledCompaction::new();
        let config = LsmConfig {
            l0_compaction_trigger: 4,
            level_size_ratio: 10,
            ..Default::default()
        };

        let mut state = ManifestState::new(7);

        // Fill L1 beyond target size
        for i in 0..20 {
            let start = format!("{:02}", i * 5);
            let end = format!("{:02}", i * 5 + 4);
            state.levels[1].push(make_meta(i, 1, start.as_bytes(), end.as_bytes(), 1024 * 1024));
        }

        let task = strategy.pick_compaction(&state, &config);
        // Should pick L1 -> L2 compaction
        if let Some(task) = task {
            assert_eq!(task.input_level, 1);
            assert_eq!(task.output_level, 2);
        }
    }

    #[test]
    fn test_compaction_task() {
        let task = CompactionTask {
            inputs: vec![make_meta(1, 0, b"a", b"m", 1000)],
            outputs: vec![
                make_meta(2, 1, b"a", b"f", 1000),
                make_meta(3, 1, b"g", b"n", 1000),
            ],
            input_level: 0,
            output_level: 1,
            compaction_type: CompactionType::LevelCompaction,
        };

        assert_eq!(task.inputs.len(), 1);
        assert_eq!(task.outputs.len(), 2);
    }
}
