//! Manifest management for LSM state.
//!
//! The manifest tracks:
//! - Current SSTable files and their levels
//! - Next SSTable ID
//! - Compaction state
//!
//! # Manifest Format
//!
//! The manifest is a log of changes:
//! - Add SSTable to level X
//! - Remove SSTable from level X
//! - Compaction markers

use crate::sstable::SSTableMeta;
use crate::{Key, LsmError, Result};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// A change to the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ManifestChange {
    /// Add an SSTable to a level
    AddSSTable {
        level: u32,
        id: u64,
        first_key: Bytes,
        last_key: Bytes,
        file_size: u64,
    },

    /// Remove an SSTable from a level
    RemoveSSTable { level: u32, id: u64 },

    /// Update next SSTable ID
    NextId(u64),

    /// Compaction started
    CompactionStart { level: u32, inputs: Vec<u64> },

    /// Compaction completed
    CompactionComplete { level: u32 },
}

/// Manifest file handler.
pub struct Manifest {
    /// Current state
    state: ManifestState,

    /// Manifest file writer
    writer: BufWriter<File>,

    /// Manifest file path
    path: String,
}

/// In-memory manifest state.
#[derive(Debug, Clone, Default)]
pub struct ManifestState {
    /// SSTables at each level
    pub levels: Vec<Vec<SSTableMeta>>,

    /// Next SSTable ID
    pub next_id: u64,

    /// In-progress compactions
    pub compactions: HashMap<u32, Vec<u64>>,
}

impl Manifest {
    /// Creates a new manifest file.
    ///
    /// TODO: Implement this function
    pub fn create(path: impl AsRef<Path>, num_levels: usize) -> Result<Self> {
        todo!("implement Manifest::create")
    }

    /// Opens an existing manifest file.
    ///
    /// TODO: Implement this function
    pub fn open(path: impl AsRef<Path>, num_levels: usize) -> Result<Self> {
        todo!("implement Manifest::open")
    }

    /// Records a change to the manifest.
    ///
    /// TODO: Implement this function
    pub fn record(&mut self, change: ManifestChange) -> Result<()> {
        todo!("implement Manifest::record")
    }

    /// Returns the current state.
    pub fn state(&self) -> &ManifestState {
        &self.state
    }

    /// Returns a mutable reference to the state.
    pub fn state_mut(&mut self) -> &mut ManifestState {
        &mut self.state
    }

    /// Syncs the manifest to disk.
    ///
    /// TODO: Implement this function
    pub fn sync(&mut self) -> Result<()> {
        todo!("implement Manifest::sync")
    }

    /// Gets the next SSTable ID and increments it.
    ///
    /// TODO: Implement this function
    pub fn next_sstable_id(&mut self) -> Result<u64> {
        todo!("implement Manifest::next_sstable_id")
    }

    /// Adds an SSTable to a level.
    ///
    /// TODO: Implement this function
    pub fn add_sstable(&mut self, level: u32, meta: SSTableMeta) -> Result<()> {
        todo!("implement Manifest::add_sstable")
    }

    /// Removes an SSTable from a level.
    ///
    /// TODO: Implement this function
    pub fn remove_sstable(&mut self, level: u32, id: u64) -> Result<()> {
        todo!("implement Manifest::remove_sstable")
    }

    /// Returns SSTables at a level.
    pub fn level(&self, level: u32) -> &[SSTableMeta] {
        self.state
            .levels
            .get(level as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Returns all SSTables that overlap with the given key range.
    ///
    /// TODO: Implement this function
    pub fn overlapping_sstables(
        &self,
        level: u32,
        start_key: &[u8],
        end_key: &[u8],
    ) -> Vec<&SSTableMeta> {
        todo!("implement Manifest::overlapping_sstables")
    }
}

impl ManifestState {
    /// Creates a new state with the given number of levels.
    pub fn new(num_levels: usize) -> Self {
        ManifestState {
            levels: (0..num_levels).map(|_| Vec::new()).collect(),
            next_id: 1,
            compactions: HashMap::new(),
        }
    }

    /// Applies a change to the state.
    ///
    /// TODO: Implement this function
    pub fn apply(&mut self, change: &ManifestChange) {
        todo!("implement ManifestState::apply")
    }

    /// Returns the total number of SSTables.
    pub fn total_sstables(&self) -> usize {
        self.levels.iter().map(|l| l.len()).sum()
    }

    /// Returns the total size of SSTables at a level.
    pub fn level_size(&self, level: u32) -> u64 {
        self.levels
            .get(level as usize)
            .map(|l| l.iter().map(|s| s.file_size).sum())
            .unwrap_or(0)
    }
}

/// Reads manifest changes from a file.
fn read_manifest(path: impl AsRef<Path>) -> Result<Vec<ManifestChange>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut changes = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let change: ManifestChange = serde_json::from_str(&line)
            .map_err(|e| LsmError::InvalidData(e.to_string()))?;
        changes.push(change);
    }

    Ok(changes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_meta(id: u64, first: &[u8], last: &[u8]) -> SSTableMeta {
        SSTableMeta {
            id,
            path: format!("test_{}.sst", id),
            first_key: Bytes::copy_from_slice(first),
            last_key: Bytes::copy_from_slice(last),
            file_size: 1000,
            level: 0,
        }
    }

    #[test]
    fn test_manifest_create() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("MANIFEST");

        let manifest = Manifest::create(&path, 7).unwrap();
        assert_eq!(manifest.state().levels.len(), 7);
        assert_eq!(manifest.state().next_id, 1);
    }

    #[test]
    fn test_manifest_add_sstable() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("MANIFEST");

        let mut manifest = Manifest::create(&path, 7).unwrap();

        let meta = make_meta(1, b"a", b"z");
        manifest.add_sstable(0, meta).unwrap();

        assert_eq!(manifest.level(0).len(), 1);
        assert_eq!(manifest.level(0)[0].id, 1);
    }

    #[test]
    fn test_manifest_remove_sstable() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("MANIFEST");

        let mut manifest = Manifest::create(&path, 7).unwrap();

        manifest.add_sstable(0, make_meta(1, b"a", b"m")).unwrap();
        manifest.add_sstable(0, make_meta(2, b"n", b"z")).unwrap();

        manifest.remove_sstable(0, 1).unwrap();

        assert_eq!(manifest.level(0).len(), 1);
        assert_eq!(manifest.level(0)[0].id, 2);
    }

    #[test]
    fn test_manifest_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("MANIFEST");

        // Create and add some SSTables
        {
            let mut manifest = Manifest::create(&path, 7).unwrap();
            manifest.add_sstable(0, make_meta(1, b"a", b"m")).unwrap();
            manifest.add_sstable(0, make_meta(2, b"n", b"z")).unwrap();
            manifest.add_sstable(1, make_meta(3, b"a", b"z")).unwrap();
            manifest.sync().unwrap();
        }

        // Reopen and verify
        {
            let manifest = Manifest::open(&path, 7).unwrap();
            assert_eq!(manifest.level(0).len(), 2);
            assert_eq!(manifest.level(1).len(), 1);
            assert_eq!(manifest.state().next_id, 4);
        }
    }

    #[test]
    fn test_manifest_overlapping() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("MANIFEST");

        let mut manifest = Manifest::create(&path, 7).unwrap();

        // L0 can have overlapping ranges
        manifest.add_sstable(0, make_meta(1, b"a", b"m")).unwrap();
        manifest.add_sstable(0, make_meta(2, b"f", b"r")).unwrap();
        manifest.add_sstable(0, make_meta(3, b"p", b"z")).unwrap();

        let overlapping = manifest.overlapping_sstables(0, b"g", b"q");
        assert_eq!(overlapping.len(), 3); // All overlap with [g, q]

        let overlapping = manifest.overlapping_sstables(0, b"s", b"z");
        assert_eq!(overlapping.len(), 1); // Only SST 3
    }

    #[test]
    fn test_manifest_state_apply() {
        let mut state = ManifestState::new(7);

        state.apply(&ManifestChange::AddSSTable {
            level: 0,
            id: 1,
            first_key: Bytes::from_static(b"a"),
            last_key: Bytes::from_static(b"z"),
            file_size: 1000,
        });

        assert_eq!(state.levels[0].len(), 1);

        state.apply(&ManifestChange::RemoveSSTable { level: 0, id: 1 });

        assert_eq!(state.levels[0].len(), 0);
    }
}
