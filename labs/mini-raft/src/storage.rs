//! Persistent storage for Raft.
//!
//! Raft requires certain state to be persisted before responding to RPCs:
//! - currentTerm
//! - votedFor
//! - log entries
//!
//! This module provides a simple file-based storage implementation.

use crate::log::LogEntry;
use crate::state::PersistentState;
use crate::RaftError;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Snapshot of persistent state for storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageSnapshot {
    /// The persistent state (term, votedFor)
    pub state: PersistentState,

    /// The log entries
    pub entries: Vec<LogEntry>,
}

/// Storage interface for Raft persistence.
pub trait Storage: Send + Sync {
    /// Saves the current state.
    fn save(&mut self, snapshot: &StorageSnapshot) -> Result<(), RaftError>;

    /// Loads the saved state, or returns default if none exists.
    fn load(&self) -> Result<Option<StorageSnapshot>, RaftError>;

    /// Appends entries to the log (incremental write).
    fn append_entries(&mut self, entries: &[LogEntry]) -> Result<(), RaftError>;

    /// Syncs all data to disk.
    fn sync(&mut self) -> Result<(), RaftError>;
}

/// In-memory storage (for testing).
pub struct MemoryStorage {
    snapshot: Option<StorageSnapshot>,
}

impl MemoryStorage {
    /// Creates a new in-memory storage.
    ///
    /// TODO: Implement this function
    pub fn new() -> Self {
        todo!("implement MemoryStorage::new")
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        MemoryStorage::new()
    }
}

impl Storage for MemoryStorage {
    /// Saves the snapshot in memory.
    ///
    /// TODO: Implement this function
    fn save(&mut self, snapshot: &StorageSnapshot) -> Result<(), RaftError> {
        todo!("implement MemoryStorage::save")
    }

    /// Loads the snapshot from memory.
    ///
    /// TODO: Implement this function
    fn load(&self) -> Result<Option<StorageSnapshot>, RaftError> {
        todo!("implement MemoryStorage::load")
    }

    /// Appends entries (just updates the snapshot).
    ///
    /// TODO: Implement this function
    fn append_entries(&mut self, entries: &[LogEntry]) -> Result<(), RaftError> {
        todo!("implement MemoryStorage::append_entries")
    }

    /// No-op for memory storage.
    fn sync(&mut self) -> Result<(), RaftError> {
        Ok(())
    }
}

/// File-based storage using JSON lines format.
///
/// Format:
/// - First line: JSON of PersistentState
/// - Following lines: JSON of each LogEntry
pub struct FileStorage {
    path: String,
    state: Option<PersistentState>,
    entries: Vec<LogEntry>,
}

impl FileStorage {
    /// Creates or opens file storage at the given path.
    ///
    /// TODO: Implement this function
    pub fn new(path: &str) -> Result<Self, RaftError> {
        todo!("implement FileStorage::new")
    }

    /// Reads existing data from the file.
    ///
    /// TODO: Implement this function
    fn read_file(&mut self) -> Result<(), RaftError> {
        todo!("implement FileStorage::read_file")
    }

    /// Writes all data to the file.
    ///
    /// TODO: Implement this function
    fn write_file(&self) -> Result<(), RaftError> {
        todo!("implement FileStorage::write_file")
    }
}

impl Storage for FileStorage {
    /// Saves the snapshot to file.
    ///
    /// TODO: Implement this function
    fn save(&mut self, snapshot: &StorageSnapshot) -> Result<(), RaftError> {
        todo!("implement FileStorage::save")
    }

    /// Loads the snapshot from file.
    ///
    /// TODO: Implement this function
    fn load(&self) -> Result<Option<StorageSnapshot>, RaftError> {
        todo!("implement FileStorage::load")
    }

    /// Appends entries to the file incrementally.
    ///
    /// TODO: Implement this function
    fn append_entries(&mut self, entries: &[LogEntry]) -> Result<(), RaftError> {
        todo!("implement FileStorage::append_entries")
    }

    /// Syncs the file to disk.
    ///
    /// TODO: Implement this function
    fn sync(&mut self) -> Result<(), RaftError> {
        todo!("implement FileStorage::sync")
    }
}

/// Write-Ahead Log for durability.
///
/// Before any operation, the WAL entry is written and synced.
/// On recovery, the WAL is replayed to reconstruct state.
pub struct WriteAheadLog {
    path: String,
    file: Option<BufWriter<File>>,
}

/// WAL entry types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry {
    /// State update (term, votedFor)
    StateUpdate(PersistentState),

    /// Log append
    LogAppend(Vec<LogEntry>),

    /// Log truncate from index
    LogTruncate(u64),
}

impl WriteAheadLog {
    /// Opens or creates a WAL at the given path.
    ///
    /// TODO: Implement this function
    pub fn new(path: &str) -> Result<Self, RaftError> {
        todo!("implement WriteAheadLog::new")
    }

    /// Appends an entry to the WAL.
    ///
    /// TODO: Implement this function
    pub fn append(&mut self, entry: WalEntry) -> Result<(), RaftError> {
        todo!("implement WriteAheadLog::append")
    }

    /// Syncs the WAL to disk.
    ///
    /// TODO: Implement this function
    pub fn sync(&mut self) -> Result<(), RaftError> {
        todo!("implement WriteAheadLog::sync")
    }

    /// Replays all WAL entries.
    ///
    /// TODO: Implement this function
    pub fn replay(&self) -> Result<Vec<WalEntry>, RaftError> {
        todo!("implement WriteAheadLog::replay")
    }

    /// Compacts the WAL with the given snapshot.
    ///
    /// TODO: Implement this function (optional, advanced)
    pub fn compact(&mut self, _snapshot: &StorageSnapshot) -> Result<(), RaftError> {
        // Advanced: replace WAL with snapshot + subsequent entries
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Command;

    fn make_entry(term: u64, index: u64) -> LogEntry {
        LogEntry::new(term, index, Command::Noop)
    }

    #[test]
    fn test_memory_storage_save_load() {
        let mut storage = MemoryStorage::new();

        let snapshot = StorageSnapshot {
            state: PersistentState {
                current_term: 5,
                voted_for: Some(1),
            },
            entries: vec![make_entry(1, 1), make_entry(2, 2)],
        };

        storage.save(&snapshot).unwrap();

        let loaded = storage.load().unwrap().unwrap();
        assert_eq!(loaded.state.current_term, 5);
        assert_eq!(loaded.state.voted_for, Some(1));
        assert_eq!(loaded.entries.len(), 2);
    }

    #[test]
    fn test_memory_storage_append() {
        let mut storage = MemoryStorage::new();

        // Save initial state
        let snapshot = StorageSnapshot {
            state: PersistentState::new(),
            entries: vec![make_entry(1, 1)],
        };
        storage.save(&snapshot).unwrap();

        // Append more entries
        let new_entries = vec![make_entry(1, 2), make_entry(2, 3)];
        storage.append_entries(&new_entries).unwrap();

        let loaded = storage.load().unwrap().unwrap();
        assert_eq!(loaded.entries.len(), 3);
    }

    #[test]
    fn test_memory_storage_empty() {
        let storage = MemoryStorage::new();
        let loaded = storage.load().unwrap();
        assert!(loaded.is_none());
    }

    // File storage tests would need a temp directory
    // These are marked as integration tests

    #[test]
    #[ignore] // Requires file system
    fn test_file_storage_persistence() {
        let path = "/tmp/mini-raft-test-storage.json";

        // Clean up first
        let _ = std::fs::remove_file(path);

        {
            let mut storage = FileStorage::new(path).unwrap();
            let snapshot = StorageSnapshot {
                state: PersistentState {
                    current_term: 10,
                    voted_for: Some(2),
                },
                entries: vec![make_entry(5, 1), make_entry(10, 2)],
            };
            storage.save(&snapshot).unwrap();
            storage.sync().unwrap();
        }

        // Reopen and verify
        {
            let storage = FileStorage::new(path).unwrap();
            let loaded = storage.load().unwrap().unwrap();
            assert_eq!(loaded.state.current_term, 10);
            assert_eq!(loaded.entries.len(), 2);
        }

        // Clean up
        let _ = std::fs::remove_file(path);
    }
}
