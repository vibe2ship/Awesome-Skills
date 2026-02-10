//! Replicated log implementation for Raft.
//!
//! The log is the core data structure in Raft. It stores a sequence of
//! commands that are replicated to all nodes in the cluster.
//!
//! # Log Properties
//!
//! 1. Entries are indexed starting from 1 (0 means no entries)
//! 2. Each entry has a term number
//! 3. Log Matching Property: If two logs contain an entry with the same
//!    index and term, the logs are identical up to that index

use crate::{Command, LogIndex, Term};
use serde::{Deserialize, Serialize};

/// A single entry in the Raft log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogEntry {
    /// The term when the entry was created
    pub term: Term,

    /// The index of this entry (1-indexed)
    pub index: LogIndex,

    /// The command to apply to the state machine
    pub command: Command,
}

impl LogEntry {
    /// Creates a new log entry.
    pub fn new(term: Term, index: LogIndex, command: Command) -> Self {
        LogEntry {
            term,
            index,
            command,
        }
    }
}

/// The replicated log.
///
/// # Implementation Notes
///
/// - Log indices are 1-based (entry at index 1 is entries[0])
/// - Index 0 is a sentinel meaning "no entry"
/// - The log is append-only from the leader's perspective
/// - Followers may need to truncate on conflicts
pub struct Log {
    /// The log entries (0-indexed internally, but LogIndex is 1-based)
    entries: Vec<LogEntry>,
}

impl Log {
    /// Creates a new empty log.
    ///
    /// TODO: Implement this function
    pub fn new() -> Self {
        todo!("implement Log::new")
    }

    /// Returns the number of entries in the log.
    ///
    /// TODO: Implement this function
    pub fn len(&self) -> usize {
        todo!("implement Log::len")
    }

    /// Returns true if the log is empty.
    ///
    /// TODO: Implement this function
    pub fn is_empty(&self) -> bool {
        todo!("implement Log::is_empty")
    }

    /// Returns the index of the last log entry, or 0 if empty.
    ///
    /// TODO: Implement this function
    pub fn last_index(&self) -> LogIndex {
        todo!("implement Log::last_index")
    }

    /// Returns the term of the last log entry, or 0 if empty.
    ///
    /// TODO: Implement this function
    pub fn last_term(&self) -> Term {
        todo!("implement Log::last_term")
    }

    /// Gets an entry by index (1-based).
    /// Returns None if index is 0 or out of bounds.
    ///
    /// TODO: Implement this function
    pub fn get(&self, index: LogIndex) -> Option<&LogEntry> {
        todo!("implement Log::get")
    }

    /// Gets the term of the entry at the given index.
    /// Returns 0 if index is 0 or out of bounds.
    ///
    /// TODO: Implement this function
    pub fn term_at(&self, index: LogIndex) -> Term {
        todo!("implement Log::term_at")
    }

    /// Appends an entry to the log.
    /// The entry's index should be last_index() + 1.
    ///
    /// TODO: Implement this function
    pub fn append(&mut self, entry: LogEntry) {
        todo!("implement Log::append")
    }

    /// Appends multiple entries to the log.
    ///
    /// TODO: Implement this function
    pub fn append_entries(&mut self, entries: Vec<LogEntry>) {
        todo!("implement Log::append_entries")
    }

    /// Truncates the log, removing entries from `from_index` onwards.
    /// Used when a follower needs to remove conflicting entries.
    ///
    /// TODO: Implement this function
    pub fn truncate_from(&mut self, from_index: LogIndex) {
        todo!("implement Log::truncate_from")
    }

    /// Returns entries from start_index to the end.
    /// Used for AppendEntries RPC.
    ///
    /// TODO: Implement this function
    pub fn entries_from(&self, start_index: LogIndex) -> Vec<LogEntry> {
        todo!("implement Log::entries_from")
    }

    /// Checks if this log is at least as up-to-date as the given last log info.
    /// Used for vote decisions.
    ///
    /// A log is more up-to-date if:
    /// 1. Its last entry has a higher term, OR
    /// 2. Same term but longer log
    ///
    /// TODO: Implement this function
    pub fn is_up_to_date(&self, last_log_index: LogIndex, last_log_term: Term) -> bool {
        todo!("implement Log::is_up_to_date")
    }

    /// Checks if the log contains an entry at prev_log_index with prev_log_term.
    /// This is the consistency check for AppendEntries.
    ///
    /// Returns true if:
    /// - prev_log_index is 0 (no previous entry needed), OR
    /// - Entry at prev_log_index has term == prev_log_term
    ///
    /// TODO: Implement this function
    pub fn matches(&self, prev_log_index: LogIndex, prev_log_term: Term) -> bool {
        todo!("implement Log::matches")
    }

    /// Finds the first index where this log conflicts with the given entries.
    /// Returns None if no conflict.
    ///
    /// A conflict occurs when an entry at the same index has a different term.
    ///
    /// TODO: Implement this function
    pub fn find_conflict(&self, entries: &[LogEntry]) -> Option<LogIndex> {
        todo!("implement Log::find_conflict")
    }

    /// Returns all entries (for persistence).
    pub fn all_entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Restores log from entries (for recovery).
    ///
    /// TODO: Implement this function
    pub fn restore(&mut self, entries: Vec<LogEntry>) {
        todo!("implement Log::restore")
    }
}

impl Default for Log {
    fn default() -> Self {
        Log::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(term: Term, index: LogIndex) -> LogEntry {
        LogEntry::new(term, index, Command::Noop)
    }

    #[test]
    fn test_new_log_is_empty() {
        let log = Log::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert_eq!(log.last_index(), 0);
        assert_eq!(log.last_term(), 0);
    }

    #[test]
    fn test_append_and_get() {
        let mut log = Log::new();

        log.append(make_entry(1, 1));
        assert_eq!(log.len(), 1);
        assert_eq!(log.last_index(), 1);
        assert_eq!(log.last_term(), 1);

        let entry = log.get(1).unwrap();
        assert_eq!(entry.term, 1);
        assert_eq!(entry.index, 1);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let log = Log::new();
        assert!(log.get(0).is_none());
        assert!(log.get(1).is_none());
    }

    #[test]
    fn test_term_at() {
        let mut log = Log::new();
        log.append(make_entry(1, 1));
        log.append(make_entry(2, 2));

        assert_eq!(log.term_at(0), 0);
        assert_eq!(log.term_at(1), 1);
        assert_eq!(log.term_at(2), 2);
        assert_eq!(log.term_at(3), 0);
    }

    #[test]
    fn test_truncate() {
        let mut log = Log::new();
        log.append(make_entry(1, 1));
        log.append(make_entry(1, 2));
        log.append(make_entry(2, 3));

        log.truncate_from(2);
        assert_eq!(log.len(), 1);
        assert_eq!(log.last_index(), 1);
    }

    #[test]
    fn test_entries_from() {
        let mut log = Log::new();
        log.append(make_entry(1, 1));
        log.append(make_entry(1, 2));
        log.append(make_entry(2, 3));

        let entries = log.entries_from(2);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].index, 2);
        assert_eq!(entries[1].index, 3);
    }

    #[test]
    fn test_is_up_to_date() {
        let mut log = Log::new();
        log.append(make_entry(1, 1));
        log.append(make_entry(2, 2));

        // Our log: [term=1, term=2], last_index=2, last_term=2

        // Same as us
        assert!(log.is_up_to_date(2, 2));

        // Higher term is more up-to-date (they beat us)
        assert!(!log.is_up_to_date(1, 3));

        // Same term but longer is more up-to-date (they beat us)
        assert!(!log.is_up_to_date(3, 2));

        // Lower term, we're more up-to-date
        assert!(log.is_up_to_date(5, 1));

        // Same term but shorter, we're more up-to-date
        assert!(log.is_up_to_date(1, 2));
    }

    #[test]
    fn test_matches() {
        let mut log = Log::new();
        log.append(make_entry(1, 1));
        log.append(make_entry(2, 2));

        // prev_log_index=0 always matches (no previous entry)
        assert!(log.matches(0, 0));

        // Correct match
        assert!(log.matches(1, 1));
        assert!(log.matches(2, 2));

        // Wrong term
        assert!(!log.matches(1, 2));
        assert!(!log.matches(2, 1));

        // Index doesn't exist
        assert!(!log.matches(3, 2));
    }

    #[test]
    fn test_find_conflict() {
        let mut log = Log::new();
        log.append(make_entry(1, 1));
        log.append(make_entry(1, 2));
        log.append(make_entry(2, 3));

        // No conflict - exact match
        let entries = vec![make_entry(1, 1), make_entry(1, 2)];
        assert!(log.find_conflict(&entries).is_none());

        // Conflict at index 2 - different term
        let entries = vec![make_entry(1, 1), make_entry(2, 2)];
        assert_eq!(log.find_conflict(&entries), Some(2));

        // No conflict - entries beyond our log
        let entries = vec![make_entry(1, 1), make_entry(1, 2), make_entry(2, 3), make_entry(2, 4)];
        assert!(log.find_conflict(&entries).is_none());
    }
}
