//! Log replication implementation for Raft.
//!
//! # Replication Process
//!
//! 1. Leader receives client request
//! 2. Leader appends entry to local log
//! 3. Leader sends AppendEntries to all followers
//! 4. Followers append entries and respond
//! 5. Leader commits entry when majority have replicated
//! 6. Leader notifies followers of commit via next AppendEntries
//!
//! # Log Consistency
//!
//! The leader maintains two indices per follower:
//! - `next_index`: Next entry to send (optimistic, may be wrong)
//! - `match_index`: Highest entry known to be replicated (conservative)
//!
//! On AppendEntries failure, leader decrements next_index and retries.

use crate::log::{Log, LogEntry};
use crate::rpc::{AppendEntriesArgs, AppendEntriesReply};
use crate::state::{LeaderState, PersistentState, VolatileState};
use crate::{LogIndex, NodeId, Term};

/// Result of processing AppendEntries.
pub struct AppendResult {
    /// Whether the append was successful
    pub success: bool,

    /// Our term (for leader to update itself)
    pub term: Term,

    /// Our last log index after append
    pub last_log_index: LogIndex,

    /// Whether our term was updated
    pub term_updated: bool,

    /// New commit index (if updated)
    pub new_commit_index: Option<LogIndex>,
}

/// Handles AppendEntries RPC as a follower.
///
/// # Arguments
///
/// * `args` - The AppendEntries arguments from the leader
/// * `persistent` - Our persistent state
/// * `volatile` - Our volatile state
/// * `log` - Our log
///
/// # Algorithm (from Raft paper §5.3)
///
/// 1. Reply false if term < currentTerm
/// 2. Reply false if log doesn't contain entry at prevLogIndex matching prevLogTerm
/// 3. If existing entry conflicts with new one (same index, different term),
///    delete existing entry and all following
/// 4. Append any new entries not already in the log
/// 5. If leaderCommit > commitIndex, set commitIndex = min(leaderCommit, index of last new entry)
///
/// TODO: Implement this function
pub fn handle_append_entries(
    args: &AppendEntriesArgs,
    persistent: &mut PersistentState,
    volatile: &mut VolatileState,
    log: &mut Log,
) -> AppendResult {
    todo!("implement handle_append_entries")
}

/// Creates AppendEntries arguments for a specific follower.
///
/// # Arguments
///
/// * `term` - Current term
/// * `leader_id` - This leader's ID
/// * `follower_id` - The follower to send to
/// * `leader_state` - Leader's state (next_index, match_index)
/// * `volatile` - Volatile state (commit_index)
/// * `log` - The log
///
/// TODO: Implement this function
pub fn create_append_entries(
    term: Term,
    leader_id: NodeId,
    follower_id: NodeId,
    leader_state: &LeaderState,
    volatile: &VolatileState,
    log: &Log,
) -> AppendEntriesArgs {
    todo!("implement create_append_entries")
}

/// Processes an AppendEntries reply from a follower.
///
/// # Arguments
///
/// * `reply` - The reply from the follower
/// * `follower_id` - Which follower sent the reply
/// * `entries_sent` - Number of entries that were sent
/// * `leader_state` - Mutable leader state to update
///
/// # Returns
///
/// True if the follower's log progressed, false if we need to retry with lower index.
///
/// TODO: Implement this function
pub fn process_append_reply(
    reply: &AppendEntriesReply,
    follower_id: NodeId,
    entries_sent: usize,
    prev_log_index: LogIndex,
    leader_state: &mut LeaderState,
) -> bool {
    todo!("implement process_append_reply")
}

/// Computes the new commit index based on replicated entries.
///
/// The leader can only commit entries from its current term.
/// Once an entry from the current term is committed, all prior entries
/// are also committed (Log Matching Property).
///
/// # Arguments
///
/// * `leader_id` - This leader's ID
/// * `current_term` - Current term
/// * `leader_state` - Leader's state
/// * `volatile` - Current volatile state
/// * `log` - The log
/// * `cluster_size` - Number of nodes in cluster
///
/// # Returns
///
/// The new commit index (may be same as current if nothing new to commit).
///
/// TODO: Implement this function
pub fn compute_commit_index(
    leader_id: NodeId,
    current_term: Term,
    leader_state: &LeaderState,
    volatile: &VolatileState,
    log: &Log,
    cluster_size: usize,
) -> LogIndex {
    todo!("implement compute_commit_index")
}

/// Finds entries to send to a follower.
///
/// # Arguments
///
/// * `next_index` - The next index to send
/// * `log` - The log
/// * `max_entries` - Maximum number of entries to include
///
/// # Returns
///
/// A tuple of (prev_log_index, prev_log_term, entries).
///
/// TODO: Implement this function
pub fn get_entries_for_follower(
    next_index: LogIndex,
    log: &Log,
    max_entries: usize,
) -> (LogIndex, Term, Vec<LogEntry>) {
    todo!("implement get_entries_for_follower")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Command;

    fn make_entry(term: Term, index: LogIndex) -> LogEntry {
        LogEntry::new(term, index, Command::Noop)
    }

    #[test]
    fn test_handle_append_entries_stale_term() {
        let args = AppendEntriesArgs::new(3, 1, 0, 0, vec![], 0);
        let mut persistent = PersistentState::new();
        persistent.current_term = 5;
        let mut volatile = VolatileState::new();
        let mut log = Log::new();

        let result = handle_append_entries(&args, &mut persistent, &mut volatile, &mut log);

        assert!(!result.success);
        assert_eq!(result.term, 5);
        assert!(!result.term_updated);
    }

    #[test]
    fn test_handle_append_entries_heartbeat() {
        let args = AppendEntriesArgs::heartbeat(5, 1, 0, 0, 0);
        let mut persistent = PersistentState::new();
        persistent.current_term = 5;
        let mut volatile = VolatileState::new();
        let mut log = Log::new();

        let result = handle_append_entries(&args, &mut persistent, &mut volatile, &mut log);

        assert!(result.success);
        assert_eq!(result.term, 5);
    }

    #[test]
    fn test_handle_append_entries_missing_prev() {
        // Leader says prev is at index 5, but we have nothing
        let args = AppendEntriesArgs::new(5, 1, 5, 4, vec![make_entry(5, 6)], 0);
        let mut persistent = PersistentState::new();
        persistent.current_term = 5;
        let mut volatile = VolatileState::new();
        let mut log = Log::new();

        let result = handle_append_entries(&args, &mut persistent, &mut volatile, &mut log);

        assert!(!result.success);
    }

    #[test]
    fn test_handle_append_entries_success() {
        let entries = vec![make_entry(5, 1), make_entry(5, 2)];
        let args = AppendEntriesArgs::new(5, 1, 0, 0, entries, 1);
        let mut persistent = PersistentState::new();
        persistent.current_term = 4;
        let mut volatile = VolatileState::new();
        let mut log = Log::new();

        let result = handle_append_entries(&args, &mut persistent, &mut volatile, &mut log);

        assert!(result.success);
        assert!(result.term_updated);
        assert_eq!(log.len(), 2);
        assert_eq!(volatile.commit_index, 1);
    }

    #[test]
    fn test_handle_append_entries_conflict() {
        // We have [term=1@1, term=1@2]
        // Leader sends [term=2@2] with prev=(1, term=1)
        // Should truncate our index 2 and replace
        let mut log = Log::new();
        log.append(make_entry(1, 1));
        log.append(make_entry(1, 2)); // This conflicts

        let args = AppendEntriesArgs::new(5, 1, 1, 1, vec![make_entry(2, 2)], 0);
        let mut persistent = PersistentState::new();
        persistent.current_term = 5;
        let mut volatile = VolatileState::new();

        let result = handle_append_entries(&args, &mut persistent, &mut volatile, &mut log);

        assert!(result.success);
        assert_eq!(log.len(), 2);
        assert_eq!(log.term_at(2), 2); // Replaced with new entry
    }

    #[test]
    fn test_create_append_entries() {
        let mut log = Log::new();
        log.append(make_entry(1, 1));
        log.append(make_entry(2, 2));
        log.append(make_entry(2, 3));

        let leader_state = LeaderState::new(&[1], 3);
        let volatile = VolatileState::new();

        // For a new follower, next_index is last_log_index + 1 = 4
        // But they need entries starting from 1
        // After a failure, next_index would be decremented

        let args = create_append_entries(2, 0, 1, &leader_state, &volatile, &log);

        assert_eq!(args.term, 2);
        assert_eq!(args.leader_id, 0);
    }

    #[test]
    fn test_process_append_reply_success() {
        let mut leader_state = LeaderState::new(&[1], 5);
        let reply = AppendEntriesReply::success(5, 8);

        let progressed = process_append_reply(&reply, 1, 3, 5, &mut leader_state);

        assert!(progressed);
        assert_eq!(leader_state.get_match_index(1), 8);
        assert_eq!(leader_state.get_next_index(1), 9);
    }

    #[test]
    fn test_process_append_reply_failure() {
        let mut leader_state = LeaderState::new(&[1], 10);
        // Initially next_index = 11
        let reply = AppendEntriesReply::failure(5, Some(5));

        let progressed = process_append_reply(&reply, 1, 0, 10, &mut leader_state);

        assert!(!progressed);
        // next_index should be decremented
        assert!(leader_state.get_next_index(1) < 11);
    }

    #[test]
    fn test_compute_commit_index() {
        // 5-node cluster, we're leader (id=0)
        let mut leader_state = LeaderState::new(&[1, 2, 3, 4], 10);

        // Simulate replication progress
        leader_state.update_indices(1, 8);
        leader_state.update_indices(2, 8);
        leader_state.update_indices(3, 5);
        leader_state.update_indices(4, 3);

        let mut log = Log::new();
        for i in 1..=10 {
            // Entries 1-7 are term 1, entries 8-10 are term 2
            let term = if i <= 7 { 1 } else { 2 };
            log.append(make_entry(term, i));
        }

        let volatile = VolatileState::new();

        // Current term is 2, so we can only commit up to where majority has term=2 entries
        // match_indices: [10 (us), 8, 8, 5, 3]
        // Majority (3) have at least 8
        // Entry at 8 has term=2 (current term), so we can commit
        let new_commit = compute_commit_index(0, 2, &leader_state, &volatile, &log, 5);

        assert_eq!(new_commit, 8);
    }

    #[test]
    fn test_get_entries_for_follower() {
        let mut log = Log::new();
        log.append(make_entry(1, 1));
        log.append(make_entry(1, 2));
        log.append(make_entry(2, 3));
        log.append(make_entry(2, 4));

        // Follower needs from index 2
        let (prev_index, prev_term, entries) = get_entries_for_follower(2, &log, 10);

        assert_eq!(prev_index, 1);
        assert_eq!(prev_term, 1);
        assert_eq!(entries.len(), 3); // Entries 2, 3, 4
        assert_eq!(entries[0].index, 2);
    }
}
