//! RPC messages for Raft.
//!
//! Raft uses two RPC types:
//!
//! 1. **RequestVote**: Invoked by candidates to gather votes
//! 2. **AppendEntries**: Invoked by leader to replicate log entries and heartbeats
//!
//! # RPC Semantics
//!
//! - If RPC request or response contains term T > currentTerm:
//!   - Set currentTerm = T
//!   - Convert to follower

use crate::log::LogEntry;
use crate::{LogIndex, NodeId, Term};
use serde::{Deserialize, Serialize};

/// RequestVote RPC arguments.
///
/// Invoked by candidates to gather votes (§5.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteArgs {
    /// Candidate's term
    pub term: Term,

    /// Candidate requesting vote
    pub candidate_id: NodeId,

    /// Index of candidate's last log entry
    pub last_log_index: LogIndex,

    /// Term of candidate's last log entry
    pub last_log_term: Term,
}

impl RequestVoteArgs {
    /// Creates a new RequestVote request.
    ///
    /// TODO: Implement this function
    pub fn new(
        term: Term,
        candidate_id: NodeId,
        last_log_index: LogIndex,
        last_log_term: Term,
    ) -> Self {
        todo!("implement RequestVoteArgs::new")
    }
}

/// RequestVote RPC response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteReply {
    /// currentTerm, for candidate to update itself
    pub term: Term,

    /// true means candidate received vote
    pub vote_granted: bool,
}

impl RequestVoteReply {
    /// Creates a vote-granted reply.
    ///
    /// TODO: Implement this function
    pub fn granted(term: Term) -> Self {
        todo!("implement RequestVoteReply::granted")
    }

    /// Creates a vote-denied reply.
    ///
    /// TODO: Implement this function
    pub fn denied(term: Term) -> Self {
        todo!("implement RequestVoteReply::denied")
    }
}

/// AppendEntries RPC arguments.
///
/// Invoked by leader to replicate log entries (§5.3); also used as heartbeat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesArgs {
    /// Leader's term
    pub term: Term,

    /// So follower can redirect clients
    pub leader_id: NodeId,

    /// Index of log entry immediately preceding new ones
    pub prev_log_index: LogIndex,

    /// Term of prev_log_index entry
    pub prev_log_term: Term,

    /// Log entries to store (empty for heartbeat)
    pub entries: Vec<LogEntry>,

    /// Leader's commit_index
    pub leader_commit: LogIndex,
}

impl AppendEntriesArgs {
    /// Creates a new AppendEntries request.
    ///
    /// TODO: Implement this function
    pub fn new(
        term: Term,
        leader_id: NodeId,
        prev_log_index: LogIndex,
        prev_log_term: Term,
        entries: Vec<LogEntry>,
        leader_commit: LogIndex,
    ) -> Self {
        todo!("implement AppendEntriesArgs::new")
    }

    /// Creates a heartbeat (empty AppendEntries).
    ///
    /// TODO: Implement this function
    pub fn heartbeat(
        term: Term,
        leader_id: NodeId,
        prev_log_index: LogIndex,
        prev_log_term: Term,
        leader_commit: LogIndex,
    ) -> Self {
        todo!("implement AppendEntriesArgs::heartbeat")
    }

    /// Returns true if this is a heartbeat (no entries).
    pub fn is_heartbeat(&self) -> bool {
        self.entries.is_empty()
    }
}

/// AppendEntries RPC response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesReply {
    /// currentTerm, for leader to update itself
    pub term: Term,

    /// true if follower contained entry matching prev_log_index and prev_log_term
    pub success: bool,

    /// For optimization: the follower's last log index
    /// Helps leader quickly find the correct next_index
    pub last_log_index: Option<LogIndex>,
}

impl AppendEntriesReply {
    /// Creates a success reply.
    ///
    /// TODO: Implement this function
    pub fn success(term: Term, last_log_index: LogIndex) -> Self {
        todo!("implement AppendEntriesReply::success")
    }

    /// Creates a failure reply.
    ///
    /// TODO: Implement this function
    pub fn failure(term: Term, last_log_index: Option<LogIndex>) -> Self {
        todo!("implement AppendEntriesReply::failure")
    }
}

/// Enum for all RPC messages (for network transport).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RpcMessage {
    RequestVote(RequestVoteArgs),
    RequestVoteReply(RequestVoteReply),
    AppendEntries(AppendEntriesArgs),
    AppendEntriesReply(AppendEntriesReply),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Command;

    #[test]
    fn test_request_vote_args() {
        let args = RequestVoteArgs::new(5, 1, 10, 4);
        assert_eq!(args.term, 5);
        assert_eq!(args.candidate_id, 1);
        assert_eq!(args.last_log_index, 10);
        assert_eq!(args.last_log_term, 4);
    }

    #[test]
    fn test_request_vote_reply() {
        let granted = RequestVoteReply::granted(5);
        assert_eq!(granted.term, 5);
        assert!(granted.vote_granted);

        let denied = RequestVoteReply::denied(5);
        assert_eq!(denied.term, 5);
        assert!(!denied.vote_granted);
    }

    #[test]
    fn test_append_entries_args() {
        let entries = vec![LogEntry::new(5, 11, Command::Noop)];
        let args = AppendEntriesArgs::new(5, 1, 10, 4, entries.clone(), 9);

        assert_eq!(args.term, 5);
        assert_eq!(args.leader_id, 1);
        assert_eq!(args.prev_log_index, 10);
        assert_eq!(args.prev_log_term, 4);
        assert_eq!(args.entries.len(), 1);
        assert_eq!(args.leader_commit, 9);
        assert!(!args.is_heartbeat());
    }

    #[test]
    fn test_heartbeat() {
        let args = AppendEntriesArgs::heartbeat(5, 1, 10, 4, 9);
        assert!(args.is_heartbeat());
        assert!(args.entries.is_empty());
    }

    #[test]
    fn test_append_entries_reply() {
        let success = AppendEntriesReply::success(5, 15);
        assert_eq!(success.term, 5);
        assert!(success.success);
        assert_eq!(success.last_log_index, Some(15));

        let failure = AppendEntriesReply::failure(5, Some(10));
        assert_eq!(failure.term, 5);
        assert!(!failure.success);
        assert_eq!(failure.last_log_index, Some(10));
    }

    #[test]
    fn test_serialization() {
        let args = RequestVoteArgs::new(5, 1, 10, 4);
        let json = serde_json::to_string(&args).unwrap();
        let decoded: RequestVoteArgs = serde_json::from_str(&json).unwrap();
        assert_eq!(args.term, decoded.term);
        assert_eq!(args.candidate_id, decoded.candidate_id);
    }
}
