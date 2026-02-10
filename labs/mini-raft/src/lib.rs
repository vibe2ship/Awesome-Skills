//! Mini-Raft: A pedagogical implementation of the Raft consensus algorithm.
//!
//! # Overview
//!
//! Raft is a consensus algorithm that allows a cluster of servers to agree
//! on a sequence of commands, even if some servers fail.
//!
//! # Key Components
//!
//! - **Node**: A server in the Raft cluster
//! - **Log**: The replicated log of commands
//! - **State**: Persistent state (term, votedFor, log)
//! - **RPC**: RequestVote and AppendEntries messages

pub mod election;
pub mod log;
pub mod node;
pub mod replication;
pub mod rpc;
pub mod state;
pub mod storage;

use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// Unique identifier for a node in the cluster.
pub type NodeId = u64;

/// Term number (logical clock).
/// Terms are used to detect stale leaders and ensure consistency.
pub type Term = u64;

/// Index into the log (1-indexed, 0 means no entries).
pub type LogIndex = u64;

/// A command to be replicated and applied to the state machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Command {
    /// Set a key to a value
    Set { key: String, value: Vec<u8> },
    /// Delete a key
    Delete { key: String },
    /// No-op command (used for committing entries from previous terms)
    Noop,
}

/// The state of a Raft node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    /// Follower: Passive, responds to RPCs from leader/candidates
    Follower,
    /// Candidate: Actively seeking votes to become leader
    Candidate,
    /// Leader: Handles client requests and replicates log
    Leader,
}

impl Default for NodeState {
    fn default() -> Self {
        NodeState::Follower
    }
}

/// Configuration for a Raft node.
#[derive(Debug, Clone)]
pub struct Config {
    /// This node's ID
    pub id: NodeId,

    /// IDs of all nodes in the cluster (including self)
    pub peers: Vec<NodeId>,

    /// Election timeout range (randomized between min and max)
    pub election_timeout_min: Duration,
    pub election_timeout_max: Duration,

    /// Heartbeat interval (leader sends AppendEntries)
    pub heartbeat_interval: Duration,

    /// Storage path for persistent state
    pub storage_path: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            id: 0,
            peers: vec![],
            election_timeout_min: Duration::from_millis(150),
            election_timeout_max: Duration::from_millis(300),
            heartbeat_interval: Duration::from_millis(50),
            storage_path: None,
        }
    }
}

/// Errors that can occur in Raft operations.
#[derive(Debug, Error)]
pub enum RaftError {
    #[error("not the leader, leader is {leader:?}")]
    NotLeader { leader: Option<NodeId> },

    #[error("log entry not found at index {0}")]
    LogEntryNotFound(LogIndex),

    #[error("term mismatch: expected {expected}, got {actual}")]
    TermMismatch { expected: Term, actual: Term },

    #[error("storage error: {0}")]
    Storage(String),

    #[error("node not in cluster: {0}")]
    UnknownNode(NodeId),

    #[error("cluster is unavailable (no quorum)")]
    NoQuorum,

    #[error("operation timeout")]
    Timeout,

    #[error("node is shutting down")]
    Shutdown,
}

/// Result type for Raft operations.
pub type Result<T> = std::result::Result<T, RaftError>;

/// Returns the quorum size for a cluster of n nodes.
/// Quorum is majority: (n / 2) + 1
///
/// # Examples
///
/// ```
/// use mini_raft::quorum_size;
/// assert_eq!(quorum_size(5), 3);
/// assert_eq!(quorum_size(3), 2);
/// assert_eq!(quorum_size(1), 1);
/// ```
pub fn quorum_size(cluster_size: usize) -> usize {
    cluster_size / 2 + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quorum_size() {
        assert_eq!(quorum_size(1), 1);
        assert_eq!(quorum_size(2), 2);
        assert_eq!(quorum_size(3), 2);
        assert_eq!(quorum_size(4), 3);
        assert_eq!(quorum_size(5), 3);
        assert_eq!(quorum_size(6), 4);
        assert_eq!(quorum_size(7), 4);
    }

    #[test]
    fn test_node_state_default() {
        let state = NodeState::default();
        assert_eq!(state, NodeState::Follower);
    }

    #[test]
    fn test_command_serialization() {
        let cmd = Command::Set {
            key: "foo".to_string(),
            value: b"bar".to_vec(),
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let decoded: Command = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, decoded);
    }
}
