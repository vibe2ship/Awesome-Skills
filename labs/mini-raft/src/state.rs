//! Persistent state for Raft.
//!
//! Raft requires certain state to be persisted to stable storage before
//! responding to RPCs. This ensures consistency after crashes.
//!
//! # Persistent State (must be saved before responding)
//!
//! - `current_term`: Latest term server has seen
//! - `voted_for`: CandidateId that received vote in current term (or None)
//! - `log[]`: Log entries
//!
//! # Volatile State (can be reconstructed)
//!
//! - `commit_index`: Index of highest log entry known to be committed
//! - `last_applied`: Index of highest log entry applied to state machine
//!
//! # Volatile State on Leaders (reinitialized after election)
//!
//! - `next_index[]`: For each server, index of next log entry to send
//! - `match_index[]`: For each server, index of highest replicated entry

use crate::{LogIndex, NodeId, Term};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Persistent state that must be saved to stable storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentState {
    /// Latest term this server has seen (initialized to 0, increases monotonically)
    pub current_term: Term,

    /// CandidateId that received vote in current term (None if none)
    pub voted_for: Option<NodeId>,
}

impl PersistentState {
    /// Creates new persistent state with initial values.
    ///
    /// TODO: Implement this function
    pub fn new() -> Self {
        todo!("implement PersistentState::new")
    }

    /// Updates the term if the new term is higher.
    /// Also resets voted_for when term changes.
    ///
    /// Returns true if term was updated.
    ///
    /// TODO: Implement this function
    pub fn update_term(&mut self, term: Term) -> bool {
        todo!("implement PersistentState::update_term")
    }

    /// Records a vote for a candidate in the current term.
    /// Returns false if already voted for a different candidate.
    ///
    /// TODO: Implement this function
    pub fn vote_for(&mut self, candidate_id: NodeId) -> bool {
        todo!("implement PersistentState::vote_for")
    }

    /// Checks if we can vote for a candidate.
    /// We can vote if:
    /// - We haven't voted this term, OR
    /// - We already voted for this candidate
    ///
    /// TODO: Implement this function
    pub fn can_vote_for(&self, candidate_id: NodeId) -> bool {
        todo!("implement PersistentState::can_vote_for")
    }
}

impl Default for PersistentState {
    fn default() -> Self {
        PersistentState::new()
    }
}

/// Volatile state that exists on all servers.
#[derive(Debug, Clone)]
pub struct VolatileState {
    /// Index of highest log entry known to be committed (initialized to 0)
    pub commit_index: LogIndex,

    /// Index of highest log entry applied to state machine (initialized to 0)
    pub last_applied: LogIndex,
}

impl VolatileState {
    /// Creates new volatile state with initial values.
    ///
    /// TODO: Implement this function
    pub fn new() -> Self {
        todo!("implement VolatileState::new")
    }

    /// Updates commit_index if the new value is higher.
    /// Returns true if commit_index was updated.
    ///
    /// TODO: Implement this function
    pub fn update_commit_index(&mut self, index: LogIndex) -> bool {
        todo!("implement VolatileState::update_commit_index")
    }

    /// Advances last_applied by one, returning the new value.
    /// Should only be called when last_applied < commit_index.
    ///
    /// TODO: Implement this function
    pub fn advance_applied(&mut self) -> LogIndex {
        todo!("implement VolatileState::advance_applied")
    }

    /// Returns true if there are committed entries that haven't been applied.
    ///
    /// TODO: Implement this function
    pub fn has_unapplied(&self) -> bool {
        todo!("implement VolatileState::has_unapplied")
    }
}

impl Default for VolatileState {
    fn default() -> Self {
        VolatileState::new()
    }
}

/// Volatile state maintained only on leaders.
/// Reinitialized after election.
#[derive(Debug, Clone)]
pub struct LeaderState {
    /// For each server, index of the next log entry to send to that server
    /// (initialized to leader's last log index + 1)
    pub next_index: HashMap<NodeId, LogIndex>,

    /// For each server, index of highest log entry known to be replicated
    /// (initialized to 0)
    pub match_index: HashMap<NodeId, LogIndex>,
}

impl LeaderState {
    /// Creates new leader state, initializing next_index and match_index for all peers.
    ///
    /// # Arguments
    ///
    /// * `peers` - All peer node IDs (excluding self)
    /// * `last_log_index` - The leader's current last log index
    ///
    /// TODO: Implement this function
    pub fn new(peers: &[NodeId], last_log_index: LogIndex) -> Self {
        todo!("implement LeaderState::new")
    }

    /// Gets the next_index for a peer.
    ///
    /// TODO: Implement this function
    pub fn get_next_index(&self, peer: NodeId) -> LogIndex {
        todo!("implement LeaderState::get_next_index")
    }

    /// Gets the match_index for a peer.
    ///
    /// TODO: Implement this function
    pub fn get_match_index(&self, peer: NodeId) -> LogIndex {
        todo!("implement LeaderState::get_match_index")
    }

    /// Decrements next_index for a peer (on AppendEntries failure).
    /// Does not go below 1.
    ///
    /// TODO: Implement this function
    pub fn decrement_next_index(&mut self, peer: NodeId) {
        todo!("implement LeaderState::decrement_next_index")
    }

    /// Updates next_index and match_index for a peer after successful append.
    ///
    /// TODO: Implement this function
    pub fn update_indices(&mut self, peer: NodeId, match_index: LogIndex) {
        todo!("implement LeaderState::update_indices")
    }

    /// Computes the highest index that has been replicated to a majority.
    /// Used to update commit_index.
    ///
    /// # Arguments
    ///
    /// * `my_id` - The leader's node ID
    /// * `my_last_index` - The leader's last log index
    /// * `cluster_size` - Total number of nodes in the cluster
    ///
    /// TODO: Implement this function
    pub fn majority_match_index(
        &self,
        my_id: NodeId,
        my_last_index: LogIndex,
        cluster_size: usize,
    ) -> LogIndex {
        todo!("implement LeaderState::majority_match_index")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persistent_state_new() {
        let state = PersistentState::new();
        assert_eq!(state.current_term, 0);
        assert_eq!(state.voted_for, None);
    }

    #[test]
    fn test_update_term() {
        let mut state = PersistentState::new();

        // Higher term should update
        assert!(state.update_term(5));
        assert_eq!(state.current_term, 5);

        // Same term should not update
        assert!(!state.update_term(5));

        // Lower term should not update
        assert!(!state.update_term(3));
        assert_eq!(state.current_term, 5);
    }

    #[test]
    fn test_vote_for() {
        let mut state = PersistentState::new();
        state.current_term = 1;

        // First vote should succeed
        assert!(state.vote_for(42));
        assert_eq!(state.voted_for, Some(42));

        // Same candidate should succeed
        assert!(state.vote_for(42));

        // Different candidate should fail
        assert!(!state.vote_for(43));
        assert_eq!(state.voted_for, Some(42));
    }

    #[test]
    fn test_term_update_clears_vote() {
        let mut state = PersistentState::new();
        state.vote_for(42);
        assert_eq!(state.voted_for, Some(42));

        state.update_term(1);
        assert_eq!(state.voted_for, None);
    }

    #[test]
    fn test_volatile_state() {
        let mut state = VolatileState::new();
        assert_eq!(state.commit_index, 0);
        assert_eq!(state.last_applied, 0);
        assert!(!state.has_unapplied());

        // Update commit index
        assert!(state.update_commit_index(5));
        assert_eq!(state.commit_index, 5);
        assert!(state.has_unapplied());

        // Advance applied
        let applied = state.advance_applied();
        assert_eq!(applied, 1);
        assert_eq!(state.last_applied, 1);
    }

    #[test]
    fn test_leader_state_initialization() {
        let peers = vec![1, 2, 3];
        let state = LeaderState::new(&peers, 10);

        // next_index should be last_log_index + 1
        assert_eq!(state.get_next_index(1), 11);
        assert_eq!(state.get_next_index(2), 11);
        assert_eq!(state.get_next_index(3), 11);

        // match_index should be 0
        assert_eq!(state.get_match_index(1), 0);
        assert_eq!(state.get_match_index(2), 0);
        assert_eq!(state.get_match_index(3), 0);
    }

    #[test]
    fn test_leader_state_decrement() {
        let peers = vec![1];
        let mut state = LeaderState::new(&peers, 10);

        state.decrement_next_index(1);
        assert_eq!(state.get_next_index(1), 10);

        // Should not go below 1
        for _ in 0..20 {
            state.decrement_next_index(1);
        }
        assert_eq!(state.get_next_index(1), 1);
    }

    #[test]
    fn test_majority_match_index() {
        // Cluster of 5: my_id=0, peers=[1,2,3,4]
        let peers = vec![1, 2, 3, 4];
        let mut state = LeaderState::new(&peers, 10);

        // Leader has index 10
        // Simulate: node 1 at 8, node 2 at 8, node 3 at 5, node 4 at 3
        state.update_indices(1, 8);
        state.update_indices(2, 8);
        state.update_indices(3, 5);
        state.update_indices(4, 3);

        // match_indices: [10 (leader), 8, 8, 5, 3]
        // Sorted: [10, 8, 8, 5, 3]
        // Majority (3/5) have at least index 8
        let majority = state.majority_match_index(0, 10, 5);
        assert_eq!(majority, 8);
    }
}
