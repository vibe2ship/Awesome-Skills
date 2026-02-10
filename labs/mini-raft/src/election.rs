//! Leader election implementation for Raft.
//!
//! # Election Process
//!
//! 1. Follower times out waiting for heartbeat
//! 2. Follower becomes candidate, increments term, votes for self
//! 3. Candidate sends RequestVote to all peers
//! 4. Nodes grant vote if:
//!    - Candidate's term >= currentTerm
//!    - Haven't voted this term (or voted for this candidate)
//!    - Candidate's log is at least as up-to-date
//! 5. Candidate becomes leader if receives majority votes
//!
//! # Election Safety
//!
//! - Each node votes for at most one candidate per term
//! - Only candidate with up-to-date log can win
//! - At most one leader per term

use crate::log::Log;
use crate::rpc::{RequestVoteArgs, RequestVoteReply};
use crate::state::PersistentState;
use crate::{quorum_size, LogIndex, NodeId, NodeState, Term};
use std::collections::HashSet;

/// Tracks the state of an ongoing election.
pub struct ElectionState {
    /// The term this election is for
    pub term: Term,

    /// Nodes that granted their vote
    votes_received: HashSet<NodeId>,

    /// Total number of nodes in the cluster
    cluster_size: usize,
}

impl ElectionState {
    /// Creates a new election state.
    /// The candidate automatically votes for itself.
    ///
    /// TODO: Implement this function
    pub fn new(term: Term, self_id: NodeId, cluster_size: usize) -> Self {
        todo!("implement ElectionState::new")
    }

    /// Records a vote from a node.
    ///
    /// TODO: Implement this function
    pub fn record_vote(&mut self, node_id: NodeId) {
        todo!("implement ElectionState::record_vote")
    }

    /// Returns true if we have received enough votes to become leader.
    ///
    /// TODO: Implement this function
    pub fn has_won(&self) -> bool {
        todo!("implement ElectionState::has_won")
    }

    /// Returns the number of votes received.
    pub fn vote_count(&self) -> usize {
        self.votes_received.len()
    }

    /// Returns the number of votes needed to win.
    pub fn votes_needed(&self) -> usize {
        quorum_size(self.cluster_size)
    }
}

/// Handles RequestVote RPC as a receiver (not a candidate).
///
/// # Arguments
///
/// * `args` - The RequestVote arguments from the candidate
/// * `state` - Our persistent state (term, votedFor)
/// * `log` - Our log (to check up-to-date)
///
/// # Returns
///
/// * The reply to send back
/// * Whether our term was updated (should step down if leader)
///
/// # Algorithm (from Raft paper §5.2, §5.4)
///
/// 1. Reply false if term < currentTerm
/// 2. If votedFor is null or candidateId, and candidate's log is at
///    least as up-to-date as receiver's log, grant vote
///
/// TODO: Implement this function
pub fn handle_request_vote(
    args: &RequestVoteArgs,
    state: &mut PersistentState,
    log: &Log,
) -> (RequestVoteReply, bool) {
    todo!("implement handle_request_vote")
}

/// Determines if the candidate's log is at least as up-to-date as ours.
///
/// A log is more up-to-date if:
/// - Its last entry has a higher term, OR
/// - Same last term but equal or longer length
///
/// TODO: Implement this function
pub fn is_candidate_log_up_to_date(
    candidate_last_index: LogIndex,
    candidate_last_term: Term,
    our_last_index: LogIndex,
    our_last_term: Term,
) -> bool {
    todo!("implement is_candidate_log_up_to_date")
}

/// Generates the RequestVote arguments for starting an election.
///
/// TODO: Implement this function
pub fn create_request_vote_args(
    term: Term,
    candidate_id: NodeId,
    log: &Log,
) -> RequestVoteArgs {
    todo!("implement create_request_vote_args")
}

/// Determines the state transition based on a RequestVote reply.
///
/// # Arguments
///
/// * `reply` - The RequestVote reply from a peer
/// * `current_term` - Our current term
/// * `election` - Our election state (if we're a candidate)
///
/// # Returns
///
/// The new node state:
/// - Leader if we won the election
/// - Follower if we discovered a higher term
/// - Candidate if we're still waiting for votes
///
/// TODO: Implement this function
pub fn process_vote_reply(
    reply: &RequestVoteReply,
    current_term: Term,
    election: &mut ElectionState,
) -> NodeState {
    todo!("implement process_vote_reply")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::LogEntry;
    use crate::Command;

    #[test]
    fn test_election_state_new() {
        let state = ElectionState::new(5, 0, 5);
        assert_eq!(state.term, 5);
        assert_eq!(state.vote_count(), 1); // Self vote
        assert!(!state.has_won()); // Need 3/5
    }

    #[test]
    fn test_election_win() {
        let mut state = ElectionState::new(5, 0, 5);

        // Self vote (already counted)
        assert_eq!(state.vote_count(), 1);

        // Two more votes needed for majority (3/5)
        state.record_vote(1);
        assert_eq!(state.vote_count(), 2);
        assert!(!state.has_won());

        state.record_vote(2);
        assert_eq!(state.vote_count(), 3);
        assert!(state.has_won());
    }

    #[test]
    fn test_duplicate_votes() {
        let mut state = ElectionState::new(5, 0, 5);
        state.record_vote(1);
        state.record_vote(1); // Duplicate
        assert_eq!(state.vote_count(), 2); // Still 2 (self + node 1)
    }

    #[test]
    fn test_handle_request_vote_stale_term() {
        let args = RequestVoteArgs::new(3, 1, 10, 2);
        let mut state = PersistentState::new();
        state.current_term = 5;
        let log = Log::new();

        let (reply, term_updated) = handle_request_vote(&args, &mut state, &log);

        assert!(!reply.vote_granted);
        assert_eq!(reply.term, 5);
        assert!(!term_updated);
    }

    #[test]
    fn test_handle_request_vote_grant() {
        let args = RequestVoteArgs::new(5, 1, 10, 4);
        let mut state = PersistentState::new();
        state.current_term = 4;
        let mut log = Log::new();
        // Our log is less up-to-date
        log.append(LogEntry::new(3, 1, Command::Noop));

        let (reply, term_updated) = handle_request_vote(&args, &mut state, &log);

        assert!(reply.vote_granted);
        assert_eq!(reply.term, 5);
        assert!(term_updated);
        assert_eq!(state.voted_for, Some(1));
    }

    #[test]
    fn test_handle_request_vote_already_voted() {
        let args = RequestVoteArgs::new(5, 2, 10, 4);
        let mut state = PersistentState::new();
        state.current_term = 5;
        state.voted_for = Some(1); // Already voted for node 1
        let log = Log::new();

        let (reply, _) = handle_request_vote(&args, &mut state, &log);

        assert!(!reply.vote_granted); // Can't vote for node 2
    }

    #[test]
    fn test_handle_request_vote_same_candidate() {
        let args = RequestVoteArgs::new(5, 1, 10, 4);
        let mut state = PersistentState::new();
        state.current_term = 5;
        state.voted_for = Some(1); // Already voted for this candidate
        let log = Log::new();

        let (reply, _) = handle_request_vote(&args, &mut state, &log);

        assert!(reply.vote_granted); // Can vote for same candidate again
    }

    #[test]
    fn test_is_candidate_log_up_to_date() {
        // Candidate has higher term
        assert!(is_candidate_log_up_to_date(5, 3, 10, 2));

        // Same term, candidate has longer log
        assert!(is_candidate_log_up_to_date(10, 3, 5, 3));

        // Same term, same length
        assert!(is_candidate_log_up_to_date(5, 3, 5, 3));

        // Same term, candidate has shorter log
        assert!(!is_candidate_log_up_to_date(5, 3, 10, 3));

        // Candidate has lower term
        assert!(!is_candidate_log_up_to_date(10, 2, 5, 3));
    }

    #[test]
    fn test_process_vote_reply_win() {
        let mut election = ElectionState::new(5, 0, 3);
        let reply = RequestVoteReply::granted(5);

        let new_state = process_vote_reply(&reply, 5, &mut election);

        // 2/3 votes (self + this vote) = majority
        assert_eq!(new_state, NodeState::Leader);
    }

    #[test]
    fn test_process_vote_reply_higher_term() {
        let mut election = ElectionState::new(5, 0, 5);
        let reply = RequestVoteReply::denied(6); // Higher term

        let new_state = process_vote_reply(&reply, 5, &mut election);

        assert_eq!(new_state, NodeState::Follower);
    }
}
