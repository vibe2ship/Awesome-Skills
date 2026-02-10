//! Raft node implementation.
//!
//! A RaftNode is a single server in the Raft cluster. It coordinates:
//! - State transitions (Follower -> Candidate -> Leader)
//! - Election timeouts and heartbeats
//! - Log replication
//! - Client request handling
//!
//! # State Machine
//!
//! ```text
//!                    timeout
//!         ┌─────────────────────────────┐
//!         │                             │
//!         ▼                             │
//!    ┌──────────┐    receives votes    ┌┴─────────┐
//!    │ Follower │───────────────────►  │Candidate │
//!    └──────────┘    from majority     └──────────┘
//!         ▲                                  │
//!         │  discovers leader               │
//!         │  or higher term                 │
//!         │                                 ▼
//!         │                           ┌──────────┐
//!         └───────────────────────────│  Leader  │
//!                                     └──────────┘
//! ```

use crate::election::ElectionState;
use crate::log::Log;
use crate::rpc::{AppendEntriesArgs, AppendEntriesReply, RequestVoteArgs, RequestVoteReply};
use crate::state::{LeaderState, PersistentState, VolatileState};
use crate::{Command, Config, LogIndex, NodeId, NodeState, RaftError, Result, Term};
use std::time::{Duration, Instant};

/// A Raft node.
pub struct RaftNode {
    /// This node's ID
    id: NodeId,

    /// IDs of peer nodes
    peers: Vec<NodeId>,

    /// Current state (Follower, Candidate, Leader)
    state: NodeState,

    /// Persistent state (term, votedFor)
    persistent: PersistentState,

    /// Volatile state (commit_index, last_applied)
    volatile: VolatileState,

    /// The replicated log
    log: Log,

    /// Leader state (only valid when leader)
    leader_state: Option<LeaderState>,

    /// Election state (only valid when candidate)
    election_state: Option<ElectionState>,

    /// Known leader ID (for redirecting clients)
    leader_id: Option<NodeId>,

    /// Last time we heard from the leader
    last_heartbeat: Instant,

    /// Election timeout (randomized)
    election_timeout: Duration,

    /// Configuration
    config: Config,
}

impl RaftNode {
    /// Creates a new Raft node.
    ///
    /// TODO: Implement this function
    /// - Initialize all state
    /// - Set random election timeout
    pub fn new(config: Config) -> Self {
        todo!("implement RaftNode::new")
    }

    /// Returns this node's ID.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Returns the current state.
    pub fn state(&self) -> NodeState {
        self.state
    }

    /// Returns the current term.
    pub fn current_term(&self) -> Term {
        self.persistent.current_term
    }

    /// Returns the known leader ID.
    pub fn leader_id(&self) -> Option<NodeId> {
        self.leader_id
    }

    /// Returns true if this node is the leader.
    pub fn is_leader(&self) -> bool {
        self.state == NodeState::Leader
    }

    /// Returns the commit index.
    pub fn commit_index(&self) -> LogIndex {
        self.volatile.commit_index
    }

    /// Returns the last applied index.
    pub fn last_applied(&self) -> LogIndex {
        self.volatile.last_applied
    }

    /// Generates a random election timeout.
    ///
    /// TODO: Implement this function
    fn random_election_timeout(&self) -> Duration {
        todo!("implement RaftNode::random_election_timeout")
    }

    /// Checks if election timeout has elapsed.
    ///
    /// TODO: Implement this function
    pub fn election_timeout_elapsed(&self) -> bool {
        todo!("implement RaftNode::election_timeout_elapsed")
    }

    /// Resets the election timeout (called when hearing from leader).
    ///
    /// TODO: Implement this function
    pub fn reset_election_timeout(&mut self) {
        todo!("implement RaftNode::reset_election_timeout")
    }

    /// Starts an election (become candidate).
    ///
    /// # Algorithm
    ///
    /// 1. Increment currentTerm
    /// 2. Vote for self
    /// 3. Reset election timer
    /// 4. Send RequestVote RPCs to all other servers
    ///
    /// TODO: Implement this function
    pub fn start_election(&mut self) -> Vec<(NodeId, RequestVoteArgs)> {
        todo!("implement RaftNode::start_election")
    }

    /// Handles a RequestVote RPC.
    ///
    /// TODO: Implement this function
    pub fn handle_request_vote(&mut self, args: RequestVoteArgs) -> RequestVoteReply {
        todo!("implement RaftNode::handle_request_vote")
    }

    /// Processes a RequestVote reply.
    ///
    /// TODO: Implement this function
    /// Returns true if we became leader.
    pub fn process_request_vote_reply(&mut self, from: NodeId, reply: RequestVoteReply) -> bool {
        todo!("implement RaftNode::process_request_vote_reply")
    }

    /// Handles an AppendEntries RPC.
    ///
    /// TODO: Implement this function
    pub fn handle_append_entries(&mut self, args: AppendEntriesArgs) -> AppendEntriesReply {
        todo!("implement RaftNode::handle_append_entries")
    }

    /// Processes an AppendEntries reply.
    ///
    /// TODO: Implement this function
    /// Returns true if commit_index advanced.
    pub fn process_append_entries_reply(
        &mut self,
        from: NodeId,
        reply: AppendEntriesReply,
        entries_sent: usize,
        prev_log_index: LogIndex,
    ) -> bool {
        todo!("implement RaftNode::process_append_entries_reply")
    }

    /// Generates heartbeat messages to send to all followers.
    /// Should be called periodically when leader.
    ///
    /// TODO: Implement this function
    pub fn generate_heartbeats(&self) -> Vec<(NodeId, AppendEntriesArgs)> {
        todo!("implement RaftNode::generate_heartbeats")
    }

    /// Generates AppendEntries messages for replication.
    /// Includes any pending log entries.
    ///
    /// TODO: Implement this function
    pub fn generate_append_entries(&self) -> Vec<(NodeId, AppendEntriesArgs)> {
        todo!("implement RaftNode::generate_append_entries")
    }

    /// Submits a command to be replicated (leader only).
    ///
    /// TODO: Implement this function
    /// Returns the log index where the command will be committed, or error if not leader.
    pub fn submit_command(&mut self, command: Command) -> Result<LogIndex> {
        todo!("implement RaftNode::submit_command")
    }

    /// Becomes the leader.
    ///
    /// TODO: Implement this function
    /// - Initialize leader state (next_index, match_index)
    /// - Append a no-op entry to establish leadership
    fn become_leader(&mut self) {
        todo!("implement RaftNode::become_leader")
    }

    /// Steps down to follower.
    ///
    /// TODO: Implement this function
    /// - Clear leader/election state
    /// - Reset election timeout
    fn become_follower(&mut self, term: Term, leader_id: Option<NodeId>) {
        todo!("implement RaftNode::become_follower")
    }

    /// Updates commit_index based on replicated entries.
    ///
    /// TODO: Implement this function
    fn update_commit_index(&mut self) {
        todo!("implement RaftNode::update_commit_index")
    }

    /// Applies committed entries to the state machine.
    /// Returns the commands that were applied.
    ///
    /// TODO: Implement this function
    pub fn apply_committed(&mut self) -> Vec<(LogIndex, Command)> {
        todo!("implement RaftNode::apply_committed")
    }

    /// Returns entries that need to be persisted.
    /// Should be called before responding to RPCs.
    pub fn entries_to_persist(&self) -> &[crate::log::LogEntry] {
        self.log.all_entries()
    }

    /// Restores state from persistent storage.
    ///
    /// TODO: Implement this function
    pub fn restore(&mut self, persistent: PersistentState, entries: Vec<crate::log::LogEntry>) {
        todo!("implement RaftNode::restore")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(id: NodeId, peers: Vec<NodeId>) -> Config {
        Config {
            id,
            peers,
            election_timeout_min: Duration::from_millis(150),
            election_timeout_max: Duration::from_millis(300),
            heartbeat_interval: Duration::from_millis(50),
            storage_path: None,
        }
    }

    #[test]
    fn test_new_node() {
        let config = make_config(0, vec![0, 1, 2]);
        let node = RaftNode::new(config);

        assert_eq!(node.id(), 0);
        assert_eq!(node.state(), NodeState::Follower);
        assert_eq!(node.current_term(), 0);
        assert_eq!(node.leader_id(), None);
    }

    #[test]
    fn test_start_election() {
        let config = make_config(0, vec![0, 1, 2]);
        let mut node = RaftNode::new(config);

        let requests = node.start_election();

        assert_eq!(node.state(), NodeState::Candidate);
        assert_eq!(node.current_term(), 1);
        assert_eq!(requests.len(), 2); // To peers 1 and 2
    }

    #[test]
    fn test_election_win() {
        let config = make_config(0, vec![0, 1, 2]);
        let mut node = RaftNode::new(config);

        node.start_election();
        assert_eq!(node.current_term(), 1);

        // Self vote is automatic, need one more for majority (2/3)
        let reply = RequestVoteReply::granted(1);
        let became_leader = node.process_request_vote_reply(1, reply);

        assert!(became_leader);
        assert_eq!(node.state(), NodeState::Leader);
    }

    #[test]
    fn test_handle_request_vote_grant() {
        let config = make_config(0, vec![0, 1, 2]);
        let mut node = RaftNode::new(config);

        let args = RequestVoteArgs::new(1, 1, 0, 0);
        let reply = node.handle_request_vote(args);

        assert!(reply.vote_granted);
        assert_eq!(node.current_term(), 1);
    }

    #[test]
    fn test_handle_append_entries_updates_term() {
        let config = make_config(0, vec![0, 1, 2]);
        let mut node = RaftNode::new(config);

        let args = AppendEntriesArgs::heartbeat(5, 1, 0, 0, 0);
        let reply = node.handle_append_entries(args);

        assert!(reply.success);
        assert_eq!(node.current_term(), 5);
        assert_eq!(node.leader_id(), Some(1));
    }

    #[test]
    fn test_step_down_on_higher_term() {
        let config = make_config(0, vec![0, 1, 2]);
        let mut node = RaftNode::new(config);

        // Become leader
        node.start_election();
        node.process_request_vote_reply(1, RequestVoteReply::granted(1));
        assert_eq!(node.state(), NodeState::Leader);

        // Receive AppendEntries with higher term
        let args = AppendEntriesArgs::heartbeat(5, 1, 0, 0, 0);
        node.handle_append_entries(args);

        assert_eq!(node.state(), NodeState::Follower);
        assert_eq!(node.current_term(), 5);
    }

    #[test]
    fn test_submit_command() {
        let config = make_config(0, vec![0, 1, 2]);
        let mut node = RaftNode::new(config);

        // Not leader - should fail
        let result = node.submit_command(Command::Noop);
        assert!(matches!(result, Err(RaftError::NotLeader { .. })));

        // Become leader
        node.start_election();
        node.process_request_vote_reply(1, RequestVoteReply::granted(1));

        // Now should succeed
        let result = node.submit_command(Command::Set {
            key: "foo".to_string(),
            value: b"bar".to_vec(),
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_heartbeats() {
        let config = make_config(0, vec![0, 1, 2]);
        let mut node = RaftNode::new(config);

        // Become leader
        node.start_election();
        node.process_request_vote_reply(1, RequestVoteReply::granted(1));

        let heartbeats = node.generate_heartbeats();
        assert_eq!(heartbeats.len(), 2); // To peers 1 and 2

        for (_, args) in heartbeats {
            assert!(args.is_heartbeat());
            assert_eq!(args.term, 1);
            assert_eq!(args.leader_id, 0);
        }
    }
}
