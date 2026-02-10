# Mini-Raft: A Pedagogical Raft Consensus Implementation

Mini-Raft is an educational implementation of the [Raft consensus algorithm](https://raft.github.io/), designed to teach distributed consensus concepts.

## Learning Objectives

By implementing this project, you will learn:

- **Consensus**: How distributed nodes agree on a single value
- **Leader Election**: Timeout-based leader election with term numbers
- **Log Replication**: Replicated state machines with consistency
- **Safety**: Why Raft is safe under various failure scenarios
- **Persistence**: WAL and state recovery

## The Raft Algorithm

### Overview

Raft is a consensus algorithm that ensures a cluster of servers agrees on a sequence of commands, even if some servers fail.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Raft Cluster (5 nodes)                      │
│                                                                 │
│    ┌──────────┐      ┌──────────┐      ┌──────────┐            │
│    │ Follower │      │  Leader  │      │ Follower │            │
│    │  Node 1  │◄────►│  Node 2  │◄────►│  Node 3  │            │
│    └──────────┘      └──────────┘      └──────────┘            │
│         ▲                 ▲                 ▲                   │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│    ┌──────────┐      ┌──────────┐                              │
│    │ Follower │◄────►│ Follower │                              │
│    │  Node 4  │      │  Node 5  │                              │
│    └──────────┘      └──────────┘                              │
│                                                                 │
│  All nodes maintain:                                            │
│  - Log: [cmd1, cmd2, cmd3, ...]                                │
│  - Current term                                                 │
│  - Voted for                                                    │
│  - Commit index                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Node States

```
                    timeout/
                    starts election
         ┌─────────────────────────────┐
         │                             │
         ▼                             │
    ┌──────────┐    receives votes    ┌┴─────────┐
    │ Follower │───────────────────►  │Candidate │
    └──────────┘    from majority     └──────────┘
         ▲                                  │
         │  discovers leader               │
         │  or higher term                 │
         │                                 ▼
         │                           ┌──────────┐
         └───────────────────────────│  Leader  │
                                     └──────────┘
```

### Key Concepts

#### 1. Terms

Terms act as logical clocks. Each term has at most one leader.

```
Term 1          Term 2          Term 3
│ Leader A │    │ Leader B │    │ Leader C │
│  ─────►  │    │  ─────►  │    │  ─────►  │
│ election │    │ election │    │ election │
```

#### 2. Log Replication

The leader replicates its log to all followers:

```
Leader Log:    [1:x←3] [1:y←1] [2:x←2] [3:z←5]
                  │       │       │       │
                  ▼       ▼       ▼       ▼
Follower 1:    [1:x←3] [1:y←1] [2:x←2] [3:z←5]  ✓ Up to date
Follower 2:    [1:x←3] [1:y←1] [2:x←2]          ✓ Catching up
Follower 3:    [1:x←3] [1:y←1]                  ✓ Catching up
```

#### 3. Commit Rules

An entry is committed when:
1. Stored on a majority of servers
2. At least one entry from leader's current term is committed

```
Entry committed when majority (3/5) have it:
Node 1: [A][B][C]  ←── C is committed
Node 2: [A][B][C]
Node 3: [A][B][C]
Node 4: [A][B]
Node 5: [A]
```

## Project Structure

```
mini-raft/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs           # Library root
│   ├── node.rs          # Raft node implementation
│   ├── log.rs           # Replicated log
│   ├── state.rs         # Persistent state
│   ├── rpc.rs           # RPC messages
│   ├── election.rs      # Leader election
│   ├── replication.rs   # Log replication
│   └── storage.rs       # Persistent storage
└── tests/
    ├── election_test.rs
    ├── replication_test.rs
    └── integration_test.rs
```

## Core Types

### Node State

```rust
pub enum NodeState {
    Follower,
    Candidate,
    Leader,
}
```

### Log Entry

```rust
pub struct LogEntry {
    pub term: u64,
    pub index: u64,
    pub command: Command,
}
```

### RPC Messages

```rust
// RequestVote RPC
pub struct RequestVoteArgs {
    pub term: u64,
    pub candidate_id: NodeId,
    pub last_log_index: u64,
    pub last_log_term: u64,
}

// AppendEntries RPC
pub struct AppendEntriesArgs {
    pub term: u64,
    pub leader_id: NodeId,
    pub prev_log_index: u64,
    pub prev_log_term: u64,
    pub entries: Vec<LogEntry>,
    pub leader_commit: u64,
}
```

## Implementation TODO List

### Milestone 1: Core Types
- [ ] Implement `src/lib.rs` - Basic types (NodeId, Term, LogIndex)
- [ ] Implement `src/log.rs` - Log entry and log operations
- [ ] Implement `src/state.rs` - Persistent state
- [ ] Run: `cargo test`

### Milestone 2: RPC Messages
- [ ] Implement `src/rpc.rs` - RequestVote and AppendEntries
- [ ] Implement request/response handling
- [ ] Run: `cargo test rpc`

### Milestone 3: Leader Election
- [ ] Implement `src/election.rs` - Election timeout and voting
- [ ] Implement term management
- [ ] Implement vote counting
- [ ] Run: `cargo test election`

### Milestone 4: Log Replication
- [ ] Implement `src/replication.rs` - AppendEntries logic
- [ ] Implement log consistency check
- [ ] Implement commit index advancement
- [ ] Run: `cargo test replication`

### Milestone 5: Raft Node
- [ ] Implement `src/node.rs` - Full Raft node
- [ ] Integrate election and replication
- [ ] Handle all state transitions
- [ ] Run: `cargo test node`

### Milestone 6: Integration
- [ ] Implement cluster simulation
- [ ] Test with network partitions
- [ ] Test leader failure scenarios
- [ ] Run: `cargo test --test integration`

## Safety Properties

Raft guarantees these safety properties:

1. **Election Safety**: At most one leader per term
2. **Leader Append-Only**: Leader never overwrites/deletes entries
3. **Log Matching**: If two logs have same index/term, all prior entries match
4. **Leader Completeness**: Committed entries will be in future leaders' logs
5. **State Machine Safety**: All nodes apply same command at same index

## Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test election

# Run with logging
RUST_LOG=debug cargo test

# Run integration tests
cargo test --test integration
```

## References

- [Raft Paper](https://raft.github.io/raft.pdf)
- [Raft Visualization](https://raft.github.io/)
- [Students' Guide to Raft](https://thesquareplanet.com/blog/students-guide-to-raft/)
