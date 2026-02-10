# Mini-Ray: A Pedagogical Distributed Computing Framework

Mini-Ray is an educational implementation of a distributed computing framework inspired by [Ray](https://ray.io/). It demonstrates core concepts of distributed task execution and actor-based programming.

## Learning Objectives

By implementing this project, you will learn:

- **Distributed Systems Fundamentals**: RPC, serialization, fault tolerance
- **Actor Model**: Stateful distributed objects
- **Task Scheduling**: DAG execution, dependency resolution
- **Object Store**: Distributed shared memory
- **Resource Management**: Worker lifecycle, heartbeat mechanism

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           Driver                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Task Submit │  │Actor Create │  │  ray.Get()  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Scheduler                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Task Queue    │  Worker Registry  │  Object Directory  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
     ┌───────────┐     ┌───────────┐     ┌───────────┐
     │  Worker 1 │     │  Worker 2 │     │  Worker 3 │
     │ ┌───────┐ │     │ ┌───────┐ │     │ ┌───────┐ │
     │ │ Tasks │ │     │ │ Tasks │ │     │ │ Tasks │ │
     │ ├───────┤ │     │ ├───────┤ │     │ ├───────┤ │
     │ │Actors │ │     │ │Actors │ │     │ │Actors │ │
     │ ├───────┤ │     │ ├───────┤ │     │ ├───────┤ │
     │ │ Store │ │     │ │ Store │ │     │ │ Store │ │
     │ └───────┘ │     │ └───────┘ │     │ └───────┘ │
     └───────────┘     └───────────┘     └───────────┘
```

## Project Structure

```
mini-ray/
├── cmd/
│   ├── driver/           # Driver process entry
│   └── worker/           # Worker process entry
├── pkg/
│   ├── core/             # Core types and interfaces
│   │   ├── types.go      # ObjectID, TaskID, ActorID
│   │   ├── object.go     # ObjectRef, Future
│   │   └── errors.go     # Error types
│   ├── task/             # Task execution
│   │   ├── task.go       # Task definition
│   │   ├── executor.go   # Task executor
│   │   └── dag.go        # DAG dependency tracking
│   ├── actor/            # Actor system
│   │   ├── actor.go      # Actor definition
│   │   ├── handle.go     # Actor handle (proxy)
│   │   └── registry.go   # Actor registry
│   ├── store/            # Object store
│   │   ├── store.go      # Object store interface
│   │   ├── memory.go     # In-memory implementation
│   │   └── plasma.go     # Shared memory (optional)
│   ├── scheduler/        # Task scheduling
│   │   ├── scheduler.go  # Scheduler interface
│   │   ├── simple.go     # Simple FIFO scheduler
│   │   └── locality.go   # Locality-aware scheduler
│   └── rpc/              # RPC layer
│       ├── server.go     # gRPC server
│       └── client.go     # gRPC client
├── internal/
│   ├── protocol/         # Wire protocol (protobuf)
│   └── serialization/    # Object serialization
├── examples/
│   ├── wordcount/        # Distributed word count
│   ├── pi/               # Monte Carlo Pi estimation
│   └── actor_counter/    # Actor example
└── tests/
    ├── task_test.go
    ├── actor_test.go
    └── integration_test.go
```

## Core Concepts

### 1. Tasks (Stateless Functions)

```go
// Define a remote function
func Add(a, b int) int {
    return a + b
}

// Submit task and get future
future := ray.Submit(Add, 1, 2)

// Get result (blocks until ready)
result := ray.Get(future).(int)  // 3
```

### 2. Actors (Stateful Objects)

```go
// Define an actor
type Counter struct {
    value int
}

func (c *Counter) Increment() int {
    c.value++
    return c.value
}

// Create actor
counter := ray.CreateActor(&Counter{})

// Call actor method
future := counter.Call("Increment")
result := ray.Get(future).(int)
```

### 3. Object Store

```go
// Put object in store
objRef := ray.Put(largeData)

// Pass reference to task (zero-copy)
future := ray.Submit(ProcessData, objRef)
```

## Implementation TODO List

### Milestone 1: Core Types & Serialization
- [ ] Implement `pkg/core/types.go` - ObjectID, TaskID, ActorID
- [ ] Implement `internal/serialization/` - msgpack serialization
- [ ] Run: `go test ./pkg/core/...`

### Milestone 2: Object Store
- [ ] Implement `pkg/store/store.go` - Store interface
- [ ] Implement `pkg/store/memory.go` - In-memory store
- [ ] Run: `go test ./pkg/store/...`

### Milestone 3: Task Execution
- [ ] Implement `pkg/task/task.go` - Task definition
- [ ] Implement `pkg/task/executor.go` - Local executor
- [ ] Implement `pkg/task/dag.go` - Dependency tracking
- [ ] Run: `go test ./pkg/task/...`

### Milestone 4: Actor System
- [ ] Implement `pkg/actor/actor.go` - Actor interface
- [ ] Implement `pkg/actor/handle.go` - Actor proxy
- [ ] Implement `pkg/actor/registry.go` - Actor management
- [ ] Run: `go test ./pkg/actor/...`

### Milestone 5: Scheduler
- [ ] Implement `pkg/scheduler/scheduler.go` - Scheduler interface
- [ ] Implement `pkg/scheduler/simple.go` - FIFO scheduler
- [ ] Run: `go test ./pkg/scheduler/...`

### Milestone 6: RPC & Distribution
- [ ] Define protobuf messages
- [ ] Implement gRPC server/client
- [ ] Implement worker registration
- [ ] Run: `go test ./...`

### Milestone 7: Integration
- [ ] Implement driver CLI
- [ ] Implement worker process
- [ ] Run examples

## Key Design Decisions

### Why Message Passing?
- Workers are separate processes
- Objects are serialized and transferred
- Enables scaling across machines

### Why Actors?
- Encapsulate state
- Sequential method execution (no locks needed)
- Natural model for stateful services

### Why Object Store?
- Avoid copying large objects
- Enable data sharing between tasks
- Zero-copy when possible (shared memory)

## Running Tests

```bash
# Run all tests
go test ./...

# Run specific package
go test ./pkg/task/...

# Run with verbose output
go test -v ./...

# Run integration tests
go test ./tests/... -tags=integration
```

## References

- [Ray Paper](https://www.usenix.org/system/files/osdi18-moritz.pdf)
- [Ray Architecture](https://docs.ray.io/en/latest/ray-core/ray-internals.html)
- [Actor Model](https://en.wikipedia.org/wiki/Actor_model)
