// Package core provides fundamental types for Mini-Ray.
//
// This package defines the core identifiers and types used throughout
// the distributed computing framework:
// - ObjectID: Unique identifier for objects in the object store
// - TaskID: Unique identifier for submitted tasks
// - ActorID: Unique identifier for actors
// - WorkerID: Unique identifier for worker processes
package core

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
)

// IDSize is the size of IDs in bytes.
const IDSize = 16

// ObjectID uniquely identifies an object in the object store.
// Objects are immutable once created.
type ObjectID [IDSize]byte

// TaskID uniquely identifies a task submission.
type TaskID [IDSize]byte

// ActorID uniquely identifies an actor instance.
type ActorID [IDSize]byte

// WorkerID uniquely identifies a worker process.
type WorkerID [IDSize]byte

// NewObjectID generates a new random ObjectID.
//
// TODO: Implement this function
// - Use crypto/rand to generate random bytes
// - Return the new ObjectID
func NewObjectID() ObjectID {
	panic("TODO: implement NewObjectID")
}

// NewTaskID generates a new random TaskID.
//
// TODO: Implement this function
func NewTaskID() TaskID {
	panic("TODO: implement NewTaskID")
}

// NewActorID generates a new random ActorID.
//
// TODO: Implement this function
func NewActorID() ActorID {
	panic("TODO: implement NewActorID")
}

// NewWorkerID generates a new random WorkerID.
//
// TODO: Implement this function
func NewWorkerID() WorkerID {
	panic("TODO: implement NewWorkerID")
}

// String returns the hex string representation of ObjectID.
//
// TODO: Implement this function
// - Use hex.EncodeToString
func (id ObjectID) String() string {
	panic("TODO: implement ObjectID.String")
}

// String returns the hex string representation of TaskID.
func (id TaskID) String() string {
	panic("TODO: implement TaskID.String")
}

// String returns the hex string representation of ActorID.
func (id ActorID) String() string {
	panic("TODO: implement ActorID.String")
}

// String returns the hex string representation of WorkerID.
func (id WorkerID) String() string {
	panic("TODO: implement WorkerID.String")
}

// IsNil returns true if the ObjectID is all zeros.
//
// TODO: Implement this function
func (id ObjectID) IsNil() bool {
	panic("TODO: implement ObjectID.IsNil")
}

// ParseObjectID parses a hex string into an ObjectID.
//
// TODO: Implement this function
// - Use hex.DecodeString
// - Return error if string is invalid
func ParseObjectID(s string) (ObjectID, error) {
	panic("TODO: implement ParseObjectID")
}

// ObjectIDFromBytes creates an ObjectID from a byte slice.
//
// TODO: Implement this function
// - Validate length
// - Copy bytes into ObjectID
func ObjectIDFromBytes(b []byte) (ObjectID, error) {
	panic("TODO: implement ObjectIDFromBytes")
}

// Bytes returns the ObjectID as a byte slice.
func (id ObjectID) Bytes() []byte {
	return id[:]
}

// TaskStatus represents the execution status of a task.
type TaskStatus int

const (
	TaskStatusPending TaskStatus = iota
	TaskStatusScheduled
	TaskStatusRunning
	TaskStatusFinished
	TaskStatusFailed
)

// String returns the string representation of TaskStatus.
//
// TODO: Implement this function
func (s TaskStatus) String() string {
	panic("TODO: implement TaskStatus.String")
}

// ActorStatus represents the lifecycle status of an actor.
type ActorStatus int

const (
	ActorStatusCreating ActorStatus = iota
	ActorStatusAlive
	ActorStatusDead
)

// String returns the string representation of ActorStatus.
func (s ActorStatus) String() string {
	panic("TODO: implement ActorStatus.String")
}

// WorkerStatus represents the status of a worker.
type WorkerStatus int

const (
	WorkerStatusStarting WorkerStatus = iota
	WorkerStatusIdle
	WorkerStatusBusy
	WorkerStatusDead
)

// Resources represents the resources available on a worker.
type Resources struct {
	CPU    float64 // Number of CPU cores
	Memory int64   // Memory in bytes
	Custom map[string]float64
}

// NewResources creates a new Resources with the given CPU and memory.
//
// TODO: Implement this function
func NewResources(cpu float64, memory int64) Resources {
	panic("TODO: implement NewResources")
}

// CanSatisfy returns true if these resources can satisfy the request.
//
// TODO: Implement this function
// - Check if all requested resources are available
func (r Resources) CanSatisfy(request Resources) bool {
	panic("TODO: implement Resources.CanSatisfy")
}

// Subtract returns new Resources with the request subtracted.
//
// TODO: Implement this function
func (r Resources) Subtract(request Resources) Resources {
	panic("TODO: implement Resources.Subtract")
}

// Add returns new Resources with the other added.
//
// TODO: Implement this function
func (r Resources) Add(other Resources) Resources {
	panic("TODO: implement Resources.Add")
}

// Ensure interfaces are implemented correctly
var (
	_ fmt.Stringer = ObjectID{}
	_ fmt.Stringer = TaskID{}
	_ fmt.Stringer = ActorID{}
	_ fmt.Stringer = WorkerID{}
	_ fmt.Stringer = TaskStatus(0)
)
