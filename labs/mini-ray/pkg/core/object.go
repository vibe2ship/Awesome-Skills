// Package core provides the ObjectRef type for referencing objects in the store.
package core

import (
	"context"
	"sync"
	"time"
)

// ObjectRef is a reference to an object in the object store.
// It acts as a future - the object may not be available yet.
//
// ObjectRefs are the primary way to pass data between tasks.
// They enable:
// - Lazy evaluation (tasks run when dependencies are ready)
// - Zero-copy data sharing (via object store)
// - Distributed execution (objects can be on remote workers)
type ObjectRef struct {
	id ObjectID

	// mu protects the following fields
	mu sync.RWMutex

	// ready is closed when the object is available
	ready chan struct{}

	// value holds the deserialized object (cached)
	value interface{}

	// err holds any error that occurred
	err error

	// resolved indicates if the future has been resolved
	resolved bool

	// ownerWorker is the worker that owns this object
	ownerWorker WorkerID

	// size is the serialized size in bytes
	size int64
}

// NewObjectRef creates a new ObjectRef with the given ID.
//
// TODO: Implement this function
// - Initialize the ObjectRef
// - Create the ready channel
func NewObjectRef(id ObjectID) *ObjectRef {
	panic("TODO: implement NewObjectRef")
}

// ID returns the ObjectID of this reference.
func (ref *ObjectRef) ID() ObjectID {
	return ref.id
}

// IsReady returns true if the object is available.
//
// TODO: Implement this function
// - Use non-blocking channel receive to check
func (ref *ObjectRef) IsReady() bool {
	panic("TODO: implement ObjectRef.IsReady")
}

// Wait blocks until the object is ready or context is cancelled.
//
// TODO: Implement this function
// - Select on ready channel and context.Done()
func (ref *ObjectRef) Wait(ctx context.Context) error {
	panic("TODO: implement ObjectRef.Wait")
}

// WaitTimeout blocks until the object is ready or timeout.
//
// TODO: Implement this function
// - Use context.WithTimeout
func (ref *ObjectRef) WaitTimeout(timeout time.Duration) error {
	panic("TODO: implement ObjectRef.WaitTimeout")
}

// Get returns the object value, blocking until ready.
// Returns the cached value if already resolved.
//
// TODO: Implement this function
// - Wait for ready
// - Return value or error
func (ref *ObjectRef) Get(ctx context.Context) (interface{}, error) {
	panic("TODO: implement ObjectRef.Get")
}

// Resolve marks the object as ready with the given value.
// This is called by the object store when the object is available.
//
// TODO: Implement this function
// - Set value
// - Close ready channel
// - Mark as resolved
// - Handle double-resolve (should be idempotent or error)
func (ref *ObjectRef) Resolve(value interface{}) error {
	panic("TODO: implement ObjectRef.Resolve")
}

// Fail marks the object as failed with the given error.
//
// TODO: Implement this function
// - Set error
// - Close ready channel
// - Mark as resolved
func (ref *ObjectRef) Fail(err error) error {
	panic("TODO: implement ObjectRef.Fail")
}

// SetOwner sets the worker that owns this object.
func (ref *ObjectRef) SetOwner(worker WorkerID) {
	ref.mu.Lock()
	defer ref.mu.Unlock()
	ref.ownerWorker = worker
}

// Owner returns the worker that owns this object.
func (ref *ObjectRef) Owner() WorkerID {
	ref.mu.RLock()
	defer ref.mu.RUnlock()
	return ref.ownerWorker
}

// SetSize sets the serialized size of the object.
func (ref *ObjectRef) SetSize(size int64) {
	ref.mu.Lock()
	defer ref.mu.Unlock()
	ref.size = size
}

// Size returns the serialized size of the object.
func (ref *ObjectRef) Size() int64 {
	ref.mu.RLock()
	defer ref.mu.RUnlock()
	return ref.size
}

// ObjectRefGroup represents a group of ObjectRefs.
// Useful for waiting on multiple objects.
type ObjectRefGroup struct {
	refs []*ObjectRef
}

// NewObjectRefGroup creates a new group from the given refs.
func NewObjectRefGroup(refs ...*ObjectRef) *ObjectRefGroup {
	return &ObjectRefGroup{refs: refs}
}

// WaitAll blocks until all objects are ready.
//
// TODO: Implement this function
// - Wait for each ref
// - Return first error encountered
func (g *ObjectRefGroup) WaitAll(ctx context.Context) error {
	panic("TODO: implement ObjectRefGroup.WaitAll")
}

// WaitAny blocks until any object is ready.
// Returns the index of the ready object.
//
// TODO: Implement this function
// - Use select with all ready channels
// - Return index of first ready
func (g *ObjectRefGroup) WaitAny(ctx context.Context) (int, error) {
	panic("TODO: implement ObjectRefGroup.WaitAny")
}

// GetAll returns all object values, blocking until all are ready.
//
// TODO: Implement this function
// - Wait for all
// - Collect values
func (g *ObjectRefGroup) GetAll(ctx context.Context) ([]interface{}, error) {
	panic("TODO: implement ObjectRefGroup.GetAll")
}

// ReadyCount returns the number of ready objects.
//
// TODO: Implement this function
func (g *ObjectRefGroup) ReadyCount() int {
	panic("TODO: implement ObjectRefGroup.ReadyCount")
}
