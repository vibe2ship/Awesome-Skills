// Package store provides the object store for Mini-Ray.
//
// The object store is a key-value store for immutable objects.
// Objects are identified by ObjectID and can be:
// - Put: Store an object
// - Get: Retrieve an object
// - Delete: Remove an object (for garbage collection)
//
// The store supports:
// - Local storage (in-memory or on disk)
// - Remote fetching (from other workers)
// - Reference counting for garbage collection
package store

import (
	"context"
	"io"

	"github.com/student/mini-ray/pkg/core"
)

// Store is the interface for object storage.
type Store interface {
	// Put stores an object with the given ID.
	// If the object already exists, this is a no-op.
	Put(ctx context.Context, id core.ObjectID, data []byte) error

	// Get retrieves an object by ID.
	// Returns ErrObjectNotFound if not present.
	Get(ctx context.Context, id core.ObjectID) ([]byte, error)

	// Contains returns true if the object exists locally.
	Contains(id core.ObjectID) bool

	// Delete removes an object from the store.
	Delete(id core.ObjectID) error

	// Size returns the size of an object in bytes.
	// Returns 0 if object doesn't exist.
	Size(id core.ObjectID) int64

	// List returns all object IDs in the store.
	List() []core.ObjectID

	// Stats returns store statistics.
	Stats() StoreStats

	// Close closes the store and releases resources.
	Close() error
}

// StoreStats holds statistics about the object store.
type StoreStats struct {
	ObjectCount   int64 // Number of objects
	TotalBytes    int64 // Total bytes stored
	AvailableBytes int64 // Available capacity
	GetCount      int64 // Number of Get operations
	PutCount      int64 // Number of Put operations
	HitCount      int64 // Cache hits
	MissCount     int64 // Cache misses
}

// ObjectMetadata holds metadata about an object.
type ObjectMetadata struct {
	ID         core.ObjectID
	Size       int64
	CreateTime int64 // Unix timestamp
	RefCount   int32
	OwnerWorker core.WorkerID
}

// StoreConfig configures the object store.
type StoreConfig struct {
	// MaxBytes is the maximum capacity in bytes.
	// If 0, no limit is applied.
	MaxBytes int64

	// EvictionPolicy determines how objects are evicted when full.
	EvictionPolicy EvictionPolicy

	// Directory for disk-based storage (if applicable).
	Directory string
}

// EvictionPolicy determines eviction behavior.
type EvictionPolicy int

const (
	// EvictionLRU evicts least recently used objects.
	EvictionLRU EvictionPolicy = iota

	// EvictionLFU evicts least frequently used objects.
	EvictionLFU

	// EvictionNone disables eviction (fail on full).
	EvictionNone
)

// DefaultStoreConfig returns sensible defaults.
func DefaultStoreConfig() StoreConfig {
	return StoreConfig{
		MaxBytes:       1 << 30, // 1 GB
		EvictionPolicy: EvictionLRU,
	}
}

// ObjectReader provides streaming access to large objects.
type ObjectReader interface {
	io.Reader
	io.Closer
	Size() int64
}

// ObjectWriter provides streaming writes for large objects.
type ObjectWriter interface {
	io.Writer
	io.Closer
	// Commit finalizes the write and makes the object available.
	Commit() error
	// Abort cancels the write.
	Abort() error
}

// StreamingStore extends Store with streaming support.
type StreamingStore interface {
	Store

	// GetReader returns a reader for streaming large objects.
	GetReader(ctx context.Context, id core.ObjectID) (ObjectReader, error)

	// PutWriter returns a writer for streaming large objects.
	PutWriter(ctx context.Context, id core.ObjectID) (ObjectWriter, error)
}

// DistributedStore extends Store with remote fetching.
type DistributedStore interface {
	Store

	// FetchRemote fetches an object from a remote worker.
	FetchRemote(ctx context.Context, id core.ObjectID, from core.WorkerID) error

	// GetLocation returns the worker(s) that have the object.
	GetLocation(id core.ObjectID) []core.WorkerID

	// Pin prevents an object from being evicted.
	Pin(id core.ObjectID) error

	// Unpin allows an object to be evicted.
	Unpin(id core.ObjectID) error
}

// ObjectDirectory tracks object locations across workers.
type ObjectDirectory interface {
	// Register records that a worker has an object.
	Register(id core.ObjectID, worker core.WorkerID) error

	// Unregister removes a worker's record for an object.
	Unregister(id core.ObjectID, worker core.WorkerID) error

	// Lookup returns workers that have the object.
	Lookup(id core.ObjectID) []core.WorkerID

	// Subscribe receives notifications when an object becomes available.
	Subscribe(id core.ObjectID) <-chan core.WorkerID
}
