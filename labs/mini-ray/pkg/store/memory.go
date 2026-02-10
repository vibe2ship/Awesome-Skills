// Package store provides an in-memory object store implementation.
package store

import (
	"container/list"
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/student/mini-ray/pkg/core"
)

// MemoryStore is an in-memory implementation of the Store interface.
// It uses LRU eviction when the capacity is exceeded.
type MemoryStore struct {
	mu sync.RWMutex

	// objects maps ObjectID to stored data
	objects map[core.ObjectID]*memoryObject

	// lru tracks access order for LRU eviction
	lru *list.List

	// lruIndex maps ObjectID to LRU list element
	lruIndex map[core.ObjectID]*list.Element

	// config
	config StoreConfig

	// stats
	totalBytes   int64
	getCount     int64
	putCount     int64
	hitCount     int64
	missCount    int64

	// closed indicates if the store is closed
	closed bool
}

// memoryObject holds an object and its metadata.
type memoryObject struct {
	id         core.ObjectID
	data       []byte
	size       int64
	createTime time.Time
	accessTime time.Time
	refCount   int32
	pinned     bool
}

// NewMemoryStore creates a new in-memory store.
//
// TODO: Implement this function
// - Initialize all maps and the LRU list
// - Store the config
func NewMemoryStore(config StoreConfig) *MemoryStore {
	panic("TODO: implement NewMemoryStore")
}

// Put stores an object.
//
// TODO: Implement this function
// - Check if already exists (no-op)
// - Check capacity, evict if needed
// - Store the object
// - Update LRU
// - Update stats
func (s *MemoryStore) Put(ctx context.Context, id core.ObjectID, data []byte) error {
	panic("TODO: implement MemoryStore.Put")
}

// Get retrieves an object.
//
// TODO: Implement this function
// - Check if exists
// - Update access time and LRU position
// - Update stats
// - Return data (copy or reference?)
func (s *MemoryStore) Get(ctx context.Context, id core.ObjectID) ([]byte, error) {
	panic("TODO: implement MemoryStore.Get")
}

// Contains returns true if the object exists.
//
// TODO: Implement this function
func (s *MemoryStore) Contains(id core.ObjectID) bool {
	panic("TODO: implement MemoryStore.Contains")
}

// Delete removes an object.
//
// TODO: Implement this function
// - Check if pinned (refuse if pinned)
// - Remove from maps and LRU
// - Update stats
func (s *MemoryStore) Delete(id core.ObjectID) error {
	panic("TODO: implement MemoryStore.Delete")
}

// Size returns the size of an object.
//
// TODO: Implement this function
func (s *MemoryStore) Size(id core.ObjectID) int64 {
	panic("TODO: implement MemoryStore.Size")
}

// List returns all object IDs.
//
// TODO: Implement this function
func (s *MemoryStore) List() []core.ObjectID {
	panic("TODO: implement MemoryStore.List")
}

// Stats returns store statistics.
func (s *MemoryStore) Stats() StoreStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return StoreStats{
		ObjectCount:    int64(len(s.objects)),
		TotalBytes:     atomic.LoadInt64(&s.totalBytes),
		AvailableBytes: s.config.MaxBytes - atomic.LoadInt64(&s.totalBytes),
		GetCount:       atomic.LoadInt64(&s.getCount),
		PutCount:       atomic.LoadInt64(&s.putCount),
		HitCount:       atomic.LoadInt64(&s.hitCount),
		MissCount:      atomic.LoadInt64(&s.missCount),
	}
}

// Close closes the store.
//
// TODO: Implement this function
// - Mark as closed
// - Clear all data
func (s *MemoryStore) Close() error {
	panic("TODO: implement MemoryStore.Close")
}

// evict removes objects until there's enough space.
// Must be called with lock held.
//
// TODO: Implement this function
// - Use LRU list to find eviction candidates
// - Skip pinned objects
// - Remove objects until we have enough space
func (s *MemoryStore) evict(needed int64) error {
	panic("TODO: implement MemoryStore.evict")
}

// updateLRU moves an object to the front of the LRU list.
// Must be called with lock held.
//
// TODO: Implement this function
func (s *MemoryStore) updateLRU(id core.ObjectID) {
	panic("TODO: implement MemoryStore.updateLRU")
}

// Pin prevents an object from being evicted.
//
// TODO: Implement this function
func (s *MemoryStore) Pin(id core.ObjectID) error {
	panic("TODO: implement MemoryStore.Pin")
}

// Unpin allows an object to be evicted.
//
// TODO: Implement this function
func (s *MemoryStore) Unpin(id core.ObjectID) error {
	panic("TODO: implement MemoryStore.Unpin")
}

// Ensure MemoryStore implements Store interface.
var _ Store = (*MemoryStore)(nil)
