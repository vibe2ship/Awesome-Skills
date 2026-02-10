package tests

import (
	"context"
	"testing"

	"github.com/student/mini-ray/pkg/core"
	"github.com/student/mini-ray/pkg/store"
)

func TestMemoryStore(t *testing.T) {
	t.Run("Put and Get", func(t *testing.T) {
		s := store.NewMemoryStore(store.DefaultStoreConfig())
		defer s.Close()

		ctx := context.Background()
		id := core.NewObjectID()
		data := []byte("hello world")

		err := s.Put(ctx, id, data)
		if err != nil {
			t.Fatalf("Put failed: %v", err)
		}

		got, err := s.Get(ctx, id)
		if err != nil {
			t.Fatalf("Get failed: %v", err)
		}

		if string(got) != string(data) {
			t.Errorf("Expected %q, got %q", data, got)
		}
	})

	t.Run("Get non-existent returns error", func(t *testing.T) {
		s := store.NewMemoryStore(store.DefaultStoreConfig())
		defer s.Close()

		ctx := context.Background()
		id := core.NewObjectID()

		_, err := s.Get(ctx, id)
		if err != core.ErrObjectNotFound {
			t.Errorf("Expected ErrObjectNotFound, got %v", err)
		}
	})

	t.Run("Contains", func(t *testing.T) {
		s := store.NewMemoryStore(store.DefaultStoreConfig())
		defer s.Close()

		ctx := context.Background()
		id := core.NewObjectID()

		if s.Contains(id) {
			t.Error("Expected Contains to return false")
		}

		s.Put(ctx, id, []byte("data"))

		if !s.Contains(id) {
			t.Error("Expected Contains to return true")
		}
	})

	t.Run("Delete", func(t *testing.T) {
		s := store.NewMemoryStore(store.DefaultStoreConfig())
		defer s.Close()

		ctx := context.Background()
		id := core.NewObjectID()

		s.Put(ctx, id, []byte("data"))

		err := s.Delete(id)
		if err != nil {
			t.Fatalf("Delete failed: %v", err)
		}

		if s.Contains(id) {
			t.Error("Expected object to be deleted")
		}
	})

	t.Run("Size", func(t *testing.T) {
		s := store.NewMemoryStore(store.DefaultStoreConfig())
		defer s.Close()

		ctx := context.Background()
		id := core.NewObjectID()
		data := []byte("hello world")

		s.Put(ctx, id, data)

		size := s.Size(id)
		if size != int64(len(data)) {
			t.Errorf("Expected size %d, got %d", len(data), size)
		}
	})

	t.Run("List", func(t *testing.T) {
		s := store.NewMemoryStore(store.DefaultStoreConfig())
		defer s.Close()

		ctx := context.Background()

		ids := make([]core.ObjectID, 3)
		for i := range ids {
			ids[i] = core.NewObjectID()
			s.Put(ctx, ids[i], []byte("data"))
		}

		listed := s.List()
		if len(listed) != 3 {
			t.Errorf("Expected 3 objects, got %d", len(listed))
		}
	})

	t.Run("Stats", func(t *testing.T) {
		s := store.NewMemoryStore(store.DefaultStoreConfig())
		defer s.Close()

		ctx := context.Background()

		for i := 0; i < 5; i++ {
			id := core.NewObjectID()
			s.Put(ctx, id, []byte("data"))
		}

		stats := s.Stats()
		if stats.ObjectCount != 5 {
			t.Errorf("Expected 5 objects, got %d", stats.ObjectCount)
		}
		if stats.PutCount != 5 {
			t.Errorf("Expected 5 puts, got %d", stats.PutCount)
		}
	})

	t.Run("LRU eviction", func(t *testing.T) {
		config := store.StoreConfig{
			MaxBytes:       100, // Very small
			EvictionPolicy: store.EvictionLRU,
		}
		s := store.NewMemoryStore(config)
		defer s.Close()

		ctx := context.Background()

		// Put multiple objects that exceed capacity
		ids := make([]core.ObjectID, 5)
		for i := range ids {
			ids[i] = core.NewObjectID()
			s.Put(ctx, ids[i], make([]byte, 30)) // 30 bytes each
		}

		// Early objects should be evicted
		// With 100 bytes max and 30 bytes each, can fit ~3
		count := 0
		for _, id := range ids {
			if s.Contains(id) {
				count++
			}
		}

		if count > 3 {
			t.Errorf("Expected at most 3 objects, got %d", count)
		}
	})

	t.Run("Pin prevents eviction", func(t *testing.T) {
		config := store.StoreConfig{
			MaxBytes:       100,
			EvictionPolicy: store.EvictionLRU,
		}
		s := store.NewMemoryStore(config)
		defer s.Close()

		ctx := context.Background()

		// Put and pin first object
		id1 := core.NewObjectID()
		s.Put(ctx, id1, make([]byte, 30))
		s.Pin(id1)

		// Put more objects to trigger eviction
		for i := 0; i < 5; i++ {
			id := core.NewObjectID()
			s.Put(ctx, id, make([]byte, 30))
		}

		// Pinned object should still be there
		if !s.Contains(id1) {
			t.Error("Pinned object was evicted")
		}
	})
}
