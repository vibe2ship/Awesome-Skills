package tests

import (
	"context"
	"testing"
	"time"

	"github.com/student/mini-ray/pkg/core"
)

func TestObjectID(t *testing.T) {
	t.Run("NewObjectID generates unique IDs", func(t *testing.T) {
		id1 := core.NewObjectID()
		id2 := core.NewObjectID()

		if id1 == id2 {
			t.Error("Expected unique IDs, got same ID")
		}
	})

	t.Run("ObjectID String returns hex", func(t *testing.T) {
		id := core.NewObjectID()
		s := id.String()

		if len(s) != 32 { // 16 bytes = 32 hex chars
			t.Errorf("Expected 32 char hex string, got %d chars", len(s))
		}
	})

	t.Run("ParseObjectID roundtrip", func(t *testing.T) {
		original := core.NewObjectID()
		s := original.String()

		parsed, err := core.ParseObjectID(s)
		if err != nil {
			t.Fatalf("ParseObjectID failed: %v", err)
		}

		if parsed != original {
			t.Error("Parsed ID doesn't match original")
		}
	})

	t.Run("IsNil detects zero ID", func(t *testing.T) {
		var nilID core.ObjectID
		if !nilID.IsNil() {
			t.Error("Expected zero ID to be nil")
		}

		id := core.NewObjectID()
		if id.IsNil() {
			t.Error("Expected non-zero ID to not be nil")
		}
	})
}

func TestTaskID(t *testing.T) {
	t.Run("NewTaskID generates unique IDs", func(t *testing.T) {
		id1 := core.NewTaskID()
		id2 := core.NewTaskID()

		if id1 == id2 {
			t.Error("Expected unique IDs")
		}
	})
}

func TestResources(t *testing.T) {
	t.Run("NewResources creates resources", func(t *testing.T) {
		r := core.NewResources(4.0, 1024)

		if r.CPU != 4.0 {
			t.Errorf("Expected CPU 4.0, got %f", r.CPU)
		}
		if r.Memory != 1024 {
			t.Errorf("Expected Memory 1024, got %d", r.Memory)
		}
	})

	t.Run("CanSatisfy checks resource availability", func(t *testing.T) {
		available := core.NewResources(8.0, 2048)
		request1 := core.NewResources(4.0, 1024)
		request2 := core.NewResources(16.0, 1024)

		if !available.CanSatisfy(request1) {
			t.Error("Expected to satisfy request1")
		}
		if available.CanSatisfy(request2) {
			t.Error("Expected not to satisfy request2 (CPU too high)")
		}
	})

	t.Run("Subtract reduces resources", func(t *testing.T) {
		r := core.NewResources(8.0, 2048)
		request := core.NewResources(2.0, 512)

		remaining := r.Subtract(request)

		if remaining.CPU != 6.0 {
			t.Errorf("Expected CPU 6.0, got %f", remaining.CPU)
		}
		if remaining.Memory != 1536 {
			t.Errorf("Expected Memory 1536, got %d", remaining.Memory)
		}
	})

	t.Run("Add combines resources", func(t *testing.T) {
		r1 := core.NewResources(4.0, 1024)
		r2 := core.NewResources(2.0, 512)

		combined := r1.Add(r2)

		if combined.CPU != 6.0 {
			t.Errorf("Expected CPU 6.0, got %f", combined.CPU)
		}
		if combined.Memory != 1536 {
			t.Errorf("Expected Memory 1536, got %d", combined.Memory)
		}
	})
}

func TestTaskStatus(t *testing.T) {
	tests := []struct {
		status   core.TaskStatus
		expected string
	}{
		{core.TaskStatusPending, "pending"},
		{core.TaskStatusScheduled, "scheduled"},
		{core.TaskStatusRunning, "running"},
		{core.TaskStatusFinished, "finished"},
		{core.TaskStatusFailed, "failed"},
	}

	for _, tc := range tests {
		t.Run(tc.expected, func(t *testing.T) {
			s := tc.status.String()
			if s != tc.expected {
				t.Errorf("Expected %q, got %q", tc.expected, s)
			}
		})
	}
}

func TestObjectRef(t *testing.T) {
	t.Run("NewObjectRef creates unresolved ref", func(t *testing.T) {
		id := core.NewObjectID()
		ref := core.NewObjectRef(id)

		if ref.ID() != id {
			t.Error("ID mismatch")
		}
		if ref.IsReady() {
			t.Error("Expected ref to not be ready")
		}
	})

	t.Run("Resolve makes ref ready", func(t *testing.T) {
		id := core.NewObjectID()
		ref := core.NewObjectRef(id)

		err := ref.Resolve("test value")
		if err != nil {
			t.Fatalf("Resolve failed: %v", err)
		}

		if !ref.IsReady() {
			t.Error("Expected ref to be ready")
		}
	})

	t.Run("Get returns resolved value", func(t *testing.T) {
		id := core.NewObjectID()
		ref := core.NewObjectRef(id)

		expected := "test value"
		ref.Resolve(expected)

		ctx := context.Background()
		value, err := ref.Get(ctx)
		if err != nil {
			t.Fatalf("Get failed: %v", err)
		}

		if value != expected {
			t.Errorf("Expected %q, got %q", expected, value)
		}
	})

	t.Run("Get blocks until ready", func(t *testing.T) {
		id := core.NewObjectID()
		ref := core.NewObjectRef(id)

		done := make(chan bool)
		go func() {
			ctx := context.Background()
			ref.Get(ctx)
			done <- true
		}()

		// Should not be done yet
		select {
		case <-done:
			t.Error("Get returned before resolve")
		case <-time.After(50 * time.Millisecond):
			// Expected
		}

		// Resolve and check
		ref.Resolve("value")

		select {
		case <-done:
			// Expected
		case <-time.After(100 * time.Millisecond):
			t.Error("Get didn't return after resolve")
		}
	})

	t.Run("WaitTimeout returns error on timeout", func(t *testing.T) {
		id := core.NewObjectID()
		ref := core.NewObjectRef(id)

		err := ref.WaitTimeout(50 * time.Millisecond)
		if err == nil {
			t.Error("Expected timeout error")
		}
	})

	t.Run("Fail marks ref as failed", func(t *testing.T) {
		id := core.NewObjectID()
		ref := core.NewObjectRef(id)

		expectedErr := core.ErrWorkerDead
		ref.Fail(expectedErr)

		if !ref.IsReady() {
			t.Error("Expected ref to be ready (failed)")
		}

		ctx := context.Background()
		_, err := ref.Get(ctx)
		if err != expectedErr {
			t.Errorf("Expected %v, got %v", expectedErr, err)
		}
	})
}

func TestObjectRefGroup(t *testing.T) {
	t.Run("WaitAll waits for all refs", func(t *testing.T) {
		refs := make([]*core.ObjectRef, 3)
		for i := range refs {
			refs[i] = core.NewObjectRef(core.NewObjectID())
		}

		group := core.NewObjectRefGroup(refs...)

		done := make(chan error)
		go func() {
			ctx := context.Background()
			done <- group.WaitAll(ctx)
		}()

		// Resolve refs one by one
		for i, ref := range refs {
			ref.Resolve(i)
		}

		select {
		case err := <-done:
			if err != nil {
				t.Errorf("WaitAll failed: %v", err)
			}
		case <-time.After(100 * time.Millisecond):
			t.Error("WaitAll didn't return")
		}
	})

	t.Run("ReadyCount tracks ready refs", func(t *testing.T) {
		refs := make([]*core.ObjectRef, 3)
		for i := range refs {
			refs[i] = core.NewObjectRef(core.NewObjectID())
		}

		group := core.NewObjectRefGroup(refs...)

		if group.ReadyCount() != 0 {
			t.Errorf("Expected 0 ready, got %d", group.ReadyCount())
		}

		refs[0].Resolve("a")
		if group.ReadyCount() != 1 {
			t.Errorf("Expected 1 ready, got %d", group.ReadyCount())
		}

		refs[1].Resolve("b")
		refs[2].Resolve("c")
		if group.ReadyCount() != 3 {
			t.Errorf("Expected 3 ready, got %d", group.ReadyCount())
		}
	})
}
