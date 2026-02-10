// Package task provides the task executor for Mini-Ray.
package task

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/student/mini-ray/pkg/core"
)

// Executor executes tasks on a worker.
type Executor struct {
	// registry holds registered functions
	registry *FunctionRegistry

	// store is the object store for fetching dependencies
	store ObjectStore

	// serializer handles object serialization
	serializer Serializer

	// workerID identifies this worker
	workerID core.WorkerID

	// running tracks currently running tasks
	running   map[core.TaskID]*runningTask
	runningMu sync.RWMutex

	// maxConcurrent limits concurrent task execution
	maxConcurrent int

	// semaphore for concurrency control
	semaphore chan struct{}

	// stats
	completedTasks int64
	failedTasks    int64
	totalDuration  time.Duration
	statsMu        sync.Mutex
}

// ObjectStore interface for fetching objects.
type ObjectStore interface {
	Get(ctx context.Context, id core.ObjectID) ([]byte, error)
	Put(ctx context.Context, id core.ObjectID, data []byte) error
}

// Serializer interface for object serialization.
type Serializer interface {
	Serialize(v interface{}) ([]byte, error)
	Deserialize(data []byte, v interface{}) error
}

// runningTask tracks a running task.
type runningTask struct {
	task   *Task
	cancel context.CancelFunc
}

// ExecutorConfig configures the executor.
type ExecutorConfig struct {
	MaxConcurrent int
	WorkerID      core.WorkerID
}

// DefaultExecutorConfig returns sensible defaults.
func DefaultExecutorConfig() ExecutorConfig {
	return ExecutorConfig{
		MaxConcurrent: 4,
	}
}

// NewExecutor creates a new task executor.
//
// TODO: Implement this function
// - Initialize all fields
// - Create semaphore channel
func NewExecutor(registry *FunctionRegistry, store ObjectStore, serializer Serializer, config ExecutorConfig) *Executor {
	panic("TODO: implement NewExecutor")
}

// Execute runs a task and stores the result.
//
// This is the main execution loop:
// 1. Acquire semaphore slot
// 2. Fetch dependencies from object store
// 3. Deserialize arguments
// 4. Call the function
// 5. Serialize result
// 6. Store result in object store
// 7. Release semaphore slot
//
// TODO: Implement this function
func (e *Executor) Execute(ctx context.Context, task *Task) error {
	panic("TODO: implement Executor.Execute")
}

// fetchDependencies fetches all task dependencies from the object store.
//
// TODO: Implement this function
// - For each dependency, fetch from store
// - Return map of ObjectID -> data
func (e *Executor) fetchDependencies(ctx context.Context, deps []core.ObjectID) (map[core.ObjectID][]byte, error) {
	panic("TODO: implement Executor.fetchDependencies")
}

// deserializeArgs deserializes task arguments.
// ObjectRef arguments are replaced with the actual values from dependencies.
//
// TODO: Implement this function
// - Deserialize each arg based on type
// - Replace ObjectRefs with fetched data
func (e *Executor) deserializeArgs(task *Task, deps map[core.ObjectID][]byte) ([]interface{}, error) {
	panic("TODO: implement Executor.deserializeArgs")
}

// invokeFunction calls the function with the given arguments.
//
// TODO: Implement this function
// - Get function from registry
// - Use reflect.Call
// - Handle panics gracefully
func (e *Executor) invokeFunction(name string, args []interface{}) (interface{}, error) {
	panic("TODO: implement Executor.invokeFunction")
}

// Cancel cancels a running task.
//
// TODO: Implement this function
func (e *Executor) Cancel(taskID core.TaskID) error {
	panic("TODO: implement Executor.Cancel")
}

// RunningTasks returns the IDs of currently running tasks.
//
// TODO: Implement this function
func (e *Executor) RunningTasks() []core.TaskID {
	panic("TODO: implement Executor.RunningTasks")
}

// ExecutorStats holds executor statistics.
type ExecutorStats struct {
	RunningTasks   int
	CompletedTasks int64
	FailedTasks    int64
	TotalDuration  time.Duration
}

// Stats returns current executor statistics.
func (e *Executor) Stats() ExecutorStats {
	e.runningMu.RLock()
	running := len(e.running)
	e.runningMu.RUnlock()

	e.statsMu.Lock()
	defer e.statsMu.Unlock()

	return ExecutorStats{
		RunningTasks:   running,
		CompletedTasks: e.completedTasks,
		FailedTasks:    e.failedTasks,
		TotalDuration:  e.totalDuration,
	}
}

// recoverFromPanic helps recover from panics during function execution.
func recoverFromPanic(fn func() (interface{}, error)) (result interface{}, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic during execution: %v", r)
		}
	}()
	return fn()
}
