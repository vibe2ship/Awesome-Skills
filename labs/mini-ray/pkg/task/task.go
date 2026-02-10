// Package task provides task definition and management for Mini-Ray.
//
// A Task represents a unit of work that can be executed on a worker.
// Tasks are:
// - Stateless functions
// - May have dependencies on other tasks (via ObjectRefs)
// - Produce a single output (stored in object store)
package task

import (
	"context"
	"reflect"
	"time"

	"github.com/student/mini-ray/pkg/core"
)

// Task represents a unit of work to be executed.
type Task struct {
	// ID uniquely identifies this task
	ID core.TaskID

	// FunctionName is the name of the function to execute
	FunctionName string

	// Args are the serialized arguments
	// Some args may be ObjectRefs (dependencies)
	Args [][]byte

	// ArgTypes help with deserialization
	ArgTypes []string

	// Dependencies are ObjectIDs that must be resolved before execution
	Dependencies []core.ObjectID

	// ResultID is the ObjectID where the result will be stored
	ResultID core.ObjectID

	// Resources required by this task
	Resources core.Resources

	// Status of the task
	Status core.TaskStatus

	// WorkerID is the worker executing this task (if scheduled)
	WorkerID core.WorkerID

	// SubmitTime is when the task was submitted
	SubmitTime time.Time

	// StartTime is when execution started
	StartTime time.Time

	// EndTime is when execution completed
	EndTime time.Time

	// RetryCount is the number of times this task has been retried
	RetryCount int

	// MaxRetries is the maximum number of retries
	MaxRetries int

	// Error message if task failed
	Error string
}

// TaskOptions configures task submission.
type TaskOptions struct {
	// Resources required by the task
	Resources core.Resources

	// MaxRetries before giving up
	MaxRetries int

	// Name for debugging
	Name string

	// PlacementGroup for co-location
	PlacementGroup string
}

// DefaultTaskOptions returns sensible defaults.
func DefaultTaskOptions() TaskOptions {
	return TaskOptions{
		Resources: core.Resources{
			CPU:    1.0,
			Memory: 256 * 1024 * 1024, // 256MB
		},
		MaxRetries: 3,
	}
}

// NewTask creates a new Task.
//
// TODO: Implement this function
// - Generate new TaskID
// - Generate ResultID for the output
// - Initialize timestamps and status
func NewTask(functionName string, args [][]byte, deps []core.ObjectID, opts TaskOptions) *Task {
	panic("TODO: implement NewTask")
}

// IsPending returns true if the task is waiting to be scheduled.
func (t *Task) IsPending() bool {
	return t.Status == core.TaskStatusPending
}

// IsRunning returns true if the task is currently executing.
func (t *Task) IsRunning() bool {
	return t.Status == core.TaskStatusRunning
}

// IsFinished returns true if the task has completed (success or failure).
func (t *Task) IsFinished() bool {
	return t.Status == core.TaskStatusFinished || t.Status == core.TaskStatusFailed
}

// CanRetry returns true if the task can be retried.
//
// TODO: Implement this function
func (t *Task) CanRetry() bool {
	panic("TODO: implement Task.CanRetry")
}

// MarkScheduled marks the task as scheduled on a worker.
//
// TODO: Implement this function
func (t *Task) MarkScheduled(workerID core.WorkerID) {
	panic("TODO: implement Task.MarkScheduled")
}

// MarkRunning marks the task as running.
//
// TODO: Implement this function
func (t *Task) MarkRunning() {
	panic("TODO: implement Task.MarkRunning")
}

// MarkFinished marks the task as successfully completed.
//
// TODO: Implement this function
func (t *Task) MarkFinished() {
	panic("TODO: implement Task.MarkFinished")
}

// MarkFailed marks the task as failed.
//
// TODO: Implement this function
func (t *Task) MarkFailed(err string) {
	panic("TODO: implement Task.MarkFailed")
}

// Duration returns the execution duration (0 if not finished).
//
// TODO: Implement this function
func (t *Task) Duration() time.Duration {
	panic("TODO: implement Task.Duration")
}

// FunctionRegistry holds registered remote functions.
type FunctionRegistry struct {
	functions map[string]interface{}
}

// NewFunctionRegistry creates a new registry.
func NewFunctionRegistry() *FunctionRegistry {
	return &FunctionRegistry{
		functions: make(map[string]interface{}),
	}
}

// Register adds a function to the registry.
//
// TODO: Implement this function
// - Validate that fn is a function
// - Store by name
func (r *FunctionRegistry) Register(name string, fn interface{}) error {
	panic("TODO: implement FunctionRegistry.Register")
}

// Get retrieves a function by name.
//
// TODO: Implement this function
func (r *FunctionRegistry) Get(name string) (interface{}, bool) {
	panic("TODO: implement FunctionRegistry.Get")
}

// Call invokes a registered function with the given arguments.
//
// TODO: Implement this function
// - Get function from registry
// - Use reflect to call with args
// - Return results
func (r *FunctionRegistry) Call(name string, args []interface{}) (interface{}, error) {
	panic("TODO: implement FunctionRegistry.Call")
}

// GetFunctionSignature returns the argument and return types of a function.
//
// TODO: Implement this function
// - Use reflect to extract type information
func GetFunctionSignature(fn interface{}) (argTypes []reflect.Type, returnTypes []reflect.Type, err error) {
	panic("TODO: implement GetFunctionSignature")
}
