// Package core provides error types for Mini-Ray.
package core

import (
	"errors"
	"fmt"
)

// Sentinel errors for common conditions.
var (
	// ErrObjectNotFound is returned when an object is not in the store.
	ErrObjectNotFound = errors.New("object not found")

	// ErrWorkerNotFound is returned when a worker is not registered.
	ErrWorkerNotFound = errors.New("worker not found")

	// ErrActorNotFound is returned when an actor is not found.
	ErrActorNotFound = errors.New("actor not found")

	// ErrTaskNotFound is returned when a task is not found.
	ErrTaskNotFound = errors.New("task not found")

	// ErrAlreadyResolved is returned when trying to resolve an already resolved future.
	ErrAlreadyResolved = errors.New("object already resolved")

	// ErrTimeout is returned when an operation times out.
	ErrTimeout = errors.New("operation timed out")

	// ErrWorkerDead is returned when a worker has died.
	ErrWorkerDead = errors.New("worker is dead")

	// ErrActorDead is returned when an actor has died.
	ErrActorDead = errors.New("actor is dead")

	// ErrInsufficientResources is returned when there aren't enough resources.
	ErrInsufficientResources = errors.New("insufficient resources")

	// ErrSerializationFailed is returned when serialization fails.
	ErrSerializationFailed = errors.New("serialization failed")

	// ErrDeserializationFailed is returned when deserialization fails.
	ErrDeserializationFailed = errors.New("deserialization failed")
)

// TaskError represents an error that occurred during task execution.
type TaskError struct {
	TaskID  TaskID
	Message string
	Cause   error
}

// Error implements the error interface.
func (e *TaskError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("task %s failed: %s: %v", e.TaskID, e.Message, e.Cause)
	}
	return fmt.Sprintf("task %s failed: %s", e.TaskID, e.Message)
}

// Unwrap returns the underlying cause.
func (e *TaskError) Unwrap() error {
	return e.Cause
}

// NewTaskError creates a new TaskError.
func NewTaskError(taskID TaskID, message string, cause error) *TaskError {
	return &TaskError{
		TaskID:  taskID,
		Message: message,
		Cause:   cause,
	}
}

// ActorError represents an error that occurred during actor execution.
type ActorError struct {
	ActorID ActorID
	Method  string
	Message string
	Cause   error
}

// Error implements the error interface.
func (e *ActorError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("actor %s method %s failed: %s: %v",
			e.ActorID, e.Method, e.Message, e.Cause)
	}
	return fmt.Sprintf("actor %s method %s failed: %s",
		e.ActorID, e.Method, e.Message)
}

// Unwrap returns the underlying cause.
func (e *ActorError) Unwrap() error {
	return e.Cause
}

// NewActorError creates a new ActorError.
func NewActorError(actorID ActorID, method, message string, cause error) *ActorError {
	return &ActorError{
		ActorID: actorID,
		Method:  method,
		Message: message,
		Cause:   cause,
	}
}

// WorkerError represents an error from a worker.
type WorkerError struct {
	WorkerID WorkerID
	Message  string
	Cause    error
}

// Error implements the error interface.
func (e *WorkerError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("worker %s error: %s: %v", e.WorkerID, e.Message, e.Cause)
	}
	return fmt.Sprintf("worker %s error: %s", e.WorkerID, e.Message)
}

// Unwrap returns the underlying cause.
func (e *WorkerError) Unwrap() error {
	return e.Cause
}

// IsRetryable returns true if the error is retryable.
//
// TODO: Implement this function
// - Check if error is a network error
// - Check if error is a temporary failure
// - ErrWorkerDead is retryable (reschedule on another worker)
// - ErrTimeout might be retryable
func IsRetryable(err error) bool {
	panic("TODO: implement IsRetryable")
}

// IsFatal returns true if the error is fatal and should stop execution.
//
// TODO: Implement this function
// - Serialization errors are usually fatal
// - User code errors are fatal
func IsFatal(err error) bool {
	panic("TODO: implement IsFatal")
}
