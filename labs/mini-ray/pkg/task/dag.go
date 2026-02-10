// Package task provides DAG (Directed Acyclic Graph) tracking for task dependencies.
package task

import (
	"sync"

	"github.com/student/mini-ray/pkg/core"
)

// DAG tracks task dependencies and execution order.
//
// In Mini-Ray, tasks can depend on other tasks via ObjectRefs.
// The DAG ensures:
// - Tasks run only when dependencies are ready
// - Cycles are detected and rejected
// - Ready tasks can be efficiently queried
type DAG struct {
	mu sync.RWMutex

	// tasks maps TaskID to Task
	tasks map[core.TaskID]*Task

	// dependencies maps TaskID to its dependency ObjectIDs
	dependencies map[core.TaskID][]core.ObjectID

	// dependents maps ObjectID to tasks waiting for it
	dependents map[core.ObjectID][]core.TaskID

	// resolvedObjects tracks which objects are ready
	resolvedObjects map[core.ObjectID]bool

	// readyTasks are tasks with all dependencies resolved
	readyTasks map[core.TaskID]bool
}

// NewDAG creates a new DAG.
//
// TODO: Implement this function
func NewDAG() *DAG {
	panic("TODO: implement NewDAG")
}

// AddTask adds a task to the DAG.
//
// TODO: Implement this function
// - Add task to tasks map
// - Record dependencies
// - Add to dependents map for each dependency
// - Check if task is immediately ready (no deps or all deps resolved)
func (d *DAG) AddTask(task *Task) error {
	panic("TODO: implement DAG.AddTask")
}

// RemoveTask removes a task from the DAG.
//
// TODO: Implement this function
// - Remove from tasks map
// - Clean up dependency tracking
// - Remove from ready set
func (d *DAG) RemoveTask(taskID core.TaskID) error {
	panic("TODO: implement DAG.RemoveTask")
}

// ResolveObject marks an object as resolved and updates ready tasks.
//
// This is called when:
// - A task completes and its output is available
// - An object is put directly into the store
//
// TODO: Implement this function
// - Mark object as resolved
// - For each dependent task, check if now ready
// - Add newly ready tasks to readyTasks
func (d *DAG) ResolveObject(objectID core.ObjectID) []core.TaskID {
	panic("TODO: implement DAG.ResolveObject")
}

// GetReadyTasks returns all tasks ready for execution.
//
// TODO: Implement this function
func (d *DAG) GetReadyTasks() []*Task {
	panic("TODO: implement DAG.GetReadyTasks")
}

// PopReadyTask removes and returns a ready task.
//
// TODO: Implement this function
// - Return nil if no ready tasks
// - Remove from ready set
func (d *DAG) PopReadyTask() *Task {
	panic("TODO: implement DAG.PopReadyTask")
}

// IsReady returns true if the task is ready to execute.
//
// TODO: Implement this function
// - Check if all dependencies are resolved
func (d *DAG) IsReady(taskID core.TaskID) bool {
	panic("TODO: implement DAG.IsReady")
}

// GetTask returns the task with the given ID.
//
// TODO: Implement this function
func (d *DAG) GetTask(taskID core.TaskID) (*Task, bool) {
	panic("TODO: implement DAG.GetTask")
}

// PendingCount returns the number of pending tasks.
func (d *DAG) PendingCount() int {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return len(d.tasks) - len(d.readyTasks)
}

// ReadyCount returns the number of ready tasks.
func (d *DAG) ReadyCount() int {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return len(d.readyTasks)
}

// TotalCount returns the total number of tasks.
func (d *DAG) TotalCount() int {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return len(d.tasks)
}

// GetDependencies returns the dependencies of a task.
//
// TODO: Implement this function
func (d *DAG) GetDependencies(taskID core.TaskID) []core.ObjectID {
	panic("TODO: implement DAG.GetDependencies")
}

// GetDependents returns tasks that depend on an object.
//
// TODO: Implement this function
func (d *DAG) GetDependents(objectID core.ObjectID) []core.TaskID {
	panic("TODO: implement DAG.GetDependents")
}

// HasCycle checks if adding the given dependencies would create a cycle.
// This is called before adding a task.
//
// TODO: Implement this function (optional, advanced)
// - Use DFS to detect cycles
// - Track the task -> output object -> dependent task chain
func (d *DAG) HasCycle(taskID core.TaskID, deps []core.ObjectID) bool {
	// For simplicity, assume no cycles in basic implementation
	return false
}

// TopologicalOrder returns tasks in topological order.
// Useful for debugging and visualization.
//
// TODO: Implement this function (optional)
// - Use Kahn's algorithm or DFS
func (d *DAG) TopologicalOrder() []*Task {
	panic("TODO: implement DAG.TopologicalOrder")
}
