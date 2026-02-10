// Package scheduler provides task scheduling for Mini-Ray.
//
// The scheduler is responsible for:
// - Matching tasks to workers based on resources
// - Handling data locality (prefer workers with data)
// - Load balancing across workers
// - Handling worker failures
package scheduler

import (
	"context"
	"sync"

	"github.com/student/mini-ray/pkg/core"
	"github.com/student/mini-ray/pkg/task"
)

// Scheduler is the interface for task scheduling.
type Scheduler interface {
	// Schedule assigns a task to a worker.
	// Returns the assigned worker ID, or error if no worker available.
	Schedule(ctx context.Context, t *task.Task) (core.WorkerID, error)

	// RegisterWorker adds a worker to the scheduler.
	RegisterWorker(worker *WorkerInfo) error

	// UnregisterWorker removes a worker.
	UnregisterWorker(workerID core.WorkerID) error

	// UpdateWorker updates worker status/resources.
	UpdateWorker(workerID core.WorkerID, update WorkerUpdate) error

	// GetWorker returns info about a worker.
	GetWorker(workerID core.WorkerID) (*WorkerInfo, error)

	// ListWorkers returns all registered workers.
	ListWorkers() []*WorkerInfo

	// Stats returns scheduler statistics.
	Stats() SchedulerStats
}

// WorkerInfo holds information about a worker.
type WorkerInfo struct {
	ID              core.WorkerID
	Address         string
	Status          core.WorkerStatus
	TotalResources  core.Resources
	AvailableResources core.Resources
	RunningTasks    []core.TaskID
	RunningActors   []core.ActorID
	LastHeartbeat   int64 // Unix timestamp
	Labels          map[string]string
}

// WorkerUpdate is an update to worker state.
type WorkerUpdate struct {
	Status             *core.WorkerStatus
	AvailableResources *core.Resources
	LastHeartbeat      *int64
}

// SchedulerStats holds scheduling statistics.
type SchedulerStats struct {
	TotalWorkers     int
	HealthyWorkers   int
	TotalTasks       int64
	ScheduledTasks   int64
	FailedSchedules  int64
	AverageWaitTime  float64 // milliseconds
}

// SchedulerConfig configures the scheduler.
type SchedulerConfig struct {
	// HeartbeatTimeout is how long before a worker is considered dead
	HeartbeatTimeoutMs int64

	// SchedulingPolicy determines scheduling behavior
	Policy SchedulingPolicy

	// EnableLocality enables data locality awareness
	EnableLocality bool

	// LocalityWeight is the bonus for data locality (0-1)
	LocalityWeight float64
}

// SchedulingPolicy determines how tasks are assigned.
type SchedulingPolicy int

const (
	// PolicyFIFO schedules tasks in submission order
	PolicyFIFO SchedulingPolicy = iota

	// PolicyRandom randomly selects a capable worker
	PolicyRandom

	// PolicyLeastLoaded selects the worker with most available resources
	PolicyLeastLoaded

	// PolicyLocality prefers workers with task data
	PolicyLocality
)

// DefaultSchedulerConfig returns sensible defaults.
func DefaultSchedulerConfig() SchedulerConfig {
	return SchedulerConfig{
		HeartbeatTimeoutMs: 10000, // 10 seconds
		Policy:             PolicyLeastLoaded,
		EnableLocality:     true,
		LocalityWeight:     0.5,
	}
}

// SimpleScheduler is a basic FIFO scheduler.
type SimpleScheduler struct {
	mu sync.RWMutex

	// workers maps WorkerID to WorkerInfo
	workers map[core.WorkerID]*WorkerInfo

	// config
	config SchedulerConfig

	// stats
	totalTasks      int64
	scheduledTasks  int64
	failedSchedules int64

	// objectLocations tracks which workers have which objects
	objectLocations map[core.ObjectID][]core.WorkerID
}

// NewSimpleScheduler creates a new simple scheduler.
//
// TODO: Implement this function
func NewSimpleScheduler(config SchedulerConfig) *SimpleScheduler {
	panic("TODO: implement NewSimpleScheduler")
}

// Schedule assigns a task to a worker.
//
// TODO: Implement this function
// Algorithm:
// 1. Filter workers that have enough resources
// 2. If locality enabled, prefer workers with task data
// 3. Apply scheduling policy to select worker
// 4. Reserve resources on selected worker
func (s *SimpleScheduler) Schedule(ctx context.Context, t *task.Task) (core.WorkerID, error) {
	panic("TODO: implement SimpleScheduler.Schedule")
}

// RegisterWorker adds a worker.
//
// TODO: Implement this function
func (s *SimpleScheduler) RegisterWorker(worker *WorkerInfo) error {
	panic("TODO: implement SimpleScheduler.RegisterWorker")
}

// UnregisterWorker removes a worker.
//
// TODO: Implement this function
// - Remove from workers map
// - Tasks on this worker need to be rescheduled (handled elsewhere)
func (s *SimpleScheduler) UnregisterWorker(workerID core.WorkerID) error {
	panic("TODO: implement SimpleScheduler.UnregisterWorker")
}

// UpdateWorker updates worker info.
//
// TODO: Implement this function
func (s *SimpleScheduler) UpdateWorker(workerID core.WorkerID, update WorkerUpdate) error {
	panic("TODO: implement SimpleScheduler.UpdateWorker")
}

// GetWorker returns worker info.
//
// TODO: Implement this function
func (s *SimpleScheduler) GetWorker(workerID core.WorkerID) (*WorkerInfo, error) {
	panic("TODO: implement SimpleScheduler.GetWorker")
}

// ListWorkers returns all workers.
//
// TODO: Implement this function
func (s *SimpleScheduler) ListWorkers() []*WorkerInfo {
	panic("TODO: implement SimpleScheduler.ListWorkers")
}

// Stats returns scheduler statistics.
func (s *SimpleScheduler) Stats() SchedulerStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	healthy := 0
	for _, w := range s.workers {
		if w.Status == core.WorkerStatusIdle || w.Status == core.WorkerStatusBusy {
			healthy++
		}
	}

	return SchedulerStats{
		TotalWorkers:    len(s.workers),
		HealthyWorkers:  healthy,
		TotalTasks:      s.totalTasks,
		ScheduledTasks:  s.scheduledTasks,
		FailedSchedules: s.failedSchedules,
	}
}

// filterWorkers returns workers that can handle the task.
//
// TODO: Implement this function
// - Check resource availability
// - Check worker status (must be healthy)
func (s *SimpleScheduler) filterWorkers(t *task.Task) []*WorkerInfo {
	panic("TODO: implement SimpleScheduler.filterWorkers")
}

// selectWorker selects a worker from candidates using the policy.
//
// TODO: Implement this function
func (s *SimpleScheduler) selectWorker(candidates []*WorkerInfo, t *task.Task) *WorkerInfo {
	panic("TODO: implement SimpleScheduler.selectWorker")
}

// RegisterObjectLocation records that a worker has an object.
//
// TODO: Implement this function
func (s *SimpleScheduler) RegisterObjectLocation(objectID core.ObjectID, workerID core.WorkerID) {
	panic("TODO: implement SimpleScheduler.RegisterObjectLocation")
}

// GetObjectLocations returns workers that have an object.
//
// TODO: Implement this function
func (s *SimpleScheduler) GetObjectLocations(objectID core.ObjectID) []core.WorkerID {
	panic("TODO: implement SimpleScheduler.GetObjectLocations")
}

// Ensure SimpleScheduler implements Scheduler.
var _ Scheduler = (*SimpleScheduler)(nil)
