// Package actor provides the actor system for Mini-Ray.
//
// Actors are stateful, distributed objects. Unlike tasks:
// - Actors maintain state between method calls
// - Method calls are serialized (one at a time)
// - Actors run on a specific worker
//
// Example:
//
//	type Counter struct {
//	    value int
//	}
//
//	func (c *Counter) Increment() int {
//	    c.value++
//	    return c.value
//	}
//
//	// Create actor
//	handle := ray.CreateActor(&Counter{})
//
//	// Call method
//	future := handle.Call("Increment")
//	result := ray.Get(future)
package actor

import (
	"context"
	"reflect"
	"sync"

	"github.com/student/mini-ray/pkg/core"
)

// Actor is the interface that all actors must implement.
// In practice, any struct with methods can be an actor.
type Actor interface{}

// ActorInstance represents a running actor instance.
type ActorInstance struct {
	// ID uniquely identifies this actor
	ID core.ActorID

	// Name is the type name of the actor
	Name string

	// State is the actual actor object
	State Actor

	// Status of the actor
	Status core.ActorStatus

	// WorkerID is the worker running this actor
	WorkerID core.WorkerID

	// methods caches reflected methods
	methods map[string]reflect.Method

	// mu protects state access
	mu sync.Mutex

	// mailbox holds pending method calls
	mailbox chan *MethodCall

	// done signals actor shutdown
	done chan struct{}
}

// MethodCall represents a method call on an actor.
type MethodCall struct {
	// ID uniquely identifies this call
	ID core.TaskID

	// Method name to call
	Method string

	// Args are serialized arguments
	Args [][]byte

	// ResultID is where to store the result
	ResultID core.ObjectID

	// Response channel for the caller
	Response chan *MethodResult
}

// MethodResult is the result of a method call.
type MethodResult struct {
	// Value is the serialized return value
	Value []byte

	// Error if the call failed
	Error error
}

// ActorOptions configures actor creation.
type ActorOptions struct {
	// Name for the actor (defaults to type name)
	Name string

	// Resources required
	Resources core.Resources

	// MaxConcurrency limits concurrent method calls (default 1)
	MaxConcurrency int

	// Lifetime controls when the actor is garbage collected
	Lifetime ActorLifetime
}

// ActorLifetime determines when an actor is garbage collected.
type ActorLifetime int

const (
	// LifetimeDetached - actor lives until explicitly killed
	LifetimeDetached ActorLifetime = iota

	// LifetimeLinked - actor dies when creator dies
	LifetimeLinked
)

// DefaultActorOptions returns sensible defaults.
func DefaultActorOptions() ActorOptions {
	return ActorOptions{
		Resources: core.Resources{
			CPU:    1.0,
			Memory: 256 * 1024 * 1024,
		},
		MaxConcurrency: 1,
		Lifetime:       LifetimeDetached,
	}
}

// NewActorInstance creates a new actor instance.
//
// TODO: Implement this function
// - Generate ActorID
// - Reflect on actor to cache methods
// - Initialize mailbox
func NewActorInstance(actor Actor, opts ActorOptions) (*ActorInstance, error) {
	panic("TODO: implement NewActorInstance")
}

// Start begins processing method calls.
//
// TODO: Implement this function
// - Start goroutine to process mailbox
// - Handle method calls sequentially
func (a *ActorInstance) Start(ctx context.Context) {
	panic("TODO: implement ActorInstance.Start")
}

// Call invokes a method on the actor.
//
// TODO: Implement this function
// - Create MethodCall
// - Send to mailbox
// - Return future for result
func (a *ActorInstance) Call(method string, args [][]byte) (*core.ObjectRef, error) {
	panic("TODO: implement ActorInstance.Call")
}

// Stop shuts down the actor.
//
// TODO: Implement this function
// - Signal done
// - Drain mailbox
// - Clean up resources
func (a *ActorInstance) Stop() error {
	panic("TODO: implement ActorInstance.Stop")
}

// invokeMethod calls a method on the actor state.
//
// TODO: Implement this function
// - Get method from cache
// - Deserialize args
// - Use reflect.Call
// - Serialize result
func (a *ActorInstance) invokeMethod(call *MethodCall) *MethodResult {
	panic("TODO: implement ActorInstance.invokeMethod")
}

// GetMethod returns the reflected method for the given name.
//
// TODO: Implement this function
func (a *ActorInstance) GetMethod(name string) (reflect.Method, bool) {
	panic("TODO: implement ActorInstance.GetMethod")
}

// ListMethods returns all available method names.
//
// TODO: Implement this function
func (a *ActorInstance) ListMethods() []string {
	panic("TODO: implement ActorInstance.ListMethods")
}

// reflectMethods extracts all public methods from the actor.
//
// TODO: Implement this function
// - Use reflect.TypeOf to get methods
// - Filter to exported methods
func reflectMethods(actor Actor) map[string]reflect.Method {
	panic("TODO: implement reflectMethods")
}
