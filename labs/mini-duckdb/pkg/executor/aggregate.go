// Package executor provides aggregation operators.
package executor

import (
	"github.com/student/mini-duckdb/pkg/vector"
)

// HashAggregate implements hash-based aggregation.
type HashAggregate struct {
	// Child is the input operator
	Child Operator

	// GroupBy are the group-by column indices
	GroupBy []int

	// Aggregates are the aggregate expressions
	Aggregates []*AggregateExpr

	// Types are the output column types
	Types []vector.DataType

	// groups maps group key hash to group state
	groups map[uint64]*groupState

	// finished indicates if input is exhausted
	finished bool

	// output is the materialized output
	output *vector.ChunkCollection

	// outputChunk is current output chunk index
	outputChunk int
}

// groupState holds state for one group.
type groupState struct {
	// Key is the group key values
	Key []interface{}

	// States are the aggregate states
	States []vector.AggregateState
}

// NewHashAggregate creates a new hash aggregate operator.
//
// TODO: Implement this function
func NewHashAggregate(child Operator, groupBy []int, aggregates []*AggregateExpr) *HashAggregate {
	panic("TODO: implement NewHashAggregate")
}

// Init implements Operator.
func (a *HashAggregate) Init() error {
	a.groups = make(map[uint64]*groupState)
	a.finished = false
	a.output = nil
	a.outputChunk = 0
	return a.Child.Init()
}

// Next implements Operator.
//
// TODO: Implement this function
// 1. If not finished, consume all input and build hash table
// 2. Output groups one chunk at a time
func (a *HashAggregate) Next() (*vector.DataChunk, error) {
	panic("TODO: implement HashAggregate.Next")
}

// consumeInput reads all input and populates groups.
//
// TODO: Implement this function
func (a *HashAggregate) consumeInput() error {
	panic("TODO: implement HashAggregate.consumeInput")
}

// processChunk processes one input chunk.
//
// TODO: Implement this function
// 1. For each row, compute group key hash
// 2. Find or create group state
// 3. Update aggregate states
func (a *HashAggregate) processChunk(chunk *vector.DataChunk) {
	panic("TODO: implement HashAggregate.processChunk")
}

// materializeOutput creates output chunks from groups.
//
// TODO: Implement this function
func (a *HashAggregate) materializeOutput() {
	panic("TODO: implement HashAggregate.materializeOutput")
}

// Close implements Operator.
func (a *HashAggregate) Close() error {
	return a.Child.Close()
}

// GetTypes implements Operator.
func (a *HashAggregate) GetTypes() []vector.DataType {
	return a.Types
}

// SimpleAggregate computes aggregates without grouping.
type SimpleAggregate struct {
	// Child is the input operator
	Child Operator

	// Aggregates are the aggregate expressions
	Aggregates []*AggregateExpr

	// Types are the output column types
	Types []vector.DataType

	// states are the aggregate states
	states []vector.AggregateState

	// finished indicates if result has been returned
	finished bool
}

// NewSimpleAggregate creates a new simple aggregate operator.
//
// TODO: Implement this function
func NewSimpleAggregate(child Operator, aggregates []*AggregateExpr) *SimpleAggregate {
	panic("TODO: implement NewSimpleAggregate")
}

// Init implements Operator.
func (a *SimpleAggregate) Init() error {
	a.finished = false
	a.states = make([]vector.AggregateState, len(a.Aggregates))
	for i, agg := range a.Aggregates {
		a.states[i] = NewAggregateState(agg.Function)
		a.states[i].Init()
	}
	return a.Child.Init()
}

// Next implements Operator.
//
// TODO: Implement this function
// 1. Consume all input
// 2. Return single row with aggregate results
func (a *SimpleAggregate) Next() (*vector.DataChunk, error) {
	panic("TODO: implement SimpleAggregate.Next")
}

// Close implements Operator.
func (a *SimpleAggregate) Close() error {
	return a.Child.Close()
}

// GetTypes implements Operator.
func (a *SimpleAggregate) GetTypes() []vector.DataType {
	return a.Types
}

// computeGroupKey computes the hash for group-by columns.
func computeGroupKey(chunk *vector.DataChunk, row int, groupBy []int) uint64 {
	hash := uint64(0)
	for _, col := range groupBy {
		h := chunk.GetColumn(col).Hash(row)
		hash = hash*31 + h
	}
	return hash
}

// extractGroupKey extracts group-by values for a row.
func extractGroupKey(chunk *vector.DataChunk, row int, groupBy []int) []interface{} {
	key := make([]interface{}, len(groupBy))
	for i, col := range groupBy {
		vec := chunk.GetColumn(col)
		if vec.IsNull(row) {
			key[i] = nil
		} else {
			switch vec.Type {
			case vector.TypeInt64:
				key[i] = vec.GetInt64(row)
			case vector.TypeFloat64:
				key[i] = vec.GetFloat64(row)
			case vector.TypeString:
				key[i] = vec.GetString(row)
			case vector.TypeBoolean:
				key[i] = vec.GetBool(row)
			}
		}
	}
	return key
}
