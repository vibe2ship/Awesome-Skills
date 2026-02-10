// Package executor provides vectorized query execution.
package executor

import (
	"errors"

	"github.com/student/mini-duckdb/pkg/vector"
)

var (
	// ErrEndOfData indicates no more data available.
	ErrEndOfData = errors.New("end of data")
)

// Operator is the interface for query operators.
type Operator interface {
	// Init initializes the operator
	Init() error

	// Next returns the next chunk of data
	// Returns nil when no more data
	Next() (*vector.DataChunk, error)

	// Close releases resources
	Close() error

	// GetTypes returns output column types
	GetTypes() []vector.DataType
}

// ExecutionContext provides context for query execution.
type ExecutionContext struct {
	// ChunkSize is the size of data chunks
	ChunkSize int

	// Parallelism is the number of parallel threads
	Parallelism int

	// Stats tracks execution statistics
	Stats *ExecutionStats
}

// ExecutionStats tracks query execution statistics.
type ExecutionStats struct {
	RowsProcessed   int64
	ChunksProcessed int64
	BytesProcessed  int64
	ExecutionTimeMs int64
}

// NewExecutionContext creates a new execution context.
func NewExecutionContext() *ExecutionContext {
	return &ExecutionContext{
		ChunkSize:   vector.VectorSize,
		Parallelism: 1,
		Stats:       &ExecutionStats{},
	}
}

// Pipeline represents a query execution pipeline.
type Pipeline struct {
	// Source is the data source operator
	Source Operator

	// Operators are the pipeline operators
	Operators []Operator

	// Sink receives the final results
	Sink ResultSink
}

// ResultSink receives query results.
type ResultSink interface {
	// Init initializes the sink
	Init(types []vector.DataType) error

	// Sink receives a data chunk
	Sink(chunk *vector.DataChunk) error

	// Finalize completes the sink
	Finalize() error
}

// Execute executes the pipeline.
//
// TODO: Implement this function
// 1. Initialize all operators
// 2. Pull chunks through pipeline
// 3. Push to sink
func (p *Pipeline) Execute() error {
	panic("TODO: implement Pipeline.Execute")
}

// MaterializeSink collects all results in memory.
type MaterializeSink struct {
	// Results are the collected chunks
	Results *vector.ChunkCollection

	// Types are the column types
	Types []vector.DataType
}

// Init implements ResultSink.
func (s *MaterializeSink) Init(types []vector.DataType) error {
	s.Types = types
	s.Results = vector.NewChunkCollection(types)
	return nil
}

// Sink implements ResultSink.
//
// TODO: Implement this function
func (s *MaterializeSink) Sink(chunk *vector.DataChunk) error {
	panic("TODO: implement MaterializeSink.Sink")
}

// Finalize implements ResultSink.
func (s *MaterializeSink) Finalize() error {
	return nil
}

// PrintSink prints results to stdout.
type PrintSink struct {
	Types []vector.DataType
	Count int64
}

// Init implements ResultSink.
func (s *PrintSink) Init(types []vector.DataType) error {
	s.Types = types
	return nil
}

// Sink implements ResultSink.
//
// TODO: Implement this function
func (s *PrintSink) Sink(chunk *vector.DataChunk) error {
	panic("TODO: implement PrintSink.Sink")
}

// Finalize implements ResultSink.
func (s *PrintSink) Finalize() error {
	return nil
}

// Expression represents a computed expression.
type Expression interface {
	// Evaluate evaluates the expression on a chunk
	Evaluate(chunk *vector.DataChunk) *vector.Vector

	// GetType returns the result type
	GetType() vector.DataType
}

// ColumnRefExpr references a column.
type ColumnRefExpr struct {
	Index int
	Type  vector.DataType
}

// Evaluate implements Expression.
func (e *ColumnRefExpr) Evaluate(chunk *vector.DataChunk) *vector.Vector {
	return chunk.GetColumn(e.Index)
}

// GetType implements Expression.
func (e *ColumnRefExpr) GetType() vector.DataType {
	return e.Type
}

// ConstantExpr is a constant value.
type ConstantExpr struct {
	Value interface{}
	Type  vector.DataType
}

// Evaluate implements Expression.
//
// TODO: Implement this function
// Create a vector filled with the constant value
func (e *ConstantExpr) Evaluate(chunk *vector.DataChunk) *vector.Vector {
	panic("TODO: implement ConstantExpr.Evaluate")
}

// GetType implements Expression.
func (e *ConstantExpr) GetType() vector.DataType {
	return e.Type
}

// BinaryExpr is a binary operation.
type BinaryExpr struct {
	Left  Expression
	Right Expression
	Op    vector.ArithOp
	Type  vector.DataType
}

// Evaluate implements Expression.
//
// TODO: Implement this function
func (e *BinaryExpr) Evaluate(chunk *vector.DataChunk) *vector.Vector {
	panic("TODO: implement BinaryExpr.Evaluate")
}

// GetType implements Expression.
func (e *BinaryExpr) GetType() vector.DataType {
	return e.Type
}

// CompareExpr is a comparison operation.
type CompareExpr struct {
	Left  Expression
	Right Expression
	Op    vector.CompareOp
}

// Evaluate implements Expression.
//
// TODO: Implement this function
func (e *CompareExpr) Evaluate(chunk *vector.DataChunk) *vector.Vector {
	panic("TODO: implement CompareExpr.Evaluate")
}

// GetType implements Expression.
func (e *CompareExpr) GetType() vector.DataType {
	return vector.TypeBoolean
}

// AggregateExpr is an aggregate function.
type AggregateExpr struct {
	Function AggFunc
	Input    Expression
	Type     vector.DataType
}

// AggFunc identifies aggregate functions.
type AggFunc int

const (
	AggSum AggFunc = iota
	AggCount
	AggAvg
	AggMin
	AggMax
)

// NewAggregateState creates state for an aggregate function.
//
// TODO: Implement this function
func NewAggregateState(fn AggFunc) vector.AggregateState {
	panic("TODO: implement NewAggregateState")
}
