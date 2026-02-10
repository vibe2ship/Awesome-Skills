// Package executor provides table scan operator.
package executor

import (
	"github.com/student/mini-duckdb/pkg/storage"
	"github.com/student/mini-duckdb/pkg/vector"
)

// TableScan scans a table and produces data chunks.
type TableScan struct {
	// Table is the table to scan
	Table *storage.Table

	// Columns are the columns to scan (nil = all)
	Columns []int

	// Types are the output column types
	Types []vector.DataType

	// currentRow is the current position
	currentRow int

	// chunkSize is the chunk size
	chunkSize int
}

// NewTableScan creates a new table scan operator.
//
// TODO: Implement this function
func NewTableScan(table *storage.Table, columns []int) *TableScan {
	panic("TODO: implement NewTableScan")
}

// Init implements Operator.
func (s *TableScan) Init() error {
	s.currentRow = 0
	return nil
}

// Next implements Operator.
//
// TODO: Implement this function
// 1. Read next chunk of rows from table
// 2. Project to requested columns
// 3. Return chunk
func (s *TableScan) Next() (*vector.DataChunk, error) {
	panic("TODO: implement TableScan.Next")
}

// Close implements Operator.
func (s *TableScan) Close() error {
	return nil
}

// GetTypes implements Operator.
func (s *TableScan) GetTypes() []vector.DataType {
	return s.Types
}

// ChunkScan scans from an in-memory chunk collection.
type ChunkScan struct {
	// Source is the chunk collection to scan
	Source *vector.ChunkCollection

	// Types are the output types
	Types []vector.DataType

	// currentChunk is the current chunk index
	currentChunk int
}

// NewChunkScan creates a new chunk scan operator.
//
// TODO: Implement this function
func NewChunkScan(source *vector.ChunkCollection) *ChunkScan {
	panic("TODO: implement NewChunkScan")
}

// Init implements Operator.
func (s *ChunkScan) Init() error {
	s.currentChunk = 0
	return nil
}

// Next implements Operator.
//
// TODO: Implement this function
func (s *ChunkScan) Next() (*vector.DataChunk, error) {
	panic("TODO: implement ChunkScan.Next")
}

// Close implements Operator.
func (s *ChunkScan) Close() error {
	return nil
}

// GetTypes implements Operator.
func (s *ChunkScan) GetTypes() []vector.DataType {
	return s.Types
}

// ValuesScan returns constant values.
type ValuesScan struct {
	// Values are the rows to return
	Values [][]interface{}

	// Types are the column types
	Types []vector.DataType

	// returned tracks if values have been returned
	returned bool
}

// NewValuesScan creates a new values scan.
//
// TODO: Implement this function
func NewValuesScan(values [][]interface{}, types []vector.DataType) *ValuesScan {
	panic("TODO: implement NewValuesScan")
}

// Init implements Operator.
func (s *ValuesScan) Init() error {
	s.returned = false
	return nil
}

// Next implements Operator.
//
// TODO: Implement this function
func (s *ValuesScan) Next() (*vector.DataChunk, error) {
	panic("TODO: implement ValuesScan.Next")
}

// Close implements Operator.
func (s *ValuesScan) Close() error {
	return nil
}

// GetTypes implements Operator.
func (s *ValuesScan) GetTypes() []vector.DataType {
	return s.Types
}
