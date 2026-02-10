// Package storage provides columnar storage for Mini-DuckDB.
package storage

import (
	"github.com/student/mini-duckdb/pkg/vector"
)

// ColumnSegment is a contiguous segment of column data.
type ColumnSegment struct {
	// Type is the data type
	Type vector.DataType

	// Data is the raw data
	Data []byte

	// Validity is the null bitmap
	Validity []uint64

	// Count is the number of values
	Count int

	// Compression is the compression type
	Compression CompressionType

	// Stats are statistics for this segment
	Stats *ColumnStats
}

// CompressionType identifies compression algorithm.
type CompressionType uint8

const (
	CompressionNone CompressionType = iota
	CompressionRLE
	CompressionDictionary
	CompressionDelta
	CompressionBitpacking
)

// ColumnStats holds statistics about a column segment.
type ColumnStats struct {
	// NullCount is the number of null values
	NullCount int

	// MinValue is the minimum value (for numeric types)
	MinValue interface{}

	// MaxValue is the maximum value (for numeric types)
	MaxValue interface{}

	// DistinctCount is the approximate number of distinct values
	DistinctCount int

	// HasNull indicates if segment contains nulls
	HasNull bool
}

// NewColumnSegment creates a new column segment.
//
// TODO: Implement this function
func NewColumnSegment(typ vector.DataType, capacity int) *ColumnSegment {
	panic("TODO: implement NewColumnSegment")
}

// Append appends values to the segment.
//
// TODO: Implement this function
func (s *ColumnSegment) Append(vec *vector.Vector) error {
	panic("TODO: implement ColumnSegment.Append")
}

// ToVector converts the segment to a vector.
//
// TODO: Implement this function
func (s *ColumnSegment) ToVector() *vector.Vector {
	panic("TODO: implement ColumnSegment.ToVector")
}

// Scan reads values from the segment.
//
// TODO: Implement this function
func (s *ColumnSegment) Scan(start, count int) *vector.Vector {
	panic("TODO: implement ColumnSegment.Scan")
}

// Compress compresses the segment.
//
// TODO: Implement this function
// Chooses best compression based on data characteristics
func (s *ColumnSegment) Compress() error {
	panic("TODO: implement ColumnSegment.Compress")
}

// Decompress decompresses the segment.
//
// TODO: Implement this function
func (s *ColumnSegment) Decompress() error {
	panic("TODO: implement ColumnSegment.Decompress")
}

// ComputeStats computes statistics for the segment.
//
// TODO: Implement this function
func (s *ColumnSegment) ComputeStats() *ColumnStats {
	panic("TODO: implement ColumnSegment.ComputeStats")
}

// ColumnChunk is a collection of segments forming a column.
type ColumnChunk struct {
	// Name is the column name
	Name string

	// Type is the data type
	Type vector.DataType

	// Segments are the column segments
	Segments []*ColumnSegment

	// TotalCount is the total number of values
	TotalCount int
}

// NewColumnChunk creates a new column chunk.
//
// TODO: Implement this function
func NewColumnChunk(name string, typ vector.DataType) *ColumnChunk {
	panic("TODO: implement NewColumnChunk")
}

// Append appends values to the column.
//
// TODO: Implement this function
func (c *ColumnChunk) Append(vec *vector.Vector) error {
	panic("TODO: implement ColumnChunk.Append")
}

// Scan reads values from the column.
//
// TODO: Implement this function
func (c *ColumnChunk) Scan(start, count int) *vector.Vector {
	panic("TODO: implement ColumnChunk.Scan")
}

// ScanAll reads all values from the column.
//
// TODO: Implement this function
func (c *ColumnChunk) ScanAll() *vector.Vector {
	panic("TODO: implement ColumnChunk.ScanAll")
}
