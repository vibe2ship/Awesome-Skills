// Package storage provides table storage for Mini-DuckDB.
package storage

import (
	"errors"
	"sync"

	"github.com/student/mini-duckdb/pkg/vector"
)

var (
	// ErrColumnNotFound indicates column doesn't exist.
	ErrColumnNotFound = errors.New("column not found")

	// ErrTableNotFound indicates table doesn't exist.
	ErrTableNotFound = errors.New("table not found")
)

// Table represents a columnar table.
type Table struct {
	// Name is the table name
	Name string

	// Columns are the column chunks
	Columns []*ColumnChunk

	// Schema describes the table structure
	Schema *TableSchema

	// RowCount is the number of rows
	RowCount int

	mu sync.RWMutex
}

// TableSchema describes a table's structure.
type TableSchema struct {
	// Name is the table name
	Name string

	// Columns are the column definitions
	Columns []ColumnDef
}

// ColumnDef defines a column.
type ColumnDef struct {
	Name     string
	Type     vector.DataType
	Nullable bool
}

// NewTable creates a new table.
//
// TODO: Implement this function
func NewTable(schema *TableSchema) *Table {
	panic("TODO: implement NewTable")
}

// Insert inserts a data chunk into the table.
//
// TODO: Implement this function
func (t *Table) Insert(chunk *vector.DataChunk) error {
	panic("TODO: implement Table.Insert")
}

// Scan scans all rows from the table.
//
// TODO: Implement this function
// Returns chunks one at a time
func (t *Table) Scan(columns []int) ([]*vector.DataChunk, error) {
	panic("TODO: implement Table.Scan")
}

// ScanRange scans a range of rows.
//
// TODO: Implement this function
func (t *Table) ScanRange(start, count int, columns []int) (*vector.DataChunk, error) {
	panic("TODO: implement Table.ScanRange")
}

// GetColumn returns a column by name.
//
// TODO: Implement this function
func (t *Table) GetColumn(name string) (*ColumnChunk, error) {
	panic("TODO: implement Table.GetColumn")
}

// GetColumnIndex returns the index of a column.
//
// TODO: Implement this function
func (t *Table) GetColumnIndex(name string) (int, error) {
	panic("TODO: implement Table.GetColumnIndex")
}

// GetColumnTypes returns all column types.
//
// TODO: Implement this function
func (t *Table) GetColumnTypes() []vector.DataType {
	panic("TODO: implement Table.GetColumnTypes")
}

// Catalog manages database objects (tables).
type Catalog struct {
	// Tables maps table name to table
	Tables map[string]*Table

	mu sync.RWMutex
}

// NewCatalog creates a new catalog.
func NewCatalog() *Catalog {
	return &Catalog{
		Tables: make(map[string]*Table),
	}
}

// CreateTable creates a new table.
//
// TODO: Implement this function
func (c *Catalog) CreateTable(schema *TableSchema) (*Table, error) {
	panic("TODO: implement Catalog.CreateTable")
}

// DropTable drops a table.
//
// TODO: Implement this function
func (c *Catalog) DropTable(name string) error {
	panic("TODO: implement Catalog.DropTable")
}

// GetTable returns a table by name.
//
// TODO: Implement this function
func (c *Catalog) GetTable(name string) (*Table, error) {
	panic("TODO: implement Catalog.GetTable")
}

// ListTables returns all table names.
//
// TODO: Implement this function
func (c *Catalog) ListTables() []string {
	panic("TODO: implement Catalog.ListTables")
}

// DataFile represents a columnar data file.
type DataFile struct {
	// Path is the file path
	Path string

	// Schema is the file schema
	Schema *TableSchema

	// RowGroups are the row groups in the file
	RowGroups []*RowGroup

	// Metadata is file metadata
	Metadata *FileMetadata
}

// RowGroup is a horizontal partition of data.
type RowGroup struct {
	// Columns are the column chunks
	Columns []*ColumnChunk

	// RowCount is the number of rows
	RowCount int

	// Offset is the file offset
	Offset int64

	// Size is the size in bytes
	Size int64
}

// FileMetadata holds file-level metadata.
type FileMetadata struct {
	// Version is the file format version
	Version int

	// RowCount is the total number of rows
	RowCount int64

	// NumRowGroups is the number of row groups
	NumRowGroups int

	// Schema is the file schema
	Schema *TableSchema
}

// WriteDataFile writes a table to a data file.
//
// TODO: Implement this function
func WriteDataFile(path string, table *Table) error {
	panic("TODO: implement WriteDataFile")
}

// ReadDataFile reads a data file into a table.
//
// TODO: Implement this function
func ReadDataFile(path string) (*Table, error) {
	panic("TODO: implement ReadDataFile")
}
