// Package vector provides data batches for vectorized execution.
package vector

// DataChunk is a collection of vectors (columns) with the same length.
// This is the unit of data transfer between operators.
type DataChunk struct {
	// Columns are the data vectors
	Columns []*Vector

	// Count is the number of rows
	Count int

	// Capacity is the maximum number of rows
	Capacity int
}

// NewDataChunk creates a new data chunk with given column types.
//
// TODO: Implement this function
func NewDataChunk(types []DataType, capacity int) *DataChunk {
	panic("TODO: implement NewDataChunk")
}

// Reset clears the chunk for reuse.
//
// TODO: Implement this function
func (c *DataChunk) Reset() {
	panic("TODO: implement DataChunk.Reset")
}

// SetCount sets the row count.
func (c *DataChunk) SetCount(count int) {
	c.Count = count
}

// GetColumn returns a column by index.
func (c *DataChunk) GetColumn(idx int) *Vector {
	return c.Columns[idx]
}

// ColumnCount returns the number of columns.
func (c *DataChunk) ColumnCount() int {
	return len(c.Columns)
}

// Append appends a row to the chunk.
//
// TODO: Implement this function
// values is a slice of values, one per column
func (c *DataChunk) Append(values []interface{}) error {
	panic("TODO: implement DataChunk.Append")
}

// Slice creates a new chunk from selected rows.
//
// TODO: Implement this function
func (c *DataChunk) Slice(sel *SelectionVector) *DataChunk {
	panic("TODO: implement DataChunk.Slice")
}

// Reference creates a reference to another chunk (zero-copy).
//
// TODO: Implement this function
func (c *DataChunk) Reference(other *DataChunk) {
	panic("TODO: implement DataChunk.Reference")
}

// Copy copies data from another chunk.
//
// TODO: Implement this function
func (c *DataChunk) Copy(other *DataChunk) {
	panic("TODO: implement DataChunk.Copy")
}

// Print prints the chunk for debugging.
//
// TODO: Implement this function
func (c *DataChunk) Print() string {
	panic("TODO: implement DataChunk.Print")
}

// ChunkCollection is a collection of data chunks.
type ChunkCollection struct {
	// Chunks are the stored chunks
	Chunks []*DataChunk

	// Types are the column types
	Types []DataType

	// TotalCount is the total number of rows
	TotalCount int
}

// NewChunkCollection creates a new chunk collection.
//
// TODO: Implement this function
func NewChunkCollection(types []DataType) *ChunkCollection {
	panic("TODO: implement NewChunkCollection")
}

// Append appends a chunk to the collection.
//
// TODO: Implement this function
func (cc *ChunkCollection) Append(chunk *DataChunk) {
	panic("TODO: implement ChunkCollection.Append")
}

// GetChunk returns chunk at index.
func (cc *ChunkCollection) GetChunk(idx int) *DataChunk {
	return cc.Chunks[idx]
}

// ChunkCount returns the number of chunks.
func (cc *ChunkCollection) ChunkCount() int {
	return len(cc.Chunks)
}

// Iterator creates an iterator over all rows.
//
// TODO: Implement this function
func (cc *ChunkCollection) Iterator() *ChunkIterator {
	panic("TODO: implement ChunkCollection.Iterator")
}

// ChunkIterator iterates over rows in a chunk collection.
type ChunkIterator struct {
	collection *ChunkCollection
	chunkIdx   int
	rowIdx     int
}

// Next returns the next row.
//
// TODO: Implement this function
// Returns (values, valid) where valid is false at end
func (it *ChunkIterator) Next() ([]interface{}, bool) {
	panic("TODO: implement ChunkIterator.Next")
}

// Reset resets the iterator to the beginning.
func (it *ChunkIterator) Reset() {
	it.chunkIdx = 0
	it.rowIdx = 0
}

// ColumnBindings maps column names to indices.
type ColumnBindings struct {
	// Names maps name to column index
	Names map[string]int

	// Types are the column types
	Types []DataType
}

// NewColumnBindings creates new column bindings.
func NewColumnBindings() *ColumnBindings {
	return &ColumnBindings{
		Names: make(map[string]int),
		Types: make([]DataType, 0),
	}
}

// Add adds a column binding.
func (b *ColumnBindings) Add(name string, typ DataType) int {
	idx := len(b.Types)
	b.Names[name] = idx
	b.Types = append(b.Types, typ)
	return idx
}

// GetIndex returns the index for a column name.
func (b *ColumnBindings) GetIndex(name string) (int, bool) {
	idx, ok := b.Names[name]
	return idx, ok
}

// GetType returns the type for a column index.
func (b *ColumnBindings) GetType(idx int) DataType {
	return b.Types[idx]
}
