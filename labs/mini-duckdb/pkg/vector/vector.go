// Package vector provides vectorized data types for Mini-DuckDB.
package vector

import (
	"errors"
	"math"
)

const (
	// VectorSize is the standard vector size (SIMD-friendly).
	VectorSize = 1024
)

var (
	// ErrTypeMismatch indicates incompatible types.
	ErrTypeMismatch = errors.New("type mismatch")

	// ErrOutOfBounds indicates index out of bounds.
	ErrOutOfBounds = errors.New("index out of bounds")
)

// DataType represents the type of data in a vector.
type DataType uint8

const (
	TypeInvalid DataType = iota
	TypeBoolean
	TypeInt8
	TypeInt16
	TypeInt32
	TypeInt64
	TypeFloat32
	TypeFloat64
	TypeString
	TypeDate
	TypeTimestamp
)

// TypeSize returns the size in bytes for fixed-size types.
func TypeSize(t DataType) int {
	switch t {
	case TypeBoolean, TypeInt8:
		return 1
	case TypeInt16:
		return 2
	case TypeInt32, TypeFloat32, TypeDate:
		return 4
	case TypeInt64, TypeFloat64, TypeTimestamp:
		return 8
	default:
		return 0 // Variable size
	}
}

// Vector is a columnar data container.
//
// Layout:
// - Fixed-size types: contiguous array of values
// - Strings: offsets array + data buffer
// - Null values tracked via validity bitmap
type Vector struct {
	// Type is the data type
	Type DataType

	// Data is the value buffer
	Data []byte

	// Validity is the null bitmap (1 = valid, 0 = null)
	// Bit i corresponds to row i
	Validity []uint64

	// Length is the number of values
	Length int

	// For strings: offsets into Data
	Offsets []uint32
}

// NewVector creates a new vector of the given type and capacity.
//
// TODO: Implement this function
func NewVector(typ DataType, capacity int) *Vector {
	panic("TODO: implement NewVector")
}

// NewFlatVector creates a vector from existing data.
//
// TODO: Implement this function
func NewFlatVector(typ DataType, data []byte, length int) *Vector {
	panic("TODO: implement NewFlatVector")
}

// IsNull checks if a value is null.
//
// TODO: Implement this function
func (v *Vector) IsNull(idx int) bool {
	panic("TODO: implement Vector.IsNull")
}

// SetNull marks a value as null.
//
// TODO: Implement this function
func (v *Vector) SetNull(idx int) {
	panic("TODO: implement Vector.SetNull")
}

// SetValid marks a value as valid (not null).
//
// TODO: Implement this function
func (v *Vector) SetValid(idx int) {
	panic("TODO: implement Vector.SetValid")
}

// NullCount returns the number of null values.
//
// TODO: Implement this function
func (v *Vector) NullCount() int {
	panic("TODO: implement Vector.NullCount")
}

// GetInt64 returns the int64 value at index.
//
// TODO: Implement this function
func (v *Vector) GetInt64(idx int) int64 {
	panic("TODO: implement Vector.GetInt64")
}

// SetInt64 sets the int64 value at index.
//
// TODO: Implement this function
func (v *Vector) SetInt64(idx int, val int64) {
	panic("TODO: implement Vector.SetInt64")
}

// GetFloat64 returns the float64 value at index.
//
// TODO: Implement this function
func (v *Vector) GetFloat64(idx int) float64 {
	panic("TODO: implement Vector.GetFloat64")
}

// SetFloat64 sets the float64 value at index.
//
// TODO: Implement this function
func (v *Vector) SetFloat64(idx int, val float64) {
	panic("TODO: implement Vector.SetFloat64")
}

// GetBool returns the boolean value at index.
//
// TODO: Implement this function
func (v *Vector) GetBool(idx int) bool {
	panic("TODO: implement Vector.GetBool")
}

// SetBool sets the boolean value at index.
//
// TODO: Implement this function
func (v *Vector) SetBool(idx int, val bool) {
	panic("TODO: implement Vector.SetBool")
}

// GetString returns the string value at index.
//
// TODO: Implement this function
func (v *Vector) GetString(idx int) string {
	panic("TODO: implement Vector.GetString")
}

// SetString sets the string value at index.
//
// TODO: Implement this function
func (v *Vector) SetString(idx int, val string) {
	panic("TODO: implement Vector.SetString")
}

// AppendInt64 appends an int64 value.
//
// TODO: Implement this function
func (v *Vector) AppendInt64(val int64) {
	panic("TODO: implement Vector.AppendInt64")
}

// AppendFloat64 appends a float64 value.
//
// TODO: Implement this function
func (v *Vector) AppendFloat64(val float64) {
	panic("TODO: implement Vector.AppendFloat64")
}

// AppendString appends a string value.
//
// TODO: Implement this function
func (v *Vector) AppendString(val string) {
	panic("TODO: implement Vector.AppendString")
}

// AppendNull appends a null value.
//
// TODO: Implement this function
func (v *Vector) AppendNull() {
	panic("TODO: implement Vector.AppendNull")
}

// Slice creates a new vector from selected indices.
//
// TODO: Implement this function
func (v *Vector) Slice(sel []int) *Vector {
	panic("TODO: implement Vector.Slice")
}

// Copy copies values from another vector.
//
// TODO: Implement this function
func (v *Vector) Copy(src *Vector) {
	panic("TODO: implement Vector.Copy")
}

// Hash computes a hash of the value at index.
//
// TODO: Implement this function
// Used for hash aggregation and hash joins
func (v *Vector) Hash(idx int) uint64 {
	panic("TODO: implement Vector.Hash")
}

// Compare compares values at two indices.
//
// TODO: Implement this function
// Returns: -1 (less), 0 (equal), 1 (greater)
func (v *Vector) Compare(i, j int) int {
	panic("TODO: implement Vector.Compare")
}

// SelectionVector tracks which rows are active.
type SelectionVector struct {
	// Indices contains the selected row indices
	Indices []int

	// Count is the number of selected rows
	Count int
}

// NewSelectionVector creates a selection vector.
//
// TODO: Implement this function
func NewSelectionVector(capacity int) *SelectionVector {
	panic("TODO: implement NewSelectionVector")
}

// SetAll selects all rows from 0 to count-1.
//
// TODO: Implement this function
func (s *SelectionVector) SetAll(count int) {
	panic("TODO: implement SelectionVector.SetAll")
}

// Apply applies the selection to get selected indices.
//
// TODO: Implement this function
func (s *SelectionVector) Apply(indices []int) {
	panic("TODO: implement SelectionVector.Apply")
}

// validityBitmapSize returns the number of uint64s needed for n bits.
func validityBitmapSize(n int) int {
	return (n + 63) / 64
}

// setBit sets bit i in the bitmap.
func setBit(bitmap []uint64, i int) {
	bitmap[i/64] |= 1 << (i % 64)
}

// clearBit clears bit i in the bitmap.
func clearBit(bitmap []uint64, i int) {
	bitmap[i/64] &^= 1 << (i % 64)
}

// testBit tests bit i in the bitmap.
func testBit(bitmap []uint64, i int) bool {
	return bitmap[i/64]&(1<<(i%64)) != 0
}
