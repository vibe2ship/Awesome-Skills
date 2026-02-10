// Package storage provides compression for columnar data.
package storage

import (
	"encoding/binary"

	"github.com/student/mini-duckdb/pkg/vector"
)

// Compressor compresses column data.
type Compressor interface {
	// Compress compresses the data
	Compress(data []byte, typ vector.DataType) ([]byte, error)

	// Decompress decompresses the data
	Decompress(data []byte, typ vector.DataType, count int) ([]byte, error)

	// Type returns the compression type
	Type() CompressionType
}

// RLECompressor implements Run-Length Encoding.
//
// Format: [value][run_length][value][run_length]...
//
// Best for: Low-cardinality, sorted columns
type RLECompressor struct{}

// Compress implements Compressor.
//
// TODO: Implement this function
// Example: [1,1,1,2,2,3] -> [(1,3),(2,2),(3,1)]
func (c *RLECompressor) Compress(data []byte, typ vector.DataType) ([]byte, error) {
	panic("TODO: implement RLECompressor.Compress")
}

// Decompress implements Compressor.
//
// TODO: Implement this function
func (c *RLECompressor) Decompress(data []byte, typ vector.DataType, count int) ([]byte, error) {
	panic("TODO: implement RLECompressor.Decompress")
}

// Type implements Compressor.
func (c *RLECompressor) Type() CompressionType {
	return CompressionRLE
}

// DictionaryCompressor implements dictionary encoding.
//
// Format: [dictionary_size][dictionary...][encoded_values...]
//
// Best for: Low-cardinality string columns
type DictionaryCompressor struct{}

// Compress implements Compressor.
//
// TODO: Implement this function
// 1. Build dictionary of unique values
// 2. Replace values with dictionary indices
// Example: [NYC,LA,NYC,LA] -> dict={NYC:0,LA:1}, data=[0,1,0,1]
func (c *DictionaryCompressor) Compress(data []byte, typ vector.DataType) ([]byte, error) {
	panic("TODO: implement DictionaryCompressor.Compress")
}

// Decompress implements Compressor.
//
// TODO: Implement this function
func (c *DictionaryCompressor) Decompress(data []byte, typ vector.DataType, count int) ([]byte, error) {
	panic("TODO: implement DictionaryCompressor.Decompress")
}

// Type implements Compressor.
func (c *DictionaryCompressor) Type() CompressionType {
	return CompressionDictionary
}

// DeltaCompressor implements delta encoding.
//
// Format: [base_value][delta1][delta2]...
//
// Best for: Sorted numeric columns (timestamps, IDs)
type DeltaCompressor struct{}

// Compress implements Compressor.
//
// TODO: Implement this function
// Example: [100,102,105,106] -> base=100, deltas=[0,2,3,1]
func (c *DeltaCompressor) Compress(data []byte, typ vector.DataType) ([]byte, error) {
	panic("TODO: implement DeltaCompressor.Compress")
}

// Decompress implements Compressor.
//
// TODO: Implement this function
func (c *DeltaCompressor) Decompress(data []byte, typ vector.DataType, count int) ([]byte, error) {
	panic("TODO: implement DeltaCompressor.Decompress")
}

// Type implements Compressor.
func (c *DeltaCompressor) Type() CompressionType {
	return CompressionDelta
}

// BitpackingCompressor packs small integers tightly.
//
// Format: [bit_width][packed_data...]
//
// Best for: Integers with limited range
type BitpackingCompressor struct{}

// Compress implements Compressor.
//
// TODO: Implement this function
// Pack values using minimum required bits
// Example: values 0-15 need only 4 bits each
func (c *BitpackingCompressor) Compress(data []byte, typ vector.DataType) ([]byte, error) {
	panic("TODO: implement BitpackingCompressor.Compress")
}

// Decompress implements Compressor.
//
// TODO: Implement this function
func (c *BitpackingCompressor) Decompress(data []byte, typ vector.DataType, count int) ([]byte, error) {
	panic("TODO: implement BitpackingCompressor.Decompress")
}

// Type implements Compressor.
func (c *BitpackingCompressor) Type() CompressionType {
	return CompressionBitpacking
}

// ChooseBestCompression analyzes data and chooses best compression.
//
// TODO: Implement this function
// Heuristics:
// - RLE: Good if many consecutive duplicates
// - Dictionary: Good if few unique values
// - Delta: Good if values are sorted/sequential
// - Bitpacking: Good if values have limited range
func ChooseBestCompression(data []byte, typ vector.DataType, count int) CompressionType {
	panic("TODO: implement ChooseBestCompression")
}

// CompressionRatio calculates the compression ratio.
func CompressionRatio(original, compressed int) float64 {
	if compressed == 0 {
		return 0
	}
	return float64(original) / float64(compressed)
}

// Helper functions

func minBitsRequired(maxValue uint64) int {
	if maxValue == 0 {
		return 1
	}
	bits := 0
	for maxValue > 0 {
		bits++
		maxValue >>= 1
	}
	return bits
}

func packBits(values []uint64, bitWidth int) []byte {
	totalBits := len(values) * bitWidth
	result := make([]byte, (totalBits+7)/8)

	bitPos := 0
	for _, v := range values {
		for b := 0; b < bitWidth; b++ {
			if v&(1<<b) != 0 {
				bytePos := bitPos / 8
				bitOffset := bitPos % 8
				result[bytePos] |= 1 << bitOffset
			}
			bitPos++
		}
	}

	return result
}

func unpackBits(data []byte, bitWidth, count int) []uint64 {
	result := make([]uint64, count)

	bitPos := 0
	for i := 0; i < count; i++ {
		var v uint64
		for b := 0; b < bitWidth; b++ {
			bytePos := bitPos / 8
			bitOffset := bitPos % 8
			if data[bytePos]&(1<<bitOffset) != 0 {
				v |= 1 << b
			}
			bitPos++
		}
		result[i] = v
	}

	return result
}
