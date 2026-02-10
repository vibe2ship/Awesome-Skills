//! Bloom filter implementation.
//!
//! A Bloom filter is a probabilistic data structure that can tell you:
//! - "Definitely not in set" (100% accurate)
//! - "Probably in set" (may have false positives)
//!
//! # Properties
//!
//! - Space-efficient: uses bits instead of storing actual keys
//! - Fast: O(k) where k is number of hash functions
//! - False positive rate: approximately (1 - e^(-kn/m))^k
//!   where n = items, m = bits, k = hash functions
//!
//! # Usage in LSM
//!
//! Before reading a block from disk, check the bloom filter.
//! If the key is definitely not present, skip the block read.

use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};

/// A Bloom filter for fast key existence checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilter {
    /// Bit array
    bits: Vec<u8>,

    /// Number of hash functions (k)
    k: u32,
}

impl BloomFilter {
    /// Creates a new Bloom filter with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `num_keys` - Expected number of keys
    /// * `bits_per_key` - Bits per key (higher = lower false positive rate)
    ///
    /// TODO: Implement this function
    /// - Calculate optimal number of bits
    /// - Calculate optimal k (number of hash functions)
    pub fn new(num_keys: usize, bits_per_key: usize) -> Self {
        todo!("implement BloomFilter::new")
    }

    /// Creates a Bloom filter from existing bits.
    pub fn from_bits(bits: Vec<u8>, k: u32) -> Self {
        BloomFilter { bits, k }
    }

    /// Adds a key to the filter.
    ///
    /// TODO: Implement this function
    /// - Compute k hash values
    /// - Set corresponding bits
    pub fn insert(&mut self, key: &[u8]) {
        todo!("implement BloomFilter::insert")
    }

    /// Checks if a key might be in the filter.
    ///
    /// Returns:
    /// - `false`: Key is definitely NOT in the set
    /// - `true`: Key is PROBABLY in the set (may be false positive)
    ///
    /// TODO: Implement this function
    pub fn may_contain(&self, key: &[u8]) -> bool {
        todo!("implement BloomFilter::may_contain")
    }

    /// Returns the bit array.
    pub fn bits(&self) -> &[u8] {
        &self.bits
    }

    /// Returns the number of hash functions.
    pub fn k(&self) -> u32 {
        self.k
    }

    /// Returns the size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.bits.len()
    }

    /// Encodes the bloom filter to bytes.
    ///
    /// TODO: Implement this function
    /// Format: [k: u32][bits_len: u32][bits...]
    pub fn encode(&self) -> Bytes {
        todo!("implement BloomFilter::encode")
    }

    /// Decodes a bloom filter from bytes.
    ///
    /// TODO: Implement this function
    pub fn decode(data: &[u8]) -> Self {
        todo!("implement BloomFilter::decode")
    }

    /// Computes the hash values for a key.
    ///
    /// Uses double hashing: h(i) = h1 + i * h2
    /// where h1 and h2 are computed from the key.
    ///
    /// TODO: Implement this function
    fn hash_values(&self, key: &[u8]) -> impl Iterator<Item = usize> + '_ {
        // Placeholder - implement double hashing
        let h1 = hash1(key);
        let h2 = hash2(key);
        let num_bits = self.bits.len() * 8;

        (0..self.k).map(move |i| {
            let h = h1.wrapping_add((i as u64).wrapping_mul(h2));
            (h as usize) % num_bits
        })
    }
}

/// First hash function (based on FNV-1a).
fn hash1(key: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in key {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Second hash function (based on murmur-like).
fn hash2(key: &[u8]) -> u64 {
    let mut hash: u64 = 0;
    for &byte in key {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
    }
    hash | 1 // Ensure odd for better distribution
}

/// Builder for creating Bloom filters from multiple keys.
pub struct BloomFilterBuilder {
    keys: Vec<Bytes>,
    bits_per_key: usize,
}

impl BloomFilterBuilder {
    /// Creates a new builder.
    pub fn new(bits_per_key: usize) -> Self {
        BloomFilterBuilder {
            keys: Vec::new(),
            bits_per_key,
        }
    }

    /// Adds a key.
    pub fn add(&mut self, key: impl Into<Bytes>) {
        self.keys.push(key.into());
    }

    /// Builds the Bloom filter.
    ///
    /// TODO: Implement this function
    pub fn build(self) -> BloomFilter {
        todo!("implement BloomFilterBuilder::build")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_new() {
        let bloom = BloomFilter::new(100, 10);
        assert!(bloom.bits.len() > 0);
        assert!(bloom.k > 0);
    }

    #[test]
    fn test_bloom_filter_insert_contains() {
        let mut bloom = BloomFilter::new(100, 10);

        bloom.insert(b"hello");
        bloom.insert(b"world");
        bloom.insert(b"foo");

        // Inserted keys should be found
        assert!(bloom.may_contain(b"hello"));
        assert!(bloom.may_contain(b"world"));
        assert!(bloom.may_contain(b"foo"));
    }

    #[test]
    fn test_bloom_filter_false_positives() {
        let mut bloom = BloomFilter::new(100, 10);

        // Insert some keys
        for i in 0..100 {
            let key = format!("key{}", i);
            bloom.insert(key.as_bytes());
        }

        // Check false positive rate
        let mut false_positives = 0;
        for i in 100..200 {
            let key = format!("other{}", i);
            if bloom.may_contain(key.as_bytes()) {
                false_positives += 1;
            }
        }

        // With 10 bits per key, false positive rate should be around 1%
        // Allow up to 10% for statistical variation
        assert!(false_positives < 10, "Too many false positives: {}", false_positives);
    }

    #[test]
    fn test_bloom_filter_encode_decode() {
        let mut bloom = BloomFilter::new(100, 10);
        bloom.insert(b"test1");
        bloom.insert(b"test2");

        let encoded = bloom.encode();
        let decoded = BloomFilter::decode(&encoded);

        assert_eq!(bloom.bits, decoded.bits);
        assert_eq!(bloom.k, decoded.k);
        assert!(decoded.may_contain(b"test1"));
        assert!(decoded.may_contain(b"test2"));
    }

    #[test]
    fn test_bloom_filter_builder() {
        let mut builder = BloomFilterBuilder::new(10);
        builder.add(b"key1".as_slice());
        builder.add(b"key2".as_slice());
        builder.add(b"key3".as_slice());

        let bloom = builder.build();

        assert!(bloom.may_contain(b"key1"));
        assert!(bloom.may_contain(b"key2"));
        assert!(bloom.may_contain(b"key3"));
    }

    #[test]
    fn test_optimal_k() {
        // For bits_per_key = 10, optimal k ≈ 10 * ln(2) ≈ 6.9
        let bloom = BloomFilter::new(100, 10);
        assert!(bloom.k >= 6 && bloom.k <= 8);
    }

    #[test]
    fn test_hash_functions() {
        let h1a = hash1(b"hello");
        let h1b = hash1(b"hello");
        let h1c = hash1(b"world");

        // Same input -> same hash
        assert_eq!(h1a, h1b);
        // Different input -> different hash (usually)
        assert_ne!(h1a, h1c);
    }
}
