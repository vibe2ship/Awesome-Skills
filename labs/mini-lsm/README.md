# Mini-LSM: A Pedagogical Log-Structured Merge Tree

Mini-LSM is an educational implementation of an LSM-tree storage engine, inspired by [LevelDB](https://github.com/google/leveldb), [RocksDB](https://rocksdb.org/), and [TiKV](https://tikv.org/).

## Learning Objectives

By implementing this project, you will learn:

- **LSM-Tree Architecture**: MemTable, SSTable, compaction
- **Write-Ahead Logging**: Durability through WAL
- **Sorted String Tables**: Block-based file format
- **Compaction Strategies**: Leveled, Tiered, FIFO
- **Bloom Filters**: Probabilistic data structures
- **Block Cache**: LRU caching for reads

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Mini-LSM Engine                          │
│                                                                 │
│  Write Path:                                                    │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │  Write  │───►│    WAL      │───►│      MemTable           │ │
│  └─────────┘    │ (append-only│    │ (skip list in memory)   │ │
│                 │    log)     │    └───────────┬─────────────┘ │
│                 └─────────────┘                │ (when full)   │
│                                                ▼               │
│                                    ┌───────────────────────┐   │
│                                    │   Immutable MemTable  │   │
│                                    └───────────┬───────────┘   │
│                                                │ (flush)       │
│                                                ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    SSTable Levels                       │   │
│  │  L0: [SST] [SST] [SST]  (unsorted, may overlap)        │   │
│  │  L1: [   SST   ] [   SST   ]  (sorted, non-overlapping)│   │
│  │  L2: [    SST    ] [    SST    ] [    SST    ]         │   │
│  │  ...                                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Read Path:                                                     │
│  ┌─────────┐                                                    │
│  │  Read   │──► MemTable ──► Imm MemTable ──► L0 ──► L1 ──► ... │
│  └─────────┘   (newest first, stop on first match)             │
└─────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### 1. MemTable

In-memory sorted data structure (skip list):
- Fast writes: O(log n)
- Fast point reads: O(log n)
- Range scans: iterate in order
- Flushed to SSTable when full

### 2. SSTable (Sorted String Table)

Immutable on-disk file format:

```
┌────────────────────────────────────────┐
│              Data Blocks               │
│  ┌──────────────────────────────────┐  │
│  │ Block 0: [key1|val1][key2|val2]  │  │
│  ├──────────────────────────────────┤  │
│  │ Block 1: [key3|val3][key4|val4]  │  │
│  ├──────────────────────────────────┤  │
│  │ ...                              │  │
│  └──────────────────────────────────┘  │
├────────────────────────────────────────┤
│           Meta Blocks                  │
│  ┌──────────────────────────────────┐  │
│  │ Bloom Filter                     │  │
│  └──────────────────────────────────┘  │
├────────────────────────────────────────┤
│             Index Block               │
│  [block0_last_key|offset]             │
│  [block1_last_key|offset]             │
│  ...                                  │
├────────────────────────────────────────┤
│              Footer                   │
│  [meta_offset|index_offset|magic]     │
└────────────────────────────────────────┘
```

### 3. Write-Ahead Log (WAL)

Durability mechanism:
1. Write operation appended to WAL
2. WAL synced to disk
3. Write applied to MemTable
4. On crash, replay WAL to recover

### 4. Compaction

Background process to:
- Merge overlapping SSTables
- Remove deleted keys (tombstones)
- Reduce read amplification

```
Before Compaction:
L0: [a-z] [b-y] [c-x]  (overlapping)

After Compaction:
L1: [a-m] [n-z]        (non-overlapping, sorted)
```

## Project Structure

```
mini-lsm/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs           # Library root
│   ├── memtable.rs      # MemTable implementation
│   ├── sstable/
│   │   ├── mod.rs       # SSTable module
│   │   ├── builder.rs   # SSTable builder
│   │   ├── iterator.rs  # SSTable iterator
│   │   └── block.rs     # Block format
│   ├── wal.rs           # Write-ahead log
│   ├── manifest.rs      # Metadata management
│   ├── compaction.rs    # Compaction strategies
│   ├── bloom.rs         # Bloom filter
│   ├── iterator.rs      # Merge iterator
│   └── lsm.rs           # LSM engine
└── tests/
    ├── memtable_test.rs
    ├── sstable_test.rs
    └── integration_test.rs
```

## Implementation TODO List

### Milestone 1: Core Types
- [ ] Implement `src/lib.rs` - Basic types and errors
- [ ] Implement `src/memtable.rs` - SkipList-based MemTable
- [ ] Run: `cargo test memtable`

### Milestone 2: Block Format
- [ ] Implement `src/sstable/block.rs` - Block encoding/decoding
- [ ] Implement key-value encoding
- [ ] Run: `cargo test block`

### Milestone 3: SSTable
- [ ] Implement `src/sstable/builder.rs` - SSTable builder
- [ ] Implement `src/sstable/mod.rs` - SSTable reader
- [ ] Implement `src/sstable/iterator.rs` - SSTable iterator
- [ ] Run: `cargo test sstable`

### Milestone 4: WAL
- [ ] Implement `src/wal.rs` - Write-ahead log
- [ ] Implement recovery logic
- [ ] Run: `cargo test wal`

### Milestone 5: Bloom Filter
- [ ] Implement `src/bloom.rs` - Bloom filter
- [ ] Integrate with SSTable
- [ ] Run: `cargo test bloom`

### Milestone 6: Iterator
- [ ] Implement `src/iterator.rs` - Merge iterator
- [ ] Support seeking and scanning
- [ ] Run: `cargo test iterator`

### Milestone 7: Compaction
- [ ] Implement `src/manifest.rs` - Level metadata
- [ ] Implement `src/compaction.rs` - Leveled compaction
- [ ] Run: `cargo test compaction`

### Milestone 8: LSM Engine
- [ ] Implement `src/lsm.rs` - Full engine
- [ ] Integrate all components
- [ ] Run: `cargo test --test integration`

## API

```rust
// Create or open a database
let db = LsmDb::open("./data")?;

// Write operations
db.put(b"key1", b"value1")?;
db.delete(b"key2")?;

// Read operations
let value = db.get(b"key1")?;

// Range scan
let iter = db.scan(b"a", b"z");
for (key, value) in iter {
    println!("{:?} = {:?}", key, value);
}

// Flush memtable to disk
db.flush()?;

// Trigger compaction
db.compact()?;
```

## Key Algorithms

### Binary Search in Block

```rust
// Find key in sorted block
fn binary_search(block: &[Entry], key: &[u8]) -> Option<&Entry> {
    let mut lo = 0;
    let mut hi = block.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        match block[mid].key.cmp(key) {
            Less => lo = mid + 1,
            Greater => hi = mid,
            Equal => return Some(&block[mid]),
        }
    }
    None
}
```

### Bloom Filter

```rust
// Check if key might exist
fn may_contain(&self, key: &[u8]) -> bool {
    for i in 0..self.k {
        let hash = hash_with_seed(key, i);
        let bit = hash % self.bits.len();
        if !self.bits[bit] {
            return false;  // Definitely not present
        }
    }
    true  // Might be present (could be false positive)
}
```

### Merge Iterator

```rust
// Merge multiple sorted iterators
fn next(&mut self) -> Option<(Key, Value)> {
    // Find iterator with smallest current key
    let min_idx = self.iters
        .iter()
        .enumerate()
        .filter_map(|(i, it)| it.peek().map(|k| (i, k)))
        .min_by_key(|(_, k)| k)?
        .0;

    self.iters[min_idx].next()
}
```

## Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test memtable

# Run with logging
RUST_LOG=debug cargo test

# Run benchmarks
cargo bench
```

## References

- [LevelDB Implementation](https://github.com/google/leveldb/blob/main/doc/impl.md)
- [LSM-based Storage Techniques](https://arxiv.org/abs/1812.07527)
- [RocksDB Wiki](https://github.com/facebook/rocksdb/wiki)
- [Mini-LSM Tutorial](https://skyzh.github.io/mini-lsm/)
