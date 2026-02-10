# Mini-DuckDB: A Pedagogical Analytical Database

Mini-DuckDB is an educational implementation of an analytical (OLAP) database, focusing on columnar storage and vectorized execution.

## Learning Objectives

By implementing this project, you will learn:

- **Columnar Storage**: Column-oriented data layout
- **Vectorized Execution**: Processing data in batches
- **Compression**: Run-length, dictionary, delta encoding
- **Query Optimization**: Predicate pushdown, projection pruning
- **Parallel Execution**: Multi-threaded query processing

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Mini-DuckDB                                │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      SQL Interface                           │   │
│  │   "SELECT city, SUM(amount) FROM sales GROUP BY city"       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Query Parser                              │   │
│  │              SQL → Abstract Syntax Tree                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   Query Optimizer                            │   │
│  │    - Predicate Pushdown                                      │   │
│  │    - Projection Pruning                                      │   │
│  │    - Join Reordering                                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  Vectorized Executor                         │   │
│  │                                                              │   │
│  │   ┌──────────┐   ┌──────────┐   ┌──────────┐               │   │
│  │   │  Scan    │──▶│  Filter  │──▶│  Agg     │               │   │
│  │   │ (1024    │   │ (vector  │   │ (hash    │               │   │
│  │   │  rows)   │   │  at once)│   │  agg)    │               │   │
│  │   └──────────┘   └──────────┘   └──────────┘               │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   Columnar Storage                           │   │
│  │                                                              │   │
│  │   Table: sales                                               │   │
│  │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │   │
│  │   │   id     │ │   city   │ │  amount  │ │   date   │      │   │
│  │   │ (int64)  │ │ (string) │ │ (float64)│ │  (date)  │      │   │
│  │   ├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤      │   │
│  │   │ 1        │ │ "NYC"    │ │ 100.0    │ │2024-01-01│      │   │
│  │   │ 2        │ │ "LA"     │ │ 150.0    │ │2024-01-02│      │   │
│  │   │ 3        │ │ "NYC"    │ │ 200.0    │ │2024-01-03│      │   │
│  │   │ ...      │ │ ...      │ │ ...      │ │ ...      │      │   │
│  │   └──────────┘ └──────────┘ └──────────┘ └──────────┘      │   │
│  │                                                              │   │
│  │   Compression: RLE, Dictionary, Delta                        │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Columnar vs Row Storage

```
Row Storage (OLTP):              Columnar Storage (OLAP):
┌────┬────┬────┬────┐           ┌────┬────┬────┬────┐
│ id │name│age │city│           │ id │ id │ id │ id │
├────┼────┼────┼────┤           │ 1  │ 2  │ 3  │ 4  │
│ 1  │Alice│25 │NYC │           ├────┴────┴────┴────┤
├────┼────┼────┼────┤           │name│name│name│name│
│ 2  │Bob │30 │LA  │           │Alice│Bob│Carol│Dave│
├────┼────┼────┼────┤           ├────┴────┴────┴────┤
│ 3  │Carol│28│CHI │           │age │age │age │age │
├────┼────┼────┼────┤           │ 25 │ 30 │ 28 │ 35 │
│ 4  │Dave│35 │NYC │           └────┴────┴────┴────┘
└────┴────┴────┴────┘

Best for: Point queries         Best for: Aggregations
SELECT * WHERE id=1             SELECT city, AVG(age)
                                GROUP BY city
```

## Project Structure

```
mini-duckdb/
├── cmd/
│   └── duckdb/          # Main CLI
├── pkg/
│   ├── storage/         # Columnar storage
│   │   ├── column.go    # Column types
│   │   ├── chunk.go     # Data chunks
│   │   ├── table.go     # Table storage
│   │   └── compress.go  # Compression
│   ├── vector/          # Vectorized types
│   │   ├── vector.go    # Vector type
│   │   ├── ops.go       # Vector operations
│   │   └── batch.go     # Data batch
│   ├── parser/          # SQL parser
│   │   ├── lexer.go     # Tokenizer
│   │   ├── parser.go    # Parser
│   │   └── ast.go       # AST nodes
│   ├── executor/        # Query execution
│   │   ├── executor.go  # Executor framework
│   │   ├── scan.go      # Table scan
│   │   ├── filter.go    # Filter operator
│   │   ├── project.go   # Projection
│   │   ├── aggregate.go # Aggregation
│   │   ├── join.go      # Join operators
│   │   └── sort.go      # Sort operator
│   └── catalog/         # Schema management
│       ├── catalog.go   # Catalog
│       └── table.go     # Table metadata
└── tests/
    ├── storage_test.go
    ├── vector_test.go
    └── integration_test.go
```

## Core Concepts

### 1. Vectors (SIMD-friendly batches)

```go
type Vector struct {
    Type     DataType
    Data     []byte      // Flat data buffer
    Validity []uint64    // Null bitmap
    Length   int
}

// Process 1024 values at once
const VectorSize = 1024
```

### 2. Columnar Chunks

```go
type DataChunk struct {
    Columns []*Vector
    Count   int  // Number of rows
}
```

### 3. Vectorized Operators

```go
// Filter: Process entire vector at once
func (f *FilterOperator) Execute(input *DataChunk) *DataChunk {
    // Create selection vector
    sel := make([]int, 0, input.Count)
    for i := 0; i < input.Count; i++ {
        if f.predicate.Evaluate(input, i) {
            sel = append(sel, i)
        }
    }
    // Apply selection to all columns
    return input.Slice(sel)
}
```

## Implementation TODO List

### Milestone 1: Vector Types
- [ ] Implement `pkg/vector/vector.go` - Vector type
- [ ] Implement null bitmap handling
- [ ] Implement `pkg/vector/ops.go` - Vector operations
- [ ] Run: `go test ./pkg/vector/...`

### Milestone 2: Data Chunks
- [ ] Implement `pkg/vector/batch.go` - Data chunks
- [ ] Implement selection vectors
- [ ] Run: `go test ./pkg/vector/...`

### Milestone 3: Columnar Storage
- [ ] Implement `pkg/storage/column.go` - Column types
- [ ] Implement `pkg/storage/chunk.go` - Storage chunks
- [ ] Implement `pkg/storage/table.go` - Table storage
- [ ] Run: `go test ./pkg/storage/...`

### Milestone 4: Compression
- [ ] Implement `pkg/storage/compress.go` - Compression
- [ ] Implement RLE, dictionary, delta encoding
- [ ] Run: `go test ./pkg/storage/...`

### Milestone 5: SQL Parser
- [ ] Implement `pkg/parser/lexer.go` - Tokenizer
- [ ] Implement `pkg/parser/parser.go` - Parser
- [ ] Support: SELECT, FROM, WHERE, GROUP BY, ORDER BY
- [ ] Run: `go test ./pkg/parser/...`

### Milestone 6: Vectorized Operators
- [ ] Implement `pkg/executor/scan.go` - Table scan
- [ ] Implement `pkg/executor/filter.go` - Filter
- [ ] Implement `pkg/executor/project.go` - Projection
- [ ] Implement `pkg/executor/aggregate.go` - Aggregation
- [ ] Run: `go test ./pkg/executor/...`

### Milestone 7: Join & Sort
- [ ] Implement `pkg/executor/join.go` - Hash join
- [ ] Implement `pkg/executor/sort.go` - External sort
- [ ] Run: `go test ./pkg/executor/...`

### Milestone 8: Integration
- [ ] End-to-end query tests
- [ ] TPC-H benchmark queries
- [ ] Parallel execution

## Compression Techniques

### Run-Length Encoding (RLE)
```
Original:   [NYC, NYC, NYC, LA, LA, NYC, NYC]
Compressed: [(NYC, 3), (LA, 2), (NYC, 2)]

Best for: Low-cardinality, sorted columns
```

### Dictionary Encoding
```
Original:   [NYC, LA, NYC, CHI, LA, NYC]
Dictionary: {0: NYC, 1: LA, 2: CHI}
Compressed: [0, 1, 0, 2, 1, 0]

Best for: Low-cardinality string columns
```

### Delta Encoding
```
Original:   [100, 102, 105, 106, 110]
Deltas:     [100, 2, 3, 1, 4]

Best for: Sorted numeric columns (timestamps, IDs)
```

## Vectorized Aggregation

```go
// Hash aggregation with vectorized processing
type HashAggregator struct {
    groups  map[uint64]*AggState  // Hash -> aggregate state
    keyCol  int                    // Group-by column index
    aggFunc AggregateFunction
}

func (a *HashAggregator) Process(chunk *DataChunk) {
    keys := chunk.Columns[a.keyCol]
    values := chunk.Columns[a.valueCol]

    // Process in batches
    for i := 0; i < chunk.Count; i++ {
        hash := keys.Hash(i)
        state := a.groups[hash]
        if state == nil {
            state = a.aggFunc.NewState()
            a.groups[hash] = state
        }
        a.aggFunc.Update(state, values, i)
    }
}
```

## Testing

```bash
# Run all tests
go test ./...

# Run specific package
go test ./pkg/vector/...

# Run benchmarks
go test -bench=. ./...

# Run with race detector
go test -race ./...
```

## Example Queries

```sql
-- Simple aggregation
SELECT city, COUNT(*), SUM(amount)
FROM sales
WHERE year = 2024
GROUP BY city;

-- Join with filter
SELECT c.name, SUM(o.amount)
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE o.date >= '2024-01-01'
GROUP BY c.name
ORDER BY SUM(o.amount) DESC
LIMIT 10;
```

## References

- [DuckDB Internals](https://duckdb.org/internals/overview)
- [MonetDB/X100](https://www.cidrdb.org/cidr2005/papers/P19.pdf)
- [Vectorized Query Execution](https://15721.courses.cs.cmu.edu/spring2024/slides/06-vectorization.pdf)
- [Column-Stores vs Row-Stores](https://stratos.seas.harvard.edu/files/stratos/files/columnstores.pdf)
