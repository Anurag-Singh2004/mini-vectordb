# mini-vectordb

A vector database built from scratch to understand how systems like ChromaDB and Pinecone work internally.

## Features
- [x] In-memory vector storage
- [x] Brute-force cosine similarity search
- [x] Metadata filtering
- [x] Persistence (save/load)

## Performance & Limitations

Brute-force search scales linearly (O(n)) with the number of stored vectors — confirmed empirically:

| n (records) | search time (s) |
|---|---|
| 1,000       | 0.0048 |
| 10,000      | 0.0251 |
| 50,000      | 0.1181 |
| 100,000     | 0.2361 |

At ~2.4 microseconds per record (dim=128), extrapolating linearly to a production-scale
10 million vectors would mean roughly **24 seconds per search query** — far too slow for
real-time use. This is exactly the problem Approximate Nearest Neighbor (ANN) algorithms
like HNSW (Hierarchical Navigable Small World graphs) solve: by building a graph structure
that allows searching a small, smart subset of vectors instead of all of them, they achieve
sub-linear (roughly logarithmic) search time, at the cost of a small, tunable loss in
recall (you may miss a few of the true top-K results in exchange for massive speed gains).

This project implements brute-force search deliberately, to build a first-principles
understanding of the problem HNSW is designed to solve, before reaching for it as a
pre-built solution.

## Known Limitations

- **Tie-breaking in search results**: when two records have the *exact same* cosine similarity
  score, the current implementation breaks ties by comparing record IDs (alphabetical order),
  which is arbitrary and unrelated to relevance. This is unlikely to matter in practice with
  real embeddings (e.g. 384+ dimensional sentence-transformer outputs), where exact score ties
  are extremely rare due to floating-point precision. It's more visible with small, hand-crafted,
  low-dimensional vectors (as used in `examples/basic_usage.py`), where clean ties like `0.0` occur
  naturally. Documented here as a known tradeoff rather than "fixed" to avoid unnecessary complexity.