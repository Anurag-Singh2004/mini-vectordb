# mini-vectordb

A vector database built from scratch to understand how systems like ChromaDB and Pinecone work internally.

## Features
- [x] In-memory vector storage
- [x] Brute-force cosine similarity search
- [x] Metadata filtering
- [x] Persistence (save/load)

## Known Limitations

- **Tie-breaking in search results**: when two records have the *exact same* cosine similarity
  score, the current implementation breaks ties by comparing record IDs (alphabetical order),
  which is arbitrary and unrelated to relevance. This is unlikely to matter in practice with
  real embeddings (e.g. 384+ dimensional sentence-transformer outputs), where exact score ties
  are extremely rare due to floating-point precision. It's more visible with small, hand-crafted,
  low-dimensional vectors (as used in `examples/basic_usage.py`), where clean ties like `0.0` occur
  naturally. Documented here as a known tradeoff rather than "fixed" to avoid unnecessary complexity.