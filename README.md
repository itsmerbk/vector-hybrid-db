# Vector-Hybrid-DB: LSH Vector & Okapi BM25 Retrieval Engine

Zero-dependency hybrid search engine built in Python combining dense vector similarity with sparse lexical ranking for Retrieval-Augmented Generation (RAG) pipelines.

## Architecture
- **Random-Projection LSH Index:** Hashes normalized embeddings into binary hyperplane buckets with 1-bit Hamming neighbor probing for fast candidate filtering.
- **Cosine Similarity & Okapi BM25:** Computes dense semantic similarity alongside IDF-weighted lexical term saturation scores.
- **Reciprocal Rank Fusion (RRF):** Merges dense and sparse rankings into a unified relevance score without requiring score calibration.
- **Snapshot Persistence:** Serializes hyperplane projections, document vectors, and inverted term statistics to portable JSON snapshots.

## Run Tests & Benchmark
    python3 -m unittest test_engine.py
