import tempfile, time, unittest
from pathlib import Path
from engine import HybridVectorEngine


class TestHybridVectorEngine(unittest.TestCase):
    def test_hybrid_rrf_ranking_and_persistence(self):
        eng = HybridVectorEngine(dim=64, num_planes=8)
        eng.add("doc_lsm", "LSM-tree storage engine with write-ahead log and bloom filters")
        eng.add("doc_raft", "Distributed Raft consensus leader election and replicated log")
        eng.add("doc_rag", "Hybrid vector search combining BM25 lexical scoring with cosine similarity")

        hits = eng.search("BM25 cosine vector search", top_k=2)
        self.assertEqual(hits[0]["id"], "doc_rag")

        with tempfile.TemporaryDirectory() as tmp:
            snap = str(Path(tmp) / "index.json")
            eng.save(snap)
            restored = HybridVectorEngine.load(snap)
            self.assertEqual(restored.search("Raft leader election", top_k=1)[0]["id"], "doc_raft")

    def test_throughput_benchmark(self):
        eng = HybridVectorEngine(dim=64, num_planes=8)
        for i in range(1000):
            eng.add(f"item_{i}", f"Distributed database shard partition replica node {i % 25}")
        t0 = time.perf_counter()
        for _ in range(200):
            eng.search("database shard replica", top_k=5)
        elapsed = max(time.perf_counter() - t0, 1e-6)
        print(f"\n[BENCH] 1,000 docs indexed | Hybrid RRF Search: {int(200 / elapsed):,} queries/sec")


if __name__ == "__main__":
    unittest.main()
