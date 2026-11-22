from __future__ import annotations
import json, math, random, re
from collections import Counter, defaultdict
from pathlib import Path


class HybridVectorEngine:
    def __init__(self, dim: int = 64, num_planes: int = 8, seed: int = 42):
        self.dim = dim
        rng = random.Random(seed)
        self.planes = [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(num_planes)]
        self.docs: dict[str, dict] = {}
        self.lsh: dict[int, set[str]] = defaultdict(set)
        self.df: Counter[str] = Counter()
        self.total_len = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in self._tokenize(text):
            h = hash(tok)
            vec[abs(h) % self.dim] += 1.0 if (h & 1) == 0 else -1.0
        return self._normalize(vec)

    def _lsh_hash(self, vec: list[float]) -> int:
        sig = 0
        for i, plane in enumerate(self.planes):
            if sum(a * b for a, b in zip(vec, plane)) >= 0:
                sig |= 1 << i
        return sig

    def add(self, doc_id: str, text: str, vector: list[float] | None = None) -> None:
        vec = self._normalize(vector) if vector is not None else self.embed_text(text)
        tokens = Counter(self._tokenize(text))
        doc_len = sum(tokens.values())
        self.docs[doc_id] = {"text": text, "vec": vec, "tokens": tokens, "len": doc_len}
        self.lsh[self._lsh_hash(vec)].add(doc_id)
        for term in tokens:
            self.df[term] += 1
        self.total_len += doc_len

    def _bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
        q_tokens = self._tokenize(query)
        n_docs = len(self.docs)
        if not n_docs or not q_tokens:
            return {}
        avgdl = self.total_len / n_docs
        scores: dict[str, float] = defaultdict(float)
        for term in q_tokens:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for did, doc in self.docs.items():
                tf = doc["tokens"].get(term, 0)
                if tf > 0:
                    denom = tf + k1 * (1.0 - b + b * (doc["len"] / avgdl))
                    scores[did] += idf * ((tf * (k1 + 1.0)) / denom)
        return scores

    def search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list[dict]:
        q_vec = self.embed_text(query)
        q_sig = self._lsh_hash(q_vec)
        candidates = set(self.lsh.get(q_sig, set()))
        for bit in range(len(self.planes)):
            candidates |= self.lsh.get(q_sig ^ (1 << bit), set())
        if len(candidates) < top_k:
            candidates = set(self.docs.keys())

        dense_ranked = sorted(
            candidates,
            key=lambda did: sum(a * b for a, b in zip(q_vec, self.docs[did]["vec"])),
            reverse=True,
        )
        bm25_map = self._bm25_scores(query)
        sparse_ranked = sorted(bm25_map.keys(), key=lambda did: bm25_map[did], reverse=True)

        rrf: dict[str, float] = defaultdict(float)
        for rank, did in enumerate(dense_ranked, start=1):
            rrf[did] += 1.0 / (rrf_k + rank)
        for rank, did in enumerate(sparse_ranked, start=1):
            rrf[did] += 1.0 / (rrf_k + rank)

        top_ids = sorted(rrf.keys(), key=lambda did: rrf[did], reverse=True)[:top_k]
        return [{"id": did, "score": round(rrf[did], 5), "text": self.docs[did]["text"]} for did in top_ids]

    def save(self, path: str) -> None:
        payload = {
            "dim": self.dim,
            "planes": self.planes,
            "docs": {k: {"text": v["text"], "vec": v["vec"]} for k, v in self.docs.items()},
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> HybridVectorEngine:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        eng = cls(dim=raw["dim"], num_planes=len(raw["planes"]))
        eng.planes = raw["planes"]
        for did, item in raw["docs"].items():
            eng.add(did, item["text"], vector=item["vec"])
        return eng
