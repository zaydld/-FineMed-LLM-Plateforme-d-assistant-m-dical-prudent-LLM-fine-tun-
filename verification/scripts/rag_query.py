import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "rag_index"

FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.jsonl"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class RAGIndex:
    def __init__(self):
        if not FAISS_PATH.exists():
            raise FileNotFoundError(f"Index FAISS introuvable: {FAISS_PATH}")
        if not META_PATH.exists():
            raise FileNotFoundError(f"Meta introuvable: {META_PATH}")

        self.index = faiss.read_index(str(FAISS_PATH))

        # meta.jsonl = 1 chunk par ligne (json)
        self.chunks = []
        for line in META_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            self.chunks.append(json.loads(line))

        if len(self.chunks) == 0:
            raise ValueError("meta.jsonl est vide (aucun chunk).")

        # sanity: le nombre de vecteurs FAISS doit correspondre au nb de chunks
        if self.index.ntotal != len(self.chunks):
            print(
                f"⚠️ Attention: FAISS ntotal={self.index.ntotal} "
                f"mais chunks={len(self.chunks)} (ça peut venir d'un rebuild partiel)."
            )

        self.model = SentenceTransformer(EMBED_MODEL_NAME)

    def embed(self, text: str) -> np.ndarray:
        v = self.model.encode([text], normalize_embeddings=True)
        return np.array(v, dtype=np.float32)

    def search(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        q = self.embed(query)
        D, I = self.index.search(q, top_k)

        results = []
        for rank, idx in enumerate(I[0]):
            if idx < 0:
                continue
            ch = self.chunks[int(idx)]
            results.append({
                "rank": rank + 1,
                "score": float(D[0][rank]),
                "source": ch.get("source", ""),
                "path": ch.get("path", ""),
                "chunk_id": ch.get("chunk_id", idx),
                "text": ch.get("text", "")
            })
        return results


_rag_singleton = None

def get_rag() -> RAGIndex:
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = RAGIndex()
    return _rag_singleton
