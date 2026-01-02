import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
RAG_SOURCES = BASE_DIR / "rag_sources"

OUT_DIR = BASE_DIR / "rag_index"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = OUT_DIR / "faiss.index"
META_PATH = OUT_DIR / "meta.jsonl"

# Modèle d'embedding (multilingue FR/EN)
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Chunking (simple et robuste)
CHUNK_SIZE = 1200      # caractères
CHUNK_OVERLAP = 200    # caractères


def iter_txt_files():
    for source in ["has", "oms"]:
        src_root = RAG_SOURCES / source
        if not src_root.exists():
            continue
        for txt in src_root.rglob("*.txt"):
            yield source, txt


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end].strip()
        if len(chunk) >= 200:  # évite les chunks trop petits
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def infer_topic(source: str, txt_path: Path) -> str:
    # ex: rag_sources/has/avc/doc.txt -> "avc"
    # ex: rag_sources/oms/fever/doc.txt -> "fever"
    try:
        rel = txt_path.relative_to(RAG_SOURCES / source)
        # premier dossier = topic
        parts = rel.parts
        return parts[0] if len(parts) > 1 else "general"
    except Exception:
        return "general"


def main():
    model = SentenceTransformer(EMBED_MODEL_NAME)

    all_chunks = []
    meta = []

    print("🔎 Scan des fichiers .txt ...")
    for source, txt_path in iter_txt_files():
        try:
            text = txt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # fallback au cas où un txt a été écrit différemment
            text = txt_path.read_text(encoding="utf-8", errors="replace")

        topic = infer_topic(source, txt_path)
        chunks = chunk_text(text)

        if not chunks:
            print(f"⚠️ Aucun chunk valide: {txt_path}")
            continue

        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            meta.append({
                "source": source,              # has / oms
                "topic": topic,                # avc / brulure / fever / etc.
                "file": str(txt_path),         # chemin du .txt
                "chunk_id": i,
                "text": ch                      # on garde le texte pour retrieval rapide
            })

        print(f"✅ {source}/{topic} -> {txt_path.name} | chunks={len(chunks)}")

    if not all_chunks:
        raise RuntimeError("Aucun chunk trouvé. Vérifie que les .txt existent bien dans rag_sources.")

    print(f"\n🧠 Embedding de {len(all_chunks)} chunks ... (ça peut prendre un peu de temps)")
    embeddings = model.encode(all_chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via vecteurs normalisés
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("\n✅ Index FAISS créé :")
    print("   -", INDEX_PATH)
    print("   -", META_PATH)
    print(f"📌 Total vectors = {index.ntotal} | dim = {dim}")


if __name__ == "__main__":
    main()
