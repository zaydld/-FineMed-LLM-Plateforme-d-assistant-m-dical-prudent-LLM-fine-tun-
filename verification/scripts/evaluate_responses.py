import csv
import json
import time
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from rag_query import get_rag
from judge_provider import judge_scientific

BASE_DIR = Path(__file__).resolve().parent.parent

# INPUT dans partie_2
INPUT = BASE_DIR.parent / "generation" / "reponses" / "reponses_llms.csv"
# OUTPUT dans partie_3
OUTPUT = BASE_DIR / "outputs" / "evaluations_scientifiques11.csv"

TOP_K = 6

# ✅ BATCH
BATCH_START = 1
BATCH_END = 2

# ✅ retry rate-limit
MAX_RETRIES = 5


def normalize_key(s: str) -> str:
    """Normalise un nom de colonne"""
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("\ufeff", "")  # BOM UTF-8
    s = "".join(ch for ch in s if ch.isprintable())
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def detect_encoding(path: Path):
    """Essaie plusieurs encodings"""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                f.readline()
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def read_csv_with_excel(path: Path):
    """Lit le CSV en gérant le format Excel"""
    enc = detect_encoding(path)
    
    with open(path, "r", encoding=enc, newline="") as f:
        # Détecter le délimiteur
        sample = f.read(2048)
        f.seek(0)
        
        # Excel utilise souvent des tabulations ou des virgules
        if '\t' in sample:
            delimiter = '\t'
        elif ';' in sample:
            delimiter = ';'
        else:
            delimiter = ','
        
        reader = csv.DictReader(f, delimiter=delimiter)
        raw_fieldnames = reader.fieldnames or []
        
        print(f"🔍 Colonnes détectées: {raw_fieldnames}")
        print(f"📄 Encoding: {enc}, Delimiter: '{delimiter}'")
        
        # Créer un mapping des colonnes
        col_map = {}
        for col in raw_fieldnames:
            norm = normalize_key(col)
            col_map[col] = norm
            print(f"   '{col}' -> '{norm}'")
        
        rows = []
        for r in reader:
            row_dict = {}
            for raw_col, value in r.items():
                norm_col = col_map.get(raw_col, normalize_key(raw_col))
                # Nettoyer la valeur
                if value:
                    value = str(value).strip()
                    # Enlever les \n littéraux
                    value = value.replace('\\n', '\n')
                row_dict[norm_col] = value
            rows.append(row_dict)
    
    print(f"✅ {len(rows)} lignes lues")
    
    # Debug: afficher la première ligne
    if rows:
        print("\n🔍 PREMIÈRE LIGNE:")
        for k, v in rows[0].items():
            display_val = v[:150] if v else '(vide)'
            print(f"   {k}: {display_val}")
    
    return rows


def pick(row: dict, *keys: str) -> str:
    """Retourne la première valeur non vide"""
    for k in keys:
        v = row.get(k, "")
        if not v:
            continue
        v = str(v).strip()
        if v and v.lower() not in ["nan", "none", "null", ""]:
            return v
    return ""


def build_evidence_text(evidence):
    blocks = []
    sources = []
    for e in evidence:
        src = e.get("source", "")
        path = e.get("path", "")
        chunk_id = e.get("chunk_id")
        score = float(e.get("score", 0.0))

        sources.append(f"{src}:{path}#chunk={chunk_id}")
        blocks.append(
            f"[{src}] {path} (chunk={chunk_id}, score={score:.4f})\n{e.get('text','')}"
        )
    return "\n\n---\n\n".join(blocks), " | ".join(sources)


def load_already_done():
    done = set()
    if not OUTPUT.exists():
        return done
    with open(OUTPUT, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            done.add((row.get("id_cas", ""), row.get("modele", ""), row.get("sample_id", "")))
    print(f"✅ {len(done)} évaluations déjà faites")
    return done


def call_judge_with_retry(prompt, response, evidence_text):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return judge_scientific(prompt, response, evidence_text)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Rate limit" in msg or "rate_limit" in msg:
                wait_s = min(60 * attempt, 300)
                print(f"⚠️ Rate limit. Attente {wait_s}s (retry {attempt}/{MAX_RETRIES})...")
                time.sleep(wait_s)
                continue
            raise
    raise RuntimeError("Trop de retries. Réduisez le batch ou attendez.")


def run():
    if not INPUT.exists():
        raise FileNotFoundError(f"Fichier introuvable: {INPUT}")

    print("🚀 Chargement RAG...")
    rag = get_rag()
    print("✅ RAG chargé\n")
    
    done = load_already_done()

    rows = read_csv_with_excel(INPUT)
    total = len(rows)

    out_fields = [
        "id_cas",
        "modele",
        "sample_id",
        "categorie_cas",
        "timestamp_reponse",
        "verdict_scientifique",
        "justification",
        "sources_utilisees",
        "topk",
        "batch_start",
        "batch_end",
        "timestamp_eval",
    ]

    file_exists = OUTPUT.exists()
    with open(OUTPUT, "a", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fields, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()

        for i, row in enumerate(rows, start=1):
            if not (BATCH_START <= i <= BATCH_END):
                continue

            # ✅ ORDRE CORRECT selon votre fichier
            # Colonnes: id_cas | categorie_cas | modele | sample_id | prompt | reponse_modele
            id_cas = pick(row, "id_cas", "idcas")
            categorie = pick(row, "categorie_cas", "categorie")
            modele = pick(row, "modele", "model")
            sample_id = pick(row, "sample_id", "sampleid")
            prompt = pick(row, "prompt", "question")
            response = pick(row, "reponse_modele", "reponse_model", "reponse")

            print(f"\n{'='*70}")
            print(f"📝 Ligne {i}/{total} (Batch {BATCH_START}-{BATCH_END})")
            print(f"   id_cas: '{id_cas}'")
            print(f"   categorie: '{categorie}'")
            print(f"   modele: '{modele}'")
            print(f"   sample_id: '{sample_id}'")
            print(f"   prompt: {prompt[:80] if prompt else '(VIDE)'}...")
            print(f"   response: {response[:80] if response else '(VIDE)'}...")

            key = (id_cas, modele, sample_id)
            if key in done and id_cas and modele and sample_id:
                print(f"↩️ Déjà évalué, skip")
                continue

            # Cas réponse vide
            if not response or not response.strip():
                writer.writerow({
                    "id_cas": id_cas,
                    "modele": modele,
                    "sample_id": sample_id,
                    "categorie_cas": categorie,
                    "timestamp_reponse": "",
                    "verdict_scientifique": "non_prouvee",
                    "justification": "Réponse vide: impossible d'évaluer.",
                    "sources_utilisees": "",
                    "topk": str(TOP_K),
                    "batch_start": str(BATCH_START),
                    "batch_end": str(BATCH_END),
                    "timestamp_eval": datetime.now().isoformat(timespec="seconds"),
                })
                print(f"⚠️ Réponse vide => non_prouvee")
                continue

            # Construire la requête RAG
            if prompt and prompt.strip():
                rag_query = f"{prompt}\n\n{response}"
            else:
                rag_query = response
            
            print(f"🔍 Recherche RAG (top_k={TOP_K})...")
            try:
                evidence = rag.search(rag_query, top_k=TOP_K)
                evidence_text, sources_str = build_evidence_text(evidence)
                print(f"✅ {len(evidence)} chunks trouvés")
            except Exception as e:
                print(f"❌ Erreur RAG: {e}")
                evidence_text = ""
                sources_str = ""

            # Appeler le juge
            print(f"⚖️ Évaluation par le juge...")
            try:
                raw = call_judge_with_retry(prompt, response, evidence_text)
                print(f"📄 Réponse brute: {raw[:150]}...")
            except Exception as e:
                print(f"❌ Erreur juge: {e}")
                raw = '{"verdict": "non_prouvee", "justification": "Erreur évaluation"}'

            # Parser le JSON
            verdict = "non_prouvee"
            justification = ""
            try:
                obj = json.loads(raw)
                verdict = obj.get("verdict", "non_prouvee")
                justification = obj.get("justification", "")
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON invalide: {e}")
                justification = f"(JSON invalide) {raw[:300]}"

            # Écrire le résultat
            writer.writerow({
                "id_cas": id_cas,
                "modele": modele,
                "sample_id": sample_id,
                "categorie_cas": categorie,
                "timestamp_reponse": "",
                "verdict_scientifique": verdict,
                "justification": justification,
                "sources_utilisees": sources_str,
                "topk": str(TOP_K),
                "batch_start": str(BATCH_START),
                "batch_end": str(BATCH_END),
                "timestamp_eval": datetime.now().isoformat(timespec="seconds"),
            })
            f_out.flush()  # Forcer l'écriture immédiate

            print(f"✅ Verdict: {verdict}")
            print(f"   Justification: {justification[:100]}...")

    print("\n" + "="*70)
    print(f"✅ TERMINÉ - Fichier: {OUTPUT}")


if __name__ == "__main__":
    run()