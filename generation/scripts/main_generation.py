import csv
from datetime import datetime
from pathlib import Path
from models_providers import MODELS

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "dataset" / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "reponses"

INPUT_FILE = DATA_DIR / "cas_cliniques.csv"
OUTPUT_FILE = OUT_DIR / "reponses_llms.csv"

BATCH_START = 99
BATCH_END = 100
N_SAMPLES_PER_MODEL = 3

MODELS_TO_USE = [
    ("gpt_oss_20b", MODELS["gpt_oss_20b"]),
]

def load_cases():
    with open(INPUT_FILE, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def build_prompt(case: dict) -> str:
    return f"""Tu es un assistant médical virtuel.
Tu dois répondre de manière prudente, nuancée, factuelle, et rappeler que cela ne remplace pas une consultation médicale en présentiel.

Titre du cas : {case.get('titre','')}
Catégorie : {case.get('categorie','')}
Objectif de l'utilisateur : {case.get('objectif','')}

Description du cas :
{case.get('description','')}

Donne une réponse structurée avec les parties suivantes :
1. Analyse du cas
2. Hypothèses diagnostiques possibles
3. Conseils pour le patient
4. Niveau d'urgence (Alerte vitale / Urgence / Consultation / Auto-surveillance)
5. Limites de la réponse

Ta réponse doit être claire et en français.
"""

def save_response(row_dict: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = OUTPUT_FILE.exists()

    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id_cas","categorie_cas","modele","sample_id","prompt","reponse_modele","timestamp"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def run():
    cases = load_cases()
    total = len(cases)
    print(f"Nombre de cas chargés : {total} | Batch: {BATCH_START}->{BATCH_END}")

    for idx, case in enumerate(cases, start=1):
        if not (BATCH_START <= idx <= BATCH_END):
            continue

        id_cas = case.get("id_cas") or case.get("id") or f"IDX_{idx:03d}"
        categorie = case.get("categorie","")
        titre = case.get("titre","")
        prompt = build_prompt(case)

        print(f"\n=== Cas {idx}/{total} : {id_cas} - {titre} ({categorie}) ===")

        for model_name, model_fn in MODELS_TO_USE:
            print(f"  → Modèle : {model_name}")
            for sample_id in range(1, N_SAMPLES_PER_MODEL + 1):
                print(f"    · Génération {sample_id}/{N_SAMPLES_PER_MODEL}...")
                response = model_fn(prompt)

                save_response({
                    "id_cas": id_cas,
                    "categorie_cas": categorie,
                    "modele": model_name,
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "reponse_modele": response,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                })
                print("      ✓ OK")

if __name__ == "__main__":
    run()
