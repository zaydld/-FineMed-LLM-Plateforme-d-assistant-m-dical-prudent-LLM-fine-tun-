"""
Script 2 - Préparation du Dataset de Fine-tuning
Objectif: Créer les paires (input → output) pour entraîner le modèle
          à partir des corrections manuelles
"""

import pandas as pd
import json
from pathlib import Path
import re
import random

# ==================== CONFIGURATION ====================

racine = Path(r"C:\Users\ZAID\OneDrive\Documents\3eme_gds\DL\DeepLearning_1")
dataset_complet = racine / "dataset_complet.csv"
finetuning_dir = racine / "finetuning_data"

# Fichiers d'entrée
template_correction_file = finetuning_dir / "01_llama_non_prouvees_TEMPLATE_CORRECTION.json"

# Fichiers de sortie
dataset_finetuning = finetuning_dir / "02_dataset_finetuning.jsonl"
dataset_train = finetuning_dir / "02_dataset_train.jsonl"
dataset_validation = finetuning_dir / "02_dataset_validation.jsonl"
rapport_dataset = finetuning_dir / "02_rapport_dataset.txt"

SYSTEM_PROMPT = (
    "Tu es un assistant médical expert. "
    "Tu réponds UNIQUEMENT avec des informations scientifiquement prouvées. "
    "Tu cites toujours tes sources (HAS, OMS, PubMed, études scientifiques). "
    "Tu es empathique et rassurant tout en restant précis et factuel."
)

MODELE_LLAMA = "meta-llama/llama-4-scout-17b-16e-instruct"

# ==================== UTILITAIRES ROBUSTES ====================

def safe_int_from_id(value, fallback=None):
    """
    Convertit id_cas en entier si possible.
    Ex: 'CAS_004' -> 4, '18' -> 18, sinon fallback.
    """
    if value is None or pd.isna(value):
        return fallback
    s = str(value).strip()
    if s.isdigit():
        return int(s)
    m = re.search(r"\d+", s)
    if m:
        return int(m.group())
    return fallback

def safe_text(x, default="N/A"):
    """Retourne un texte propre même si NaN."""
    if x is None or pd.isna(x):
        return default
    return str(x)

# ==================== VÉRIFICATION ====================

print("="*70)
print("📂 PRÉPARATION DU DATASET DE FINE-TUNING")
print("="*70)

if not template_correction_file.exists():
    print(f"\n❌ ERREUR: Fichier de corrections introuvable!")
    print(f"   Attendu: {template_correction_file}")
    print(f"\n📋 ÉTAPES À SUIVRE:")
    print("   1. Exécutez d'abord le Script 1 (extraction)")
    print("   2. Remplissez les corrections manuellement dans le JSON")
    print("   3. Relancez ce script")
    raise SystemExit(1)

print("✅ Fichier de corrections trouvé")

# ==================== CHARGEMENT DES CORRECTIONS ====================

print("\n" + "="*70)
print("📥 CHARGEMENT DES CORRECTIONS")
print("="*70)

with open(template_correction_file, 'r', encoding='utf-8') as f:
    corrections = json.load(f)

print(f"✅ {len(corrections)} cas chargés")

# Vérifier si les corrections ont été faites
nb_corriges = 0
nb_non_corriges = 0

for corr in corrections:
    texte_corrige = corr.get('reponse_corrigee', {}).get('texte', '')
    if "À REMPLIR MANUELLEMENT" not in safe_text(texte_corrige, ""):
        nb_corriges += 1
    else:
        nb_non_corriges += 1

print(f"\n📊 État des corrections:")
print(f"   ✅ Corrigés: {nb_corriges}")
print(f"   ⏳ À corriger: {nb_non_corriges}")

if nb_corriges == 0:
    print(f"\n⚠️  ATTENTION: Aucune correction n'a été faite!")
    print(f"   Veuillez d'abord corriger les cas dans:")
    print(f"   {template_correction_file}")
    print(f"\n   Cherchez 'À REMPLIR MANUELLEMENT' et remplacez par votre correction")
    raise SystemExit(1)

# ==================== CRÉATION DATASET FINE-TUNING ====================

print("\n" + "="*70)
print("🔧 CRÉATION DU DATASET DE FINE-TUNING")
print("="*70)

dataset_entries = []

for corr in corrections:
    texte_corrige = corr.get('reponse_corrigee', {}).get('texte', '')
    if "À REMPLIR MANUELLEMENT" in safe_text(texte_corrige, ""):
        continue

    entry = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": safe_text(corr.get('description_cas'))},
            {"role": "assistant", "content": safe_text(texte_corrige)}
        ],
        "metadata": {
            # ✅ cas_id peut être int ou string selon ton JSON; on le garde tel quel + on sécurise
            "cas_id": corr.get("cas_id"),
            "cas_id_original": corr.get("cas_id_original", None),
            "categorie": corr.get('categorie', 'N/A'),
            "sources": corr.get('reponse_corrigee', {}).get('sources_ajoutees', []),
            "type_correction": "non_prouvee_vers_prouvee"
        }
    }

    dataset_entries.append(entry)

print(f"✅ {len(dataset_entries)} paires créées (input → output corrigé)")

# ==================== AJOUT DES EXEMPLES POSITIFS ====================

print("\n" + "="*70)
print("➕ AJOUT DES EXEMPLES POSITIFS (réponses déjà bonnes)")
print("="*70)

df = pd.read_csv(dataset_complet, encoding='utf-8-sig')
llama = df[df['modele'] == MODELE_LLAMA].copy()

def normaliser_verdict(verdict):
    if pd.isna(verdict):
        return None
    v = str(verdict).lower().strip()
    return v.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('_', ' ')

def est_prouvee(verdict):
    v = normaliser_verdict(verdict)
    if v is None:
        return False
    # prouvee/validee/valide mais pas "non ..."
    return any(mot in v for mot in ['prouvee', 'prouve', 'validee', 'valide']) and 'non' not in v

llama['est_prouvee'] = llama['verdict_scientifique'].apply(est_prouvee)
prouvees = llama[llama['est_prouvee']].copy()

print(f"📊 Réponses prouvées de Llama: {len(prouvees)}")

nb_exemples = min(30, len(prouvees))
if nb_exemples == 0:
    print("⚠️ Aucun exemple positif trouvé (aucune réponse prouvée détectée).")
else:
    prouvees_sample = prouvees.sample(n=nb_exemples, random_state=42)
    print(f"✅ Ajout de {nb_exemples} exemples positifs")

    for idx, row in prouvees_sample.iterrows():
        id_cas_val = row.get('id_cas', None)
        cas_id_int = safe_int_from_id(id_cas_val, fallback=None)

        entry = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": safe_text(row.get('description'))},
                {"role": "assistant", "content": safe_text(row.get('reponse_texte'))}
            ],
            "metadata": {
                # ✅ correction du bug ici (plus de int('CAS_004'))
                "cas_id": cas_id_int,
                "cas_id_original": None if pd.isna(id_cas_val) else str(id_cas_val),
                "categorie": safe_text(row.get('categorie')),
                "type_correction": "exemple_positif"
            }
        }

        dataset_entries.append(entry)

print(f"\n📊 Dataset total: {len(dataset_entries)} exemples")
print(f"   • Corrections (non prouvée → prouvée): {nb_corriges}")
print(f"   • Exemples positifs (déjà prouvées): {nb_exemples}")

# ==================== SPLIT TRAIN / VALIDATION ====================

print("\n" + "="*70)
print("✂️  SPLIT TRAIN / VALIDATION")
print("="*70)

random.seed(42)
random.shuffle(dataset_entries)

split_idx = int(len(dataset_entries) * 0.8)
train_data = dataset_entries[:split_idx]
val_data = dataset_entries[split_idx:]

print(f"✅ Train: {len(train_data)} exemples (80%)")
print(f"✅ Validation: {len(val_data)} exemples (20%)")

# ==================== SAUVEGARDE ====================

print("\n" + "="*70)
print("💾 SAUVEGARDE DES DATASETS")
print("="*70)

def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

save_jsonl(dataset_entries, dataset_finetuning)
print(f"✅ Dataset complet: {dataset_finetuning.name}")

save_jsonl(train_data, dataset_train)
print(f"✅ Dataset train: {dataset_train.name}")

save_jsonl(val_data, dataset_validation)
print(f"✅ Dataset validation: {dataset_validation.name}")

# ==================== RAPPORT ====================

rapport = []
rapport.append("="*70)
rapport.append("RAPPORT DU DATASET DE FINE-TUNING")
rapport.append("="*70)
rapport.append("")
rapport.append("📊 Statistiques:")
rapport.append(f"   • Total exemples: {len(dataset_entries)}")
rapport.append(f"   • Corrections (non prouvée → prouvée): {nb_corriges}")
rapport.append(f"   • Exemples positifs: {nb_exemples}")
rapport.append("")
rapport.append("✂️  Split:")
rapport.append(f"   • Train: {len(train_data)} exemples (80%)")
rapport.append(f"   • Validation: {len(val_data)} exemples (20%)")
rapport.append("")
rapport.append("📁 Fichiers créés:")
rapport.append(f"   • {dataset_finetuning.name}")
rapport.append(f"   • {dataset_train.name}")
rapport.append(f"   • {dataset_validation.name}")
rapport.append("")
rapport.append("🎯 Objectifs du fine-tuning:")
rapport.append("   1. Éliminer les réponses non prouvées")
rapport.append("   2. Toujours citer des sources scientifiques")
rapport.append("   3. Maintenir l'empathie et le ton approprié")
rapport.append("")
rapport.append("📋 Format du dataset:")
rapport.append("   • Format: JSONL (JSON Lines)")
rapport.append("   • Structure: messages conversationnels")
rapport.append("   • Système prompt: instructions pour validation scientifique")
rapport.append("")
rapport.append("🔧 Prochaine étape:")
rapport.append("   Script 3 - Configuration et lancement du fine-tuning")
rapport.append("")

with open(rapport_dataset, 'w', encoding='utf-8') as f:
    f.write('\n'.join(rapport))

print(f"✅ Rapport: {rapport_dataset.name}")

# ==================== APERÇU ====================

print("\n" + "="*70)
print("👀 APERÇU DU DATASET")
print("="*70)

if len(train_data) > 0:
    exemple = train_data[0]
    print("\n📋 Exemple d'entrée de dataset:")
    print("-" * 70)
    print(f"SYSTÈME: {exemple['messages'][0]['content'][:100]}...")
    print(f"\nUSER: {exemple['messages'][1]['content'][:150]}...")
    print(f"\nASSISTANT: {exemple['messages'][2]['content'][:200]}...")
    print("-" * 70)

# ==================== INSTRUCTIONS FINALES ====================

print("\n" + "="*70)
print("📋 PROCHAINES ÉTAPES")
print("="*70)

print(f"""
✅ Dataset de fine-tuning prêt!

📂 Fichiers créés dans: {finetuning_dir}
   • {dataset_train.name} ({len(train_data)} exemples)
   • {dataset_validation.name} ({len(val_data)} exemples)

🎯 Ce dataset permettra à Llama d'apprendre à:
   1. ✅ Ne jamais générer de réponses non prouvées
   2. ✅ Toujours valider scientifiquement ses réponses
   3. ✅ Citer des sources fiables
   4. ✅ Maintenir un ton empathique

🚀 Prochaine étape: Script 3
   → Configuration du fine-tuning (LoRA/QLoRA)
   → Choix des hyperparamètres
   → Lancement de l'entraînement

💡 Conseil:
   Avant de fine-tuner, vérifiez que toutes vos corrections
   dans le fichier JSON sont de qualité et citent des sources
   scientifiques fiables!
""")

print("\n" + "="*70)
print("✅ PRÉPARATION DU DATASET TERMINÉE")
print("="*70)
