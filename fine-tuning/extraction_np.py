"""
Script 1 - Extraction des 7 Réponses NON PROUVÉES de Llama
Objectif: Identifier et analyser ces cas pour comprendre pourquoi
          le modèle a généré des réponses non validées
"""

import pandas as pd
from pathlib import Path
import json
import re

# ==================== CONFIGURATION ====================

racine = Path(r"C:\Users\ZAID\OneDrive\Documents\3eme_gds\DL\DeepLearning_1")
dataset_complet = racine / "dataset_complet.csv"
finetuning_dir = racine / "finetuning_data"
finetuning_dir.mkdir(parents=True, exist_ok=True)

# Fichiers de sortie
extraction_file = finetuning_dir / "01_llama_non_prouvees_EXTRACTION.csv"
analyse_file = finetuning_dir / "01_llama_non_prouvees_ANALYSE.txt"
template_correction_file = finetuning_dir / "01_llama_non_prouvees_TEMPLATE_CORRECTION.json"

# ==================== DÉTECTION VERDICTS ====================

def normaliser_verdict(verdict):
    if pd.isna(verdict):
        return None
    v = str(verdict).lower().strip()
    v = v.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('_', ' ')
    return v

def est_non_prouvee(verdict):
    v = normaliser_verdict(verdict)
    if v is None:
        return False
    return any(mot in v for mot in ['non prouvee', 'non prouve', 'non validee'])

# ==================== UTILITAIRES ROBUSTES ====================

def safe_float(x, default=0.0):
    """Convertit en float sans planter si NaN / vide / string."""
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def safe_int_from_id(value, fallback):
    """
    Convertit id_cas en entier si possible.
    Exemple: 'CAS_018' -> 18, '18' -> 18, sinon fallback.
    """
    if pd.isna(value):
        return fallback
    s = str(value).strip()
    # Si c'est déjà un nombre (ex: '18')
    if s.isdigit():
        return int(s)
    # Extraire le premier groupe de chiffres (ex: 'CAS_018' -> 018)
    m = re.search(r"\d+", s)
    if m:
        return int(m.group())
    return fallback

# ==================== CHARGEMENT ====================

print("="*70)
print("📂 EXTRACTION DES NON PROUVÉES DE LLAMA")
print("="*70)

df = pd.read_csv(dataset_complet, encoding='utf-8-sig')
print(f"✅ Dataset chargé: {len(df)} lignes")

# Filtrer Llama
modele_llama = "meta-llama/llama-4-scout-17b-16e-instruct"
llama = df[df['modele'] == modele_llama].copy()
print(f"✅ Réponses Llama: {len(llama)}")

# ==================== EXTRACTION ====================

print("\n" + "="*70)
print("🔍 IDENTIFICATION DES NON PROUVÉES")
print("="*70)

# Identifier les non prouvées
llama['est_non_prouvee'] = llama['verdict_scientifique'].apply(est_non_prouvee)
non_prouvees = llama[llama['est_non_prouvee']].copy()

print(f"\n🚨 Nombre de non prouvées trouvées: {len(non_prouvees)}")

if len(non_prouvees) == 0:
    print("\n✅ Aucune non prouvée détectée!")
    print("   Vérifiez les verdicts manuellement si vous pensez qu'il devrait y en avoir.")
    raise SystemExit(0)

# ==================== ANALYSE DÉTAILLÉE ====================

print("\n" + "="*70)
print("📋 ANALYSE DÉTAILLÉE DE CHAQUE CAS")
print("="*70)

colonnes_importantes = [
    'id_cas', 'titre', 'description', 'categorie', 'objectif',
    'prompt', 'reponse_texte', 'verdict_scientifique', 'justification',
    'sources_utilisees', 'ton_dominant', 'credibilite_percue',
    'score_certitude', 'score_empathie', 'niveau_anxiete',
    'longueur_reponse', 'nb_mots'
]

# Garder seulement les colonnes qui existent
colonnes_existantes = [col for col in colonnes_importantes if col in non_prouvees.columns]
non_prouvees_export = non_prouvees[colonnes_existantes].copy()

# Ajouter un numéro d'ordre
non_prouvees_export.insert(0, 'numero', range(1, len(non_prouvees_export) + 1))

# Afficher chaque cas
rapport_analyse = []
rapport_analyse.append("="*70)
rapport_analyse.append("ANALYSE DES RÉPONSES NON PROUVÉES DE LLAMA")
rapport_analyse.append("="*70)
rapport_analyse.append("")

for idx, row in non_prouvees_export.iterrows():
    num = row['numero']
    rapport_analyse.append(f"\n{'='*70}")
    rapport_analyse.append(f"CAS #{num}")
    rapport_analyse.append(f"{'='*70}")

    # Informations du cas
    if 'titre' in non_prouvees_export.columns:
        rapport_analyse.append(f"\n📋 Titre: {row.get('titre', 'N/A')}")
    if 'categorie' in non_prouvees_export.columns:
        rapport_analyse.append(f"📂 Catégorie: {row.get('categorie', 'N/A')}")
    if 'objectif' in non_prouvees_export.columns:
        rapport_analyse.append(f"🎯 Objectif: {row.get('objectif', 'N/A')}")

    # Description du cas
    if 'description' in non_prouvees_export.columns:
        rapport_analyse.append(f"\n📝 Description du cas:")
        rapport_analyse.append(f"{row.get('description', 'N/A')}")

    # Réponse de Llama
    if 'reponse_texte' in non_prouvees_export.columns:
        reponse = row.get('reponse_texte', '')
        reponse = '' if pd.isna(reponse) else str(reponse)
        rapport_analyse.append(f"\n🤖 Réponse de Llama:")
        rapport_analyse.append(f"{reponse[:500]}...")
        rapport_analyse.append(f"   Longueur: {row.get('longueur_reponse', 'N/A')} caractères")
        rapport_analyse.append(f"   Mots: {row.get('nb_mots', 'N/A')}")

    # Verdict et justification
    rapport_analyse.append(f"\n❌ Verdict: {row.get('verdict_scientifique', 'N/A')}")
    if 'justification' in non_prouvees_export.columns and pd.notna(row.get('justification', None)):
        rapport_analyse.append(f"\n📖 Justification:")
        rapport_analyse.append(f"{row.get('justification')}")

    if 'sources_utilisees' in non_prouvees_export.columns and pd.notna(row.get('sources_utilisees', None)):
        rapport_analyse.append(f"\n📚 Sources utilisées:")
        rapport_analyse.append(f"{row.get('sources_utilisees')}")

    # Impact psychologique
    rapport_analyse.append(f"\n💭 Impact Psychologique:")
    if 'credibilite_percue' in non_prouvees_export.columns:
        rapport_analyse.append(f"   • Crédibilité perçue: {safe_float(row.get('credibilite_percue')):.2f}/10")
    if 'score_certitude' in non_prouvees_export.columns:
        rapport_analyse.append(f"   • Certitude: {safe_float(row.get('score_certitude')):.2f}/10")
    if 'score_empathie' in non_prouvees_export.columns:
        rapport_analyse.append(f"   • Empathie: {safe_float(row.get('score_empathie')):.2f}/10")
    if 'ton_dominant' in non_prouvees_export.columns:
        rapport_analyse.append(f"   • Ton dominant: {row.get('ton_dominant', 'N/A')}")
    if 'niveau_anxiete' in non_prouvees_export.columns:
        rapport_analyse.append(f"   • Anxiété induite: {row.get('niveau_anxiete', 'N/A')}")

    rapport_analyse.append("")
    rapport_analyse.append("🔧 ACTION REQUISE: Corriger cette réponse manuellement")
    rapport_analyse.append("   → Ajouter validation scientifique")
    rapport_analyse.append("   → Citer sources fiables (HAS, OMS, PubMed)")
    rapport_analyse.append("")

    # Affichage console
    print(f"\n{'─'*70}")
    print(f"CAS #{num}: {row.get('titre', 'Sans titre')}")
    print(f"Catégorie: {row.get('categorie', 'N/A')}")
    print(f"Crédibilité: {safe_float(row.get('credibilite_percue')):.2f}/10")
    print(f"Longueur: {row.get('nb_mots', 0)} mots")

# ==================== CRÉATION TEMPLATE CORRECTION ====================

print("\n" + "="*70)
print("📝 CRÉATION DU TEMPLATE DE CORRECTION")
print("="*70)

template_corrections = []

for idx, row in non_prouvees_export.iterrows():
    cas_id_value = row.get('id_cas', None) if 'id_cas' in non_prouvees_export.columns else None
    cas_id_int = safe_int_from_id(cas_id_value, fallback=int(row['numero']))

    template = {
        # ✅ correction ici : plus de int('CAS_018')
        "cas_id": cas_id_int,
        # bonus: garder l'id original pour traçabilité
        "cas_id_original": None if pd.isna(cas_id_value) else str(cas_id_value),

        "numero": int(row['numero']),
        "categorie": row.get('categorie', 'N/A'),
        "titre": row.get('titre', 'N/A'),
        "description_cas": row.get('description', 'N/A'),

        # Réponse actuelle (non prouvée)
        "reponse_actuelle": {
            "texte": row.get('reponse_texte', 'N/A'),
            "probleme": "Non prouvée scientifiquement",
            "justification_probleme": row.get('justification', 'N/A')
        },

        # Template pour la correction
        "reponse_corrigee": {
            "texte": "À REMPLIR MANUELLEMENT - Réécrire la réponse avec validation scientifique",
            "sources_ajoutees": [
                "À AJOUTER - Source 1 (HAS, OMS, PubMed)",
                "À AJOUTER - Source 2",
                "À AJOUTER - Source 3"
            ],
            "modifications_effectuees": [
                "Ajout de validation scientifique",
                "Citation de sources fiables",
                "Reformulation pour plus de précision"
            ]
        },

        # Infos supplémentaires
        "impact_psycho": {
            "credibilite": safe_float(row.get('credibilite_percue', 0)),
            "ton": row.get('ton_dominant', 'N/A'),
            "anxiete": row.get('niveau_anxiete', 'N/A')
        },

        "instructions_correction": [
            "1. Rechercher des sources scientifiques fiables (PubMed, HAS, OMS, Cochrane)",
            "2. Vérifier la validité de l'information",
            "3. Réécrire la réponse en citant les sources",
            "4. Conserver le ton empathique si présent",
            "5. Ajouter les références dans 'sources_ajoutees'"
        ]
    }

    template_corrections.append(template)

# ==================== SAUVEGARDE ====================

print("\n" + "="*70)
print("💾 SAUVEGARDE DES FICHIERS")
print("="*70)

# 1. CSV avec toutes les données
non_prouvees_export.to_csv(extraction_file, index=False, encoding='utf-8-sig')
print(f"✅ CSV d'extraction: {extraction_file.name}")

# 2. Rapport d'analyse
with open(analyse_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(rapport_analyse))
print(f"✅ Rapport d'analyse: {analyse_file.name}")

# 3. Template JSON pour corrections
with open(template_correction_file, 'w', encoding='utf-8') as f:
    json.dump(template_corrections, f, indent=2, ensure_ascii=False)
print(f"✅ Template de correction: {template_correction_file.name}")

# ==================== STATISTIQUES ====================

print("\n" + "="*70)
print("📊 STATISTIQUES DES NON PROUVÉES")
print("="*70)

if 'categorie' in non_prouvees_export.columns:
    print("\n📂 Répartition par catégorie:")
    cat_counts = non_prouvees_export['categorie'].value_counts()
    for cat, count in cat_counts.items():
        print(f"   • {cat}: {count} cas")

if 'credibilite_percue' in non_prouvees_export.columns:
    print(f"\n💡 Crédibilité perçue:")
    print(f"   • Moyenne: {non_prouvees_export['credibilite_percue'].apply(safe_float).mean():.2f}/10")
    print(f"   • Min: {non_prouvees_export['credibilite_percue'].apply(safe_float).min():.2f}")
    print(f"   • Max: {non_prouvees_export['credibilite_percue'].apply(safe_float).max():.2f}")

    nb_credibles = (non_prouvees_export['credibilite_percue'].apply(safe_float) > 5).sum()
    print(f"   • Cas avec crédibilité > 5: {nb_credibles}")

if 'nb_mots' in non_prouvees_export.columns:
    print(f"\n📝 Longueur des réponses:")
    print(f"   • Moyenne: {non_prouvees_export['nb_mots'].mean():.0f} mots")
    print(f"   • Min: {non_prouvees_export['nb_mots'].min():.0f} mots")
    print(f"   • Max: {non_prouvees_export['nb_mots'].max():.0f} mots")

# ==================== INSTRUCTIONS FINALES ====================

print("\n" + "="*70)
print("📋 PROCHAINES ÉTAPES")
print("="*70)

print(f"""
✅ 3 fichiers créés dans: {finetuning_dir}

1️⃣ {extraction_file.name}
   → Ouvrir avec Excel pour voir tous les détails

2️⃣ {analyse_file.name}
   → Lire pour comprendre chaque cas

3️⃣ {template_correction_file.name}
   → Remplir manuellement les corrections

📝 COMMENT CORRIGER:

Pour chaque cas dans le fichier JSON:
1. Lire la description du cas
2. Lire la réponse actuelle (non prouvée)
3. Rechercher des sources scientifiques:
   • PubMed: https://pubmed.ncbi.nlm.nih.gov/
   • HAS: https://www.has-sante.fr/
   • OMS: https://www.who.int/
4. Réécrire la réponse avec validation scientifique
5. Ajouter les sources dans "sources_ajoutees"
6. Remplir "reponse_corrigee" > "texte"

Une fois les corrections terminées, passez au Script 2 pour créer
le dataset de fine-tuning!
""")

print("\n" + "="*70)
print("✅ EXTRACTION TERMINÉE")
print("="*70)
