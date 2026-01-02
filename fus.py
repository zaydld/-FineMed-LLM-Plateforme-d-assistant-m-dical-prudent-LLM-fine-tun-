import os
from pathlib import Path
import pandas as pd
import yaml
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION DES CHEMINS ====================

class ProjetConfig:
    """Configuration centralisée du projet avec chemins réels"""
    
    def __init__(self, racine_projet=None):
        """
        Initialise la configuration avec vos chemins existants
        
        Args:
            racine_projet: Chemin vers la racine du projet. 
                          Par défaut: C:/Users/ZAID/OneDrive/Documents/3eme_gds/DL/DeepLearning_1
        """
        if racine_projet is None:
            # Utiliser votre chemin par défaut
            self.racine = Path(r"C:\Users\ZAID\OneDrive\Documents\3eme_gds\DL\DeepLearning_1")
        else:
            self.racine = Path(racine_projet)
        
        # Définir tous les chemins
        self._definir_chemins()
    
    def _definir_chemins(self):
        """Définit tous les chemins du projet"""
        
        # ========== DOSSIERS EXISTANTS ==========
        self.dataset_dir = self.racine / "dataset" / "data"
        self.generation_dir = self.racine / "generation" / "reponses"
        self.analyse_psycho_dir = self.racine / "analyse_psychologique" / "outputs"
        
        # ========== FICHIERS SOURCES (VOS DONNÉES) ==========
        self.cas_cliniques = self.dataset_dir / "cas_cliniques.csv"
        self.reponses_llms = self.generation_dir / "reponses_llms_structured.csv"
        self.evaluations = self.analyse_psycho_dir / "analyse_complete_CLEAN.csv"
        
        # ========== NOUVEAU DOSSIER POUR ANALYSE ==========
        self.analyse_dir = self.racine / "analyse_finale"
        
        # Sous-dossiers analyse
        self.data_processed = self.analyse_dir / "data_processed"
        self.data_exports = self.analyse_dir / "data_exports"
        self.resultats = self.analyse_dir / "resultats"
        self.figures = self.resultats / "figures"
        self.tableaux = self.resultats / "tableaux"
        self.rapports = self.resultats / "rapports"
        self.models = self.analyse_dir / "models"
        self.scripts = self.analyse_dir / "scripts"
        self.notebooks = self.analyse_dir / "notebooks"
        self.config_dir = self.analyse_dir / "config"
        
        # ========== FICHIERS TRAITÉS ==========
        self.dataset_complet = self.data_processed / "dataset_complet.csv"
        self.dataset_complet_xlsx = self.data_processed / "dataset_complet.xlsx"
        self.dataset_nettoye = self.data_processed / "dataset_nettoye.csv"
        
        # ========== EXPORTS D'ANALYSES ==========
        self.stats_par_modele = self.data_exports / "stats_par_modele.csv"
        self.stats_par_categorie = self.data_exports / "stats_par_categorie.csv"
        self.analyses_croisees = self.data_exports / "analyses_croisees.csv"
        self.correlations = self.data_exports / "correlations.csv"
        self.patterns_problematiques = self.data_exports / "patterns_problematiques.csv"
        self.scores_modeles = self.data_exports / "scores_modeles.csv"
        self.benchmark_final = self.data_exports / "benchmark_final.csv"
        
        # ========== FIGURES ==========
        self.fig_comparaison = self.figures / "comparaison_modeles.png"
        self.fig_heatmap = self.figures / "heatmap_performance.png"
        self.fig_radar = self.figures / "radar_chart.png"
        self.fig_distribution_verdicts = self.figures / "distribution_verdicts.png"
        self.fig_correlation = self.figures / "correlation_validite_psycho.png"
        self.fig_patterns = self.figures / "patterns_problematiques.png"
        self.fig_performance_categorie = self.figures / "performance_par_categorie.png"
        
        # ========== TABLEAUX ==========
        self.tableau_benchmark = self.tableaux / "benchmark_final.xlsx"
        self.tableau_stats_desc = self.tableaux / "statistiques_descriptives.xlsx"
        self.tableau_analyse_croisee = self.tableaux / "analyse_croisee.xlsx"
        self.tableau_scores = self.tableaux / "scores_modeles.xlsx"
        
        # ========== RAPPORTS ==========
        self.rapport_selection = self.rapports / "rapport_selection_modele.pdf"
        self.rapport_patterns = self.rapports / "analyse_patterns_problematiques.pdf"
        self.rapport_final = self.rapports / "rapport_final_complet.pdf"
        
        # ========== CONFIG ==========
        self.config_yaml = self.config_dir / "config.yaml"
    
    def creer_structure_analyse(self):
        """Crée la structure de dossiers pour l'analyse finale"""
        
        dossiers = [
            self.analyse_dir,
            self.data_processed,
            self.data_exports,
            self.resultats,
            self.figures,
            self.tableaux,
            self.rapports,
            self.models,
            self.scripts,
            self.notebooks,
            self.config_dir
        ]
        
        print("="*70)
        print("📁 CRÉATION DE LA STRUCTURE D'ANALYSE")
        print("="*70)
        
        for dossier in dossiers:
            dossier.mkdir(parents=True, exist_ok=True)
            try:
                chemin_relatif = dossier.relative_to(self.racine)
            except:
                chemin_relatif = dossier
            print(f"   ✅ {chemin_relatif}/")
        
        print(f"\n✅ Structure créée dans: {self.analyse_dir}")
    
    def verifier_fichiers_sources(self):
        """Vérifie la présence de VOS fichiers sources"""
        
        fichiers_sources = {
            'Cas cliniques': self.cas_cliniques,
            'Réponses LLMs': self.reponses_llms,
            'Analyses complètes': self.evaluations
        }
        
        print("\n" + "="*70)
        print("🔍 VÉRIFICATION DES FICHIERS SOURCES")
        print("="*70)
        
        tous_presents = True
        
        for nom, chemin in fichiers_sources.items():
            if chemin.exists():
                taille = chemin.stat().st_size / 1024  # Ko
                print(f"   ✅ {nom}")
                print(f"      {chemin}")
                print(f"      Taille: {taille:.1f} Ko")
            else:
                print(f"   ❌ {nom} - INTROUVABLE")
                print(f"      Cherché ici: {chemin}")
                tous_presents = False
        
        print("="*70)
        return tous_presents
    
    def afficher_chemins(self):
        """Affiche tous les chemins du projet"""
        
        print("\n" + "="*70)
        print("📂 CHEMINS DU PROJET")
        print("="*70)
        
        print("\n🗂️  RACINE DU PROJET:")
        print(f"   {self.racine}")
        
        print("\n📥 FICHIERS SOURCES (existants):")
        print(f"   • Cas cliniques:")
        print(f"     {self.cas_cliniques}")
        print(f"   • Réponses LLMs:")
        print(f"     {self.reponses_llms}")
        print(f"   • Analyses psycho:")
        print(f"     {self.evaluations}")
        
        print("\n📤 DOSSIER D'ANALYSE (nouveau):")
        print(f"   {self.analyse_dir}")
        
        print("\n💾 FICHIERS QUI SERONT CRÉÉS:")
        print(f"   • Dataset complet CSV:")
        print(f"     {self.dataset_complet}")
        print(f"   • Dataset complet Excel:")
        print(f"     {self.dataset_complet_xlsx}")
        
        print("\n📊 EXPORTS D'ANALYSES:")
        print(f"   • Stats par modèle: {self.stats_par_modele.name}")
        print(f"   • Analyses croisées: {self.analyses_croisees.name}")
        print(f"   • Patterns problématiques: {self.patterns_problematiques.name}")
        print(f"   • Scores des modèles: {self.scores_modeles.name}")
        print(f"   • Benchmark final: {self.benchmark_final.name}")
        
        print("\n📈 FIGURES:")
        print(f"   • Comparaison modèles: {self.fig_comparaison.name}")
        print(f"   • Heatmap performance: {self.fig_heatmap.name}")
        print(f"   • Radar chart: {self.fig_radar.name}")
        print(f"   • Distribution verdicts: {self.fig_distribution_verdicts.name}")
        
        print("\n📋 TABLEAUX:")
        print(f"   • Benchmark final: {self.tableau_benchmark.name}")
        print(f"   • Stats descriptives: {self.tableau_stats_desc.name}")
        print(f"   • Analyse croisée: {self.tableau_analyse_croisee.name}")
        
        print("\n📄 RAPPORTS:")
        print(f"   • Sélection modèle: {self.rapport_selection.name}")
        print(f"   • Analyse patterns: {self.rapport_patterns.name}")
        print(f"   • Rapport final: {self.rapport_final.name}")
        
        print("="*70)
    
    def sauvegarder_config(self):
        """Sauvegarde la configuration en YAML"""
        
        config_dict = {
            'projet': {
                'nom': 'Analyse Foundation Models Médicaux',
                'version': '1.0.0',
                'racine': str(self.racine),
                'analyse_dir': str(self.analyse_dir)
            },
            'fichiers_sources': {
                'cas_cliniques': str(self.cas_cliniques),
                'reponses_llms': str(self.reponses_llms),
                'evaluations': str(self.evaluations)
            },
            'dossiers_sortie': {
                'data_processed': str(self.data_processed),
                'data_exports': str(self.data_exports),
                'figures': str(self.figures),
                'tableaux': str(self.tableaux),
                'rapports': str(self.rapports)
            },
            'fichiers_sortie': {
                'dataset_complet': str(self.dataset_complet),
                'stats_par_modele': str(self.stats_par_modele),
                'benchmark': str(self.benchmark_final)
            }
        }
        
        with open(self.config_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n💾 Configuration sauvegardée: {self.config_yaml}")


# ==================== FUSION DES DONNÉES ====================

def nettoyer_colonnes(df):
    """Nettoie les noms de colonnes (supprime caractères invisibles)"""
    df.columns = df.columns.str.strip().str.replace('﻿', '').str.replace('\ufeff', '')
    return df


def fusionner_donnees(config):
    """
    Fusionne vos 3 fichiers CSV existants
    
    Args:
        config: Instance de ProjetConfig
    
    Returns:
        DataFrame fusionné ou None en cas d'erreur
    """
    
    print("\n" + "="*70)
    print("🚀 FUSION DES DONNÉES")
    print("="*70)
    
    # ========== VÉRIFICATION ==========
    if not config.verifier_fichiers_sources():
        print("\n❌ ERREUR: Fichiers sources introuvables!")
        print("   Vérifiez que les chemins sont corrects.")
        return None
    
    # ========== CHARGEMENT ==========
    print("\n📂 Étape 1/7: Chargement des fichiers...")
    
    try:
        print(f"\n   Lecture: {config.cas_cliniques.name}")
        cas = pd.read_csv(config.cas_cliniques, encoding='utf-8-sig')
        print(f"   ✅ {len(cas)} lignes, {len(cas.columns)} colonnes")
        print(f"      Colonnes: {list(cas.columns)}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return None
    
    try:
        print(f"\n   Lecture: {config.reponses_llms.name}")
        reponses = pd.read_csv(config.reponses_llms, encoding='utf-8-sig')
        print(f"   ✅ {len(reponses)} lignes, {len(reponses.columns)} colonnes")
        print(f"      Colonnes: {list(reponses.columns)}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return None
    
    try:
        print(f"\n   Lecture: {config.evaluations.name}")
        eval_complete = pd.read_csv(config.evaluations, encoding='utf-8-sig')
        print(f"   ✅ {len(eval_complete)} lignes, {len(eval_complete.columns)} colonnes")
        print(f"      Colonnes: {list(eval_complete.columns[:10])}... (et {len(eval_complete.columns)-10} autres)")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return None
    
    # ========== NETTOYAGE ==========
    print("\n🧹 Étape 2/7: Nettoyage des colonnes...")
    
    cas = nettoyer_colonnes(cas)
    reponses = nettoyer_colonnes(reponses)
    eval_complete = nettoyer_colonnes(eval_complete)
    
    print(f"   ✅ Nettoyage terminé")
    
    # ========== STANDARDISATION ==========
    print("\n🔄 Étape 3/7: Standardisation des noms...")
    
    # Renommer 'id' en 'id_cas' dans cas_cliniques
    if 'id' in cas.columns and 'id_cas' not in cas.columns:
        cas = cas.rename(columns={'id': 'id_cas'})
        print(f"   ✅ Renommage: 'id' → 'id_cas'")
    
    # Renommer 'reponse_modele' en 'reponse_texte' si existe
    if 'reponse_modele' in reponses.columns:
        reponses = reponses.rename(columns={'reponse_modele': 'reponse_texte'})
        print(f"   ✅ Renommage: 'reponse_modele' → 'reponse_texte'")
    
    # ========== VÉRIFICATION COHÉRENCE ==========
    print("\n🔍 Étape 4/7: Vérification de la cohérence...")
    
    # Vérifier les colonnes clés
    print(f"\n   Colonnes dans cas_cliniques: {list(cas.columns)}")
    print(f"   Colonnes dans reponses_llms: {list(reponses.columns)}")
    print(f"   Colonnes dans evaluations: {list(eval_complete.columns[:15])}...")
    
    # Identifier la colonne ID dans chaque fichier
    id_col_cas = 'id_cas' if 'id_cas' in cas.columns else 'id'
    id_col_rep = 'id_cas' if 'id_cas' in reponses.columns else None
    id_col_eval = 'id_cas' if 'id_cas' in eval_complete.columns else None
    
    if not id_col_rep:
        print(f"   ⚠️ Colonne 'id_cas' non trouvée dans reponses_llms")
        print(f"      Colonnes disponibles: {list(reponses.columns)}")
    
    if not id_col_eval:
        print(f"   ⚠️ Colonne 'id_cas' non trouvée dans evaluations")
        print(f"      Colonnes disponibles: {list(eval_complete.columns)}")
    
    cas_ids = set(cas[id_col_cas])
    rep_ids = set(reponses[id_col_rep]) if id_col_rep else set()
    eval_ids = set(eval_complete[id_col_eval]) if id_col_eval else set()
    
    print(f"\n   • Cas cliniques: {len(cas_ids)} ID uniques")
    print(f"   • Réponses: {len(rep_ids)} ID uniques")
    print(f"   • Évaluations: {len(eval_ids)} ID uniques")
    
    if 'modele' in reponses.columns:
        modeles_rep = set(reponses['modele'].unique())
        print(f"   • Modèles dans réponses: {modeles_rep}")
    
    if 'modele' in eval_complete.columns:
        modeles_eval = set(eval_complete['modele'].unique())
        print(f"   • Modèles dans évaluations: {modeles_eval}")
    
    # ========== FUSION 1: CAS + RÉPONSES ==========
    print("\n🔗 Étape 5/7: Fusion Cas + Réponses...")
    
    # Déterminer les colonnes de jointure
    if id_col_cas in cas.columns and id_col_rep in reponses.columns:
        df = cas.merge(
            reponses,
            left_on=id_col_cas,
            right_on=id_col_rep,
            how='inner',
            suffixes=('_cas', '_rep')
        )
        print(f"   ✅ Fusion réussie: {len(df)} lignes")
        print(f"      • Cas uniques: {df[id_col_cas].nunique()}")
        if 'modele' in df.columns:
            print(f"      • Modèles: {df['modele'].nunique()}")
    else:
        print(f"   ❌ Impossible de fusionner: colonnes ID incompatibles")
        return None
    
    # Standardiser le nom de la colonne ID
    if id_col_cas != 'id_cas':
        df = df.rename(columns={id_col_cas: 'id_cas'})
    
    # Gérer les colonnes dupliquées (categorie)
    if 'categorie_cas' in df.columns and 'categorie_rep' in df.columns:
        df['categorie'] = df['categorie_cas']
        df = df.drop(columns=['categorie_cas', 'categorie_rep'])
        print(f"   ✅ Colonnes 'categorie' fusionnées")
    elif 'categorie' not in df.columns:
        if 'categorie_cas' in df.columns:
            df['categorie'] = df['categorie_cas']
        elif 'categorie_rep' in df.columns:
            df['categorie'] = df['categorie_rep']
    
    # ========== FUSION 2: + ÉVALUATIONS ==========
    print("\n🔗 Étape 6/7: Fusion + Évaluations...")
    
    # Déterminer les colonnes de jointure
    join_cols = []
    if 'id_cas' in df.columns and 'id_cas' in eval_complete.columns:
        join_cols.append('id_cas')
    if 'modele' in df.columns and 'modele' in eval_complete.columns:
        join_cols.append('modele')
    if 'sample_id' in df.columns and 'sample_id' in eval_complete.columns:
        join_cols.append('sample_id')
    
    if not join_cols:
        print(f"   ❌ Aucune colonne commune pour la fusion!")
        print(f"      Colonnes df: {list(df.columns)}")
        print(f"      Colonnes eval: {list(eval_complete.columns)}")
        return None
    
    print(f"   • Jointure sur: {join_cols}")
    
    df_final = df.merge(
        eval_complete,
        on=join_cols,
        how='inner',
        suffixes=('', '_eval')
    )
    
    print(f"   ✅ Fusion réussie: {len(df_final)} lignes")
    
    # Supprimer colonnes dupliquées
    cols_dupli = [col for col in df_final.columns if col.endswith('_eval')]
    if cols_dupli:
        print(f"   🧹 Suppression de {len(cols_dupli)} colonnes dupliquées")
        df_final = df_final.drop(columns=cols_dupli)
    
    # ========== AJOUT MÉTADONNÉES ==========
    print("\n📊 Étape 7/7: Ajout de métadonnées...")
    
    # Identifier la colonne de réponse
    col_reponse = None
    for possible in ['reponse_texte', 'reponse_modele', 'reponse']:
        if possible in df_final.columns:
            col_reponse = possible
            break
    
    if col_reponse:
        df_final['longueur_reponse'] = df_final[col_reponse].astype(str).str.len()
        df_final['nb_mots'] = df_final[col_reponse].astype(str).str.split().str.len()
        print(f"   ✅ Métadonnées ajoutées (basées sur '{col_reponse}')")
        print(f"      • Longueur moyenne: {df_final['longueur_reponse'].mean():.0f} caractères")
        print(f"      • Mots moyens: {df_final['nb_mots'].mean():.0f} mots")
    else:
        print(f"   ⚠️ Colonne de réponse non trouvée")
    
    # ========== STATISTIQUES FINALES ==========
    print("\n" + "="*70)
    print("📈 STATISTIQUES DU DATASET FUSIONNÉ")
    print("="*70)
    
    print(f"\n📊 Dimensions:")
    print(f"   • Lignes totales: {len(df_final)}")
    print(f"   • Colonnes: {len(df_final.columns)}")
    print(f"   • Cas uniques: {df_final['id_cas'].nunique()}")
    
    if 'modele' in df_final.columns:
        print(f"   • Modèles: {df_final['modele'].nunique()}")
        print(f"\n🤖 Répartition par modèle:")
        for modele, count in df_final['modele'].value_counts().items():
            pct = count / len(df_final) * 100
            print(f"   • {modele}: {count} réponses ({pct:.1f}%)")
    
    if 'categorie' in df_final.columns:
        print(f"\n📁 Répartition par catégorie:")
        for cat, count in df_final['categorie'].value_counts().items():
            pct = count / len(df_final) * 100
            print(f"   • {cat}: {count} ({pct:.1f}%)")
    
    if 'verdict_scientifique' in df_final.columns:
        print(f"\n🔬 Répartition des verdicts:")
        for verdict, count in df_final['verdict_scientifique'].value_counts().items():
            pct = count / len(df_final) * 100
            print(f"   • {verdict}: {count} ({pct:.1f}%)")
    
    # Valeurs manquantes
    missing = df_final.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        print(f"\n⚠️ Données manquantes: {total_missing} valeurs")
        cols_missing = missing[missing > 0].head(10)
        for col, count in cols_missing.items():
            pct = count / len(df_final) * 100
            print(f"   • {col}: {count} ({pct:.1f}%)")
        if len(missing[missing > 0]) > 10:
            print(f"   ... et {len(missing[missing > 0]) - 10} autres colonnes")
    else:
        print(f"\n✅ Aucune donnée manquante!")
    
    # ========== SAUVEGARDE ==========
    print("\n💾 Sauvegarde du dataset fusionné...")
    
    # CSV
    df_final.to_csv(config.dataset_complet, index=False, encoding='utf-8-sig')
    taille_csv = config.dataset_complet.stat().st_size / 1024 / 1024
    print(f"   ✅ CSV créé: {config.dataset_complet.name} ({taille_csv:.2f} Mo)")
    print(f"      Chemin: {config.dataset_complet}")
    
    # Excel
    try:
        df_final.to_excel(config.dataset_complet_xlsx, index=False, engine='openpyxl')
        taille_xlsx = config.dataset_complet_xlsx.stat().st_size / 1024 / 1024
        print(f"   ✅ Excel créé: {config.dataset_complet_xlsx.name} ({taille_xlsx:.2f} Mo)")
        print(f"      Chemin: {config.dataset_complet_xlsx}")
    except Exception as e:
        print(f"   ⚠️ Excel non créé: {e}")
        print(f"      Installer: pip install openpyxl")
    
    print("\n" + "="*70)
    print("✅ FUSION TERMINÉE AVEC SUCCÈS!")
    print("="*70)
    
    return df_final


# ==================== SCRIPT PRINCIPAL ====================

def main():
    """Script principal d'initialisation et de fusion"""
    
    print("\n" + "="*70)
    print("🎯 PROJET: ANALYSE FOUNDATION MODELS MÉDICAUX")
    print("="*70)
    
    # 1. Créer la configuration avec VOS chemins
    print("\n⚙️  Initialisation de la configuration...")
    config = ProjetConfig()
    print(f"   ✅ Racine du projet: {config.racine}")
    
    # 2. Créer la structure pour l'analyse
    config.creer_structure_analyse()
    
    # 3. Afficher tous les chemins
    config.afficher_chemins()
    
    # 4. Vérifier les fichiers sources
    fichiers_ok = config.verifier_fichiers_sources()
    
    if not fichiers_ok:
        print("\n❌ Des fichiers sources sont manquants!")
        print("   Vérifiez les chemins dans la classe ProjetConfig")
        return config, None
    
    # 5. Sauvegarder la config
    config.sauvegarder_config()
    
    # 6. Demander si on fusionne
    print("\n" + "="*70)
    print("❓ LANCER LA FUSION DES DONNÉES ?")
    print("="*70)
    print("   [o] Oui, fusionner maintenant")
    print("   [n] Non, juste créer la structure")
    print()
    
    try:
        reponse = input("Votre choix (o/n): ").lower().strip()
        
        if reponse == 'o':
            print("\n🚀 Lancement de la fusion...")
            df = fusionner_donnees(config)
            
            if df is not None:
                print("\n" + "="*70)
                print("🎉 SUCCÈS COMPLET!")
                print("="*70)
                print(f"""
✅ Dataset fusionné créé!

📊 Résumé:
   • {len(df):,} lignes
   • {len(df.columns)} colonnes
   • {df['id_cas'].nunique()} cas uniques
   • {df['modele'].nunique() if 'modele' in df.columns else 'N/A'} modèles
   
📍 Fichiers créés:
   • CSV: {config.dataset_complet}
   • Excel: {config.dataset_complet_xlsx}

🎯 Prochaines étapes:
   1. Ouvrir le fichier Excel pour visualiser
   2. Exécuter les analyses statistiques
   3. Créer les visualisations
   4. Sélectionner le meilleur modèle
                """)
                return config, df
            else:
                print("\n❌ La fusion a échoué.")
                return config, None
        else:
            print("\n✅ Structure créée!")
            print(f"   Dossier d'analyse: {config.analyse_dir}")
            return config, None
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Opération annulée.")
        return config, None
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return config, None


# ==================== POINT D'ENTRÉE ====================

if __name__ == "__main__":
    config, df = main()
    
    if df is not None:
        print("\n💡 Variables disponibles:")
        print("   • config : Configuration du projet")
        print("   • df     : DataFrame fusionné")