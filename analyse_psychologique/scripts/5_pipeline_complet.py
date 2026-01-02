#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 5 : Pipeline Complet d'Analyse Psychologique
===================================================

Ce module orchestre l'ensemble du processus d'analyse :
1. Chargement des données
2. Analyse du ton
3. Analyse de la crédibilité
4. Analyse de l'empathie
5. Benchmark des modèles
6. Génération des résultats

Utilisation :
    python 5_pipeline_complet.py
    
    Ou depuis Python:
    from pipeline_complet import executer_analyse_complete
    resultats = executer_analyse_complete(
        input_csv='../verification/outputs/evaluations_scientifiques.csv'
    )
"""

import pandas as pd
import sys
import os
from datetime import datetime

# Importer les modules d'analyse
try:
    from analyseur_ton import AnalyseurTon
    from analyseur_credibilite import AnalyseurCredibilite
    from analyseur_empathie import AnalyseurEmpathie
    from benchmark import BenchmarkModeles
except ImportError:
    # Si imports locaux échouent, utiliser imports directs
    import importlib.util
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Charger chaque module
    for module_name, file_name in [
        ('analyseur_ton', '1_analyseur_ton.py'),
        ('analyseur_credibilite', '2_analyseur_credibilite.py'),
        ('analyseur_empathie', '3_analyseur_empathie.py'),
        ('benchmark', '4_benchmark.py')
    ]:
        spec = importlib.util.spec_from_file_location(
            module_name,
            os.path.join(script_dir, file_name)
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    
    from analyseur_ton import AnalyseurTon
    from analyseur_credibilite import AnalyseurCredibilite
    from analyseur_empathie import AnalyseurEmpathie
    from benchmark import BenchmarkModeles


class PipelineAnalyse:
    """Pipeline complet d'analyse psychologique."""
    
    def __init__(self, input_csv: str, output_dir: str = '../outputs'):
        """
        Initialise le pipeline.
        
        Args:
            input_csv (str): Chemin vers evaluations_scientifiques.csv
            output_dir (str): Dossier de sortie pour les résultats
        """
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.df = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Créer le dossier de sortie si nécessaire
        os.makedirs(output_dir, exist_ok=True)
    
    
    def charger_donnees(self):
        """Charge et prépare les données."""
        print("=" * 80)
        print(" ÉTAPE 1 : CHARGEMENT DES DONNÉES")
        print("=" * 80)
        
        print(f" Chargement depuis : {self.input_csv}")
        
        try:
            self.df = pd.read_csv(self.input_csv, sep=';')
            
            # Nettoyer les colonnes vides
            self.df = self.df.dropna(axis=1, how='all')
            
            print(f" Données chargées : {len(self.df)} lignes")
            print(f" Modèles détectés : {self.df['modele'].nunique()}")
            print(f"   {list(self.df['modele'].unique())}")
            print(f" Catégories : {list(self.df['categorie_cas'].unique())}")
            
        except Exception as e:
            print(f" Erreur lors du chargement : {e}")
            sys.exit(1)
    
    
    def analyser_ton(self):
        """Effectue l'analyse du ton."""
        print("\n" + "=" * 80)
        print(" ÉTAPE 2 : ANALYSE DU TON")
        print("=" * 80)
        
        analyseur = AnalyseurTon()
        
        # Analyse du ton
        print(" Analyse du ton en cours...")
        resultats_ton = self.df['justification'].apply(analyseur.analyser)
        df_ton = pd.DataFrame(resultats_ton.tolist())
        
        # Analyse de l'anxiété induite
        print(" Analyse de l'anxiété induite...")
        resultats_anxiete = self.df.apply(
            lambda row: analyseur.analyser_anxiete_induite(
                row['justification'],
                row.get('categorie_cas')
            ),
            axis=1
        )
        df_anxiete = pd.DataFrame(resultats_anxiete.tolist())
        
        # Combiner
        self.df = pd.concat([self.df, df_ton, df_anxiete], axis=1)
        
        print(" Analyse du ton terminée")
        print(f"   • Colonnes ajoutées : {list(df_ton.columns)}")
    
    
    def analyser_credibilite(self):
        """Effectue l'analyse de la crédibilité."""
        print("\n" + "=" * 80)
        print(" ÉTAPE 3 : ANALYSE DE LA CRÉDIBILITÉ")
        print("=" * 80)
        
        analyseur = AnalyseurCredibilite()
        
        # Analyse de crédibilité
        print(" Analyse de crédibilité en cours...")
        resultats = self.df.apply(
            lambda row: analyseur.analyser(
                row['justification'],
                row.get('verdict_scientifique')
            ),
            axis=1
        )
        df_cred = pd.DataFrame(resultats.tolist())
        
        # Analyse du risque d'influence
        print(" Analyse du risque d'influence...")
        resultats_influence = self.df.apply(
            lambda row: analyseur.analyser_influence(
                row['justification'],
                row.get('verdict_scientifique')
            ),
            axis=1
        )
        df_influence = pd.DataFrame(resultats_influence.tolist())
        
        # Combiner
        self.df = pd.concat([self.df, df_cred, df_influence], axis=1)
        
        print(" Analyse de crédibilité terminée")
        print(f"   • Colonnes ajoutées : {list(df_cred.columns)}")
    
    
    def analyser_empathie(self):
        """Effectue l'analyse de l'empathie."""
        print("\n" + "=" * 80)
        print(" ÉTAPE 4 : ANALYSE DE L'EMPATHIE")
        print("=" * 80)
        
        analyseur = AnalyseurEmpathie()
        
        # Analyse d'empathie
        print(" Analyse d'empathie en cours...")
        resultats = self.df['justification'].apply(analyseur.analyser)
        df_empathie = pd.DataFrame(resultats.tolist())
        
        # Impact émotionnel
        print(" Analyse de l'impact émotionnel...")
        resultats_impact = self.df.apply(
            lambda row: analyseur.analyser_impact_emotionnel(
                row['justification'],
                {
                    'ton_rassurant': row.get('ton_rassurant', 0),
                    'ton_alarmiste': row.get('ton_alarmiste', 0)
                }
            ),
            axis=1
        )
        df_impact = pd.DataFrame(resultats_impact.tolist())
        
        # Combiner
        self.df = pd.concat([self.df, df_empathie, df_impact], axis=1)
        
        print(" Analyse d'empathie terminée")
        print(f"   • Colonnes ajoutées : {list(df_empathie.columns)}")
    
    
    def calculer_score_impact_global(self):
        """Calcule un score d'impact psychologique global."""
        print("\n Calcul du score d'impact psychologique global...")
        
        # Score combiné (0-1)
        self.df['impact_psychologique_global'] = (
            self.df['credibilite_percue'] * 0.30 +
            self.df['anxiete_induite'] * 0.25 +
            (1 - self.df['empathie_globale']) * 0.20 +
            self.df['risque_influence'] * 0.25
        )
        
        self.df['impact_psychologique_global'] = self.df['impact_psychologique_global'].round(3)
        
        print(" Score d'impact global calculé")
    
    
    def sauvegarder_resultats(self):
        """Sauvegarde tous les résultats."""
        print("\n" + "=" * 80)
        print(" ÉTAPE 5 : SAUVEGARDE DES RÉSULTATS")
        print("=" * 80)
        
        # Fichier principal avec toutes les analyses
        output_file = os.path.join(
            self.output_dir,
            f'analyse_psychologique_complete_{self.timestamp}.csv'
        )
        self.df.to_csv(output_file, index=False)
        print(f" Analyse complète sauvegardée : {output_file}")
        
        # Statistiques descriptives
        stats_file = os.path.join(
            self.output_dir,
            f'statistiques_descriptives_{self.timestamp}.csv'
        )
        
        colonnes_stats = [
            'ton_rassurant', 'ton_alarmiste', 'anxiete_induite',
            'credibilite_percue', 'risque_influence',
            'empathie_globale', 'impact_psychologique_global'
        ]
        
        stats = self.df.groupby('modele')[colonnes_stats].agg(['mean', 'std', 'min', 'max']).round(3)
        stats.to_csv(stats_file)
        print(f" Statistiques sauvegardées : {stats_file}")
        
        return output_file
    
    
    def executer_benchmark(self, csv_analyse: str):
        """Exécute le benchmark des modèles."""
        print("\n" + "=" * 80)
        print(" ÉTAPE 6 : BENCHMARK DES MODÈLES")
        print("=" * 80)
        
        benchmark = BenchmarkModeles(self.df)
        benchmark.afficher_resume()
        
        # Sauvegarder les résultats du benchmark
        rapport = benchmark.generer_rapport_complet()
        
        for nom, df_resultat in rapport.items():
            if isinstance(df_resultat, pd.DataFrame) and not df_resultat.empty:
                output_file = os.path.join(
                    self.output_dir,
                    f'benchmark_{nom}_{self.timestamp}.csv'
                )
                df_resultat.to_csv(output_file)
                print(f" {nom} sauvegardé")
    
    
    def executer(self):
        """Exécute le pipeline complet."""
        print("\n PIPELINE D'ANALYSE PSYCHOLOGIQUE - DÉMARRAGE")
        
        debut = datetime.now()
        
        # Étapes du pipeline
        self.charger_donnees()
        self.analyser_ton()
        self.analyser_credibilite()
        self.analyser_empathie()
        self.calculer_score_impact_global()
        csv_resultats = self.sauvegarder_resultats()
        self.executer_benchmark(csv_resultats)
        
        # Résumé final
        duree = (datetime.now() - debut).total_seconds()
        
        
        print("ANALYSE TERMINÉE AVEC SUCCÈS !")
        print(f"\n Durée totale : {duree:.2f} secondes")
        print(f" Lignes analysées : {len(self.df)}")
        print(f" Résultats dans : {self.output_dir}")
        print("\n" + "=" * 80)


def executer_analyse_complete(
    input_csv: str = None,
    output_dir: str = '../outputs'
) -> pd.DataFrame:
    """
    Fonction principale pour exécuter l'analyse complète.
    
    Args:
        input_csv (str): Chemin vers evaluations_scientifiques.csv
        output_dir (str): Dossier de sortie
        
    Returns:
        DataFrame avec toutes les analyses
    """
    # Chemin par défaut
    if input_csv is None:
        input_csv = '../verification/outputs/evaluations_scientifiques.csv'
    
    # Créer et exécuter le pipeline
    pipeline = PipelineAnalyse(input_csv, output_dir)
    pipeline.executer()
    
    return pipeline.df


if __name__ == "__main__":
    # Exécution en ligne de commande
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline d\'analyse psychologique')
    parser.add_argument(
        '--input',
        type=str,
        default='/mnt/user-data/uploads/evaluations_scientifiques.csv',
        help='Chemin vers le CSV d\'évaluation'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/claude/analyse_psychologique/outputs',
        help='Dossier de sortie'
    )
    
    args = parser.parse_args()
    
    # Exécuter
    executer_analyse_complete(input_csv=args.input, output_dir=args.output)
