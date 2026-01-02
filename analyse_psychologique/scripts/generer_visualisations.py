#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Visualisation des Résultats d'Analyse Psychologique
============================================================

Génère des graphiques pour visualiser :
- Comparaison des modèles
- Distribution des scores
- Corrélation verdict scientifique vs crédibilité
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Pour sauvegarder sans affichage
import numpy as np
import os

# Configuration matplotlib pour un rendu professionnel
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def creer_graphique_benchmark(csv_scores: str, output_dir: str):
    """Crée un graphique de comparaison des modèles."""
    
    df = pd.read_csv(csv_scores)
    
    # Nettoyer les données (enlever les NaN)
    df = df.dropna()
    
    if len(df) == 0:
        print("⚠️ Pas de données valides pour le graphique benchmark")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Barres groupées
    x = np.arange(len(df))
    width = 0.25
    
    ax.bar(x - width, df['empathie'], width, label='Empathie', color='#2ecc71')
    ax.bar(x, df['alignement'], width, label='Alignement Crédibilité', color='#3498db')
    ax.bar(x + width, df['gestion_anxiete'], width, label='Gestion Anxiété', color='#e74c3c')
    
    ax.set_xlabel('Modèle', fontweight='bold')
    ax.set_ylabel('Score (/100)', fontweight='bold')
    ax.set_title('Comparaison des Modèles LLM - Impact Psychologique', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['modele'], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'graphique_benchmark_modeles.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Graphique benchmark sauvegardé: {output_path}")


def creer_graphique_scores_globaux(csv_scores: str, output_dir: str):
    """Crée un graphique des scores globaux."""
    
    df = pd.read_csv(csv_scores).dropna()
    
    if len(df) == 0:
        print(" Pas de données valides pour le graphique des scores")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = ['#2ecc71' if score > 60 else '#f39c12' if score > 50 else '#e74c3c' 
              for score in df['score_global']]
    
    bars = ax.barh(df['modele'], df['score_global'], color=colors)
    
    # Ajouter les valeurs sur les barres
    for i, (bar, score) in enumerate(zip(bars, df['score_global'])):
        ax.text(score + 1, i, f'{score:.1f}', va='center', fontweight='bold')
    
    ax.set_xlabel('Score Global (/100)', fontweight='bold')
    ax.set_title('Classement Général des Modèles\nScore d\'Impact Psychologique',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    
    # Légende des couleurs
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Excellent (>60)'),
        Patch(facecolor='#f39c12', label='Moyen (50-60)'),
        Patch(facecolor='#e74c3c', label='Faible (<50)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'graphique_scores_globaux.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graphique scores globaux sauvegardé: {output_path}")


def creer_graphique_distribution_tons(csv_complet: str, output_dir: str):
    """Crée un graphique de distribution des tons par modèle."""
    
    df = pd.read_csv(csv_complet)
    
    # Compter les tons dominants par modèle
    ton_counts = df.groupby(['modele', 'ton_dominant']).size().unstack(fill_value=0)
    
    # Enlever la ligne NaN si elle existe
    if np.nan in ton_counts.index:
        ton_counts = ton_counts.drop(np.nan)
    
    if len(ton_counts) == 0:
        print(" Pas de données valides pour le graphique des tons")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ton_counts.plot(kind='bar', stacked=False, ax=ax, 
                    color=['#2ecc71', '#e74c3c', '#95a5a6'])
    
    ax.set_xlabel('Modèle', fontweight='bold')
    ax.set_ylabel('Nombre de réponses', fontweight='bold')
    ax.set_title('Distribution des Tons par Modèle', fontsize=14, fontweight='bold')
    ax.legend(title='Ton Dominant')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'graphique_distribution_tons.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Graphique distribution tons sauvegardé: {output_path}")


def creer_graphique_credibilite_verdict(csv_complet: str, output_dir: str):
    """Graphique de corrélation crédibilité vs verdict scientifique."""
    
    df = pd.read_csv(csv_complet)
    df = df.dropna(subset=['verdict_scientifique', 'credibilite_percue'])
    
    if len(df) == 0:
        print(" Pas de données valides pour le graphique crédibilité/verdict")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Boxplot par verdict
    verdicts = df['verdict_scientifique'].unique()
    data_to_plot = [df[df['verdict_scientifique'] == v]['credibilite_percue'].values 
                    for v in verdicts]
    
    bp = ax.boxplot(data_to_plot, labels=verdicts, patch_artist=True)
    
    # Colorier les boîtes
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
    for patch, color in zip(bp['boxes'], colors[:len(verdicts)]):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Verdict Scientifique', fontweight='bold')
    ax.set_ylabel('Crédibilité Perçue', fontweight='bold')
    ax.set_title('Corrélation entre Verdict Scientifique et Crédibilité Perçue',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'graphique_credibilite_verdict.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graphique crédibilité/verdict sauvegardé: {output_path}")


def creer_heatmap_correlations(csv_complet: str, output_dir: str):
    """Crée une heatmap des corrélations entre variables psychologiques."""
    
    df = pd.read_csv(csv_complet)
    
    colonnes_analyse = [
        'ton_rassurant', 'ton_alarmiste', 'anxiete_induite',
        'credibilite_percue', 'risque_influence',
        'empathie_globale', 'score_reassurance'
    ]
    
    # Filtrer les colonnes qui existent
    colonnes_existantes = [col for col in colonnes_analyse if col in df.columns]
    
    if len(colonnes_existantes) < 2:
        print(" Pas assez de colonnes pour la heatmap")
        return
    
    # Calculer la matrice de corrélation
    corr_matrix = df[colonnes_existantes].corr()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # Ajouter les valeurs
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    ax.set_xticks(range(len(colonnes_existantes)))
    ax.set_yticks(range(len(colonnes_existantes)))
    ax.set_xticklabels(colonnes_existantes, rotation=45, ha='right')
    ax.set_yticklabels(colonnes_existantes)
    
    plt.colorbar(im, ax=ax, label='Corrélation')
    ax.set_title('Matrice de Corrélation - Variables Psychologiques',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'heatmap_correlations.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Heatmap corrélations sauvegardée: {output_path}")


def generer_toutes_visualisations(
    csv_complet: str,
    csv_scores: str,
    output_dir: str
):
    """Génère toutes les visualisations."""
    
    print("\n" + "="*80)
    print(" GÉNÉRATION DES VISUALISATIONS")
    print("="*80 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        creer_graphique_scores_globaux(csv_scores, output_dir)
        creer_graphique_benchmark(csv_scores, output_dir)
        creer_graphique_distribution_tons(csv_complet, output_dir)
        creer_graphique_credibilite_verdict(csv_complet, output_dir)
        creer_heatmap_correlations(csv_complet, output_dir)
    except Exception as e:
        print(f" Erreur lors de la génération: {e}")
    
    print("\n" + "="*80)
    print(" VISUALISATIONS TERMINÉES")
    print("="*80)


if __name__ == "__main__":
    output_dir = '../outputs'  
    
    import glob
    
    fichiers_complets = glob.glob(os.path.join(output_dir, 'analyse_complete_*.csv'))
    fichiers_scores = glob.glob(os.path.join(output_dir, 'benchmark_scores_*.csv'))
    
    if fichiers_complets and fichiers_scores:
        csv_complet = max(fichiers_complets, key=os.path.getctime)
        csv_scores = max(fichiers_scores, key=os.path.getctime)
        
        viz_dir = os.path.join(output_dir, '../visualizations')
        
        generer_toutes_visualisations(csv_complet, csv_scores, viz_dir)
    else:
        print(" Fichiers d'analyse introuvables. Exécutez d'abord l'analyse complète.")
