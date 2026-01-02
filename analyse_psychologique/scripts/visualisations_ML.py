#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisations avec Données ML 
==============================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import os


def creer_heatmap_correlations_ML(csv_path: str, output_dir: str):
    """
    Crée une heatmap de corrélations SANS les colonnes à variance nulle.
    """
    
    df = pd.read_csv(csv_path)
    
    # Colonnes à analyser (en excluant celles qui sont toujours à 0)
    colonnes_analyse = [
        'ton_rassurant', 'ton_alarmiste', 'anxiete_induite',
        'credibilite_percue', 'risque_influence',
        'vader_compound', 'vader_negative', 'vader_positive',
        'textblob_polarity', 'textblob_subjectivity',
        'risque_psychologique_ML'
    ]
    
    # Filtrer les colonnes qui existent ET qui ont de la variance
    colonnes_valides = []
    for col in colonnes_analyse:
        if col in df.columns:
            variance = df[col].var()
            if variance > 0.001:  # Seuil de variance minimum
                colonnes_valides.append(col)
    
    if len(colonnes_valides) < 2:
        print("⚠️ Pas assez de colonnes avec variance pour la heatmap")
        return
    
    # Calculer la matrice de corrélation
    corr_matrix = df[colonnes_valides].corr()
    
    # Créer la figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Heatmap avec seaborn (plus jolie)
    sns.heatmap(
        corr_matrix, 
        annot=True,           # Afficher les valeurs
        fmt='.2f',            # Format 2 décimales
        cmap='coolwarm',      # Palette de couleurs
        center=0,             # Centrer sur 0
        vmin=-1, vmax=1,      # Limites
        square=True,          # Cellules carrées
        linewidths=0.5,       # Lignes de séparation
        cbar_kws={"label": "Corrélation"},
        ax=ax
    )
    
    ax.set_title('Matrice de Corrélation - Variables Psychologiques ML',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'heatmap_correlations_ML.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Heatmap ML sauvegardée : {output_path}")


def creer_graphique_vader_par_verdict(csv_path: str, output_dir: str):
    """
    Graphique VADER compound par verdict scientifique.
    """
    
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Boxplot
    verdicts = df['verdict_scientifique'].unique()
    data_to_plot = [df[df['verdict_scientifique'] == v]['vader_compound'].values 
                    for v in verdicts]
    
    bp = ax.boxplot(data_to_plot, labels=verdicts, patch_artist=True)
    
    # Colorier
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
    for patch, color in zip(bp['boxes'], colors[:len(verdicts)]):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Verdict Scientifique', fontweight='bold')
    ax.set_ylabel('VADER Compound Score', fontweight='bold')
    ax.set_title('Sentiment VADER par Verdict Scientifique (ML)',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'vader_par_verdict_ML.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graphique VADER sauvegardé : {output_path}")


def creer_graphique_risque_ML_par_modele(csv_path: str, output_dir: str):
    """
    Graphique du risque psychologique ML par modèle.
    """
    
    df = pd.read_csv(csv_path)
    
    # Moyennes par modèle
    stats = df.groupby('modele')['risque_psychologique_ML'].agg(['mean', 'std']).reset_index()
    stats = stats.dropna()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Barres avec barres d'erreur
    bars = ax.bar(stats['modele'], stats['mean'], yerr=stats['std'], 
                   capsize=5, color=['#3498db', '#e74c3c', '#2ecc71'][:len(stats)])
    
    # Ajouter les valeurs
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Modèle LLM', fontweight='bold')
    ax.set_ylabel('Risque Psychologique ML', fontweight='bold')
    ax.set_title('Risque Psychologique par Modèle (Machine Learning)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'risque_ML_par_modele.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graphique risque ML sauvegardé : {output_path}")


def generer_toutes_visualisations_ML(csv_path: str, output_dir: str):
    """
    Génère toutes les visualisations ML.
    """
    
    print("\n" + "="*80)
    print("📊 GÉNÉRATION DES VISUALISATIONS ML")
    print("="*80 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        creer_heatmap_correlations_ML(csv_path, output_dir)
        creer_graphique_vader_par_verdict(csv_path, output_dir)
        creer_graphique_risque_ML_par_modele(csv_path, output_dir)
    except Exception as e:
        print(f"⚠️ Erreur : {e}")
    
    print("\n" + "="*80)
    print("✅ VISUALISATIONS ML TERMINÉES")
    print("="*80)


if __name__ == "__main__":
    csv_path = '../outputs/analyse_complete_ML.csv'
    viz_dir = '../visualizations'
    
    generer_toutes_visualisations_ML(csv_path, viz_dir)