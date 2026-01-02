#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Psychologique avec MACHINE LEARNING
===========================================

Ajoute de vraies analyses ML :
- VADER Sentiment Analysis
- TextBlob Sentiment & Subjectivity
- Scoring avancé

Usage:
    python analyse_ML.py
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')


class AnalyseurML:
    """Analyse psychologique avec Machine Learning."""
    
    def __init__(self):
        print(" Initialisation des modèles ML...")
        self.vader = SentimentIntensityAnalyzer()
        print(" VADER Sentiment Analyzer chargé")
    
    
    def analyser_sentiment_vader(self, texte: str) -> dict:
        """
        Analyse de sentiment avec VADER (spécialisé pour textes courts).
        
        Returns:
            Dict avec scores : negative, neutral, positive, compound
        """
        if not texte or pd.isna(texte):
            return {
                'vader_negative': 0.0,
                'vader_neutral': 1.0,
                'vader_positive': 0.0,
                'vader_compound': 0.0
            }
        
        scores = self.vader.polarity_scores(str(texte))
        
        return {
            'vader_negative': round(scores['neg'], 3),
            'vader_neutral': round(scores['neu'], 3),
            'vader_positive': round(scores['pos'], 3),
            'vader_compound': round(scores['compound'], 3)
        }
    
    
    def analyser_sentiment_textblob(self, texte: str) -> dict:
        """
        Analyse de sentiment avec TextBlob.
        
        Returns:
            Dict avec polarity (-1 à 1) et subjectivity (0 à 1)
        """
        if not texte or pd.isna(texte):
            return {
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0
            }
        
        try:
            blob = TextBlob(str(texte))
            return {
                'textblob_polarity': round(blob.sentiment.polarity, 3),
                'textblob_subjectivity': round(blob.sentiment.subjectivity, 3)
            }
        except:
            return {
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0
            }
    
    
    def classifier_impact_emotionnel_ML(self, vader_compound: float, 
                                        ton_alarmiste: float) -> dict:
        """
        Classification ML de l'impact émotionnel basée sur les scores.
        
        Args:
            vader_compound: Score VADER (-1 à 1)
            ton_alarmiste: Score de ton alarmiste (0 à 1)
        
        Returns:
            Classification et score de risque psychologique
        """
        # Score de risque combiné
        risque_psycho = (
            (1 - vader_compound) * 0.4 +  # Sentiment négatif
            ton_alarmiste * 0.6            # Ton alarmiste
        )
        
        # Classification
        if risque_psycho > 0.7:
            classification = 'risque_élevé'
        elif risque_psycho > 0.5:
            classification = 'risque_modéré'
        elif risque_psycho > 0.3:
            classification = 'risque_faible'
        else:
            classification = 'impact_positif'
        
        return {
            'risque_psychologique_ML': round(risque_psycho, 3),
            'classification_impact_ML': classification
        }
    
    
    def analyser_texte_complet(self, texte: str, ton_alarmiste: float = 0) -> dict:
        """
        Analyse ML complète d'un texte.
        
        Args:
            texte: Texte à analyser
            ton_alarmiste: Score de ton alarmiste (pour enrichissement)
        
        Returns:
            Dict avec tous les scores ML
        """
        # VADER
        vader = self.analyser_sentiment_vader(texte)
        
        # TextBlob
        textblob = self.analyser_sentiment_textblob(texte)
        
        # Classification ML
        classification = self.classifier_impact_emotionnel_ML(
            vader['vader_compound'],
            ton_alarmiste
        )
        
        # Combiner
        resultats = {**vader, **textblob, **classification}
        
        return resultats


def enrichir_dataset_avec_ML(csv_input: str, csv_output: str):
    """
    Enrichit le dataset avec des analyses ML.
    
    Args:
        csv_input: CSV nettoyé en entrée
        csv_output: CSV enrichi en sortie
    """
    print("\n" + "="*80)
    print(" ENRICHISSEMENT AVEC MACHINE LEARNING")
    print("="*80)
    
    # Charger le CSV
    print(f"\n Chargement : {csv_input}")
    df = pd.read_csv(csv_input)
    print(f" {len(df)} lignes chargées")
    
    # Initialiser l'analyseur ML
    analyseur = AnalyseurML()
    
    # Analyser chaque ligne
    print("\n Analyse ML en cours...")
    
    resultats_ml = []
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"   Progression: {idx}/{len(df)} lignes...")
        
        # Analyse ML
        resultat = analyseur.analyser_texte_complet(
            row['justification'],
            row.get('ton_alarmiste', 0)
        )
        resultats_ml.append(resultat)
    
    # Convertir en DataFrame
    df_ml = pd.DataFrame(resultats_ml)
    
    # Combiner avec le dataset original
    df_enrichi = pd.concat([df, df_ml], axis=1)
    
    print(f" Analyse ML terminée")
    print(f"   • {len(df_ml.columns)} nouvelles colonnes ML ajoutées")
    
    # Sauvegarder
    df_enrichi.to_csv(csv_output, index=False)
    print(f"\n Dataset enrichi sauvegardé : {csv_output}")
    
    # Statistiques
    print("\n" + "="*80)
    print(" STATISTIQUES ML")
    print("="*80)
    
    print("\n Sentiment VADER (compound) :")
    print(f"   • Moyenne : {df_enrichi['vader_compound'].mean():.3f}")
    print(f"   • Min : {df_enrichi['vader_compound'].min():.3f}")
    print(f"   • Max : {df_enrichi['vader_compound'].max():.3f}")
    
    print("\n TextBlob Polarity :")
    print(f"   • Moyenne : {df_enrichi['textblob_polarity'].mean():.3f}")
    
    print("\n Classification Impact ML :")
    print(df_enrichi['classification_impact_ML'].value_counts())
    
    print("\n Risque Psychologique ML :")
    print(f"   • Moyenne : {df_enrichi['risque_psychologique_ML'].mean():.3f}")
    
    print("\n" + "="*80)
    print(" ENRICHISSEMENT ML TERMINÉ")
    print("="*80)
    
    return df_enrichi


if __name__ == "__main__":
    # Chemins des fichiers
    csv_input = '../outputs/analyse_complete_CLEAN.csv'
    csv_output = '../outputs/analyse_complete_ML.csv'
    
    # Enrichir avec ML
    df_ml = enrichir_dataset_avec_ML(csv_input, csv_output)
    
    print("\n Vous pouvez maintenant utiliser : analyse_complete_ML.csv")
