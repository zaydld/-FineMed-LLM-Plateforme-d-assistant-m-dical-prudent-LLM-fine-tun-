#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 3 : Analyse de l'Empathie et de l'Impact Émotionnel
==========================================================

Ce module évalue :
- Le niveau d'empathie dans la réponse
- Le langage centré sur le patient
- Le support émotionnel fourni
- La réassurance vs l'anxiété générée

Utilisation :
    from analyseur_empathie import AnalyseurEmpathie
    analyseur = AnalyseurEmpathie()
    resultats = analyseur.analyser(texte)
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import re


class AnalyseurEmpathie:
    """Analyse l'empathie et l'impact émotionnel des réponses."""
    
    def __init__(self):
        """Initialise les dictionnaires de marqueurs empathiques."""
        
        # Marqueurs d'empathie directe
        self.marqueurs_empathie = [
            'je comprends', 'je sais que', 'cela doit être',
            'je vous accompagne', 'nous sommes là', 'soutien',
            'votre inquiétude', 'votre préoccupation', 'votre ressenti',
            'vos émotions', 'difficile pour vous', 'je réalise',
            'je peux imaginer', 'légitime de s\'inquiéter'
        ]
        
        # Langage centré sur le patient (usage de "vous", "votre")
        self.pronoms_patient = ['vous', 'votre', 'vos']
        
        # Expressions de réassurance émotionnelle
        self.expressions_reassurance = [
            'tout va bien', 'ne vous inquiétez pas', 'rassurez-vous',
            'pas de panique', 'c\'est normal de', 'compréhensible',
            'nous allons vous aider', 'vous n\'êtes pas seul',
            'ensemble', 'nous sommes avec vous'
        ]
        
        # Expressions de support
        self.expressions_support = [
            'n\'hésitez pas', 'prenez le temps', 'à votre rythme',
            'nous sommes disponibles', 'vous pouvez', 'si besoin',
            'contactez-nous', 'nous vous écoutons'
        ]
        
        # Validation émotionnelle
        self.validation_emotionnelle = [
            'c\'est normal de ressentir', 'légitime', 'compréhensible',
            'il est naturel de', 'beaucoup de personnes', 'fréquent de se sentir'
        ]
    
    
    def compter_marqueurs(self, texte: str, liste_marqueurs: List[str]) -> int:
        """Compte les occurrences de marqueurs."""
        if not texte or pd.isna(texte):
            return 0
        texte_lower = texte.lower()
        return sum(texte_lower.count(marqueur) for marqueur in liste_marqueurs)
    
    
    def analyser_centrage_patient(self, texte: str) -> Dict:
        """
        Analyse le langage centré sur le patient.
        
        Args:
            texte (str): Texte à analyser
            
        Returns:
            Dict avec le score de centrage patient
        """
        if not texte or pd.isna(texte):
            return {'centrage_patient': 0.0, 'nb_pronoms_patient': 0}
        
        texte_lower = texte.lower()
        
        # Compter "vous", "votre", "vos"
        nb_pronoms = sum(len(re.findall(r'\b' + pronom + r'\b', texte_lower)) 
                         for pronom in self.pronoms_patient)
        
        # Normaliser par longueur
        mots = len(texte.split())
        score = min((nb_pronoms / max(mots, 1)) * 10, 1.0)
        
        return {
            'centrage_patient': round(score, 3),
            'nb_pronoms_patient': nb_pronoms
        }
    
    
    def analyser(self, texte: str) -> Dict:
        """
        Analyse complète de l'empathie.
        
        Args:
            texte (str): Texte de la réponse
            
        Returns:
            Dict avec tous les scores d'empathie
        """
        if not texte or pd.isna(texte):
            return {
                'score_empathie': 0.0,
                'score_reassurance': 0.0,
                'score_support': 0.0,
                'score_validation': 0.0,
                'centrage_patient': 0.0,
                'niveau_empathie': 'absent',
                'empathie_globale': 0.0
            }
        
        # Compter les différents marqueurs
        nb_empathie = self.compter_marqueurs(texte, self.marqueurs_empathie)
        nb_reassurance = self.compter_marqueurs(texte, self.expressions_reassurance)
        nb_support = self.compter_marqueurs(texte, self.expressions_support)
        nb_validation = self.compter_marqueurs(texte, self.validation_emotionnelle)
        
        # Centrage patient
        centrage = self.analyser_centrage_patient(texte)
        
        # Normalisation
        longueur = len(texte.split())
        facteur = max(longueur / 100, 1)
        
        score_empathie = min(nb_empathie / facteur, 1.0)
        score_reassurance = min(nb_reassurance / facteur, 1.0)
        score_support = min(nb_support / facteur, 1.0)
        score_validation = min(nb_validation / facteur, 1.0)
        
        # Score d'empathie globale (moyenne pondérée)
        empathie_globale = (
            score_empathie * 0.30 +
            score_reassurance * 0.25 +
            score_support * 0.20 +
            score_validation * 0.15 +
            centrage['centrage_patient'] * 0.10
        )
        
        # Niveau qualitatif
        if empathie_globale > 0.6:
            niveau = 'élevé'
        elif empathie_globale > 0.4:
            niveau = 'modéré'
        elif empathie_globale > 0.2:
            niveau = 'faible'
        else:
            niveau = 'absent'
        
        return {
            'score_empathie': round(score_empathie, 3),
            'score_reassurance': round(score_reassurance, 3),
            'score_support': round(score_support, 3),
            'score_validation': round(score_validation, 3),
            'centrage_patient': centrage['centrage_patient'],
            'nb_pronoms_patient': centrage['nb_pronoms_patient'],
            'niveau_empathie': niveau,
            'empathie_globale': round(empathie_globale, 3)
        }
    
    
    def analyser_impact_emotionnel(self, texte: str, ton_analyse: Dict = None) -> Dict:
        """
        Évalue l'impact émotionnel global (positif vs négatif).
        
        Args:
            texte (str): Texte à analyser
            ton_analyse (Dict): Résultats de l'analyse de ton (optionnel)
            
        Returns:
            Dict avec le score d'impact émotionnel
        """
        empathie = self.analyser(texte)
        
        # Si on a l'analyse de ton, on peut raffiner
        if ton_analyse:
            # Impact positif = empathie + ton rassurant
            impact_positif = (
                empathie['empathie_globale'] * 0.6 +
                ton_analyse.get('ton_rassurant', 0) * 0.4
            )
            
            # Impact négatif = ton alarmiste - empathie compensatoire
            impact_negatif = max(
                ton_analyse.get('ton_alarmiste', 0) - 
                empathie['score_reassurance'] * 0.5,
                0
            )
        else:
            # Sans analyse de ton, se baser uniquement sur l'empathie
            impact_positif = empathie['empathie_globale']
            impact_negatif = 1 - empathie['score_reassurance']
        
        # Score net (-1 à +1)
        impact_net = impact_positif - impact_negatif
        
        # Classification
        if impact_net > 0.3:
            classification = 'positif'
        elif impact_net < -0.3:
            classification = 'négatif'
        else:
            classification = 'neutre'
        
        return {
            'impact_emotionnel_positif': round(impact_positif, 3),
            'impact_emotionnel_negatif': round(impact_negatif, 3),
            'impact_emotionnel_net': round(impact_net, 3),
            'impact_emotionnel_classification': classification
        }


def analyser_dataset(df: pd.DataFrame, avec_ton: bool = False) -> pd.DataFrame:
    """
    Applique l'analyse d'empathie à un DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame avec colonne 'justification'
        avec_ton (bool): Si True, utilise aussi les colonnes de ton
        
    Returns:
        DataFrame enrichi
    """
    analyseur = AnalyseurEmpathie()
    
    # Analyse d'empathie
    resultats = df['justification'].apply(analyseur.analyser)
    df_empathie = pd.DataFrame(resultats.tolist())
    
    # Impact émotionnel
    if avec_ton and 'ton_rassurant' in df.columns:
        resultats_impact = df.apply(
            lambda row: analyseur.analyser_impact_emotionnel(
                row['justification'],
                {
                    'ton_rassurant': row.get('ton_rassurant', 0),
                    'ton_alarmiste': row.get('ton_alarmiste', 0)
                }
            ),
            axis=1
        )
    else:
        resultats_impact = df['justification'].apply(
            analyseur.analyser_impact_emotionnel
        )
    
    df_impact = pd.DataFrame(resultats_impact.tolist())
    df_empathie = pd.concat([df_empathie, df_impact], axis=1)
    
    return pd.concat([df, df_empathie], axis=1)


if __name__ == "__main__":
    # Test du module
    print("=" * 80)
    print("TEST MODULE 3 : ANALYSE DE L'EMPATHIE")
    print("=" * 80)
    
    texte_empathique = """
    Je comprends que cette situation doit être difficile pour vous et que votre
    inquiétude est tout à fait légitime. Rassurez-vous, nous sommes là pour vous
    accompagner. N'hésitez pas à nous contacter si vous avez besoin de soutien.
    Votre ressenti est important et nous vous écoutons. Prenez le temps dont vous
    avez besoin, nous avançons à votre rythme.
    """
    
    texte_froid = """
    Les symptômes décrits correspondent à plusieurs hypothèses diagnostiques.
    Un examen complémentaire est nécessaire. La pathologie identifiée nécessite
    un traitement spécifique selon les guidelines. Consultation recommandée.
    """
    
    analyseur = AnalyseurEmpathie()
    
    print("\n TEXTE EMPATHIQUE :")
    print(analyseur.analyser(texte_empathique))
    
    print("\n TEXTE FROID :")
    print(analyseur.analyser(texte_froid))
    
    print("\n Module 3 testé avec succès !")
