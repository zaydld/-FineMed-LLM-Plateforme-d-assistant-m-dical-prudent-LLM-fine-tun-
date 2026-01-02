#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 1 : Analyse du Ton des Réponses IA Médicales
===================================================

Ce module analyse le ton émotionnel des réponses :
- Ton rassurant (minimise l'anxiété)
- Ton alarmiste (augmente l'anxiété)
- Ton neutre (factuel)

Utilisation :
    from analyseur_ton import AnalyseurTon
    analyseur = AnalyseurTon()
    resultats = analyseur.analyser(texte)
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import re


class AnalyseurTon:
    """Analyse le ton émotionnel des réponses médicales."""
    
    def __init__(self):
        """Initialise les dictionnaires de mots-clés."""
        
        # Mots rassurants
        self.mots_rassurants = [
            'normal', 'bénin', 'fréquent', 'courant', 'habituel', 
            'temporaire', 'passager', 'léger', 'sans gravité',
            'rassurez-vous', 'ne vous inquiétez pas', 'pas de panique',
            'généralement sans danger', 'banal', 'anodin', 'commun',
            'inoffensif', 'sans conséquence', 'bénigne', 'rassurant'
        ]
        
        # Mots alarmistes
        self.mots_alarmistes = [
            'grave', 'urgent', 'immédiatement', 'danger', 'risque',
            'sévère', 'critique', 'urgence', 'vital', 'mortel',
            'fatal', 'alarme', 'alarmant', 'préoccupant', 'inquiétant',
            'complications graves', 'potentiellement mortel', 'urgence vitale',
            'appeler le 15', 'samu', 'pompiers', '112', 'danger de mort'
        ]
        
        # Mots neutres/médicaux
        self.mots_neutres = [
            'symptôme', 'diagnostic', 'traitement', 'consultation',
            'médecin', 'examen', 'analyse', 'bilan', 'suivi'
        ]
    
    
    def compter_occurrences(self, texte: str, liste_mots: List[str]) -> int:
        """
        Compte les occurrences de mots d'une liste dans le texte.
        
        Args:
            texte (str): Texte à analyser
            liste_mots (List[str]): Liste de mots à chercher
            
        Returns:
            int: Nombre total d'occurrences
        """
        if not texte or pd.isna(texte):
            return 0
        
        texte_lower = texte.lower()
        return sum(texte_lower.count(mot) for mot in liste_mots)
    
    
    def analyser(self, texte: str) -> Dict[str, float]:
        """
        Analyse le ton d'un texte.
        
        Args:
            texte (str): Texte de la réponse IA
            
        Returns:
            Dict avec les scores de ton
        """
        if not texte or pd.isna(texte):
            return {
                'ton_rassurant': 0.0,
                'ton_alarmiste': 0.0,
                'ton_neutre': 1.0,
                'ton_dominant': 'neutre',
                'nb_mots_rassurants': 0,
                'nb_mots_alarmistes': 0
            }
        
        # Compter les occurrences
        nb_rassurant = self.compter_occurrences(texte, self.mots_rassurants)
        nb_alarmiste = self.compter_occurrences(texte, self.mots_alarmistes)
        nb_neutre = self.compter_occurrences(texte, self.mots_neutres)
        
        # Normalisation par la longueur du texte
        longueur = len(texte.split())
        
        # Scores pondérés (sur 100 mots)
        score_rassurant = (nb_rassurant / max(longueur, 1)) * 100
        score_alarmiste = (nb_alarmiste / max(longueur, 1)) * 100
        score_neutre = (nb_neutre / max(longueur, 1)) * 100
        
        # Normaliser entre 0 et 1
        total = score_rassurant + score_alarmiste + score_neutre + 1
        
        ton_rassurant = score_rassurant / total
        ton_alarmiste = score_alarmiste / total
        ton_neutre = 1.0 - (ton_rassurant + ton_alarmiste)
        
        # Déterminer le ton dominant
        if ton_rassurant > ton_alarmiste and ton_rassurant > 0.15:
            ton_dominant = 'rassurant'
        elif ton_alarmiste > ton_rassurant and ton_alarmiste > 0.15:
            ton_dominant = 'alarmiste'
        else:
            ton_dominant = 'neutre'
        
        return {
            'ton_rassurant': round(ton_rassurant, 3),
            'ton_alarmiste': round(ton_alarmiste, 3),
            'ton_neutre': round(max(ton_neutre, 0), 3),
            'ton_dominant': ton_dominant,
            'nb_mots_rassurants': nb_rassurant,
            'nb_mots_alarmistes': nb_alarmiste
        }
    
    
    def analyser_anxiete_induite(self, texte: str, categorie_cas: str = None) -> Dict:
        """
        Évalue le niveau d'anxiété potentiellement induit.
        
        Args:
            texte (str): Texte à analyser
            categorie_cas (str): Catégorie du cas (Anxiogène, Simple, etc.)
            
        Returns:
            Dict avec le score d'anxiété
        """
        ton_analyse = self.analyser(texte)
        score_base = ton_analyse['ton_alarmiste']
        
        # Ajuster selon la catégorie du cas
        if categorie_cas == 'Anxiogène':
            # Pour un cas anxiogène, un ton alarmiste peut être justifié
            score_anxiete = score_base * 0.7
        elif categorie_cas == 'Simple':
            # Pour un cas simple, un ton alarmiste est problématique
            score_anxiete = score_base * 1.5
        elif categorie_cas == 'Psychologique':
            score_anxiete = score_base * 1.2
        else:
            score_anxiete = score_base
        
        score_anxiete = min(score_anxiete, 1.0)
        
        # Niveau qualitatif
        if score_anxiete > 0.5:
            niveau = 'élevé'
        elif score_anxiete > 0.3:
            niveau = 'modéré'
        else:
            niveau = 'faible'
        
        return {
            'anxiete_induite': round(score_anxiete, 3),
            'niveau_anxiete': niveau
        }


def analyser_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique l'analyse de ton à un DataFrame complet.
    
    Args:
        df (pd.DataFrame): DataFrame avec colonne 'justification'
        
    Returns:
        DataFrame avec colonnes d'analyse ajoutées
    """
    analyseur = AnalyseurTon()
    
    # Analyse du ton
    resultats_ton = df['justification'].apply(analyseur.analyser)
    df_ton = pd.DataFrame(resultats_ton.tolist())
    
    # Analyse de l'anxiété
    if 'categorie_cas' in df.columns:
        resultats_anxiete = df.apply(
            lambda row: analyseur.analyser_anxiete_induite(
                row['justification'], 
                row.get('categorie_cas')
            ), 
            axis=1
        )
        df_anxiete = pd.DataFrame(resultats_anxiete.tolist())
        df_ton = pd.concat([df_ton, df_anxiete], axis=1)
    
    return pd.concat([df, df_ton], axis=1)


if __name__ == "__main__":
    # Test du module
    print("=" * 80)
    print("TEST MODULE 1 : ANALYSE DU TON")
    print("=" * 80)
    
    # Exemple de texte rassurant
    texte_rassurant = """
    Ne vous inquiétez pas, ces symptômes sont tout à fait normaux et fréquents.
    Il s'agit d'une condition bénigne et temporaire qui ne présente pas de gravité.
    C'est banal et sans danger.
    """
    
    # Exemple de texte alarmiste
    texte_alarmiste = """
    Attention, ces symptômes sont graves et préoccupants ! Il y a un risque urgent
    de complications sévères. Consultez immédiatement les urgences, c'est vital !
    Appelez le 15 sans délai, danger potentiellement mortel !
    """
    
    analyseur = AnalyseurTon()
    
    print("\n TEXTE RASSURANT :")
    print(analyseur.analyser(texte_rassurant))
    
    print("\n TEXTE ALARMISTE :")
    print(analyseur.analyser(texte_alarmiste))
    
    print("\n Module 1 testé avec succès !")
