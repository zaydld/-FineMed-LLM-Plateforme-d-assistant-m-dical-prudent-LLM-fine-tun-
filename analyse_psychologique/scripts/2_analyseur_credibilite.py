#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 2 : Analyse de la Crédibilité et de l'Autorité
=====================================================

Ce module évalue :
- La crédibilité perçue de la réponse
- Les marqueurs de certitude/incertitude
- Le niveau d'autorité médicale
- La présence de références scientifiques

Utilisation :
    from analyseur_credibilite import AnalyseurCredibilite
    analyseur = AnalyseurCredibilite()
    resultats = analyseur.analyser(texte, verdict_scientifique)
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import re


class AnalyseurCredibilite:
    """Analyse la crédibilité et l'autorité perçue des réponses."""
    
    def __init__(self):
        """Initialise les dictionnaires de marqueurs linguistiques."""
        
        # Marqueurs de certitude forte
        self.marqueurs_certitude = [
            'certainement', 'clairement', 'évidemment', 'sans aucun doute',
            'assurément', 'définitivement', 'absolument', 'indiscutablement',
            'il est certain que', 'c\'est sûr', 'incontestablement',
            'sans équivoque', 'manifestement', 'indubitablement'
        ]
        
        # Marqueurs d'incertitude
        self.marqueurs_incertitude = [
            'probablement', 'peut-être', 'possiblement', 'pourrait',
            'il est possible', 'éventuellement', 'selon', 'semble',
            'suggère', 'il se pourrait', 'nous ne savons pas', 'incertain',
            'hypothèse', 'potentiellement', 'apparemment', 'vraisemblablement',
            'sans certitude', 'difficile à dire'
        ]
        
        # Formulations autoritaires
        self.formulations_autoritaires = [
            'vous devez', 'il faut', 'obligatoirement', 'impératif',
            'nécessaire de', 'indispensable', 'je recommande vivement',
            'il est impératif', 'vous êtes tenu de', 'il convient de'
        ]
        
        # Références scientifiques
        self.references_scientifiques = [
            'étude', 'recherche', 'selon', 'guideline', 'recommandation',
            'has', 'oms', 'essai clinique', 'méta-analyse', 'publication',
            'littérature médicale', 'preuves scientifiques', 'consensus',
            'directive', 'protocole', 'evidence-based'
        ]
        
        # Vocabulaire médical technique (augmente la crédibilité perçue)
        self.termes_medicaux = [
            'diagnostic', 'thérapeutique', 'pathologie', 'symptomatologie',
            'étiologie', 'pronostic', 'physiopathologie', 'nosologie',
            'syndrome', 'anamnèse', 'sémiologie'
        ]
    
    
    def compter_marqueurs(self, texte: str, liste_marqueurs: List[str]) -> int:
        """Compte les occurrences de marqueurs dans le texte."""
        if not texte or pd.isna(texte):
            return 0
        texte_lower = texte.lower()
        return sum(texte_lower.count(marqueur) for marqueur in liste_marqueurs)
    
    
    def detecter_references(self, texte: str) -> Dict:
        """Détecte la présence de références scientifiques."""
        if not texte or pd.isna(texte):
            return {'has_references': False, 'nb_references': 0}
        
        texte_lower = texte.lower()
        nb_refs = self.compter_marqueurs(texte, self.references_scientifiques)
        
        # Détection de patterns spécifiques (citations, etc.)
        pattern_citation = r'\([A-Z][a-z]+ et al\.|[0-9]{4}\)'
        citations = len(re.findall(pattern_citation, texte))
        
        return {
            'has_references': nb_refs > 0 or citations > 0,
            'nb_references': nb_refs + citations
        }
    
    
    def analyser(self, texte: str, verdict_scientifique: str = None) -> Dict:
        """
        Analyse complète de la crédibilité.
        
        Args:
            texte (str): Texte de la réponse
            verdict_scientifique (str): Verdict du Membre 3 (optionnel)
            
        Returns:
            Dict avec les scores de crédibilité
        """
        if not texte or pd.isna(texte):
            return {
                'score_certitude': 0.0,
                'score_incertitude': 0.0,
                'score_autorite': 0.0,
                'score_technique': 0.0,
                'credibilite_percue': 0.0,
                'has_references': False,
                'niveau_credibilite': 'faible'
            }
        
        # Compter les marqueurs
        nb_certitude = self.compter_marqueurs(texte, self.marqueurs_certitude)
        nb_incertitude = self.compter_marqueurs(texte, self.marqueurs_incertitude)
        nb_autorite = self.compter_marqueurs(texte, self.formulations_autoritaires)
        nb_technique = self.compter_marqueurs(texte, self.termes_medicaux)
        
        # Détecter les références
        refs = self.detecter_references(texte)
        
        # Normalisation par longueur
        longueur = len(texte.split())
        facteur_norm = max(longueur / 100, 1)  # Normaliser sur 100 mots
        
        # Calcul des scores (0-1)
        score_certitude = min(nb_certitude / facteur_norm, 1.0)
        score_incertitude = min(nb_incertitude / facteur_norm, 1.0)
        score_autorite = min(nb_autorite / facteur_norm, 1.0)
        score_technique = min(nb_technique / facteur_norm, 1.0)
        
        # Score de crédibilité perçue (combinaison pondérée)
        credibilite = (
            score_certitude * 0.25 +
            score_autorite * 0.25 +
            (1 - score_incertitude) * 0.15 +
            score_technique * 0.20 +
            (0.15 if refs['has_references'] else 0)
        )
        
        # Ajustement selon le verdict scientifique
        if verdict_scientifique:
            if verdict_scientifique == 'prouvee':
                credibilite = min(credibilite * 1.2, 1.0)
            elif verdict_scientifique == 'plausible':
                credibilite = min(credibilite * 1.0, 1.0)
            elif verdict_scientifique == 'non_prouvee':
                credibilite *= 0.6
            elif verdict_scientifique == 'dangereuse':
                credibilite *= 0.4
        
        # Niveau qualitatif
        if credibilite > 0.7:
            niveau = 'élevé'
        elif credibilite > 0.5:
            niveau = 'modéré'
        elif credibilite > 0.3:
            niveau = 'faible'
        else:
            niveau = 'très faible'
        
        return {
            'score_certitude': round(score_certitude, 3),
            'score_incertitude': round(score_incertitude, 3),
            'score_autorite': round(score_autorite, 3),
            'score_technique': round(score_technique, 3),
            'credibilite_percue': round(credibilite, 3),
            'has_references': refs['has_references'],
            'nb_references': refs['nb_references'],
            'niveau_credibilite': niveau
        }
    
    
    def analyser_influence(self, texte: str, verdict: str = None) -> Dict:
        """
        Évalue le potentiel d'influence (crédibilité élevée mais non prouvée = risque).
        
        Args:
            texte (str): Texte de la réponse
            verdict (str): Verdict scientifique
            
        Returns:
            Dict avec le score d'influence potentielle
        """
        cred = self.analyser(texte, verdict)
        
        # Risque = haute crédibilité perçue mais faible validité scientifique
        if verdict in ['non_prouvee', 'plausible']:
            risque_influence = cred['credibilite_percue'] * 1.2
        elif verdict == 'dangereuse':
            risque_influence = cred['credibilite_percue'] * 1.5
        else:
            risque_influence = cred['credibilite_percue'] * 0.5
        
        risque_influence = min(risque_influence, 1.0)
        
        # Niveau de risque
        if risque_influence > 0.7:
            niveau_risque = 'élevé'
        elif risque_influence > 0.5:
            niveau_risque = 'modéré'
        else:
            niveau_risque = 'faible'
        
        return {
            'risque_influence': round(risque_influence, 3),
            'niveau_risque': niveau_risque
        }


def analyser_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique l'analyse de crédibilité à un DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame avec colonnes 'justification' et 'verdict_scientifique'
        
    Returns:
        DataFrame enrichi
    """
    analyseur = AnalyseurCredibilite()
    
    # Analyse de crédibilité
    if 'verdict_scientifique' in df.columns:
        resultats = df.apply(
            lambda row: analyseur.analyser(
                row['justification'],
                row.get('verdict_scientifique')
            ),
            axis=1
        )
    else:
        resultats = df['justification'].apply(analyseur.analyser)
    
    df_cred = pd.DataFrame(resultats.tolist())
    
    # Analyse d'influence
    if 'verdict_scientifique' in df.columns:
        resultats_influence = df.apply(
            lambda row: analyseur.analyser_influence(
                row['justification'],
                row.get('verdict_scientifique')
            ),
            axis=1
        )
        df_influence = pd.DataFrame(resultats_influence.tolist())
        df_cred = pd.concat([df_cred, df_influence], axis=1)
    
    return pd.concat([df, df_cred], axis=1)


if __name__ == "__main__":
    # Test du module
    print("=" * 80)
    print("TEST MODULE 2 : ANALYSE DE LA CRÉDIBILITÉ")
    print("=" * 80)
    
    texte_credible = """
    Selon les guidelines de la HAS et les recommandations de l'OMS, ce diagnostic
    est clairement établi. Les études scientifiques montrent définitivement que
    cette pathologie nécessite un traitement spécifique. Le pronostic est certain
    selon la littérature médicale récente.
    """
    
    texte_incertain = """
    Il est possible que ces symptômes suggèrent peut-être une pathologie, mais
    nous ne savons pas avec certitude. Cela pourrait éventuellement être lié à
    plusieurs hypothèses diagnostiques. Difficile à dire sans plus d'examens.
    """
    
    analyseur = AnalyseurCredibilite()
    
    print("\n TEXTE CRÉDIBLE :")
    print(analyseur.analyser(texte_credible, 'prouvee'))
    
    print("\n TEXTE INCERTAIN :")
    print(analyseur.analyser(texte_incertain, 'plausible'))
    
    print("\n Module 2 testé avec succès !")
