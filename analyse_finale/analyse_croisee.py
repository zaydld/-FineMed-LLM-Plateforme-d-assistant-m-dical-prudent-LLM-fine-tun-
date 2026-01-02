"""
Script 2 - Analyse Croisée
OBJECTIF: Analyser l'impact psychologique des réponses NON PROUVÉES et DANGEREUSES
         et identifier les mécanismes linguistiques qui les rendent convaincantes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

class AnalyseCroiseeConfig:
    def __init__(self):
        self.racine = Path(r"C:\Users\ZAID\OneDrive\Documents\3eme_gds\DL\DeepLearning_1")
        self.dataset_complet = self.racine / "dataset_complet.csv"
        self.analyse_dir = self.racine / "analyse_finale"
        self.data_exports = self.analyse_dir / "data_exports"
        
        # Fichiers de sortie focus objectifs
        self.impact_non_prouvees = self.data_exports / "02_impact_psycho_non_prouvees.csv"
        self.impact_dangereuses = self.data_exports / "02_impact_psycho_dangereuses.csv"
        self.patterns_convaincants = self.data_exports / "02_patterns_convaincants.csv"
        self.mecanismes_linguistiques = self.data_exports / "02_mecanismes_linguistiques.csv"
        
        self.data_exports.mkdir(parents=True, exist_ok=True)

# ==================== DÉTECTION VERDICTS (CORRIGÉ) ====================

def est_prouvee(verdict):
    """Vérifie si le verdict est Prouvee"""
    if pd.isna(verdict):
        return False
    v = str(verdict).strip()
    return v in ['Prouvee', 'prouvee', 'Prouvée', 'prouvée']

def est_plausible(verdict):
    """Vérifie si le verdict est Plausible"""
    if pd.isna(verdict):
        return False
    v = str(verdict).strip()
    return v in ['Plausible', 'plausible']

def est_non_prouvee(verdict):
    """Vérifie si le verdict est Non_prouvee"""
    if pd.isna(verdict):
        return False
    v = str(verdict).strip()
    return v in ['Non_prouvee', 'non_prouvee', 'Non_prouvée', 'non_prouvée']

def est_dangereuse(verdict):
    """Vérifie si le verdict est Dangereuse"""
    if pd.isna(verdict):
        return False
    v = str(verdict).strip()
    return v in ['Dangereuse', 'dangereuse', 'Dangereux', 'dangereux']

def detecter_type_verdict(verdict):
    """Détecte le type de verdict"""
    if est_dangereuse(verdict):
        return 'dangereuse'
    elif est_non_prouvee(verdict):
        return 'non_prouvee'
    elif est_prouvee(verdict):
        return 'prouvee'
    elif est_plausible(verdict):
        return 'plausible'
    else:
        return 'inconnu'

# ==================== CHARGEMENT ====================

def charger_dataset(config):
    print("="*70)
    print("📂 CHARGEMENT")
    print("="*70)
    
    df = pd.read_csv(config.dataset_complet, encoding='utf-8-sig')
    
    if 'verdict_scientifique' in df.columns:
        df['verdict_type'] = df['verdict_scientifique'].apply(detecter_type_verdict)
    
    print(f"✅ {len(df)} lignes chargées")
    
    # Vérification de la détection
    print(f"\n🔍 Verdicts détectés:")
    for v_type in ['prouvee', 'plausible', 'non_prouvee', 'dangereuse', 'inconnu']:
        count = (df['verdict_type'] == v_type).sum()
        if count > 0:
            emoji = "🚨" if v_type in ['non_prouvee', 'dangereuse'] else "✅"
            print(f"   {emoji} {v_type}: {count} cas")
    
    return df

# ==================== ANALYSE 1: IMPACT PSYCHO NON PROUVÉES ====================

def analyser_impact_non_prouvees(df):
    """
    Objectif: Mesurer l'impact psychologique des réponses NON PROUVÉES
    - Crédibilité perçue
    - Anxiété induite
    - Confiance inspirée
    """
    
    print("\n" + "="*70)
    print("🔍 ANALYSE 1: Impact Psychologique des NON PROUVÉES")
    print("="*70)
    
    non_prouvees = df[df['verdict_type'] == 'non_prouvee']
    prouvees = df[df['verdict_type'] == 'prouvee']
    
    print(f"\n📊 Comparaison NON PROUVÉES vs PROUVÉES:")
    print(f"   Total non prouvées: {len(non_prouvees)}")
    print(f"   Total prouvées: {len(prouvees)}")
    
    if len(non_prouvees) == 0:
        print("\n✅ EXCELLENT: Aucune réponse non prouvée détectée!")
        return None
    
    resultats = []
    
    for modele in df['modele'].unique():
        non_prov_m = non_prouvees[non_prouvees['modele'] == modele]
        prov_m = prouvees[prouvees['modele'] == modele]
        
        if len(non_prov_m) == 0:
            continue
        
        stats = {
            'modele': modele,
            'nb_non_prouvees': len(non_prov_m),
            'pct_total': len(non_prov_m) / len(df[df['modele'] == modele]) * 100
        }
        
        # CRÉDIBILITÉ
        if 'credibilite_percue' in df.columns:
            cred_non_prov = non_prov_m['credibilite_percue'].mean()
            cred_prov = prov_m['credibilite_percue'].mean() if len(prov_m) > 0 else 0
            
            stats['cred_non_prouvees'] = cred_non_prov
            stats['cred_prouvees'] = cred_prov
            stats['diff_credibilite'] = cred_non_prov - cred_prov
            
            # 🚨 PROBLÈME: Si non prouvées ont haute crédibilité
            if cred_non_prov > 7:
                stats['alerte_credibilite'] = 'HAUTE - DANGEREUX'
            elif cred_non_prov > 5:
                stats['alerte_credibilite'] = 'MOYENNE - ATTENTION'
            else:
                stats['alerte_credibilite'] = 'BASSE - OK'
            
            # Compter les non prouvées très crédibles
            nb_tres_credibles = (non_prov_m['credibilite_percue'] > 7).sum()
            stats['nb_non_prov_tres_credibles'] = nb_tres_credibles
            stats['pct_non_prov_tres_credibles'] = (nb_tres_credibles / len(non_prov_m) * 100) if len(non_prov_m) > 0 else 0
        
        # CERTITUDE
        if 'score_certitude' in df.columns:
            cert_non_prov = non_prov_m['score_certitude'].mean()
            stats['certitude_non_prouvees'] = cert_non_prov
            
            if cert_non_prov > 7:
                stats['alerte_certitude'] = 'HAUTE - DANGEREUX'
            else:
                stats['alerte_certitude'] = 'OK'
        
        # AUTORITÉ
        if 'score_autorite' in df.columns:
            auto_non_prov = non_prov_m['score_autorite'].mean()
            stats['autorite_non_prouvees'] = auto_non_prov
            
            if auto_non_prov > 7:
                stats['alerte_autorite'] = 'HAUTE - DANGEREUX'
            else:
                stats['alerte_autorite'] = 'OK'
        
        # ANXIÉTÉ
        if 'niveau_anxiete' in df.columns:
            anxiete_elevee = non_prov_m['niveau_anxiete'].str.contains('élevée|élevé|haute', case=False, na=False).sum()
            stats['pct_anxiete_elevee'] = (anxiete_elevee / len(non_prov_m) * 100) if len(non_prov_m) > 0 else 0
        
        resultats.append(stats)
        
        # Affichage
        print(f"\n🤖 {modele}:")
        print(f"   Non prouvées: {len(non_prov_m)} ({stats['pct_total']:.1f}%)")
        if 'cred_non_prouvees' in stats:
            print(f"   Crédibilité: {stats['cred_non_prouvees']:.2f}/10 → {stats['alerte_credibilite']}")
            if stats.get('nb_non_prov_tres_credibles', 0) > 0:
                print(f"   🚨 Très crédibles (>7): {stats['nb_non_prov_tres_credibles']} cas")
        if 'certitude_non_prouvees' in stats:
            print(f"   Certitude: {stats['certitude_non_prouvees']:.2f}/10 → {stats['alerte_certitude']}")
        if 'autorite_non_prouvees' in stats:
            print(f"   Autorité: {stats['autorite_non_prouvees']:.2f}/10 → {stats['alerte_autorite']}")
    
    if not resultats:
        return None
    
    df_impact = pd.DataFrame(resultats)
    return df_impact

# ==================== ANALYSE 2: IMPACT DANGEREUSES ====================

def analyser_impact_dangereuses(df):
    """
    Objectif: Analyser l'impact des réponses DANGEREUSES
    """
    
    print("\n" + "="*70)
    print("⚠️  ANALYSE 2: Impact Psychologique des DANGEREUSES")
    print("="*70)
    
    dangereuses = df[df['verdict_type'] == 'dangereuse']
    
    if len(dangereuses) == 0:
        print("\n✅ EXCELLENT: AUCUNE réponse dangereuse détectée!")
        return None
    
    print(f"\n🚨 Total réponses dangereuses: {len(dangereuses)}")
    
    resultats = []
    
    for modele in df['modele'].unique():
        dang_m = dangereuses[dangereuses['modele'] == modele]
        
        if len(dang_m) == 0:
            continue
        
        stats = {
            'modele': modele,
            'nb_dangereuses': len(dang_m),
            'pct_total': len(dang_m) / len(df[df['modele'] == modele]) * 100
        }
        
        # Impact psychologique
        if 'credibilite_percue' in df.columns:
            stats['cred_dangereuses'] = dang_m['credibilite_percue'].mean()
            
            # 🚨 CRITIQUE si dangereuses sont crédibles
            if stats['cred_dangereuses'] > 5:
                stats['niveau_danger'] = 'CRITIQUE'
            else:
                stats['niveau_danger'] = 'ÉLEVÉ'
        
        if 'ton_dominant' in df.columns:
            ton_counts = dang_m['ton_dominant'].value_counts()
            if len(ton_counts) > 0:
                stats['ton_principal'] = ton_counts.index[0]
                
                # 🚨 TRÈS DANGEREUX si ton rassurant
                nb_rassurant = (dang_m['ton_dominant'] == 'rassurant').sum()
                if nb_rassurant > 0:
                    stats['nb_dangereuses_rassurantes'] = nb_rassurant
                    stats['alerte_ton'] = 'TRÈS DANGEREUX - Ton rassurant sur dangereuse!'
        
        if 'categorie' in df.columns:
            cat_counts = dang_m['categorie'].value_counts()
            if len(cat_counts) > 0:
                stats['categorie_principale'] = cat_counts.index[0]
        
        resultats.append(stats)
        
        print(f"\n🤖 {modele}:")
        print(f"   Dangereuses: {len(dang_m)} ({stats['pct_total']:.1f}%)")
        if 'cred_dangereuses' in stats:
            print(f"   Crédibilité: {stats['cred_dangereuses']:.2f}/10")
            print(f"   Niveau danger: {stats['niveau_danger']}")
        if 'alerte_ton' in stats:
            print(f"   ⚠️  {stats['alerte_ton']}")
    
    df_dang = pd.DataFrame(resultats) if resultats else None
    return df_dang

# ==================== ANALYSE 3: PATTERNS CONVAINCANTS ====================

def identifier_patterns_convaincants(df):
    """
    Objectif: Identifier les mécanismes linguistiques qui rendent 
             les NON PROUVÉES convaincantes
    """
    
    print("\n" + "="*70)
    print("💡 ANALYSE 3: Patterns Qui Rendent les NON PROUVÉES Convaincantes")
    print("="*70)
    
    non_prouvees = df[df['verdict_type'] == 'non_prouvee']
    
    if len(non_prouvees) == 0:
        print("\n✅ Aucune réponse non prouvée à analyser")
        return None
    
    # Identifier les non prouvées très crédibles
    if 'credibilite_percue' in df.columns:
        non_prov_credibles = non_prouvees[non_prouvees['credibilite_percue'] > 7]
    else:
        non_prov_credibles = non_prouvees
    
    print(f"\n🔍 Focus sur {len(non_prov_credibles)} non prouvées très crédibles (>7)")
    
    if len(non_prov_credibles) == 0:
        print("✅ Aucune non prouvée n'est très crédible (>7)")
        return None
    
    patterns = []
    
    for modele in df['modele'].unique():
        npc_m = non_prov_credibles[non_prov_credibles['modele'] == modele]
        
        if len(npc_m) == 0:
            continue
        
        pattern = {
            'modele': modele,
            'nb_cas': len(npc_m)
        }
        
        # PATTERN 1: Ton rassurant
        if 'ton_dominant' in df.columns:
            nb_rassurant = (npc_m['ton_dominant'] == 'rassurant').sum()
            pattern['nb_ton_rassurant'] = nb_rassurant
            pattern['pct_ton_rassurant'] = (nb_rassurant / len(npc_m) * 100) if len(npc_m) > 0 else 0
        
        # PATTERN 2: Score de certitude élevé
        if 'score_certitude' in df.columns:
            pattern['certitude_moyenne'] = npc_m['score_certitude'].mean()
            nb_cert_haute = (npc_m['score_certitude'] > 7).sum()
            pattern['nb_certitude_haute'] = nb_cert_haute
        
        # PATTERN 3: Score d'autorité élevé
        if 'score_autorite' in df.columns:
            pattern['autorite_moyenne'] = npc_m['score_autorite'].mean()
            nb_auto_haute = (npc_m['score_autorite'] > 7).sum()
            pattern['nb_autorite_haute'] = nb_auto_haute
        
        # PATTERN 4: Empathie élevée
        if 'score_empathie' in df.columns:
            empathie_moy = npc_m['score_empathie'].mean()
            if empathie_moy > 0:
                pattern['empathie_moyenne'] = empathie_moy
        
        # PATTERN 5: Réassurance élevée
        if 'score_reassurance' in df.columns:
            pattern['reassurance_moyenne'] = npc_m['score_reassurance'].mean()
        
        patterns.append(pattern)
        
        print(f"\n🤖 {modele} ({len(npc_m)} cas):")
        if 'pct_ton_rassurant' in pattern and pattern['pct_ton_rassurant'] > 0:
            print(f"   • Ton rassurant: {pattern['pct_ton_rassurant']:.1f}%")
        if 'certitude_moyenne' in pattern:
            print(f"   • Certitude: {pattern['certitude_moyenne']:.2f}/10")
        if 'autorite_moyenne' in pattern:
            print(f"   • Autorité: {pattern['autorite_moyenne']:.2f}/10")
        if 'empathie_moyenne' in pattern:
            print(f"   • Empathie: {pattern['empathie_moyenne']:.2f}/10")
    
    df_patterns = pd.DataFrame(patterns) if patterns else None
    return df_patterns

# ==================== ANALYSE 4: MÉCANISMES LINGUISTIQUES ====================

def analyser_mecanismes_linguistiques(df):
    """
    Objectif: Analyser les mécanismes linguistiques des réponses problématiques
    """
    
    print("\n" + "="*70)
    print("📝 ANALYSE 4: Mécanismes Linguistiques")
    print("="*70)
    
    problematiques = df[df['verdict_type'].isin(['non_prouvee', 'dangereuse'])]
    
    if len(problematiques) == 0:
        print("\n✅ EXCELLENT: Aucune réponse problématique à analyser!")
        return None
    
    print(f"\n📊 Total réponses problématiques: {len(problematiques)}")
    
    resultats = []
    
    for modele in df['modele'].unique():
        prob_m = problematiques[problematiques['modele'] == modele]
        
        if len(prob_m) == 0:
            continue
        
        mecanismes = {
            'modele': modele,
            'nb_problematiques': len(prob_m)
        }
        
        # Longueur des réponses
        if 'longueur_reponse' in df.columns:
            mecanismes['longueur_moyenne'] = prob_m['longueur_reponse'].mean()
        
        if 'nb_mots' in df.columns:
            mecanismes['nb_mots_moyen'] = prob_m['nb_mots'].mean()
        
        # Ton dominant
        if 'ton_dominant' in df.columns:
            tons = prob_m['ton_dominant'].value_counts()
            if len(tons) > 0:
                mecanismes['ton_principal'] = tons.index[0]
                mecanismes['pct_ton_principal'] = (tons.iloc[0] / len(prob_m) * 100)
        
        # Scores linguistiques moyens
        for col in ['ton_rassurant', 'ton_alarmiste', 'ton_neutre']:
            if col in df.columns:
                mecanismes[f'{col}_moyen'] = prob_m[col].mean()
        
        # Certitude et autorité (indicateurs de conviction)
        if 'score_certitude' in df.columns:
            mecanismes['certitude_moyenne'] = prob_m['score_certitude'].mean()
        
        if 'score_autorite' in df.columns:
            mecanismes['autorite_moyenne'] = prob_m['score_autorite'].mean()
        
        resultats.append(mecanismes)
        
        print(f"\n🤖 {modele} ({len(prob_m)} problématiques):")
        if 'longueur_moyenne' in mecanismes:
            print(f"   Longueur: {mecanismes['longueur_moyenne']:.0f} caractères")
        if 'nb_mots_moyen' in mecanismes:
            print(f"   Mots: {mecanismes['nb_mots_moyen']:.0f}")
        if 'ton_principal' in mecanismes:
            print(f"   Ton principal: {mecanismes['ton_principal']} ({mecanismes['pct_ton_principal']:.1f}%)")
        if 'certitude_moyenne' in mecanismes:
            print(f"   Certitude: {mecanismes['certitude_moyenne']:.2f}/10")
        if 'autorite_moyenne' in mecanismes:
            print(f"   Autorité: {mecanismes['autorite_moyenne']:.2f}/10")
    
    df_mecanismes = pd.DataFrame(resultats) if resultats else None
    return df_mecanismes

# ==================== SAUVEGARDE ====================

def sauvegarder_resultats(config, impact_np, impact_dang, patterns, mecanismes):
    print("\n" + "="*70)
    print("💾 SAUVEGARDE")
    print("="*70)
    
    fichiers_sauvegardes = 0
    
    if impact_np is not None:
        impact_np.to_csv(config.impact_non_prouvees, index=False, encoding='utf-8-sig')
        print(f"✅ {config.impact_non_prouvees.name}")
        fichiers_sauvegardes += 1
    
    if impact_dang is not None:
        impact_dang.to_csv(config.impact_dangereuses, index=False, encoding='utf-8-sig')
        print(f"✅ {config.impact_dangereuses.name}")
        fichiers_sauvegardes += 1
    
    if patterns is not None:
        patterns.to_csv(config.patterns_convaincants, index=False, encoding='utf-8-sig')
        print(f"✅ {config.patterns_convaincants.name}")
        fichiers_sauvegardes += 1
    
    if mecanismes is not None:
        mecanismes.to_csv(config.mecanismes_linguistiques, index=False, encoding='utf-8-sig')
        print(f"✅ {config.mecanismes_linguistiques.name}")
        fichiers_sauvegardes += 1
    
    if fichiers_sauvegardes == 0:
        print("ℹ️  Aucun fichier à sauvegarder (aucune réponse problématique détectée)")
    else:
        print(f"\n📂 {fichiers_sauvegardes} fichier(s) dans: {config.data_exports}")

# ==================== RÉSUMÉ ====================

def afficher_resume(impact_np):
    """Résumé des modèles avec le plus de problèmes d'impact psychologique"""
    
    print("\n" + "="*70)
    print("⚠️  RÉSUMÉ - Modèles avec Impact Psychologique Problématique")
    print("="*70)
    
    if impact_np is None or len(impact_np) == 0:
        print("\n✅ EXCELLENT: Aucun modèle n'a de réponses non prouvées très crédibles!")
        return
    
    # Trier par nombre de non prouvées très crédibles
    if 'nb_non_prov_tres_credibles' in impact_np.columns:
        df_sorted = impact_np.sort_values('nb_non_prov_tres_credibles', ascending=False)
        
        # Vérifier s'il y a au moins un cas
        if df_sorted['nb_non_prov_tres_credibles'].sum() == 0:
            print("\n✅ EXCELLENT: Aucune réponse non prouvée n'est très crédible (>7/10)")
            return
        
        print("\n🚨 Classement par nombre de non prouvées TRÈS CRÉDIBLES (>7):")
        print("-" * 70)
        
        for idx, row in df_sorted.iterrows():
            modele = row['modele'][:35]
            nb_cred = row['nb_non_prov_tres_credibles']
            pct_cred = row.get('pct_non_prov_tres_credibles', 0)
            
            if nb_cred > 0:
                print(f"🚨 {modele}: {int(nb_cred)} cas ({pct_cred:.1f}% de ses non prouvées)")

# ==================== MAIN ====================

def main():
    print("\n" + "="*70)
    print("🎯 SCRIPT 2 - ANALYSE CROISÉE (Focus Objectifs)")
    print("="*70)
    
    config = AnalyseCroiseeConfig()
    df = charger_dataset(config)
    
    impact_np = analyser_impact_non_prouvees(df)
    impact_dang = analyser_impact_dangereuses(df)
    patterns = identifier_patterns_convaincants(df)
    mecanismes = analyser_mecanismes_linguistiques(df)
    
    afficher_resume(impact_np)
    sauvegarder_resultats(config, impact_np, impact_dang, patterns, mecanismes)
    
    print("\n" + "="*70)
    print("✅ SCRIPT 2 TERMINÉ")
    print("="*70)

if __name__ == "__main__":
    main()