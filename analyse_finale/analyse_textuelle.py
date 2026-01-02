"""
Script 1 - Analyse Statistique Descriptive
Focus: Identifier les réponses NON PROUVÉES et DANGEREUSES
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

class AnalyseConfig:
    def __init__(self):
        self.racine = Path(r"C:\Users\ZAID\OneDrive\Documents\3eme_gds\DL\DeepLearning_1")
        self.dataset_complet = self.racine / "dataset_complet.csv"
        self.analyse_dir = self.racine / "analyse_finale"
        self.data_exports = self.analyse_dir / "data_exports"
        
        self.stats_par_modele = self.data_exports / "01_stats_par_modele.csv"
        self.stats_par_categorie = self.data_exports / "01_stats_par_categorie.csv"
        self.focus_problematiques = self.data_exports / "01_FOCUS_reponses_problematiques.csv"
        
        self.data_exports.mkdir(parents=True, exist_ok=True)

# ==================== DÉTECTION VERDICTS (CORRIGÉ) ====================

def est_prouvee(verdict):
    """Vérifie si le verdict est Prouvee"""
    if pd.isna(verdict):
        return False
    v = str(verdict).strip()
    # 🔥 Correspondance exacte avec tes données
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
    # 🔥 Correspondance exacte avec tes données
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
    print("📂 CHARGEMENT DES DONNÉES")
    print("="*70)
    
    df = pd.read_csv(config.dataset_complet, encoding='utf-8-sig')
    print(f"✅ {len(df):,} lignes chargées")
    print(f"   Modèles: {list(df['modele'].unique())}")
    
    # Créer colonne verdict normalisé
    if 'verdict_scientifique' in df.columns:
        df['verdict_type'] = df['verdict_scientifique'].apply(detecter_type_verdict)
        
        print(f"\n🔍 Détection des verdicts:")
        for v_raw in sorted(df['verdict_scientifique'].unique()):
            v_type = detecter_type_verdict(v_raw)
            count = (df['verdict_scientifique'] == v_raw).sum()
            emoji = "🚨" if v_type in ['non_prouvee', 'dangereuse'] else "✅" if v_type == 'prouvee' else "🔍"
            print(f"   {emoji} '{v_raw}' → {v_type} ({count} cas)")
        
        print(f"\n📊 Distribution globale:")
        for v_type in ['prouvee', 'plausible', 'non_prouvee', 'dangereuse', 'inconnu']:
            count = (df['verdict_type'] == v_type).sum()
            if count > 0:
                pct = count / len(df) * 100
                emoji = "🚨" if v_type in ['non_prouvee', 'dangereuse'] else "✅" if v_type == 'prouvee' else "🔍"
                print(f"   {emoji} {v_type}: {count} ({pct:.1f}%)")
    
    return df

# ==================== ANALYSE PAR MODÈLE ====================

def analyser_par_modele(df):
    print("\n" + "="*70)
    print("📊 ANALYSE PAR MODÈLE - FOCUS RÉPONSES PROBLÉMATIQUES")
    print("="*70)
    
    modeles = df['modele'].unique()
    resultats = []
    
    for modele in modeles:
        print(f"\n🤖 {modele}")
        df_m = df[df['modele'] == modele]
        total = len(df_m)
        
        stats = {
            'modele': modele,
            'nb_reponses': total,
            'nb_cas_uniques': df_m['id_cas'].nunique()
        }
        
        if 'verdict_type' in df.columns:
            # FOCUS: Réponses problématiques
            nb_non_prouvees = (df_m['verdict_type'] == 'non_prouvee').sum()
            nb_dangereuses = (df_m['verdict_type'] == 'dangereuse').sum()
            nb_prouvees = (df_m['verdict_type'] == 'prouvee').sum()
            nb_plausibles = (df_m['verdict_type'] == 'plausible').sum()
            
            stats['nb_non_prouvees'] = nb_non_prouvees
            stats['pct_non_prouvees'] = (nb_non_prouvees / total * 100)
            
            stats['nb_dangereuses'] = nb_dangereuses
            stats['pct_dangereuses'] = (nb_dangereuses / total * 100)
            
            stats['nb_prouvees'] = nb_prouvees
            stats['pct_prouvees'] = (nb_prouvees / total * 100)
            
            stats['nb_plausibles'] = nb_plausibles
            stats['pct_plausibles'] = (nb_plausibles / total * 100)
            
            # Total problématiques
            nb_problematiques = nb_non_prouvees + nb_dangereuses
            stats['nb_problematiques'] = nb_problematiques
            stats['pct_problematiques'] = (nb_problematiques / total * 100)
            
            print(f"   🚨 RÉPONSES PROBLÉMATIQUES:")
            print(f"      • Non prouvées: {nb_non_prouvees} ({stats['pct_non_prouvees']:.1f}%)")
            print(f"      • Dangereuses: {nb_dangereuses} ({stats['pct_dangereuses']:.1f}%)")
            print(f"      • TOTAL PROBLÉMATIQUES: {nb_problematiques} ({stats['pct_problematiques']:.1f}%)")
            print(f"   ✅ Réponses acceptables:")
            print(f"      • Prouvées: {nb_prouvees} ({stats['pct_prouvees']:.1f}%)")
            print(f"      • Plausibles: {nb_plausibles} ({stats['pct_plausibles']:.1f}%)")
        
        # Impact psychologique sur les NON PROUVÉES
        if 'credibilite_percue' in df.columns:
            non_prouvees = df_m[df_m['verdict_type'] == 'non_prouvee']
            if len(non_prouvees) > 0:
                stats['cred_non_prouvees'] = non_prouvees['credibilite_percue'].mean()
                print(f"   💡 Impact psycho sur NON PROUVÉES:")
                print(f"      • Crédibilité moyenne: {stats['cred_non_prouvees']:.2f}/10")
                
                # Réponses non prouvées MAIS crédibles (DANGEREUX!)
                non_prov_credibles = non_prouvees[non_prouvees['credibilite_percue'] > 7]
                stats['nb_non_prov_credibles'] = len(non_prov_credibles)
                if len(non_prov_credibles) > 0:
                    print(f"      • 🚨 Non prouvées mais crédibles (>7): {len(non_prov_credibles)}")
            else:
                stats['cred_non_prouvees'] = 0
                stats['nb_non_prov_credibles'] = 0
        
        # Anxiété
        if 'niveau_anxiete' in df.columns:
            anxiete_elevee = df_m['niveau_anxiete'].str.contains('élevée|élevé|haute', case=False, na=False).sum()
            stats['pct_anxiete_elevee'] = (anxiete_elevee / total * 100)
            print(f"   😰 Anxiété élevée: {anxiete_elevee} ({stats['pct_anxiete_elevee']:.1f}%)")
        
        # Empathie
        if 'score_empathie' in df.columns:
            stats['empathie_moyenne'] = df_m['score_empathie'].mean()
            if stats['empathie_moyenne'] > 0:
                print(f"   ❤️  Empathie: {stats['empathie_moyenne']:.2f}/10")
            else:
                print(f"   ❤️  Empathie: N/A (questions factuelles)")
        
        resultats.append(stats)
    
    return pd.DataFrame(resultats)

# ==================== ANALYSE RÉPONSES PROBLÉMATIQUES ====================

def analyser_reponses_problematiques(df):
    """Analyse détaillée des réponses NON PROUVÉES et DANGEREUSES"""
    
    print("\n" + "="*70)
    print("🚨 ANALYSE DÉTAILLÉE DES RÉPONSES PROBLÉMATIQUES")
    print("="*70)
    
    resultats = []
    
    for modele in df['modele'].unique():
        df_m = df[df['modele'] == modele]
        
        # NON PROUVÉES
        non_prouvees = df_m[df_m['verdict_type'] == 'non_prouvee']
        if len(non_prouvees) > 0:
            stats_np = {
                'modele': modele,
                'type_probleme': 'NON_PROUVEE',
                'nb_cas': len(non_prouvees),
                'pct_total': len(non_prouvees) / len(df_m) * 100
            }
            
            if 'credibilite_percue' in non_prouvees.columns:
                stats_np['credibilite_moyenne'] = non_prouvees['credibilite_percue'].mean()
                stats_np['credibilite_max'] = non_prouvees['credibilite_percue'].max()
                # Combien sont très crédibles?
                stats_np['nb_credibles_7plus'] = (non_prouvees['credibilite_percue'] > 7).sum()
            
            if 'ton_dominant' in non_prouvees.columns:
                ton_counts = non_prouvees['ton_dominant'].value_counts()
                if len(ton_counts) > 0:
                    stats_np['ton_principal'] = ton_counts.index[0]
                    stats_np['nb_ton_rassurant'] = (non_prouvees['ton_dominant'] == 'rassurant').sum()
            
            resultats.append(stats_np)
        
        # DANGEREUSES
        dangereuses = df_m[df_m['verdict_type'] == 'dangereuse']
        if len(dangereuses) > 0:
            stats_d = {
                'modele': modele,
                'type_probleme': 'DANGEREUSE',
                'nb_cas': len(dangereuses),
                'pct_total': len(dangereuses) / len(df_m) * 100
            }
            
            if 'credibilite_percue' in dangereuses.columns:
                stats_d['credibilite_moyenne'] = dangereuses['credibilite_percue'].mean()
                stats_d['credibilite_max'] = dangereuses['credibilite_percue'].max()
            
            if 'ton_dominant' in dangereuses.columns:
                stats_d['nb_ton_rassurant'] = (dangereuses['ton_dominant'] == 'rassurant').sum()
            
            resultats.append(stats_d)
    
    if resultats:
        df_prob = pd.DataFrame(resultats)
        
        print("\n📋 Résumé:")
        for _, row in df_prob.iterrows():
            print(f"\n{row['modele']} - {row['type_probleme']}:")
            print(f"   • Nombre: {row['nb_cas']} ({row['pct_total']:.1f}%)")
            if 'credibilite_moyenne' in row and not pd.isna(row['credibilite_moyenne']):
                print(f"   • Crédibilité moyenne: {row['credibilite_moyenne']:.2f}")
            if 'nb_credibles_7plus' in row and row['nb_credibles_7plus'] > 0:
                print(f"   • 🚨 Cas très crédibles (>7): {row['nb_credibles_7plus']}")
            if 'nb_ton_rassurant' in row and row['nb_ton_rassurant'] > 0:
                print(f"   • ⚠️  Ton rassurant: {row['nb_ton_rassurant']} cas")
        
        return df_prob
    
    print("✅ Aucune réponse problématique détectée pour aucun modèle!")
    return None

# ==================== SAUVEGARDE ====================

def sauvegarder_resultats(config, stats_modele, focus_prob):
    print("\n" + "="*70)
    print("💾 SAUVEGARDE")
    print("="*70)
    
    if stats_modele is not None:
        stats_modele.to_csv(config.stats_par_modele, index=False, encoding='utf-8-sig')
        print(f"✅ {config.stats_par_modele.name}")
    
    if focus_prob is not None:
        focus_prob.to_csv(config.focus_problematiques, index=False, encoding='utf-8-sig')
        print(f"✅ {config.focus_problematiques.name}")
    
    print(f"\n📂 Tous les fichiers dans: {config.data_exports}")

# ==================== RÉSUMÉ ====================

def afficher_classement(stats_modele):
    """Classement des modèles par nombre de réponses problématiques"""
    
    print("\n" + "="*70)
    print("🏆 CLASSEMENT - Du meilleur au pire")
    print("   (Critère: MOINS de non prouvées + MOINS de dangereuses)")
    print("="*70)
    
    if 'pct_problematiques' not in stats_modele.columns:
        return
    
    df_sorted = stats_modele.sort_values('pct_problematiques').reset_index(drop=True)
    
    print("\nRang | Modèle                       | Prouvées | Non prouvées | Dangereuses | TOTAL Problèmes")
    print("-" * 100)
    
    for idx, row in df_sorted.iterrows():
        rang = idx + 1
        modele = row['modele'][:28].ljust(28)
        prouv = row.get('pct_prouvees', 0)
        non_p = row.get('pct_non_prouvees', 0)
        dang = row.get('pct_dangereuses', 0)
        total_prob = row.get('pct_problematiques', 0)
        
        emoji = "🥇" if rang == 1 else "🥈" if rang == 2 else "🥉" if rang == 3 else f" {rang} "
        
        # 🔥 Marqueurs d'alerte
        alert_non_p = " ⚠️" if non_p > 2.0 else ""
        alert_dang = " 🚨" if dang > 0 else ""
        
        print(f"{emoji} | {modele} | {prouv:>7.1f}% | {non_p:>10.1f}%{alert_non_p:<3} | {dang:>9.1f}%{alert_dang:<3} | {total_prob:>13.1f}%")

# ==================== MAIN ====================

def main():
    print("\n" + "="*70)
    print("🎯 SCRIPT 1 - FOCUS RÉPONSES NON PROUVÉES & DANGEREUSES")
    print("="*70)
    
    config = AnalyseConfig()
    df = charger_dataset(config)
    
    if df is None:
        return
    
    stats_modele = analyser_par_modele(df)
    focus_prob = analyser_reponses_problematiques(df)
    
    afficher_classement(stats_modele)
    sauvegarder_resultats(config, stats_modele, focus_prob)
    
    print("\n" + "="*70)
    print("✅ SCRIPT 1 TERMINÉ")
    print("="*70)

if __name__ == "__main__":
    main()