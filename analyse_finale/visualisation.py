import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration matplotlib pour un meilleur rendu
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# ==================== CONFIGURATION ====================

class VisualisationConfig:
    """Configuration pour les visualisations"""
    
    def __init__(self):
        self.racine = Path(r"C:\Users\ZAID\OneDrive\Documents\3eme_gds\DL\DeepLearning_1")
        
        # Fichiers d'entrée
        self.dataset_complet = self.racine / "dataset_complet.csv"
        self.analyse_dir = self.racine / "analyse_finale"
        self.data_exports = self.analyse_dir / "data_exports"
        
        # Dossier de sortie
        self.figures = self.analyse_dir / "figures"
        self.figures.mkdir(parents=True, exist_ok=True)
        
        # Fichiers de sortie
        self.fig_comparaison = self.figures / "04_comparaison_modeles.png"
        self.fig_heatmap = self.figures / "04_heatmap_performance.png"
        self.fig_radar = self.figures / "04_radar_chart.png"
        self.fig_verdicts = self.figures / "04_distribution_verdicts.png"
        self.fig_correlation = self.figures / "04_correlation_validite_psycho.png"
        self.fig_patterns = self.figures / "04_patterns_problematiques.png"
        self.fig_categorie = self.figures / "04_performance_categorie.png"
        self.fig_anxiete = self.figures / "04_analyse_anxiete.png"


# ==================== CHARGEMENT ====================

def charger_donnees(config):
    """Charge toutes les données nécessaires"""
    
    print("="*70)
    print("📂 CHARGEMENT")
    print("="*70)
    
    df = pd.read_csv(config.dataset_complet, encoding='utf-8-sig')
    print(f"✅ Dataset principal: {len(df)} lignes")
    
    # Charger les scores si disponibles
    scores_file = config.data_exports / "03_scores_modeles.csv"
    if scores_file.exists():
        df_scores = pd.read_csv(scores_file, encoding='utf-8-sig')
        print(f"✅ Scores des modèles: {len(df_scores)} modèles")
    else:
        df_scores = None
        print("⚠️  Scores non disponibles (exécuter script 3 d'abord)")
    
    return df, df_scores


# ==================== VIZ 1: COMPARAISON MODÈLES ====================

def viz_comparaison_modeles(df, config):
    """Graphique de comparaison des modèles"""
    
    print("\n📊 Génération: Comparaison des modèles...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparaison des Modèles - Vue d\'ensemble', fontsize=16, fontweight='bold')
    
    modeles = df['modele'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(modeles)))
    
    # 1. Distribution des verdicts
    if 'verdict_scientifique' in df.columns:
        ax = axes[0, 0]
        verdict_data = []
        for modele in modeles:
            df_m = df[df['modele'] == modele]
            verdicts = df_m['verdict_scientifique'].value_counts(normalize=True) * 100
            verdict_data.append(verdicts)
        
        verdict_df = pd.DataFrame(verdict_data, index=modeles)
        verdict_df.plot(kind='bar', ax=ax, stacked=False, color=colors)
        ax.set_title('Distribution des Verdicts Scientifiques', fontweight='bold')
        ax.set_xlabel('Modèle')
        ax.set_ylabel('Pourcentage (%)')
        ax.legend(title='Verdict', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
    
    # 2. Scores moyens
    ax = axes[0, 1]
    scores_cols = ['credibilite_percue', 'score_empathie', 'score_certitude']
    scores_data = []
    for modele in modeles:
        df_m = df[df['modele'] == modele]
        scores = [df_m[col].mean() if col in df.columns else 0 for col in scores_cols]
        scores_data.append(scores)
    
    x = np.arange(len(modeles))
    width = 0.25
    for i, col in enumerate(scores_cols):
        values = [s[i] for s in scores_data]
        ax.bar(x + i*width, values, width, label=col, color=colors[i])
    
    ax.set_title('Scores Psychologiques Moyens', fontweight='bold')
    ax.set_xlabel('Modèle')
    ax.set_ylabel('Score (0-10)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(modeles, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 10)
    
    # 3. Anxiété induite
    if 'niveau_anxiete' in df.columns:
        ax = axes[1, 0]
        anxiete_data = []
        for modele in modeles:
            df_m = df[df['modele'] == modele]
            anxiete_elevee = df_m['niveau_anxiete'].str.contains('élevée|élevé|haute', case=False, na=False).sum()
            pct = (anxiete_elevee / len(df_m) * 100) if len(df_m) > 0 else 0
            anxiete_data.append(pct)
        
        ax.bar(modeles, anxiete_data, color=colors)
        ax.set_title('Pourcentage d\'Anxiété Élevée Induite', fontweight='bold')
        ax.set_xlabel('Modèle')
        ax.set_ylabel('Pourcentage (%)')
        ax.tick_params(axis='x', rotation=45)
        
        # Ligne de seuil acceptable (15%)
        ax.axhline(y=15, color='r', linestyle='--', label='Seuil acceptable (15%)')
        ax.legend()
    
    # 4. Longueur des réponses
    if 'nb_mots' in df.columns:
        ax = axes[1, 1]
        box_data = [df[df['modele'] == m]['nb_mots'].dropna() for m in modeles]
        bp = ax.boxplot(box_data, labels=modeles, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title('Distribution de la Longueur des Réponses', fontweight='bold')
        ax.set_xlabel('Modèle')
        ax.set_ylabel('Nombre de mots')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(config.fig_comparaison, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: {config.fig_comparaison.name}")
    plt.close()


# ==================== VIZ 2: HEATMAP PERFORMANCE ====================

def viz_heatmap_performance(df, config):
    """Heatmap de performance par catégorie"""
    
    print("\n📊 Génération: Heatmap performance...")
    
    if 'categorie' not in df.columns or 'verdict_scientifique' not in df.columns:
        print("⚠️  Colonnes manquantes, skip")
        return
    
    modeles = df['modele'].unique()
    categories = df['categorie'].unique()
    
    # Calculer taux de réponses validées par modèle et catégorie
    heatmap_data = []
    for modele in modeles:
        row = []
        for cat in categories:
            df_subset = df[(df['modele'] == modele) & (df['categorie'] == cat)]
            if len(df_subset) > 0:
                validees = df_subset['verdict_scientifique'].str.contains('prouvée|validée', case=False, na=False).sum()
                taux = (validees / len(df_subset) * 100)
                row.append(taux)
            else:
                row.append(0)
        heatmap_data.append(row)
    
    df_heatmap = pd.DataFrame(heatmap_data, index=modeles, columns=categories)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_heatmap, annot=True, fmt='.1f', cmap='RdYlGn', 
                cbar_kws={'label': 'Taux de réponses validées (%)'}, 
                vmin=0, vmax=100)
    plt.title('Performance par Catégorie de Cas\n(% de réponses scientifiquement validées)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Catégorie de cas', fontsize=12)
    plt.ylabel('Modèle', fontsize=12)
    plt.tight_layout()
    plt.savefig(config.fig_heatmap, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: {config.fig_heatmap.name}")
    plt.close()


# ==================== VIZ 3: RADAR CHART ====================

def viz_radar_chart(df, df_scores, config):
    """Radar chart des dimensions de performance"""
    
    print("\n📊 Génération: Radar chart...")
    
    if df_scores is None:
        print("⚠️  Scores non disponibles, skip")
        return
    
    from math import pi
    
    modeles = df_scores['modele'].tolist()
    categories_radar = ['Validité', 'Sécurité Psycho', 'Qualité', 'Robustesse']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Angles pour les axes
    angles = [n / float(len(categories_radar)) * 2 * pi for n in range(len(categories_radar))]
    angles += angles[:1]
    
    # Tracer chaque modèle
    colors = plt.cm.Set2(np.linspace(0, 1, len(modeles)))
    
    for idx, (_, row) in enumerate(df_scores.iterrows()):
        values = [
            row['score_validite'],
            row['score_securite'],
            row['score_qualite'],
            row['score_robustesse']
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['modele'], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories_radar, size=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'])
    ax.grid(True)
    
    plt.title('Profil de Performance des Modèles\n(Scores sur 100)', 
              size=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(config.fig_radar, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: {config.fig_radar.name}")
    plt.close()


# ==================== VIZ 4: DISTRIBUTION VERDICTS ====================

def viz_distribution_verdicts(df, config):
    """Distribution détaillée des verdicts"""
    
    print("\n📊 Génération: Distribution verdicts...")
    
    if 'verdict_scientifique' not in df.columns:
        print("⚠️  Colonne manquante, skip")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Distribution globale
    verdicts_global = df['verdict_scientifique'].value_counts()
    colors_verd = plt.cm.Set3(np.linspace(0, 1, len(verdicts_global)))
    
    ax1.pie(verdicts_global.values, labels=verdicts_global.index, autopct='%1.1f%%',
            colors=colors_verd, startangle=90)
    ax1.set_title('Distribution Globale des Verdicts', fontsize=14, fontweight='bold')
    
    # Distribution par modèle (stacked bar)
    modeles = df['modele'].unique()
    verdicts_types = df['verdict_scientifique'].unique()
    
    verdict_data = {}
    for verdict in verdicts_types:
        verdict_data[verdict] = []
        for modele in modeles:
            df_m = df[df['modele'] == modele]
            count = (df_m['verdict_scientifique'] == verdict).sum()
            pct = (count / len(df_m) * 100) if len(df_m) > 0 else 0
            verdict_data[verdict].append(pct)
    
    x = np.arange(len(modeles))
    bottom = np.zeros(len(modeles))
    
    for verdict, values in verdict_data.items():
        ax2.bar(x, values, label=verdict, bottom=bottom)
        bottom += np.array(values)
    
    ax2.set_title('Distribution des Verdicts par Modèle', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Modèle')
    ax2.set_ylabel('Pourcentage (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(modeles, rotation=45, ha='right')
    ax2.legend(title='Verdict', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(config.fig_verdicts, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: {config.fig_verdicts.name}")
    plt.close()


# ==================== VIZ 5: CORRÉLATION ====================

def viz_correlation(df, config):
    """Visualisation des corrélations"""
    
    print("\n📊 Génération: Corrélations...")
    
    # Créer variable numérique pour verdict
    if 'verdict_scientifique' not in df.columns:
        print("⚠️  Colonne manquante, skip")
        return
    
    verdict_map = {}
    for verdict in df['verdict_scientifique'].unique():
        if 'prouvée' in str(verdict).lower() or 'validée' in str(verdict).lower():
            verdict_map[verdict] = 2
        elif 'danger' in str(verdict).lower():
            verdict_map[verdict] = 0
        else:
            verdict_map[verdict] = 1
    
    df['verdict_num'] = df['verdict_scientifique'].map(verdict_map)
    
    # Colonnes à corréler
    cols_corr = ['verdict_num', 'credibilite_percue', 'score_certitude', 
                 'score_empathie', 'score_autorite', 'score_reassurance']
    cols_disponibles = [col for col in cols_corr if col in df.columns]
    
    if len(cols_disponibles) < 2:
        print("⚠️  Pas assez de colonnes, skip")
        return
    
    # Matrice de corrélation
    df_corr = df[cols_disponibles].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={'label': 'Coefficient de corrélation'})
    plt.title('Corrélations: Validité Scientifique ↔ Impact Psychologique', 
              fontsize=14, fontweight='bold')
    
    # Renommer pour affichage
    labels = {
        'verdict_num': 'Validité Scientifique',
        'credibilite_percue': 'Crédibilité',
        'score_certitude': 'Certitude',
        'score_empathie': 'Empathie',
        'score_autorite': 'Autorité',
        'score_reassurance': 'Réassurance'
    }
    
    new_labels = [labels.get(col, col) for col in cols_disponibles]
    plt.xticks(np.arange(len(new_labels)) + 0.5, new_labels, rotation=45, ha='right')
    plt.yticks(np.arange(len(new_labels)) + 0.5, new_labels, rotation=0)
    
    plt.tight_layout()
    plt.savefig(config.fig_correlation, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: {config.fig_correlation.name}")
    plt.close()


# ==================== VIZ 6: PATTERNS PROBLÉMATIQUES ====================

def viz_patterns_problematiques(df, config):
    """Visualisation des patterns problématiques"""
    
    print("\n📊 Génération: Patterns problématiques...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Patterns Problématiques Identifiés', fontsize=16, fontweight='bold')
    
    modeles = df['modele'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(modeles)))
    
    # Pattern 1: Non validées mais très crédibles
    if 'verdict_scientifique' in df.columns and 'credibilite_percue' in df.columns:
        ax = axes[0, 0]
        pattern1_data = []
        for modele in modeles:
            df_m = df[df['modele'] == modele]
            non_val = df_m[df_m['verdict_scientifique'].str.contains('non_prouvée|non prouvée', case=False, na=False)]
            credibles = non_val[non_val['credibilite_percue'] > 7.5]
            pct = (len(credibles) / len(df_m) * 100) if len(df_m) > 0 else 0
            pattern1_data.append(pct)
        
        ax.bar(modeles, pattern1_data, color=colors)
        ax.set_title('Pattern 1: Non validées mais très crédibles', fontweight='bold')
        ax.set_ylabel('Pourcentage (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=5, color='r', linestyle='--', label='Seuil critique')
        ax.legend()
    
    # Pattern 2: Réponses dangereuses
    if 'verdict_scientifique' in df.columns:
        ax = axes[0, 1]
        pattern2_data = []
        for modele in modeles:
            df_m = df[df['modele'] == modele]
            dang = df_m['verdict_scientifique'].str.contains('danger', case=False, na=False).sum()
            pct = (dang / len(df_m) * 100) if len(df_m) > 0 else 0
            pattern2_data.append(pct)
        
        ax.bar(modeles, pattern2_data, color='red', alpha=0.7)
        ax.set_title('Pattern 2: Réponses dangereuses', fontweight='bold')
        ax.set_ylabel('Pourcentage (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=2, color='darkred', linestyle='--', label='Seuil acceptable')
        ax.legend()
    
    # Pattern 3: Anxiété élevée
    if 'niveau_anxiete' in df.columns:
        ax = axes[1, 0]
        pattern3_data = []
        for modele in modeles:
            df_m = df[df['modele'] == modele]
            anxiete = df_m['niveau_anxiete'].str.contains('élevée|élevé|haute', case=False, na=False).sum()
            pct = (anxiete / len(df_m) * 100) if len(df_m) > 0 else 0
            pattern3_data.append(pct)
        
        ax.bar(modeles, pattern3_data, color='orange', alpha=0.7)
        ax.set_title('Pattern 3: Forte anxiété induite', fontweight='bold')
        ax.set_ylabel('Pourcentage (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=15, color='darkorange', linestyle='--', label='Seuil acceptable')
        ax.legend()
    
    # Pattern 4: Ton rassurant + non validé
    if 'ton_dominant' in df.columns and 'verdict_scientifique' in df.columns:
        ax = axes[1, 1]
        pattern4_data = []
        for modele in modeles:
            df_m = df[df['modele'] == modele]
            pattern = df_m[
                (df_m['ton_dominant'] == 'rassurant') &
                (df_m['verdict_scientifique'].str.contains('non_prouvée|non prouvée|danger', case=False, na=False))
            ]
            pct = (len(pattern) / len(df_m) * 100) if len(df_m) > 0 else 0
            pattern4_data.append(pct)
        
        ax.bar(modeles, pattern4_data, color='purple', alpha=0.7)
        ax.set_title('Pattern 4: Rassurant mais non validé', fontweight='bold')
        ax.set_ylabel('Pourcentage (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=10, color='darkviolet', linestyle='--', label='Seuil acceptable')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(config.fig_patterns, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: {config.fig_patterns.name}")
    plt.close()


# ==================== MAIN ====================

def main():
    """Fonction principale"""
    
    print("\n" + "="*70)
    print("🎯 SCRIPT 4 - VISUALISATIONS")
    print("="*70)
    
    config = VisualisationConfig()
    df, df_scores = charger_donnees(config)
    
    viz_comparaison_modeles(df, config)
    viz_heatmap_performance(df, config)
    viz_radar_chart(df, df_scores, config)
    viz_distribution_verdicts(df, config)
    viz_correlation(df, config)
    viz_patterns_problematiques(df, config)
    
    print("\n" + "="*70)
    print("✅ SCRIPT 4 TERMINÉ")
    print("="*70)
    print(f"\n📂 Toutes les figures sont dans: {config.figures}")
    print(f"\nFichiers créés:")
    for fig_file in config.figures.glob("04_*.png"):
        print(f"   • {fig_file.name}")


if __name__ == "__main__":
    main()
