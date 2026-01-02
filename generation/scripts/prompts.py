def build_prompt(row):
    """
    Construit le prompt à partir d'une ligne du CSV cas_cliniques.
    row : pandas.Series contenant les colonnes 'titre', 'description', 'categorie', 'objectif'
    """
    return f"""
Tu es un assistant médical virtuel.
Tu dois répondre de manière prudente, nuancée, factuelle, et rappeler que cela ne remplace pas une consultation médicale en présentiel.

Titre du cas : {row['titre']}
Catégorie : {row['categorie']}
Objectif de l'utilisateur : {row['objectif']}

Description du cas :
{row['description']}

Donne une réponse structurée avec les parties suivantes :

1. Analyse du cas
2. Hypothèses diagnostiques possibles
3. Conseils pour le patient
4. Niveau d'urgence (Alerte vitale / Urgence / Consultation / Auto-surveillance)
5. Limites de la réponse (ce que l'IA ne peut pas faire, importance de consulter un professionnel)

Ta réponse doit être claire et en français.
""".strip()
