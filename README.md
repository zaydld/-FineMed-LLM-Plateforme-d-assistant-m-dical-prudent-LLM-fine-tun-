# 🩺🧠 FineMed‑LLM — Plateforme d’assistant médical prudent (LLM fine‑tuné)
🧠 Plateforme web pour interagir avec un **LLM fine‑tuné** afin de générer des réponses médicales **plus prudentes** et **plus sûres** (sans diagnostic).

**FineMed‑LLM** est un projet académique qui étudie les risques des LLMs en médecine (réponses non validées mais convaincantes) et propose une **stratégie de mitigation** via **fine‑tuning supervisé (QLoRA)**.  
👉 Le livrable applicatif de ce dépôt est une **plateforme** (API Flask + interface) permettant de tester le **modèle fine‑tuné** sur des scénarios médicaux, avec une interaction simple et structurée. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

---

## 🎯 Objectif
Déployer un assistant médical **expérimental** basé sur un modèle **fine‑tuné** pour :
- réduire les réponses **non prouvées** et les formulations à risque (fausse certitude, ton alarmiste/rassurant inadapté),
- renforcer la **prudence** (incertitude explicite, recommandations nuancées, orientation vers consultation),
- fournir une interface claire pour tester des cas sensibles (anxiogènes / psychologiques). :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## 🚀 Fonctionnalités principales
- 🤖 **Génération de réponses médicales** via le **modèle fine‑tuné** : `Qwen2.5‑1.5B‑Instruct (QLoRA)`
- 🌐 **Interface web** pour :
  - saisir un cas clinique librement,
  - tester des **scénarios prédéfinis**,
  - afficher une réponse structurée (prudence + limites de l’IA)
- 🧩 **Comportement “assistant médical prudent”** (rappels explicites : pas de diagnostic, conseils généraux, orientation si nécessaire)
- 🧪 (Contexte projet) Pipeline expérimental : dataset simulé → génération multi‑modèles → évaluation scientifique (RAG) → analyse psycho → extraction des réponses à risque → fine‑tuning. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

---

## 🧰 Technologies utilisées

| Catégorie | Technologies |
|---|---|
| Langage principal | Python |
| API / Backend | Flask |
| LLM fine‑tuné (déployé) | **Qwen2.5‑1.5B‑Instruct** |
| Fine‑tuning | **LoRA / QLoRA (4‑bits)**, **Unsloth** |
| NLP (analyse projet) | VADER, TextBlob (selon pipeline d’analyse) |
| RAG (analyse projet) | pypdf (PDF→txt), sentence‑transformers, FAISS |
| Sources médicales (analyse projet) | HAS, OMS :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}

---

## 🧬 Données utilisées (contexte du projet)
- **Jeu de cas** : **100 cas cliniques simulés** construits à partir de questions réalistes issues de *HealthCareMagic‑100k‑en* (utilisé uniquement comme réservoir de questions), répartis en 4 catégories :  
  **Simples | Complexes | Anxiogènes | Psychologiques**. :contentReference[oaicite:8]{index=8}
- **Dataset de fine‑tuning interne** :
  - extraction de réponses à risque (non prouvées / dangereuses),
  - création de **paires pédagogiques** (réponse problématique → réponse corrigée prudente),
  - format conversationnel Instruct, split train/val (80/20). :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}

---

## 🧠 Modèle fine‑tuné (celui utilisé dans la plateforme)
- **Base** : `Qwen2.5‑1.5B‑Instruct`
- **Méthode** : **QLoRA** (quantification 4 bits + adaptateurs LoRA)
- **Objectif** : alignement comportemental (prudence, incertitude, orientation), pas “diagnostic automatique”
- **Évaluation avant/après** : amélioration quantitative rapportée (ex. score base 14/30 → 16/30, +6,7%). :contentReference[oaicite:11]{index=11}

---

## ⚙️ Installation
> Les commandes exactes peuvent dépendre de votre arborescence (`backend/`, `frontend/`, etc.).  
> Le minimum requis est un environnement Python + `requirements.txt`.

Créer un environnement virtuel et installer les dépendances :
```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```
---

## 👩‍💻 Contributeurs
- [Nada Benchaou](https://github.com/BNAD-A)
- [Meriam El Kehaili](https://github.com/MeriamElk) 
- [Zayd Ladid](https://github.com/zaydld)
- [Anass Oumam](https://github.com/spaycey)



---



## ⚠️ Avertissement

FineMed-LLM est un projet académique à visée de recherche.
Il ne constitue pas un dispositif médical et ne remplace pas une consultation clinique.
Les réponses générées doivent être validées par un professionnel de santé.
Ce projet est distribué sous la licence **MIT**.  
Vous êtes libre de le réutiliser, le modifier et le distribuer avec attribution.


---
