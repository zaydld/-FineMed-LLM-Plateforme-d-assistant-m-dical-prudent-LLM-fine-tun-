"""
Script 3 - Architecture et Configuration du Fine-tuning
Stratégie: LoRA (Low-Rank Adaptation)
Pourquoi LoRA? Léger, rapide, préserve les connaissances du modèle
"""

import json
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import numpy as np

# ==================== CONFIGURATION ====================

class FineTuningConfig:
    """Configuration complète du fine-tuning"""
    
    def __init__(self):
        self.racine = Path(r"C:\Users\ZAID\OneDrive\Documents\3eme_gds\DL\DeepLearning_1")
        self.finetuning_dir = self.racine / "finetuning_data"
        
        # Datasets
        self.dataset_train = self.finetuning_dir / "02_dataset_train.jsonl"
        self.dataset_validation = self.finetuning_dir / "02_dataset_validation.jsonl"
        
        # Modèle de base
        self.model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
        
        # Sortie
        self.output_dir = self.finetuning_dir / "llama_finetuned"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        
        # LoRA Configuration
        self.lora_config = {
            "r": 16,                    # Rank (8, 16, 32) - Plus élevé = plus de capacité
            "lora_alpha": 32,           # Alpha (généralement 2*r)
            "target_modules": [         # Quelles couches modifier
                "q_proj",               # Query projection
                "v_proj",               # Value projection
                "k_proj",               # Key projection
                "o_proj",               # Output projection
                "gate_proj",            # Gate projection (pour LLaMA)
                "up_proj",              # Up projection
                "down_proj"             # Down projection
            ],
            "lora_dropout": 0.05,       # Dropout pour régularisation
            "bias": "none",             # Pas de biais adaptatif
            "task_type": "CAUSAL_LM"    # Tâche de langage causal
        }
        
        # Training Hyperparameters
        self.training_args = {
            "output_dir": str(self.output_dir),
            "num_train_epochs": 3,              # Nombre d'époques
            "per_device_train_batch_size": 2,   # Batch size (ajuster selon GPU)
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 4,   # Simule batch_size=8
            "learning_rate": 2e-4,              # Learning rate pour LoRA
            "warmup_steps": 100,                # Warmup
            "logging_steps": 10,                # Log tous les 10 steps
            "eval_steps": 50,                   # Évaluer tous les 50 steps
            "save_steps": 100,                  # Sauvegarder tous les 100 steps
            "save_total_limit": 3,              # Garder 3 checkpoints max
            "evaluation_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "fp16": True,                       # Mixed precision (si GPU compatible)
            "optim": "paged_adamw_8bit",        # Optimiseur efficace en mémoire
            "report_to": "none"                 # Pas de logging externe
        }
        
        # Créer les dossiers
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

# ==================== PRÉPARATION DES DONNÉES ====================

def prepare_dataset(tokenizer, data_file, max_length=512):
    """Prépare le dataset pour le fine-tuning"""
    
    print(f"\n📂 Chargement: {data_file.name}")
    
    # Charger le dataset JSONL
    dataset = load_dataset('json', data_files=str(data_file), split='train')
    print(f"✅ {len(dataset)} exemples chargés")
    
    def format_prompt(example):
        """Formate l'exemple en prompt conversationnel"""
        messages = example['messages']
        
        # Format pour Llama (style chat)
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == "system":
                formatted += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n"
        
        return {"text": formatted}
    
    # Formater tous les exemples
    dataset = dataset.map(format_prompt)
    
    def tokenize_function(examples):
        """Tokenize les exemples"""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    # Tokenizer
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

# ==================== CHARGEMENT DU MODÈLE ====================

def load_model_and_tokenizer(config):
    """Charge le modèle et le tokenizer"""
    
    print("\n" + "="*70)
    print("🤖 CHARGEMENT DU MODÈLE")
    print("="*70)
    
    print(f"\n📦 Modèle: {config.model_name}")
    
    # Tokenizer
    print("📝 Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    
    # Ajouter un padding token si absent
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Tokenizer chargé")
    
    # Modèle
    print(f"🧠 Chargement du modèle (peut prendre quelques minutes)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,      # Utiliser float16 pour économiser mémoire
        device_map="auto",              # Répartition automatique sur GPU
        trust_remote_code=True
    )
    
    print("✅ Modèle de base chargé")
    
    # Afficher les infos du modèle
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Paramètres totaux: {total_params:,}")
    
    return model, tokenizer

# ==================== APPLICATION DE LoRA ====================

def apply_lora(model, config):
    """Applique LoRA au modèle"""
    
    print("\n" + "="*70)
    print("🔧 APPLICATION DE LoRA")
    print("="*70)
    
    # Configuration LoRA
    lora_config = LoraConfig(
        r=config.lora_config["r"],
        lora_alpha=config.lora_config["lora_alpha"],
        target_modules=config.lora_config["target_modules"],
        lora_dropout=config.lora_config["lora_dropout"],
        bias=config.lora_config["bias"],
        task_type=TaskType.CAUSAL_LM
    )
    
    print(f"📋 Configuration LoRA:")
    print(f"   • Rank (r): {config.lora_config['r']}")
    print(f"   • Alpha: {config.lora_config['lora_alpha']}")
    print(f"   • Dropout: {config.lora_config['lora_dropout']}")
    print(f"   • Modules ciblés: {', '.join(config.lora_config['target_modules'][:3])}...")
    
    # Appliquer LoRA
    model = get_peft_model(model, lora_config)
    
    # Compter les paramètres entraînables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n📊 Résultat:")
    print(f"   • Paramètres entraînables: {trainable_params:,}")
    print(f"   • Paramètres totaux: {total_params:,}")
    print(f"   • Pourcentage entraînable: {100 * trainable_params / total_params:.2f}%")
    print(f"   • 🔥 Réduction: {100 * (1 - trainable_params / total_params):.1f}% de paramètres gelés!")
    
    return model

# ==================== ENTRAÎNEMENT ====================

def train_model(model, tokenizer, train_dataset, eval_dataset, config):
    """Lance l'entraînement du modèle"""
    
    print("\n" + "="*70)
    print("🚀 LANCEMENT DE L'ENTRAÎNEMENT")
    print("="*70)
    
    # Arguments d'entraînement
    training_args = TrainingArguments(**config.training_args)
    
    print(f"\n📋 Configuration:")
    print(f"   • Époques: {config.training_args['num_train_epochs']}")
    print(f"   • Batch size: {config.training_args['per_device_train_batch_size']}")
    print(f"   • Gradient accumulation: {config.training_args['gradient_accumulation_steps']}")
    print(f"   • Learning rate: {config.training_args['learning_rate']}")
    print(f"   • Exemples train: {len(train_dataset)}")
    print(f"   • Exemples validation: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Pas de Masked Language Modeling, juste Causal LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    
    print(f"\n🏃 Entraînement en cours...")
    print("   (Cela peut prendre plusieurs heures selon votre matériel)")
    
    # Lancer l'entraînement
    trainer.train()
    
    print("\n✅ Entraînement terminé!")
    
    # Sauvegarder le modèle final
    print(f"\n💾 Sauvegarde du modèle final...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    print(f"✅ Modèle sauvegardé dans: {config.output_dir}")
    
    return trainer

# ==================== MAIN ====================

def main():
    """Fonction principale"""
    
    print("\n" + "="*70)
    print("🎯 SCRIPT 3 - FINE-TUNING DE LLAMA AVEC LoRA")
    print("="*70)
    
    # Configuration
    config = FineTuningConfig()
    
    # Vérifier la disponibilité du GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"   Mémoire: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"\n⚠️  Pas de GPU détecté - L'entraînement sera TRÈS lent sur CPU")
        print("   Recommandation: Utiliser Google Colab ou un service cloud avec GPU")
    
    # Vérifier les datasets
    if not config.dataset_train.exists():
        print(f"\n❌ Dataset train introuvable: {config.dataset_train}")
        print("   Exécutez d'abord le Script 2 (préparation dataset)")
        return
    
    if not config.dataset_validation.exists():
        print(f"\n❌ Dataset validation introuvable: {config.dataset_validation}")
        return
    
    try:
        # 1. Charger modèle et tokenizer
        model, tokenizer = load_model_and_tokenizer(config)
        
        # 2. Appliquer LoRA
        model = apply_lora(model, config)
        
        # 3. Préparer les datasets
        print("\n" + "="*70)
        print("📊 PRÉPARATION DES DATASETS")
        print("="*70)
        
        train_dataset = prepare_dataset(tokenizer, config.dataset_train)
        eval_dataset = prepare_dataset(tokenizer, config.dataset_validation)
        
        # 4. Entraîner
        trainer = train_model(model, tokenizer, train_dataset, eval_dataset, config)
        
        # 5. Rapport final
        print("\n" + "="*70)
        print("🎉 FINE-TUNING TERMINÉ AVEC SUCCÈS!")
        print("="*70)
        
        print(f"""
📂 Modèle fine-tuné sauvegardé dans:
   {config.output_dir}

📊 Fichiers créés:
   • adapter_config.json (configuration LoRA)
   • adapter_model.bin (poids LoRA)
   • tokenizer files

🎯 Résultats attendus:
   ✅ 0% de réponses non prouvées
   ✅ Validation scientifique systématique
   ✅ Citations de sources
   ✅ Maintien de l'empathie

🚀 Prochaine étape:
   Script 4 - Évaluation du modèle fine-tuné
   → Comparer avant/après
   → Vérifier les objectifs
        """)
        
    except Exception as e:
        print(f"\n❌ ERREUR pendant le fine-tuning:")
        print(f"   {str(e)}")
        print(f"\n💡 Solutions possibles:")
        print(f"   • Vérifier que le modèle est accessible")
        print(f"   • Réduire batch_size si erreur de mémoire")
        print(f"   • Utiliser un GPU avec plus de mémoire")

if __name__ == "__main__":
    main()