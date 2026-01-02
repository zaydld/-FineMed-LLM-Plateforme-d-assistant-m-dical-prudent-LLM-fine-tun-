import math
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===================== CONFIG =====================
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LEN = 512

# On est dans: DeepLearning_1/fine-tuning/eval_perplexity_lora.py
# Donc ROOT = DeepLearning_1
ROOT = Path(__file__).resolve().parents[1]

ADAPTER_DIR = ROOT / "finetuning_data" / "qwen2_5_1_5b_unsloth_qlora"
VAL_FILE    = ROOT / "finetuning_data" / "02_dataset_validation.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("📌 ROOT:", ROOT)
print("📌 ADAPTER_DIR:", ADAPTER_DIR)
print("📌 VAL_FILE:", VAL_FILE)
print("✅ CUDA:", torch.cuda.is_available())

if not ADAPTER_DIR.exists():
    raise FileNotFoundError(f"❌ Adapter folder introuvable: {ADAPTER_DIR}")

if not VAL_FILE.exists():
    raise FileNotFoundError(f"❌ Validation file introuvable: {VAL_FILE}")

# ===================== LOAD TOKENIZER =====================
# IMPORTANT: On charge le tokenizer depuis TON dossier LoRA (il contient tokenizer.json, chat template…)
print("\n🔹 Chargement tokenizer (depuis le dossier LoRA)")
tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), use_fast=True)

# ===================== LOAD BASE MODEL + APPLY LORA =====================
print("🔹 Chargement base model (fp16)")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("🔹 Application LoRA adapter")
model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
model.eval()

# ===================== DATA =====================
print("\n🔹 Chargement dataset validation")
dataset = load_dataset("json", data_files=str(VAL_FILE), split="train")
print("✅ Nb exemples val:", len(dataset))

def messages_to_text(ex):
    return tokenizer.apply_chat_template(
        ex["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

# ===================== PERPLEXITY =====================
losses = []

for i, ex in enumerate(dataset):
    text = messages_to_text(ex)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
    ).to(device)

    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
        losses.append(out.loss.item())

mean_loss = sum(losses) / len(losses)
ppl = math.exp(mean_loss)

print("\n✅ Résultats (Validation)")
print(f"📉 Mean loss  : {mean_loss:.4f}")
print(f"📊 Perplexity : {ppl:.2f}")
