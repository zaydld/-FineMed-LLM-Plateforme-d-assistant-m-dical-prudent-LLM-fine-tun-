import os
import re
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# =========================================
# Charger .env (portable)
# =========================================
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY manquante dans .env")

client = Groq(api_key=GROQ_API_KEY)

# =========================================
# Nettoyage <think>...</think>
# =========================================
def clean_response(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# =========================================
# Appel générique Groq
# =========================================
def call_groq(prompt: str, model_name: str) -> str:
    try:
        r = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        content = r.choices[0].message.content or ""
        return clean_response(content)
    except Exception as e:
        return f"[ERREUR GROQ {model_name}] {e}"

# =========================================
# Modèles
# =========================================
def call_llama4_scout(prompt: str) -> str:
    return call_groq(prompt, "meta-llama/llama-4-scout-17b-16e-instruct")

def call_qwen3_32b(prompt: str) -> str:
    return call_groq(prompt, "qwen/qwen3-32b")
def call_gpt_oss_20b(prompt: str) -> str:
    return call_groq(prompt, "openai/gpt-oss-20b")


MODELS = {
    "llama4_scout": call_llama4_scout,
    "qwen3_32b": call_qwen3_32b,
    "gpt_oss_20b":call_gpt_oss_20b,
}
