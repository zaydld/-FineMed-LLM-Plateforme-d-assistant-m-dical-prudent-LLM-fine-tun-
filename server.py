import os
import time
import threading
from flask import Flask, jsonify, request, render_template
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# adapte si ton dossier s'appelle autrement
LORA_PATH = os.path.join("finetuning_data", "qwen2_5_1_5b_unsloth_qlora")

SYSTEM_PROMPT = (
    "Tu es un assistant médical académique. "
    "Tu ne remplaces pas un médecin. "
    "En cas de symptômes graves (douleur poitrine, essoufflement, malaise, paralysie, saignement important), "
    "recommande d'appeler les urgences. "
    "Sois prudent, clair, et propose toujours une consultation en cas de doute."
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

app = Flask(__name__, template_folder="templates", static_folder="static")

_model = None
_tok = None
_lock = threading.Lock()


def load_model():
    """Load once (thread-safe)."""
    global _model, _tok
    with _lock:
        if _model is not None and _tok is not None:
            return _model, _tok

        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=DTYPE,
            device_map="auto" if DEVICE == "cuda" else None,
        )

        model = PeftModel.from_pretrained(base, LORA_PATH)
        model.eval()

        if DEVICE == "cpu":
            model.to("cpu")

        _model, _tok = model, tok
        return _model, _tok


def build_inputs(tokenizer, user_prompt: str):
    """
    Qwen Instruct: chat template => meilleure qualité et moins d'artefacts.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt.strip()},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    return inputs


@torch.inference_mode()
def generate_text(model, tokenizer, user_prompt: str,
                  max_new_tokens=220, temperature=0.7, top_p=0.9):
    inputs = build_inputs(tokenizer, user_prompt)

    input_len = inputs["input_ids"].shape[-1]

    out = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )

    # decode seulement les nouveaux tokens (pas tout le prompt)
    new_tokens = out[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "endpoints": {
            "generate": "POST /generate  {prompt, max_new_tokens?, temperature?, top_p?}",
            "health": "GET /health"
        }
    })


@app.post("/generate")
def generate():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "Champ 'prompt' vide."}), 400

        max_new_tokens = payload.get("max_new_tokens", 220)
        temperature = payload.get("temperature", 0.7)
        top_p = payload.get("top_p", 0.9)

        model, tok = load_model()

        t0 = time.time()
        text = generate_text(
            model, tok, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        dt = int((time.time() - t0) * 1000)

        return jsonify({"answer": text, "latency_ms": dt})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Lance sur http://127.0.0.1:5000
    app.run(host="127.0.0.1", port=5000, debug=True)

