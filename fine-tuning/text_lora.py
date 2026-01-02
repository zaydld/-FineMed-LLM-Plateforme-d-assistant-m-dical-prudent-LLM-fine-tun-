import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

LORA_DIR = r"C:\Users\pc\Projects\DeepLearning_1\finetuning_data\qwen2_5_1_5b_unsloth_qlora"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading base model + tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=512,
    dtype=torch.float16,
    load_in_4bit=True,
)

print("Loading LoRA adapter...")
model.load_adapter(LORA_DIR)
FastLanguageModel.for_inference(model)

prompt = [{"role":"user","content":"Ma mère se plaint d’une douleur intense dans la poitrine qui irradie vers le bras gauche depuis 20 minutes. Elle pense que c’est juste le stress. Dois-je m’inquiéter ?"}]
text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer([text], return_tensors="pt").to(device)

out = model.generate(**inputs, max_new_tokens=120)
print("\n=== OUTPUT ===")
print(tokenizer.decode(out[0], skip_special_tokens=True))
