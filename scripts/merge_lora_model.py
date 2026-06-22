# ╔══════════════════════════════════════════════════════════════════╗
# ║  Fusion LoRA + base model → modèle mergé pour conversion GGUF  ║
# ╚══════════════════════════════════════════════════════════════════╝

import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, set_peft_model_state_dict
import safetensors.torch

HF_TOKEN = "hf_TON_TOKEN_ICI"
os.environ["HF_TOKEN"] = HF_TOKEN
from huggingface_hub import login
login(token=HF_TOKEN, add_to_git_credential=False)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CKPT_EP7   = '/marimo/checkpoints-llama31-bgem3-v2/epoch-7/adapter_model.safetensors'
MERGED_DIR = '/marimo/merged_model'
os.makedirs(MERGED_DIR, exist_ok=True)

print("Chargement modèle de base...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16,
    device_map="auto", token=HF_TOKEN
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, lora_config)

print("Injection des poids LoRA epoch-7...")
lora_weights = safetensors.torch.load_file(CKPT_EP7)
set_peft_model_state_dict(model, lora_weights)

print("Fusion LoRA → modèle final (merge_and_unload)...")
merged_model = model.merge_and_unload()

print(f"Sauvegarde du modèle fusionné → {MERGED_DIR}")
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

print("✅ Modèle fusionné sauvegardé")
print(f"   Fichiers : {os.listdir(MERGED_DIR)}")

gc.collect()
torch.cuda.empty_cache()
