# ╔══════════════════════════════════════════════════════════════════╗
# ║  Push checkpoint epoch-7 BGE-M3 v2 vers HuggingFace Hub        ║
# ╚══════════════════════════════════════════════════════════════════╝

import os
from huggingface_hub import HfApi, create_repo

HF_TOKEN = "hf_TON_VRAI_TOKEN_ICI"   # ← remplace
REPO_ID  = "kgueye001/llama31-african-health-qa-lora"
CKPT_PATH = "/marimo/checkpoints-llama31-bgem3-v2/epoch-7"

assert os.path.exists(f"{CKPT_PATH}/adapter_model.safetensors"), "❌ Checkpoint introuvable"

api = HfApi(token=HF_TOKEN)

# Crée le repo (public pour pouvoir l'utiliser facilement dans le Space)
create_repo(
    REPO_ID,
    token=HF_TOKEN,
    repo_type="model",
    private=False,
    exist_ok=True
)
print(f"✅ Repo créé/vérifié : {REPO_ID}")

# Upload le checkpoint complet
api.upload_folder(
    folder_path=CKPT_PATH,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Llama-3.1-8B LoRA epoch-7 BGE-M3 v2 — val_loss=0.0587, Zindi LB=0.6200",
)
print(f"✅ Checkpoint uploadé → https://huggingface.co/{REPO_ID}")

# Liste les fichiers uploadés pour vérification
print("\nFichiers présents :")
for f in os.listdir(CKPT_PATH):
    print(f"  {f}")
