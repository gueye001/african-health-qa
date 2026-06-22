# ╔══════════════════════════════════════════════════════════════════╗
# ║  Push Train.csv + Val.csv vers HuggingFace Dataset Hub          ║
# ╚══════════════════════════════════════════════════════════════════╝

import os
import pandas as pd
from huggingface_hub import HfApi, create_repo

HF_TOKEN = "hf_TON_TOKEN_WRITE_ICI"   # ← le token Write qui a fonctionné
DATASET_REPO = "kgueye001/african-health-qa-data"

# Vérifie les fichiers
TRAIN_PATH = '/marimo/Train.csv'
VAL_PATH   = '/marimo/Val.csv'

for path in [TRAIN_PATH, VAL_PATH]:
    assert os.path.exists(path), f"❌ Introuvable : {path}"
    size_mb = os.path.getsize(path) / 1e6
    print(f"✅ {path} — {size_mb:.2f} MB")

# Crée le repo dataset
api = HfApi(token=HF_TOKEN)
create_repo(
    DATASET_REPO,
    token=HF_TOKEN,
    repo_type="dataset",
    private=False,
    exist_ok=True
)
print(f"✅ Dataset repo créé/vérifié : {DATASET_REPO}")

# Upload les deux fichiers CSV
api.upload_file(
    path_or_fileobj=TRAIN_PATH,
    path_in_repo="Train.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",
    commit_message="Add Train.csv — 29815 ex, 8 African languages health QA"
)
print("✅ Train.csv uploadé")

api.upload_file(
    path_or_fileobj=VAL_PATH,
    path_in_repo="Val.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",
    commit_message="Add Val.csv — 6686 ex"
)
print("✅ Val.csv uploadé")

print(f"\n✅ Dataset disponible → https://huggingface.co/datasets/{DATASET_REPO}")
print("\nPour charger dans le Space :")
print(f'  train_df = pd.read_csv("hf://datasets/{DATASET_REPO}/Train.csv")')
print(f'  val_df   = pd.read_csv("hf://datasets/{DATASET_REPO}/Val.csv")')
