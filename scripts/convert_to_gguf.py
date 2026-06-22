# ╔══════════════════════════════════════════════════════════════════╗
# ║  Conversion modèle fusionné → GGUF 4-bit (Q4_K_M)              ║
# ║  Nécessite llama.cpp — clone + build                           ║
# ╚══════════════════════════════════════════════════════════════════╝

import subprocess
import os

MERGED_DIR = '/marimo/merged_model'
GGUF_DIR   = '/marimo/gguf_output'
os.makedirs(GGUF_DIR, exist_ok=True)

# 1. Clone llama.cpp si pas déjà fait
if not os.path.exists('/marimo/llama.cpp'):
    print("Clone de llama.cpp...")
    subprocess.run(
        ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", "/marimo/llama.cpp"],
        check=True
    )

# 2. Installe les dépendances Python de conversion
print("Installation dépendances...")
subprocess.run(
    ["pip", "install", "-r", "/marimo/llama.cpp/requirements.txt"],
    check=True
)

# 3. Convertit HF → GGUF (format f16 d'abord)
print("Conversion HF → GGUF f16...")
subprocess.run([
    "python", "/marimo/llama.cpp/convert_hf_to_gguf.py",
    MERGED_DIR,
    "--outfile", f"{GGUF_DIR}/model-f16.gguf",
    "--outtype", "f16"
], check=True)
print("✅ Conversion f16 terminée")

# 4. Build llama.cpp (pour avoir l'outil de quantisation)
print("Build llama.cpp (quantize tool)...")
subprocess.run(["cmake", "-B", "build"], cwd="/marimo/llama.cpp", check=True)
subprocess.run(
    ["cmake", "--build", "build", "--config", "Release", "-j", "8", "--target", "llama-quantize"],
    cwd="/marimo/llama.cpp", check=True
)
print("✅ Build terminé")

# 5. Quantise en Q4_K_M (bon compromis qualité/taille pour CPU)
print("Quantisation Q4_K_M...")
subprocess.run([
    "/marimo/llama.cpp/build/bin/llama-quantize",
    f"{GGUF_DIR}/model-f16.gguf",
    f"{GGUF_DIR}/model-Q4_K_M.gguf",
    "Q4_K_M"
], check=True)
print("✅ Quantisation terminée")

size_mb = os.path.getsize(f"{GGUF_DIR}/model-Q4_K_M.gguf") / 1e6
print(f"\n✅ Modèle GGUF prêt : {GGUF_DIR}/model-Q4_K_M.gguf")
print(f"   Taille : {size_mb:.0f} MB")
