"""
African Health QA - Llama-3.1-8B + LoRA (bf16, ZeroGPU) + BGE-M3 RAG
Multilingual medical Q&A across 8 African languages.

Genere sur GPU (ZeroGPU, A10G) au lieu du GGUF Q4_K_M quantifie CPU -> plus fidele
au modele fine-tune (pas de perte de precision liee a la quantization 4-bit).
"""

import contextlib

import numpy as np
import pandas as pd
import gradio as gr
import faiss
import spaces
import torch

from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


@contextlib.contextmanager
def _force_cpu_only():
    """FlagEmbedding/sentence-transformers re-detectent torch.cuda.is_available()
    a chaque encode() (au lieu de respecter device='cpu' fixe au constructeur).
    Sous ZeroGPU, is_available() renvoie True meme hors contexte @spaces.GPU,
    ce qui les pousse a faire model.to('cuda') et declenche un crash bas-niveau
    ("Low-level CUDA init"). On masque temporairement la disponibilite CUDA
    pendant les appels d'embedding CPU pour forcer un vrai fallback CPU.
    """
    original = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        yield
    finally:
        torch.cuda.is_available = original

BASE_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
LORA_REPO    = "kgueye001/llama31-african-health-qa-lora"
DATASET_REPO = "kgueye001/african-health-qa-data"

LANGUAGES = [
    "Aka_Gha", "Amh_Eth", "Eng_Eth", "Eng_Gha",
    "Eng_Ken", "Eng_Uga", "Lug_Uga", "Swa_Ken",
]

LANGUAGE_NAMES = {
    "Aka_Gha": "Akan (Ghana)",
    "Amh_Eth": "Amharic (Ethiopia)",
    "Eng_Eth": "English (Ethiopia)",
    "Eng_Gha": "English (Ghana)",
    "Eng_Ken": "English (Kenya)",
    "Eng_Uga": "English (Uganda)",
    "Lug_Uga": "Luganda (Uganda)",
    "Swa_Ken": "Swahili (Kenya)",
}

K_PAR_LANGUE = {
    'Eng_Uga':10, 'Eng_Gha':10, 'Aka_Gha':8,
    'Eng_Eth':8,  'Lug_Uga':5,  'Swa_Ken':5,
    'Amh_Eth':5,  'Eng_Ken':5,
}

SYSTEM_PROMPT = (
    "You are a helpful multilingual medical assistant. "
    "Answer health questions accurately and completely in the language specified. "
    "Use the provided contexts if relevant to improve your answer."
)

# --- Retrieval (CPU) : embeddings precalcules + FAISS + CrossEncoder -------

print("Loading RAG data...")
train_df = pd.read_csv(
    f"hf://datasets/{DATASET_REPO}/Train.csv"
).rename(columns={"input": "question", "output": "answer", "subset": "language"})
val_df = pd.read_csv(
    f"hf://datasets/{DATASET_REPO}/Val.csv"
).rename(columns={"input": "question", "output": "answer", "subset": "language"})
full_df = pd.concat([train_df, val_df], ignore_index=True)
print(f"Loaded {len(full_df)} examples")

print("Loading BGE-M3 (CPU) - only needed for encoding new queries at inference time...")
# Le device cible de BGEM3FlagModel est fixe UNE FOIS a la construction (pas
# recalcule a chaque encode()). Sous ZeroGPU, torch.cuda.is_available() renvoie
# True meme hors contexte @spaces.GPU, donc le kwarg device='cpu' seul ne
# suffit pas a empecher un fallback interne vers 'cuda' -- on masque aussi
# is_available() pendant la construction pour forcer le choix CPU.
with _force_cpu_only():
    embedder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device='cpu')
print("BGE-M3 ready")

print("Loading CrossEncoder (CPU)...")
with _force_cpu_only():
    reranker = CrossEncoder(
        'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
        max_length=256,
        device='cpu'
    )
print("CrossEncoder ready")


def encode_query(texts, batch_size=8):
    """Encode a small number of query texts at inference time (fast)."""
    all_vecs = []
    with _force_cpu_only():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            output = embedder.encode(
                batch, batch_size=batch_size, max_length=512,
                return_dense=True, return_sparse=False, return_colbert_vecs=False
            )
            vecs = np.array(output['dense_vecs'], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / (norms + 1e-8)
            all_vecs.append(vecs)
    return np.vstack(all_vecs)


print("Downloading precomputed BGE-M3 embeddings (Train+Val, computed offline on GPU)...")
emb_path = hf_hub_download(repo_id=DATASET_REPO, filename="bgem3_embeddings.npz", repo_type="dataset")
embeddings_npz = np.load(emb_path)
print("Precomputed embeddings loaded")

print("Building FAISS index per language (instant, reusing precomputed vectors)...")
faiss_index = {}
for langue in sorted(full_df['language'].unique()):
    subset = full_df[full_df['language'] == langue].reset_index(drop=True)
    vecs = embeddings_npz[f"emb_{langue}"].astype(np.float32)
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    faiss_index[langue] = {
        'index': idx,
        'answers': subset['answer'].fillna('').tolist(),
        'questions': subset['question'].fillna('').tolist(),
    }
    print(f"  {langue}: {len(subset)} ex")
print("FAISS index ready")


def retrieve_top3(question, langue, max_sim=0.999):
    if langue not in faiss_index:
        return []
    idx_data = faiss_index[langue]
    K = K_PAR_LANGUE.get(langue, 5)
    q_vec = encode_query([question])
    sims, idxs = idx_data['index'].search(q_vec, k=min(K * 3, len(idx_data['questions'])))
    sims, idxs = sims[0], idxs[0]
    candidats = []
    for sim, i in zip(sims, idxs):
        if i < 0 or sim < 0.10 or sim >= max_sim:
            continue
        if idx_data['questions'][i].strip() == question.strip():
            continue
        candidats.append((idx_data['questions'][i], idx_data['answers'][i], float(sim)))
        if len(candidats) >= K:
            break
    if not candidats:
        return []
    with _force_cpu_only():
        scores = reranker.predict([[question, c[0]] for c in candidats], batch_size=8)
    ranked = sorted(zip(scores, candidats), reverse=True, key=lambda x: x[0])
    return [(str(c[1])[:200], float(s)) for s, c in ranked[:3]]


# --- Generation (GPU, ZeroGPU) : Llama-3.1-8B-Instruct + LoRA (bf16) -------

print("Loading tokenizer (does not need a GPU)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Le chargement du modele de base + application du LoRA est reporte a
# l'interieur de generate() (premiere requete, mis en cache ensuite). Sous
# ZeroGPU, aucun GPU reel n'est attache hors d'une fonction decoree
# @spaces.GPU -- meme construire le PeftModel (chargement des poids LoRA via
# safetensors) echoue en dehors de ce contexte ("No CUDA GPUs are available").
_model_cache: dict = {}


def _get_model():
    if "model" not in _model_cache:
        print("Loading base model (bf16) + LoRA adapter on GPU (first call)...")
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base_model, LORA_REPO)
        model.eval()
        model.to("cuda")
        _model_cache["model"] = model
        print("Model ready")
    return _model_cache["model"]


def make_messages(question, language, ctx1, ctx2, ctx3):
    contexts = []
    for i, ctx in enumerate([ctx1, ctx2, ctx3], 1):
        if ctx and str(ctx).strip():
            contexts.append(f"Context {i}: {str(ctx).strip()}")
    user_content = f"Language: {language}\n"
    if contexts:
        user_content += "\n".join(contexts) + "\n"
    user_content += f"Question: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


@spaces.GPU(duration=120)
def generate(messages):
    model = _get_model()
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to("cuda")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return text.strip()


def answer_question(question, language):
    if not question or not question.strip():
        return "Please enter a question.", ""

    top3 = retrieve_top3(question, language)
    ctx1 = top3[0][0] if len(top3) > 0 else ""
    ctx2 = top3[1][0] if len(top3) > 1 else ""
    ctx3 = top3[2][0] if len(top3) > 2 else ""

    contexts_display = ""
    for i, (ctx, score) in enumerate(top3, 1):
        contexts_display += f"**Context {i}** (similarity score: {score:.3f})\n{ctx}\n\n"
    if not contexts_display:
        contexts_display = "_No relevant context found in the knowledge base._"

    messages = make_messages(question, language, ctx1, ctx2, ctx3)
    answer = generate(messages)

    return answer, contexts_display


EXAMPLES = [
    ["How can I prevent HIV transmission?", "Eng_Ken"],
    ["What are the symptoms of malaria?", "Eng_Uga"],
    ["Ndiyo. Ni muhimu kuanza matibabu mara moja?", "Swa_Ken"],
    ["Wo sukuu oyarehwefo anaa akwahosan ho dwumayeni ne wo beb nna ho nsem?", "Aka_Gha"],
]

with gr.Blocks(title="African Health QA - Multilingual Medical Assistant") as demo:
    gr.Markdown(
        """
        # African Health QA Assistant
        **Llama-3.1-8B fine-tuned with LoRA (bf16, GPU) + BGE-M3 RAG retrieval**

        Multilingual health Q&A across 8 African languages: Akan, Amharic, English (Ethiopia/Ghana/Kenya/Uganda), Luganda, Swahili.

        Built for the Zindi/ITU Multilingual Health QA Challenge - Final score: **0.620** (LLM Judge: 0.756).

        Pipeline: **BGE-M3 embeddings** -> **CrossEncoder reranking** -> **Llama-3.1-8B + LoRA generation (bf16, ZeroGPU)**
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Your health question",
                placeholder="e.g. How can I prevent HIV transmission?",
                lines=3
            )
            language_input = gr.Dropdown(
                choices=[(LANGUAGE_NAMES[lg], lg) for lg in LANGUAGES],
                value="Eng_Ken",
                label="Language"
            )
            submit_btn = gr.Button("Ask", variant="primary")
            gr.Examples(
                examples=EXAMPLES,
                inputs=[question_input, language_input],
            )

        with gr.Column(scale=2):
            answer_output = gr.Textbox(label="Answer", lines=6)
            contexts_output = gr.Markdown(label="Retrieved contexts (RAG)")

    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, language_input],
        outputs=[answer_output, contexts_output]
    )

    gr.Markdown(
        """
        ---
        **Base model:** [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
        **LoRA adapter:** [kgueye001/llama31-african-health-qa-lora](https://huggingface.co/kgueye001/llama31-african-health-qa-lora)
        **Data:** [kgueye001/african-health-qa-data](https://huggingface.co/datasets/kgueye001/african-health-qa-data)
        """
    )

demo.queue(max_size=10).launch(show_error=True)
