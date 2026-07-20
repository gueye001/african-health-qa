"""
African Health QA - Llama-3.1-8B (GGUF Q4_K_M) + BGE-M3 RAG
Multilingual medical Q&A across 8 African languages, running on CPU.
"""

import os
import numpy as np
import pandas as pd
import gradio as gr
import faiss

from llama_cpp import Llama
from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import hf_hub_download

GGUF_REPO    = "kgueye001/llama31-african-health-qa-gguf"
GGUF_FILE    = "model-Q4_K_M.gguf"
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
embedder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device='cpu')
print("BGE-M3 ready")

print("Loading CrossEncoder (CPU)...")
reranker = CrossEncoder(
    'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
    max_length=256,
    device='cpu'
)
print("CrossEncoder ready")


def encode_query(texts, batch_size=8):
    """Encode a small number of query texts at inference time (fast)."""
    all_vecs = []
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
    scores = reranker.predict([[question, c[0]] for c in candidats], batch_size=8)
    ranked = sorted(zip(scores, candidats), reverse=True, key=lambda x: x[0])
    return [(str(c[1])[:200], float(s)) for s, c in ranked[:3]]


print("Downloading GGUF model...")
gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
print(f"GGUF downloaded: {gguf_path}")

print("Loading llama.cpp model...")
llm = Llama(
    model_path=gguf_path,
    n_ctx=2048,
    n_threads=os.cpu_count() or 4,
    verbose=False,
)
print("llama.cpp model ready")


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


def answer_question(question, language):
    if not question or not question.strip():
        return "Please write a question first.", gr.update(value="", visible=False)

    top3 = retrieve_top3(question, language)
    ctx1 = top3[0][0] if len(top3) > 0 else ""
    ctx2 = top3[1][0] if len(top3) > 1 else ""
    ctx3 = top3[2][0] if len(top3) > 2 else ""

    similar_cases = ""
    for i, (ctx, _score) in enumerate(top3, 1):
        similar_cases += f"**{i}.** {ctx}\n\n"

    messages = make_messages(question, language, ctx1, ctx2, ctx3)
    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.0,
    )
    answer = result["choices"][0]["message"]["content"].strip()

    return answer, gr.update(value=similar_cases, visible=bool(similar_cases))


EXAMPLES = [
    ["How can I prevent HIV transmission?", "Eng_Ken"],
    ["What are the symptoms of malaria?", "Eng_Uga"],
    ["Ndiyo. Ni muhimu kuanza matibabu mara moja?", "Swa_Ken"],
    ["Wo sukuu oyarehwefo anaa akwahosan ho dwumayeni ne wo beb nna ho nsem?", "Aka_Gha"],
]

CUSTOM_CSS = """
.gradio-container {
    background: linear-gradient(135deg, #eafaf1 0%, #eef6fb 100%) !important;
}
#header-card {
    background: linear-gradient(120deg, #0f9b6c 0%, #14b8a6 100%);
    padding: 28px 32px;
    border-radius: 18px;
    color: white !important;
    margin-bottom: 18px;
    box-shadow: 0 6px 18px rgba(15, 155, 108, 0.25);
}
#header-card h1 { color: white !important; margin-bottom: 6px; }
#header-card p { color: #eafff6 !important; font-size: 1.05em; }
#ask-btn { font-size: 1.1em; font-weight: 600; }
#answer-box textarea {
    font-size: 1.15em !important;
    border: 2px solid #14b8a6 !important;
    border-radius: 12px !important;
}
#footer-note { text-align: center; color: #4b5563; margin-top: 10px; font-size: 0.9em; }
"""

with gr.Blocks(
    title="Assistant Santé Africain",
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="orange"),
    css=CUSTOM_CSS,
) as demo:
    gr.Markdown(
        """
        <div id="header-card">
        <h1>🌍 Assistant Santé Africain</h1>
        <p>Posez une question de santé dans votre langue et recevez une réponse claire.
        Disponible en 8 langues d'Afrique : Akan, Amharique, Anglais (Éthiopie, Ghana, Kenya, Ouganda), Luganda, Swahili.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="✏️ Votre question de santé",
                placeholder="Exemple : Comment puis-je me protéger du paludisme ?",
                lines=3
            )
            language_input = gr.Dropdown(
                choices=[(LANGUAGE_NAMES[lg], lg) for lg in LANGUAGES],
                value="Eng_Ken",
                label="🌐 Langue"
            )
            submit_btn = gr.Button("💬 Poser la question", variant="primary", elem_id="ask-btn")
            gr.Markdown("⏳ *La réponse peut prendre 30 à 90 secondes, merci de patienter.*")
            gr.Examples(
                examples=EXAMPLES,
                inputs=[question_input, language_input],
                label="Exemples de questions",
            )

        with gr.Column(scale=2):
            answer_output = gr.Textbox(label="✅ Réponse", lines=6, elem_id="answer-box")
            with gr.Accordion("📚 Voir des cas similaires déjà répondus", open=False):
                contexts_output = gr.Markdown(value="", visible=False)

    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, language_input],
        outputs=[answer_output, contexts_output]
    )

    gr.Markdown(
        """
        <p id="footer-note">🧑‍💻 Projet réalisé par Khadim Gueye · Assistant gratuit, à titre informatif uniquement — ne remplace pas l'avis d'un professionnel de santé.</p>
        """
    )

demo.queue(max_size=10).launch()