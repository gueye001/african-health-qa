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

LANGUAGE_FLAGS = {
    "Aka_Gha": "🇬🇭 Akan (Ghana)",
    "Amh_Eth": "🇪🇹 Amharique (Éthiopie)",
    "Eng_Eth": "🇪🇹 Anglais (Éthiopie)",
    "Eng_Gha": "🇬🇭 Anglais (Ghana)",
    "Eng_Ken": "🇰🇪 Anglais (Kenya)",
    "Eng_Uga": "🇺🇬 Anglais (Ouganda)",
    "Lug_Uga": "🇺🇬 Luganda (Ouganda)",
    "Swa_Ken": "🇰🇪 Swahili (Kenya)",
}

# Note: elem_id targets the wrapper <div> Gradio itself generates for a
# component, unlike an id typed inside raw Markdown/HTML which the
# sanitizer strips — hence styling goes through elem_id everywhere below.
#
# Design direction for this one: a warm editorial/humanitarian feel,
# closer to an NGO field-guide than a dashboard — a serif headline, a
# cream "paper" canvas, a hand-woven-inspired stripe accent, and the
# Q&A laid out as a single vertical conversation (question bubble then
# answer bubble) instead of side-by-side panels.
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@600&display=swap');
:root { color-scheme: light; }
.gradio-container {
    background: #fbf6ee !important;
    max-width: 720px !important;
}
#hero { text-align: center; padding: 10px 8px 0; }
#hero h1 {
    font-family: 'Lora', serif !important;
    font-size: 2em !important;
    font-weight: 600 !important;
    color: #3a2a1c !important;
    margin: 0 0 8px !important;
}
#hero p { color: #7a6a56 !important; font-size: 1em !important; line-height: 1.55; margin: 0 auto 14px !important; max-width: 480px; }
#stripe {
    height: 5px; max-width: 220px; margin: 0 auto 20px; border-radius: 3px;
    background: linear-gradient(90deg, #c2703d 0 25%, #b8862f 25% 50%, #3d6a52 50% 75%, #8a3b32 75% 100%);
}
/* Gradio's own component internals (Accordion, empty states, icons)
   read these CSS custom properties, which flip to dark values under
   the browser's OS dark-mode `body.dark` class regardless of our
   theme choice. Scoping light values on the conversation stage fixes
   every nested component at once instead of chasing individual classes. */
#conversation {
    --block-background-fill: #ffffff;
    --block-label-text-color: #7a6a56;
    --block-border-color: #ecd9c4;
    --border-color-primary: #ecd9c4;
    --body-text-color: #3a2a1c;
    --body-text-color-subdued: #8a7c6a;
    --input-background-fill: #ffffff;
}
.bubble {
    border-radius: 18px;
    padding: 16px 20px;
    margin-bottom: 6px;
    border: 1px solid #ecd9c4;
}
.bubble-q { background: #ffffff; border-bottom-left-radius: 4px; }
.bubble-a { background: #f7ede0; border-bottom-left-radius: 4px; }
.bubble-label { font-size: 0.85em !important; font-weight: 600 !important; color: #a5601f !important; margin-bottom: 6px !important; }
#conversation label, #conversation .label-wrap span { color: #7a6a56 !important; }
#conversation input, #conversation textarea, #conversation select {
    background: #ffffff !important;
    color: #3a2a1c !important;
    border: 1px solid #ecd9c4 !important;
}
#ask-btn {
    font-size: 1.02em !important;
    font-weight: 700 !important;
    border-radius: 999px !important;
    padding: 10px 26px !important;
    background: #c2703d !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(194, 112, 61, 0.28);
    display: block !important;
    margin: 4px auto 0 !important;
}
#ask-btn:hover { background: #a85c2f !important; }
#wait-note * { color: #a5601f !important; text-align: center; font-size: 0.88em; }
#similar-cases { color: #5c4f3f !important; }
#footer {
    text-align: center;
    margin-top: 22px;
    padding: 14px;
    border-top: 1px solid #ecd9c4;
}
#footer * { color: #a39784 !important; font-size: 0.88em; }
"""

with gr.Blocks(
    title="Assistant Santé Africain",
    theme=gr.themes.Soft(primary_hue="orange", secondary_hue="stone"),
    css=CUSTOM_CSS,
) as demo:
    gr.Markdown(
        """
# 🌍 Assistant Santé Africain

Posez une question de santé, recevez une réponse claire — dans votre langue.
        """,
        elem_id="hero",
    )
    gr.HTML('<div id="stripe"></div>')

    with gr.Column(elem_id="conversation"):
        with gr.Column(elem_classes=["bubble", "bubble-q"]):
            gr.Markdown("🧑 VOUS", elem_classes=["bubble-label"])
            language_input = gr.Dropdown(
                choices=[(LANGUAGE_FLAGS[lg], lg) for lg in LANGUAGES],
                value="Eng_Ken",
                label="🌐 Langue"
            )
            question_input = gr.Textbox(
                label="Question de santé",
                placeholder="Exemple : Comment puis-je me protéger du paludisme ?",
                lines=3
            )
            gr.Examples(
                examples=EXAMPLES,
                inputs=[question_input, language_input],
                label="💡 Exemples de questions",
            )

        submit_btn = gr.Button("💬 Poser la question", variant="primary", elem_id="ask-btn")
        gr.Markdown("⏳ *La réponse peut prendre 30 à 90 secondes, merci de patienter.*", elem_id="wait-note")

        with gr.Column(elem_classes=["bubble", "bubble-a"]):
            gr.Markdown("🌍 ASSISTANT", elem_classes=["bubble-label"])
            answer_output = gr.Textbox(label="", lines=6, elem_id="answer-box", show_label=False)
            with gr.Accordion("📚 Voir des cas similaires déjà répondus", open=False):
                contexts_output = gr.Markdown(value="", visible=False, elem_id="similar-cases")

    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, language_input],
        outputs=[answer_output, contexts_output]
    )

    gr.Markdown(
        "🧑‍💻 Projet réalisé par Khadim Gueye · Assistant gratuit, à titre informatif uniquement — "
        "ne remplace pas l'avis d'un professionnel de santé.",
        elem_id="footer",
    )

demo.queue(max_size=10).launch()