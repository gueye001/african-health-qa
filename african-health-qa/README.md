# African Health QA — Assistant médical multilingue

Solution pour le challenge Zindi/ITU "Multilingual Health Question Answering
in Low-Resource African Languages". Étant donné une question de santé
maternelle/sexuelle/reproductive dans une langue et un pays donnés, le
système produit une réponse dans la même langue.

**Score final sur le leaderboard public : 0.620** (Rang : 86e)

```
LB = 0.37 · ROUGE-1 F1  +  0.37 · ROUGE-L F1  +  0.26 · LLM-Judge
```

Une démo en ligne est déployée sur HuggingFace Spaces :
**[kgueye001/african-health-qa](https://huggingface.co/spaces/kgueye001/african-health-qa)**

## Approche

Le pipeline final combine un générateur fine-tuné avec une génération
augmentée par récupération (RAG) :

```
Question --> Embedding BGE-M3 --> Recherche FAISS (index par langue)
         --> Reranking CrossEncoder --> top-3 contextes
         --> Llama-3.1-8B-Instruct + LoRA (fine-tuné sur Train+Val)
         --> Réponse
```

| Composant | Choix |
|---|---|
| Modèle de base | `meta-llama/Llama-3.1-8B-Instruct` |
| Fine-tuning | LoRA (r=16, alpha=32) sur Train, SFT complet avec TRL |
| Retriever | `BAAI/bge-m3` (embeddings denses multilingues, 1024-dim) |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Contexte | Top-3 paires Q-A récupérées par question, index FAISS par langue |
| Décodage | Greedy pour la plupart des langues ; multi-échantillonnage (n=3) + sélection CrossEncoder pour les subsets hors amharique |

### Corrections clés qui ont fait la différence

- **`tokenizer.padding_side = "left"`** est obligatoire pour la génération
  batchée avec un modèle décodeur — sans ce fix, les sorties dégénèrent en
  tokens incohérents.
- **Ne jamais inclure `"Answer ONLY in {language}."` dans le prompt
  d'entraînement** — cette instruction n'était pas présente lors du SFT de
  base, et l'ajouter plus tard a fait chuter le score LLM-Judge à 0, même si
  le ROUGE restait correct. Le format du prompt doit être identique entre
  entraînement et inférence.
- **Sélection de l'epoch sur le score du leaderboard, pas sur la val_loss.**
  La val_loss continuait de baisser jusqu'à l'epoch 10, mais le score
  leaderboard culminait à l'epoch 7 puis se dégradait — un cas classique de
  surapprentissage sur le petit échantillon de validation (320 exemples) que
  l'early-stopping basé uniquement sur la val_loss n'aurait pas détecté.
- **Le décodage par échantillonnage n'est pas uniformément bénéfique.** La
  génération multiple + reranking CrossEncoder a amélioré le score sur 7 des
  8 langues, mais a dégradé la génération amharique (~80% de sorties
  corrompues) à cause d'une mauvaise tokenisation du script ge'ez par
  Llama-3.1. Le pipeline final route l'amharique vers un décodage greedy
  déterministe tout en gardant l'échantillonnage pour le reste.

## Historique des résultats

| Modèle | Score LB |
|---|---|
| mT0-XXL + reranking TF-IDF | 0.497 |
| Qwen2.5-7B + reranking FAISS | 0.559 |
| Llama-3.1-8B + reranking FAISS (epoch 5) | 0.596 |
| Llama-3.1-8B + reranking FAISS (epoch 8) | 0.606 |
| + retriever BGE-M3, top-3 contextes (epoch 7) | 0.612 |
| + BGE-M3 top-5 contextes | 0.613 |
| + génération multiple + sélection CrossEncoder | 0.616 |
| **+ décodage déterministe pour l'amharique (final)** | **0.620** |

Voir [`results/leaderboard_history.md`](results/leaderboard_history.md) pour
le journal complet des soumissions, y compris les pistes abandonnées.

## Ce qui n'a pas fonctionné

- **Ensemble de Qwen2.5-7B et Llama-3.1-8B** (par longueur, ou par langue) —
  toujours moins bon que Llama seul.
- **Récupération directe naïve** (retourner la réponse Train/Val la plus
  proche au-dessus d'un seuil de similarité) — score de 0.513, bien en
  dessous de la génération pure. Une récupération utile nécessite de
  conditionner le générateur sur le contexte, pas de se substituer à la
  génération.
- **Llama-3.3-70B en 4-bit (QLoRA)** — abandonné ; ~0.03 it/s en
  entraînement rendait même 5 epochs impraticables (~57h projetées).
- **Pousser l'entraînement au-delà de l'epoch optimale en val_loss** — les
  epochs 9-10 ont systématiquement dégradé le score leaderboard sur chaque
  variante testée, malgré une val_loss qui continuait de s'améliorer.

## Structure du dépôt

```
.
├── notebooks/              scripts d'entraînement et d'inférence
├── scripts/
│   ├── merge_lora_model.py     fusionne l'adaptateur LoRA dans le modèle de base
│   ├── convert_to_gguf.py      quantise le modèle fusionné en GGUF Q4_K_M
│   └── deploy/
│       └── app.py              application Gradio (HuggingFace Space, CPU)
├── results/
│   └── leaderboard_history.md  journal complet des soumissions
├── requirements.txt
└── README.md
```

## Modèles et données sur HuggingFace Hub

- Adaptateur LoRA : [kgueye001/llama31-african-health-qa-lora](https://huggingface.co/kgueye001/llama31-african-health-qa-lora)
- Modèle quantisé GGUF : [kgueye001/llama31-african-health-qa-gguf](https://huggingface.co/kgueye001/llama31-african-health-qa-gguf)
- Données d'entraînement : [kgueye001/african-health-qa-data](https://huggingface.co/datasets/kgueye001/african-health-qa-data)

## Compétition

[Multilingual Health Question Answering in Low-Resource African Languages](https://zindi.africa/) (ITU/HASH, hébergé sur Zindi).
Langues couvertes : akan (Ghana), amharique (Éthiopie), anglais (Éthiopie/Ghana/Kenya/Ouganda), luganda (Ouganda), swahili (Kenya).
