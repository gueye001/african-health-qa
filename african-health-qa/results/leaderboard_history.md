# Leaderboard submission history

Chronological log of every submission made during the competition, including
dead ends — kept so future attempts don't re-walk the same paths.

## Summary table

| # | Submission | Model | Retrieval | Decoding | LB score |
|---|---|---|---|---|---|
| 1 | mt0-xxl + TF-IDF | mT0-XXL | TF-IDF char n-gram | beam search | 0.497 |
| 2 | Qwen2.5-7B FAISS | Qwen2.5-7B-Instruct LoRA | MiniLM + FAISS + CrossEncoder | greedy | 0.559 |
| 3 | Qwen2.5-7B epoch10 | Qwen2.5-7B-Instruct LoRA (ep10) | same | greedy | 0.560 |
| 4 | Ensemble Qwen+Llama (length) | both | same | greedy | 0.580 |
| 5 | Llama FAISS epoch5 | Llama-3.1-8B-Instruct LoRA | MiniLM + FAISS + CrossEncoder | greedy | 0.596 |
| 6 | Ensemble Qwen+Llama (per-language) | both | same | greedy | 0.570 |
| 7 | Retrieval direct (sim>=0.90) | none (pure retrieval) | MiniLM + FAISS | n/a | 0.513 |
| 8 | Llama FAISS epoch8 | Llama-3.1-8B-Instruct LoRA | MiniLM + FAISS + CrossEncoder | greedy | 0.606 |
| 9 | Llama FAISS epoch10 | Llama-3.1-8B-Instruct LoRA (ep10) | same | greedy | 0.591 |
| 10 | BGE-M3 v1 (buggy prompt) | Llama-3.1-8B-Instruct LoRA | BGE-M3 top-3 | greedy | 0.408 (LLM-Judge=0) |
| 11 | BGE-M3 v2 epoch6 | Llama-3.1-8B-Instruct LoRA | BGE-M3 top-3 | greedy | 0.608 |
| 12 | BGE-M3 v2 epoch7 | Llama-3.1-8B-Instruct LoRA | BGE-M3 top-3 | greedy | 0.612 |
| 13 | BGE-M3 v2 epoch8 | Llama-3.1-8B-Instruct LoRA | BGE-M3 top-3 | greedy | 0.608 |
| 14 | BGE-M3 v2 epoch10 | Llama-3.1-8B-Instruct LoRA (ep10) | BGE-M3 top-3 | greedy | 0.605 |
| 15 | BGE-M3 v2 epoch7, top-5 context | Llama-3.1-8B-Instruct LoRA | BGE-M3 top-5 | greedy | 0.613 |
| 16 | Multi-sample + CrossEncoder selection | Llama-3.1-8B-Instruct LoRA | BGE-M3 top-5 | sample (n=3) + rerank | 0.614 |
| 17 | + Amharic fallback fix | same | same | sample (n=3) + rerank, Amharic patched | 0.616 |
| 18 | **Hybrid: deterministic Amharic + sampled rest** | Llama-3.1-8B-Instruct LoRA | BGE-M3 top-5 | **greedy (Amh_Eth), sample+rerank (rest)** | **0.620** |

## Key findings, in order of impact

### 1. Prompt format mismatch silently destroys the LLM-Judge score

The BGE-M3 v1 fine-tune included `"Answer ONLY in {language}."` in the
training prompt — a format never used in the base FAISS run. This produced
fluent-looking but judge-incoherent outputs: ROUGE stayed reasonable (0.51)
but LLM-Judge collapsed to exactly 0. ROUGE and LLM-Judge can diverge sharply
when train/inference prompts don't match — always check both before
concluding a strategy succeeded or failed.

### 2. Epoch selection must follow leaderboard score, not validation loss

Validation loss decreased monotonically through epoch 10 in every run
(Qwen, Llama+FAISS, Llama+BGE-M3), but leaderboard score peaked consistently
around epoch 7–8 and degraded afterward:

```
Llama+BGE-M3 v2:
  epoch 6:  val_loss=0.0748  LB=0.608
  epoch 7:  val_loss=0.0587  LB=0.612  <- best
  epoch 8:  val_loss=0.0491  LB=0.608
  epoch 9:  val_loss=0.0459  LB=(not submitted, projected lower)
  epoch 10: val_loss=0.0415  LB=0.605
```

This is overfitting to the small (320-example) validation monitor set, not
to the full Val.csv — the gap between "validation loss still improving" and
"leaderboard still improving" appeared consistently.

### 3. Decoding strategy is language-dependent, not universal

Switching from greedy to multi-sample (n=3, temperature=0.4) generation with
CrossEncoder-based answer selection improved scores on English, Akan,
Luganda, and Swahili subsets. On Amharic, the same strategy corrupted
~80% of outputs (replacement characters, prompt leakage) — traced to poor
Ge'ez-script tokenization in the Llama-3.1 vocabulary, which sampling
amplifies. The fix was a per-language router: greedy decoding for Amharic,
sampling + reranking for everything else.

### 4. Naive retrieval is not a shortcut

Returning the most similar Train/Val answer directly (cosine similarity
threshold approach, no generation) scored 0.513 — worse than pure LoRA
generation (0.559+) at every threshold tested. Useful retrieval requires
feeding context to a generator, not substituting for generation. (Other
solutions in this competition found stronger results by treating roughly
half the test set as a pure retrieval problem — see note below.)

### 5. Model ensembling did not help

Combining Qwen2.5-7B and Llama-3.1-8B predictions, whether by output length
heuristic or per-language routing, always scored below Llama-3.1-8B alone.
The two models did not appear to be sufficiently decorrelated in their
error patterns for ensembling to add value.

## Post-competition note

A higher-scoring public solution (LB ~0.70) took a fundamentally different
approach: it identified that 5 of the 8 language subsets reuse a closed bank
of canonical FAQ answers (high paraphrase duplication from deployed
chatbots), making them primarily a retrieval/paraphrase-matching problem
rather than a generation problem. Routing those subsets to pure embedding
retrieval (with a fine-tuned matcher, not a generic one) while reserving
fine-tuned generation for the remaining 3 genuinely generative subsets
(Akan, English-Ghana, Amharic) closed most of the remaining gap. This
single-pipeline solution did not explore that per-subset split and instead
applied one RAG+generation pipeline uniformly across all 8 languages —
the main structural difference from the top-scoring approaches.
