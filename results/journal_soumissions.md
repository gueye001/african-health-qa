# Journal des soumissions

Historique chronologique de chaque soumission effectuée pendant la
compétition, y compris les pistes abandonnées — conservé pour que de
futures tentatives ne reproduisent pas les mêmes impasses.

## Tableau récapitulatif

| # | Soumission | Modèle | Récupération | Décodage | Score LB |
|---|---|---|---|---|---|
| 1 | mt0-xxl + TF-IDF | mT0-XXL | TF-IDF (n-grammes de caractères) | beam search | 0,497 |
| 2 | Qwen2.5-7B FAISS | Qwen2.5-7B-Instruct + LoRA | MiniLM + FAISS + CrossEncoder | greedy | 0,559 |
| 3 | Qwen2.5-7B epoch10 | Qwen2.5-7B-Instruct + LoRA (ep10) | identique | greedy | 0,560 |
| 4 | Ensemble Qwen+Llama (longueur) | les deux | identique | greedy | 0,580 |
| 5 | Llama FAISS epoch5 | Llama-3.1-8B-Instruct + LoRA | MiniLM + FAISS + CrossEncoder | greedy | 0,596 |
| 6 | Ensemble Qwen+Llama (par langue) | les deux | identique | greedy | 0,570 |
| 7 | Récupération directe (sim≥0,90) | aucun (récupération pure) | MiniLM + FAISS | n/a | 0,513 |
| 8 | Llama FAISS epoch8 | Llama-3.1-8B-Instruct + LoRA | MiniLM + FAISS + CrossEncoder | greedy | 0,606 |
| 9 | Llama FAISS epoch10 | Llama-3.1-8B-Instruct + LoRA (ep10) | identique | greedy | 0,591 |
| 10 | BGE-M3 v1 (prompt buggé) | Llama-3.1-8B-Instruct + LoRA | BGE-M3 top-3 | greedy | 0,408 (Jugement LLM = 0) |
| 11 | BGE-M3 v2 epoch6 | Llama-3.1-8B-Instruct + LoRA | BGE-M3 top-3 | greedy | 0,608 |
| 12 | BGE-M3 v2 epoch7 | Llama-3.1-8B-Instruct + LoRA | BGE-M3 top-3 | greedy | 0,612 |
| 13 | BGE-M3 v2 epoch8 | Llama-3.1-8B-Instruct + LoRA | BGE-M3 top-3 | greedy | 0,608 |
| 14 | BGE-M3 v2 epoch10 | Llama-3.1-8B-Instruct + LoRA (ep10) | BGE-M3 top-3 | greedy | 0,605 |
| 15 | BGE-M3 v2 epoch7, top-5 contextes | Llama-3.1-8B-Instruct + LoRA | BGE-M3 top-5 | greedy | 0,613 |
| 16 | Échantillonnage multiple + sélection CrossEncoder | Llama-3.1-8B-Instruct + LoRA | BGE-M3 top-5 | échantillonnage (n=3) + reranking | 0,614 |
| 17 | + correctif amharique | identique | identique | échantillonnage (n=3) + reranking, amharique corrigé | 0,616 |
| 18 | **Hybride : amharique déterministe + reste échantillonné** | Llama-3.1-8B-Instruct + LoRA | BGE-M3 top-5 | **greedy (Amh_Eth), échantillonnage+reranking (reste)** | **0,620** |

## Constats principaux, par ordre d'impact

### 1. Un décalage de format de prompt détruit silencieusement le score Jugement LLM

Le fine-tuning BGE-M3 v1 incluait `"Answer ONLY in {language}."` dans le
prompt d'entraînement — un format jamais utilisé lors de l'entraînement de
référence FAISS. Cela a produit des sorties d'apparence fluide mais
incohérentes pour le juge automatique : le ROUGE restait raisonnable
(0,51) mais le score Jugement LLM s'est effondré à exactement 0. Le ROUGE
et le Jugement LLM peuvent diverger fortement lorsque les prompts
d'entraînement et d'inférence ne correspondent pas exactement — toujours
vérifier les deux avant de conclure qu'une stratégie a réussi ou échoué.

### 2. La sélection de l'epoch doit suivre le score du leaderboard, pas la perte de validation

La perte de validation a diminué de façon monotone jusqu'à l'epoch 10 dans
chaque run (Qwen, Llama+FAISS, Llama+BGE-M3), mais le score du leaderboard
a systématiquement culminé vers l'epoch 7-8 avant de se dégrader :

```
Llama+BGE-M3 v2 :
  epoch 6  : val_loss = 0,0748  LB = 0,608
  epoch 7  : val_loss = 0,0587  LB = 0,612  ← meilleur
  epoch 8  : val_loss = 0,0491  LB = 0,608
  epoch 9  : val_loss = 0,0459  LB = (non soumis, projeté plus bas)
  epoch 10 : val_loss = 0,0415  LB = 0,605
```

Il s'agit d'un surapprentissage sur le petit échantillon de suivi utilisé
pendant l'entraînement (320 exemples), non représentatif de l'ensemble
Val.csv complet — l'écart entre "la perte de validation continue de
s'améliorer" et "le score du leaderboard continue de s'améliorer" est
apparu de façon constante.

### 3. La stratégie de décodage dépend de la langue, elle n'est pas universelle

Passer d'un décodage greedy à un échantillonnage multiple (n=3,
température=0,4) avec sélection des réponses par CrossEncoder a amélioré
les scores sur les sous-ensembles anglais, akan, luganda et swahili. Sur
l'amharique, la même stratégie a corrompu environ 80 % des sorties
(caractères de remplacement, fuite du prompt) — un problème attribué à une
mauvaise tokenisation du script ge'ez dans le vocabulaire de Llama-3.1, que
l'échantillonnage amplifie. La correction a consisté à router par langue :
décodage greedy pour l'amharique, échantillonnage + reranking pour tout le
reste.

### 4. La récupération naïve n'est pas un raccourci valable

Retourner directement la réponse la plus similaire de Train/Val (seuil de
similarité cosinus, sans génération) a obtenu un score de 0,513 — inférieur
à la génération pure avec LoRA (0,559 et plus) à tous les seuils testés.
Une récupération utile nécessite de fournir le contexte à un générateur, et
non de se substituer entièrement à la génération.

### 5. L'ensemble de modèles n'a pas aidé

Combiner les prédictions de Qwen2.5-7B et Llama-3.1-8B, par heuristique de
longueur ou par routage par langue, a toujours obtenu un score inférieur à
Llama-3.1-8B seul. Les deux modèles ne semblaient pas suffisamment
décorrélés dans leurs patterns d'erreur pour que l'ensemble apporte une
valeur ajoutée.

## Remarque post-compétition

Une solution publique ayant obtenu un score plus élevé (LB ~0,70) a adopté
une approche fondamentalement différente : elle a identifié que 5 des 8
sous-ensembles de langues réutilisent une banque fermée de réponses
canoniques (forte duplication par paraphrase provenant de chatbots
déployés), ce qui en fait principalement un problème de correspondance par
paraphrase plutôt qu'un problème de génération. Router ces sous-ensembles
vers une récupération d'embeddings pure (avec un matcher spécifiquement
fine-tuné, pas un modèle générique) tout en réservant la génération
fine-tunée aux 3 sous-ensembles réellement génératifs (akan,
anglais-Ghana, amharique) a permis de combler une grande partie de l'écart
restant. Cette solution à pipeline unique n'a pas exploré cette séparation
par sous-ensemble et a appliqué un seul pipeline RAG + génération de
manière uniforme sur les 8 langues — c'est la principale différence
structurelle avec les approches les mieux classées.
