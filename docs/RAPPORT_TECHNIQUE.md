# Rapport technique — African Health QA

Ce document détaille les choix d'architecture, les corrections critiques et
les pistes explorées (réussies ou abandonnées) au cours du développement de
ce projet.

## 1. Contexte du challenge

### Objectif

Étant donné une question de santé maternelle, sexuelle ou reproductive,
posée dans une langue et un pays donnés, produire une réponse de qualité
dans cette même langue.

### Données

| Fichier | Lignes | Colonnes |
|---|---|---|
| `Train.csv` | 29 815 | `ID`, `input`, `output`, `subset` |
| `Val.csv` | 6 686 | `ID`, `input`, `output`, `subset` |
| `Test.csv` | 2 618 | `ID`, `input`, `subset` |

`input` = la question de santé, `output` = la réponse de référence (même
langue que la question), `subset` = `<Langue>_<Pays>`, l'une des huit
combinaisons suivantes :

```
Aka_Gha   akan, Ghana
Amh_Eth   amharique, Éthiopie
Eng_Eth   anglais, Éthiopie
Eng_Gha   anglais, Ghana
Eng_Ken   anglais, Kenya
Eng_Uga   anglais, Ouganda
Lug_Uga   luganda, Ouganda
Swa_Ken   swahili, Kenya
```

### Métrique d'évaluation

```
Score = 0,37 × ROUGE-1 F1  +  0,37 × ROUGE-L F1  +  0,26 × Jugement LLM
```

Le ROUGE est calculé avec une tokenisation par espace, sans stemming. Le
"Jugement LLM" (LLM-Judge) évalue la qualité sémantique de la réponse via un
modèle juge — c'est cette composante qui s'est avérée la plus sensible aux
erreurs de format de prompt (voir section 3.1).

## 2. Architecture de la solution finale

```
Question
   │
   ▼
Embedding BGE-M3 (1024 dimensions, multilingue)
   │
   ▼
Recherche FAISS — index FAISS séparé par langue/pays
   │
   ▼
Reranking CrossEncoder (cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
   │
   ▼
Sélection des 3 meilleurs contextes (paires question-réponse similaires)
   │
   ▼
Construction du prompt : système + langue + contextes + question
   │
   ▼
Llama-3.1-8B-Instruct + adaptateur LoRA (fine-tuné sur Train+Val)
   │
   ▼
Décodage adaptatif :
   - amharique       → greedy déterministe
   - autres langues  → échantillonnage (n=3, T=0,4) + sélection CrossEncoder
   │
   ▼
Réponse finale
```

### Pourquoi BGE-M3 plutôt qu'un embedder classique

Les premiers essais utilisaient `all-MiniLM-L6-v2` (384 dimensions,
principalement entraîné sur l'anglais). Le remplacement par `BAAI/bge-m3`
(1024 dimensions, entraîné sur plus de 100 langues dont plusieurs langues
africaines) a apporté un gain mesurable, en particulier sur les langues à
faible ressource comme l'amharique et le luganda où le retriever précédent
peinait à établir une correspondance sémantique fiable.

### Pourquoi top-3 contextes plutôt qu'un seul

Donner au modèle plusieurs réponses similaires plutôt qu'une seule a permis
une meilleure couverture des reformulations possibles d'une même question
de santé. Le passage à top-5 a apporté un gain supplémentaire mais marginal,
suggérant un plateau de rendement décroissant au-delà de 3 contextes.

## 3. Corrections critiques

### 3.1 — Le format de prompt doit être identique entre l'entraînement et l'inférence

**Symptôme observé.** Une version du fine-tuning incluait l'instruction
`"Answer ONLY in {language}."` dans le prompt d'entraînement — une
consigne absente du prompt utilisé pour l'entraînement de référence. Le
modèle ainsi obtenu produisait des réponses qui semblaient correctes au
premier regard (ROUGE-L F1 ≈ 0,51, score raisonnable), mais le score
LLM-Judge est tombé à exactement 0.

**Diagnostic.** Le modèle n'avait jamais vu cette instruction pendant son
entraînement initial. Lui injecter cette consigne au moment de la
génération a produit des réponses syntaxiquement correctes mais
sémantiquement incohérentes pour un juge automatique — un cas classique de
décalage train/inférence qui n'est pas détectable par le ROUGE seul.

**Leçon.** Toujours vérifier le ROUGE *et* le LLM-Judge avant de conclure
qu'une stratégie a réussi ou échoué ; les deux métriques peuvent diverger
fortement.

### 3.2 — `padding_side = "left"` est obligatoire pour la génération batchée

Avec un modèle décodeur (Llama, Qwen), un padding à droite (comportement
par défaut de certains tokenizers) produit des sorties corrompues en
génération batchée, car le modèle continue à "voir" les tokens de padding
avant le texte réel lors du calcul de l'attention causale. Ce paramètre a
été intégré systématiquement dans tous les pipelines d'inférence après sa
découverte.

### 3.3 — Sélectionner l'epoch sur le score du leaderboard, pas sur la perte de validation

La perte de validation (val_loss) a continué à décroître de façon monotone
jusqu'à l'epoch 10 sur chaque variante de modèle testée. Pourtant, le score
sur le leaderboard atteignait systématiquement son maximum vers l'epoch 7-8
puis se dégradait :

```
Modèle Llama + BGE-M3 v2 :
  epoch 6  : val_loss = 0,0748  →  score LB = 0,608
  epoch 7  : val_loss = 0,0587  →  score LB = 0,612   ← meilleur
  epoch 8  : val_loss = 0,0491  →  score LB = 0,608
  epoch 10 : val_loss = 0,0415  →  score LB = 0,605
```

Ce phénomène s'explique par un surapprentissage sur le petit échantillon de
suivi utilisé pendant l'entraînement (320 exemples), qui ne reflète pas
fidèlement la performance sur le jeu de test caché. Une stratégie d'arrêt
anticipé basée uniquement sur la val_loss aurait manqué ce signal.

### 3.4 — Le décodage par échantillonnage n'est pas universellement bénéfique

La génération multiple (3 séquences par question, température 0,4) suivie
d'une sélection par CrossEncoder a amélioré le score sur l'anglais, l'akan,
le luganda et le swahili. Sur l'amharique, la même stratégie a corrompu
environ 80 % des sorties générées (caractères de remplacement Unicode,
fuite du prompt dans la réponse).

**Cause probable.** Le tokenizer de Llama-3.1 représente mal le script
ge'ez (alphabet amharique) ; l'échantillonnage amplifie cette instabilité.

**Solution retenue.** Un routage par langue dans le pipeline d'inférence :
décodage greedy déterministe pour l'amharique, échantillonnage + reranking
pour les sept autres langues. Cette approche hybride a apporté le gain
final le plus net de la compétition (+0,004 par rapport à la meilleure
version sans routage).

## 4. Historique détaillé des résultats

Voir [`../results/journal_soumissions.md`](../results/journal_soumissions.md)
pour le tableau complet des 18 soumissions effectuées, avec scores ROUGE-1,
ROUGE-L et LLM-Judge détaillés pour chacune.

## 5. Pistes explorées sans succès

### Ensemble de modèles (Qwen2.5-7B + Llama-3.1-8B)

Plusieurs stratégies de combinaison ont été testées (sélection par longueur
de réponse, routage par langue) — toutes ont obtenu un score inférieur à
Llama-3.1-8B utilisé seul. Les deux modèles ne semblaient pas suffisamment
décorrélés dans leurs patterns d'erreur pour qu'un ensemble apporte un gain.

### Récupération directe sans génération

Retourner directement la réponse la plus similaire trouvée dans Train/Val
(au-dessus d'un seuil de similarité cosinus), sans passer par la
génération, a obtenu un score de 0,513 — nettement inférieur à la
génération pure avec LoRA (0,559 et plus). Une récupération utile nécessite
de fournir le contexte à un générateur, pas de se substituer entièrement à
la génération.

### Llama-3.3-70B en quantification 4 bits (QLoRA)

Abandonné après évaluation de la vitesse d'entraînement : environ
0,03 itération/seconde, soit une projection de ~57 heures pour seulement
5 epochs sur le matériel disponible — non viable dans le temps imparti par
la compétition.

### Poursuivre l'entraînement au-delà de l'epoch optimale

Comme détaillé en section 3.3, les epochs 9 et 10 ont systématiquement
dégradé le score du leaderboard sur chaque variante de modèle testée,
malgré une perte de validation qui continuait de s'améliorer.

## 6. Déploiement

### Fusion et quantification

Le modèle de base et l'adaptateur LoRA ont été fusionnés (`merge_and_unload`
de la bibliothèque PEFT) pour produire un modèle autonome, puis convertis
au format GGUF et quantisés en Q4_K_M via `llama.cpp` — réduisant la taille
du modèle d'environ 16 Go (bfloat16) à 4,9 Go, permettant une inférence sur
CPU sans GPU.

### Application de démonstration

Une interface Gradio interroge le pipeline complet (retrieval BGE-M3,
reranking CrossEncoder, génération via `llama-cpp-python`) et affiche à la
fois la réponse générée et les contextes récupérés, pour une transparence
totale sur le fonctionnement du système RAG. Déployée gratuitement sur
HuggingFace Spaces (CPU basique).

## 7. Pistes d'amélioration identifiées a posteriori

Une analyse de solutions tierces ayant obtenu un score sensiblement plus
élevé (~0,70) a révélé une différence structurelle majeure : cinq des huit
langues du jeu de données réutilisent en réalité une banque fermée de
réponses canoniques issues de chatbots déployés (forte duplication par
paraphrase), ce qui en fait fondamentalement un problème de correspondance
par similarité plutôt qu'un problème de génération.

Router ces cinq sous-ensembles vers une récupération pure (avec un
retriever spécifiquement fine-tuné, pas un modèle générique) tout en
réservant la génération fine-tunée aux trois sous-ensembles réellement
génératifs (akan, anglais-Ghana, amharique) aurait probablement permis de
combler une partie significative de l'écart restant. Cette piste — un
routage par sous-ensemble en amont du pipeline, plutôt qu'un pipeline RAG
unique appliqué uniformément aux huit langues — constitue la principale
direction d'amélioration pour une itération future de ce projet.
