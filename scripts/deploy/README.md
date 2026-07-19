# Deploy scripts

Deux variantes du Space Hugging Face (`kgueye001/african-health-qa`) :

- **`app.py` + `requirements.txt`** (actuellement deploye) — Llama-3.1-8B + LoRA
  fusionne, quantifie en GGUF Q4_K_M, inference CPU via `llama-cpp-python`.
  Tourne sur le hardware gratuit `cpu-basic`.

- **`app_zerogpu.py` + `requirements_zerogpu.txt`** (pret, non deploye) —
  meme modele en bf16 (non quantifie), charge via `transformers` + `peft`
  (LoRA applique a la volee sur le modele de base), inference sur GPU via
  ZeroGPU (`zero-a10g`, inclus dans l'abonnement Pro).

  Ce chemin a ete teste et debogue (voir historique de commits) mais bloque
  par une erreur infrastructure cote Hugging Face au moment du test
  (`RuntimeError: No CUDA GPUs are available` dans l'init bas-niveau du
  worker ZeroGPU, reproductible meme apres un factory reboot du Space —
  le compte Pro est pourtant bien actif). A retenter : pousser
  `app_zerogpu.py` -> `app.py` et `requirements_zerogpu.txt` ->
  `requirements.txt` sur le Space, puis `request_space_hardware(hardware="zero-a10g")`.

## Notes techniques (variante ZeroGPU)

Deux pieges rencontres et corriges dans `app_zerogpu.py` :

1. **`BGEM3FlagModel`/`CrossEncoder` ignorent `device='cpu'` a l'usage** :
   le device cible est fixe a la construction en re-detectant
   `torch.cuda.is_available()`, qui renvoie `True` sous ZeroGPU meme hors
   contexte `@spaces.GPU`. Solution : masquer temporairement
   `torch.cuda.is_available` pendant la construction de ces modeles
   (`_force_cpu_only()`).

2. **Chargement paresseux obligatoire pour le modele GPU** : construire le
   `PeftModel` (chargement des poids LoRA) en dehors d'une fonction decoree
   `@spaces.GPU` echoue ("No CUDA GPUs are available"), car aucun GPU reel
   n'est attache au process en dehors de ce contexte. Le modele de base +
   LoRA doivent etre charges a l'interieur de la fonction decoree (mis en
   cache apres le premier appel).
