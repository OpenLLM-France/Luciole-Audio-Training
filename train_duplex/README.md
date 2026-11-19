# Test duplex : encodeur Nemotron 3.5 ASR streaming + Luciole

Tests de faisabilité (CPU, 2026-09-30) avant un entraînement de type NemotronVoiceChat :
d'abord un `DuplexSTTModel` (voix → texte en streaming full-duplex), puis plus tard un EAR-TTS branché
sur le canal texte de l'agent.

- Encodeur : [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
  (FastConformer cache-aware RNNT, 24 couches, d_model 1024, 40 langues dont le français)
- LLM : `Luciole-1B-Instruct-1.1`
- Code : fork `~/NeMo` (clone local), branche **`luciole_duplexstt_3.1`** (partie de `luciole_speech.3.1.0-rc0`).
  Elle ajoute `trainable_head_rows` à `DuplexSTTModel` (`parts/special_rows_head.py`) et le type
  d'entrée `conversation_as_duplex` (`common/data/lhotse/cutset.py`).

## Entraînement sur JZ (une phase, 10k pas)

| Fichier | Rôle |
|---|---|
| `conf/duplex_luciole.yaml` | La config : encodeur et LLM gelés ; on entraîne la projection, LoRA r=128 sur toutes les projections et les lignes pad/bos/eos de la tête |
| `conf/data/input_cfg_duplex_fr_en.yaml` | Mix de test : 13 manifests, fr + en, QA 40 % / ASR 40 % / AST 20 % ; l'ASR pointe vers les copies alignées de `duplex/nemo` (MLS en recasepunc) |
| `conf/data/val_cfg_duplex.yaml` | Jeux de validation (asr_fr, asr_en, qa_fr, qa_slue_unseen), passés comme l’input_cfg (`--val-cfg`) |
| `conf/data/input_cfg_duplex_fr.yaml` | Mix français complet (46 manifests), **généré** par `make_input_cfg.py` à partir du v4 |
| `run_duplex.slurm` | Lance `duplex_train.py` avec le NeMo du fork (vérifie la branche) sur 1 nœud de 4×H100 |
| `duplex_train.py` | `examples/speechlm2/stt_duplex_train.py` du fork + validation au step 0 et `validation_schedule` repris de `../training/salm_train.py` |
| `tests/smoke_config_cpu.py` | Répétition à blanc sur CPU de la vraie config (voir plus bas) |

Les autres sous-dossiers :

| Dossier | Contenu |
|---|---|
| `alignment/` | Horodatage par mot des manifests ASR, pour la tête ASR (`predict_user_text`) : voir [Alignement ASR](#alignement-asr) |
| `tests/` | Tests CPU de l'encodeur, du modèle, du parseur multi-tours, et la répétition à blanc de la config |
| `inference/` | `stream_duplex.py` : inférence en streaming d'un checkpoint HF exporté |

Prérequis sur JZ : la branche `luciole_duplexstt_3.1` du fork, copiée dans `$HOME/NeMo_Duplex` par `git archive`
(sans `.git` ; la branche et le commit sont dans `GIT_REVISION`). `$HOME/NeMo`, installé en éditable dans l'env, reste
intact pour les runs SALM : le slurm met `NeMo_Duplex` en tête du `PYTHONPATH` et vérifie d'où `nemo` est importé.
Pour mettre à jour la copie depuis le poste :

```bash
cd ~/NeMo && ssh jz 'rm -rf ~/NeMo_Duplex && mkdir ~/NeMo_Duplex' && git archive luciole_duplexstt_3.1 | ssh jz 'tar -x -C ~/NeMo_Duplex' \
  && ssh jz "echo luciole_duplexstt_3.1 $(git rev-parse --short luciole_duplexstt_3.1) > ~/NeMo_Duplex/GIT_REVISION"
```

Il faut aussi l'encodeur dans
`$SCRATCH/speechlm/models/nemotron-3.5-asr-streaming-0.6b.nemo` (copié le 2026-09-30).

Les index `.idx` du duplex sont à part de ceux de SALM, sous `$ALL_CCFRSCRATCH/audio/duplex/idx`
(`indexes_root` de `conf/duplex_luciole.yaml`). Il faut les construire une fois par mix avant d'entraîner :
sinon le job les crée à la volée, depuis chaque rang, au démarrage. `build_indexes.py` ignore le type
`conversation_as_duplex`, donc `build_indexes_duplex.slurm` lui passe des copies où il devient
`multimodal_conversation`. Un `.idx` ne contient que les positions des lignes du jsonl, donc le résultat est le même.

```bash
cd train_duplex && mkdir -p logs
sbatch build_indexes_duplex.slurm   # mix de test + mix fr (780 + 1875 fichiers, les doublons sont sautés)
```

```bash
cd train_duplex
# Test court d'abord (dev QoS, 1 GPU, 50 pas)
python duplex_launcher.py --job-name smoke --qos dev --gpus 1 --time 00:30:00 \
    --set ++trainer.max_steps=50 --set ++trainer.val_check_interval=25
python duplex_launcher.py --job-name duplex_test_v1                       # 10k pas, mix de test
python duplex_launcher.py --job-name duplex_fr_v1 --input-cfg conf/data/input_cfg_duplex_fr.yaml
python duplex_launcher.py --job-name duplex_test_v1 --resume              # reprend au dernier checkpoint
```

Chaque xp a son dossier `$ALL_CCFRSCRATCH/audio/training/duplex/<job-name>/`, créé par `duplex_launcher.py`
avant la soumission (Slurm ouvre `--output` avant le début du job). On y trouve le `.slurm` matérialisé
(`sbatch` dessus relance la même xp), les `.out`/`.err`, les copies des yaml et la sortie d'`exp_manager`
(checkpoints, TensorBoard). `--dry-run` crée le dossier et le `.slurm` sans soumettre.

**Données.** Il n'existe pas encore de dialogues bi-canal en français. `conversation_as_duplex` réutilise
donc les manifests SALM à un seul tour audio : l'instruction texte devient le prompt système,
l'audio le tour de l'utilisateur (complété par du silence, en padding lhotse paresseux, sans rien charger
en RAM), et la réponse le tour de l'agent. L'agent commence 0,32 à 0,56 s après la fin de l'audio,
avec un token par trame, puis EOS.

**Multi-tours.** Une conversation devient une suite d'échanges (un ou plusieurs audios utilisateur,
puis une réponse) placés sur une même timeline. Voici les règles :
- l'utilisateur reprend 0,32 à 0,56 s après l'EOS de l'agent ;
- le texte utilisateur situé avant la première réponse devient le prompt système ;
- des audios consécutifs (question + contexte audio) forment un seul tour utilisateur ;
- la conversation est coupée au premier tour impossible à placer (texte utilisateur après la première
  réponse, réponse de plus de `max_agent_tokens` tokens).

Vérifié sur 2 000 conversations par fichier (`tests/test_multiturn_cpu.py`, audio réel de `/data-server`) :

| dataset | conservés | exemples à plusieurs échanges | durée médiane / max |
|---|---|---|---|
| UltraChat-300K-SLAM-Omni (en) | 100 % (52 % avant) | 48 %, jusqu'à 7 | 22 s / 137 s |
| HeySQuAD_human (en) | 100 % (~3 % avant) | 90 %, jusqu'à 19 | 32 s / 142 s |
| slue-phase-2-sqa5 (en, `AAR`) | 100 % (0 % avant) | 1 échange, question + contexte | 47 s / 63 s |

`max_duration` passe donc à 1 200 s (15k trames), comme les runs SALM 1B (`max_tokens: 16384`).

Ces données apprennent à se taire, puis à répondre au bon moment. Elles n'apprennent **pas** les
interruptions ni les backchannels : il faudra de vrais dialogues bi-canal (ou synthétiques, via la TTS),
à ajouter comme groupes `lhotse_shar` dans le mix.

**Répétition à blanc sur CPU** (`tests/smoke_config_cpu.py`, avec la vraie config, le fork sur la branche et un mini-manifest local) :
- le modèle entraîne 105,9M paramètres (LoRA, deltas de la tête, projection) ;
- le parseur passe par le vrai dataloader lhotse, avec workers en spawn et bucketing ;
- les prompts système sont bien placés, et les timelines ont la forme `....B xxxx ...E....` ;
- la loss est finie et les gradients arrivent aux deltas et à LoRA ;
- un checkpoint se recharge par `init_from_checkpoint`, deltas et LoRA compris.

Bugs trouvés et corrigés par cette répétition :
- `task_type: CAUSAL_LM` plante, parce que le duplex ne garde que `llm.model`, sans tête LM ;
  il faut `FEATURE_EXTRACTION` ;
- `CutSet.map` ignorait les conversations : il faut `apply_fn=None`.

**Pas encore vérifié sur GPU :**
- le batch (`batch_duration: 1200`, à calibrer avec l'OOMptimizer) ;
- la présence des sidecars `.idx` pour les manifests `conversation_as_duplex` (mêmes fichiers que le v4) ;
- les performances du dataloader : la tokenisation de la réponse se fait dans les workers.

## Alignement ASR

La tête ASR (`predict_user_text`) écrit la transcription de l'utilisateur pendant qu'il parle : il lui faut
l'instant de chaque mot. `alignment/align_asr_manifests.py` aligne la transcription de référence sur l'audio
avec la branche CTC d'un FastConformer hybride monolingue (115M, trames de 80 ms comme le duplex) :
`linto_stt_fr_fastconformer_pc` (fr) et `stt_en_fastconformer_hybrid_large_pc` (en), dans `$SCRATCH/speechlm/models/`.
Chaque ligne gardée est recopiée telle quelle, plus `custom.user_words` (`[mot, début, fin]` par tour audio,
ponctuation et casse intactes) et `custom.align_wer`.

- L'aligneur ne voit pas la ponctuation : un « . » final collé au dernier mot l'étirait jusqu'à la fin de l'audio.
- Les dates d'un mot viennent des pics CTC de son premier et de son dernier token (Viterbi étire les tokens sur le silence).
- Sans token chiffre dans les aligneurs, un mot à chiffres (« 2017, ») prend l'intervalle entre ses voisins.
- Sont écartées les lignes dont la transcription CTC est à plus de 30 % de WER de la référence (référence fausse ou partielle).
- MLS fr est pris dans `Multilingual_LibriSpeech_recasepunc` : toutes les instructions de la version minuscule
  demandent de transcrire sans majuscules ni ponctuation, alors que la tête ASR, sans prompt, doit apprendre un seul style.

Test sur 200 lignes (CV fr, MLS fr, CV en, job 697044) : 97–98 % gardées, aucune date non monotone ou négative,
200 lignes en 2 à 4 s.

```bash
cd train_duplex/alignment
mkdir -p $SCRATCH/speechlm/duplex_align_test && sbatch align_asr_test.slurm      # 200 lignes par jeu, dans $SCRATCH
bash align_asr.slurm --count                                                     # 453 fichiers -> 38 tâches
mkdir -p $SCRATCH/speechlm/duplex_align/logs && sbatch --array=0-37 align_asr.slurm
```

`align_asr.slurm` écrit sous `$ALL_CCFRSCRATCH/audio/duplex/nemo/`, avec l'arborescence de `audio/nemo/`
(CV fr, VoxPopuli fr, MLS fr recasepunc, CV en, VoxPopuli en, plus les `dev.jsonl` de CV). Une sortie existante est
sautée : relancer la même commande ne refait que les fichiers manquants.
`align_debug.slurm` affiche les tokens alignés et l'argmax CTC par trame de quelques lignes.

**Version `clean/` (2026-11-19).** Le mix `conf/data/input_cfg_duplex_fr_en_clean.yaml` (et `input_cfg_duplex_asr_only_clean.yaml`)
prend l'ASR et l'AST dans les manifests nettoyés de base (`audio/nemo/<tâche>/<langue>/clean/`, les mêmes que le
v5_transcribe), réalignés sous `audio/duplex/nemo/.../clean/`. Ils sont sans tags d'événements et sans hésitations,
y compris pour le spontané : la tête ASR n'apprend plus le verbatim. CFPP2000 revient dans le mix, et `strip_tags_inplace`
et `fix_first_word_inplace` ne sont plus nécessaires (`fix_early_first_word` est appelé dans `align_asr_manifests.py`).
La validation garde les dev `context/` déjà alignés (`asr/fr/clean/CommonVoice` n'a pas de `dev.jsonl`).

```bash
cd train_duplex/alignment
sbatch align_asr_test.slurm 200 clean; sbatch align_asr_test.slurm 200 clean_spont   # test, dans $SCRATCH
sbatch --export=ALL,SET=clean --array=0-37 align_asr.slurm         # lu : CV fr/en, VoxPopuli fr/en, MLS fr recasepunc (449 fichiers)
sbatch --export=ALL,SET=clean_spont --array=0-9 align_asr.slurm    # spontané, MAX_WER 0.8 : ESLO, CFPP2000, LVL-Atril (116 fichiers)
sbatch --dependency=afterok:<job SET=clean> --export=ALL,VARIANT=clean join_ast_words.slurm   # AST CV fr-en / en-fr
cd .. && sbatch build_indexes_duplex.slurm conf/data/input_cfg_duplex_fr_en_clean.yaml conf/data/input_cfg_duplex_asr_only_clean.yaml
```

## Tests CPU

```bash
# venv CPU jetable (le .venv du fork NeMo ne contient que torch)
uv venv -p 3.12 venv_duplex
uv pip install -p venv_duplex/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install -p venv_duplex/bin/python -e "$HOME/NeMo[speechlm2]"

ASR=~/.cache/huggingface/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b/snapshots/*/nemotron-3.5-asr-streaming-0.6b.nemo
LLM=/data-server/public/models/OpenLLM-France/Luciole-1B-Instruct-1.1

python tests/test_encoder.py --asr $ASR --wav <un wav français 16 kHz>
PYTHONUNBUFFERED=1 python tests/test_duplex_model.py --asr $ASR --llm $LLM \
    --wav1 <split_dir>/ACSYNT_001_03*.wav \
    --wav2 <split_dir>/ACSYNT_003_03*.wav --steps 100 --workdir /tmp/duplex_run
```

## `tests/test_encoder.py` : l'encodeur

| lookahead (`att_context_size`) | chunk | causalité | streaming = offline |
|---|---|---|---|
| [56, 0] | 80 ms | OK | écart relatif de 3,5e-2 (premières trames) |
| **[56, 3]** (défaut) | 320 ms | OK | **6e-7** |
| [56, 6] | 560 ms | OK | 6e-6 |
| [56, 13] | 1 120 ms | OK | 1e-6 |

- Se charge via `load_pretrained_nemo` de speechlm2 (`EncDecRNNTBPEModelWithPrompt`).
- Transcription `fr-FR` exacte sur un extrait LibriVox.
- 1 trame toutes les 80 ms (79,5 ms mesurés), dim 1024 : compatible avec `frame_length: 0.08` du duplex.
- Le prompt de langue est concaténé **après** l'encodeur. Speechlm2 ne garde que préprocesseur et encodeur,
  donc le LLM reçoit des représentations indépendantes de la langue.
- `asr.transcribe(paths, target_lang=...)` plante dans le fork (langue du cut lhotse à `None`).
  Le script appelle `forward(prompt_indices=...)` à la place.

## `tests/test_duplex_model.py` : le modèle complet

Il fabrique un dialogue synthétique (user / agent / user / agent, 17,6 s, 220 trames), passe par
`DuplexSTTDataset`, lance `training_step`, surapprend ce dialogue, puis appelle
`DuplexSTTStreamingInference.offline_inference`, qui décode trame par trame avec cache KV.

Résultat au bout de 100 pas (entre 4 et 5 s par pas sur CPU) :

```
gen  .......................................................................xxxxxxxxxxxx...........................
     ....................................................BBxxxxxxxxxx..............................................
ref  .......................................................................Bxxxxxxxxxxxxx.........................
     ...........E..........................................Bxxxxxxxxxx.......................................E.....
```

Le modèle se tait pendant que l'utilisateur parle et répond dans les créneaux de l'agent au bon moment.
Il n'a pas convergé (répétitions, pas encore d'EOS), ce qui est attendu d'un test de fumée.

## Pièges trouvés, à reporter dans une vraie config

1. **`perception.modality_adapter.d_model: 1024`.** `duplex_stt.yaml` met 512 (FastConformer-L).
2. **Fixer `perception.encoder.att_context_size`, par ex. `[56, 3]`.** Le checkpoint est multi-lookahead et
   `ConformerEncoder` tire un lookahead au hasard à chaque pas d'entraînement si on le laisse en liste.
3. **Le chemin du LLM ne doit pas contenir « Nemotron ».** `DuplexSTTModel` active alors un hack de cache prévu
   pour Nemotron-H (`cache_params`). Luciole est un `NemotronForCausalLM` transformer classique, qui doit
   passer par `past_key_values`.
4. **Embeddings liés de Luciole (`tie_word_embeddings`) et canal `<pad>`.** C'est le point critique :
   - les poids sont petits (rms 5,5e-3) et **la ligne `<pad>` est quasi nulle (7e-4)** ;
   - tout entraîner (embedding et tête), même à 5e-5 : le modèle s'effondre sur `<pad>` partout ;
   - tête découplée et entraînée, même à 1e-5 : même effondrement, car Adam baisse les 128k lignes non cibles
     et le logit de `<pad>` (environ 0) gagne ;
   - tête gelée : le modèle apprend le texte mais n'émet jamais `<pad>`, donc il parle sans arrêt ;
   - **ce qui marche** : embedding et tête gelés, avec un delta entraînable seulement sur les lignes
     `<pad>`, `<s>` et `<|im_end|>`. C'est `model.trainable_head_rows: [pad, bos, eos]` dans la branche
     `luciole_duplexstt_3.1` du fork. Dans `duplex_luciole.yaml`, le BOS est ensuite passé à `<|im_start|>`
     (`model.bos_token`) : le tokenizer de Luciole Instruct garde le `<s>` du pré-entraînement comme BOS,
     mais `<|im_end|>` comme EOS. Les lignes entraînées deviennent donc `<pad>`=3, `<|im_start|>`=260 et
     `<|im_end|>`=261. L'alternative serait un token de silence dédié, avec une ligne
     initialisée normalement.
5. **L'audio projeté est 5,5 fois plus fort que les embeddings texte** (rms 0,031 contre 0,0057) dans la somme
   par trame. Pour rééquilibrer si besoin : `duplex_user_channel_weight` / `duplex_text_channel_weight`.
6. **Le streaming de `DuplexSTTStreamingInference` est simulé** : l'audio est encodé d'un coup, puis le LLM
   avance trame par trame. Pour la démo temps réel, il faut encoder avec `encoder.cache_aware_stream_step`
   (voir `streaming_encode` dans `tests/test_encoder.py`, validé identique à l'offline).
