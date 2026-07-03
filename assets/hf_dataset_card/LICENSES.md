# Licenses of the source datasets

The **Luciole Audio Training Dataset** is an *aggregation of many third-party datasets*, each
under its **own license**. This file documents, for every source dataset, its license, origin,
and whether **audio redistribution** appears to be permitted.

> ⚠️ **Read this before redistributing anything.**
> - **No single license covers the whole collection.** Always apply the terms of each source
>   dataset to the corresponding split (the split name maps 1:1 to a source dataset; see
>   `dataset_index.json`).
> - The columns below were compiled from public sources and **may be inaccurate or out of
>   date**. Confidence is noted per row. **Verify before any commercial use or redistribution.**
> - This table is **informational, not legal advice**.

## ⛔ Do **not** redistribute the audio of these (blockers / unknown rights)

| Dataset | Why |
|---|---|
| **LINAGORA Meetings** | Internal Linagora corpus, no public license. Needs written permission. |
| **LeVoiceLab CTFAR / CTFNN1** | Commercial data marketplace; proprietary terms. Get clearance. |
| **YouTubeFr** | Self-asserted CC0 but not verifiable per-video; YouTube ToS + embedded music rights. |
| **MELD** | Audio = *Friends* TV clips (Warner Bros). GPL covers code/annotations only, not the media. |
| **MusicCaps / LP-MusicCaps / LP-MusicCaps-MTT / CompA-R** | Audio not distributed by source — YouTube/AudioSet IDs only; underlying audio copyrighted. |
| **MECAT-QA / MECAT-Caption** | Audio derived from ACAV100M (YouTube); CC tag likely covers annotations only. |
| **MusicQA** (bundled MusicCaps part) | Inherits MusicCaps (YouTube) restrictions. |
| **amuvarma/10k-filtered-tune-audio**, **liangtianle/science-question**, **liangtianle/unsafety-question**, **VoxPopuli-QA** (annotation layer) | No verifiable license. |

## 🔒 Non-commercial / research-only (redistribution allowed but restricted)

mTEDx (NC-ND), CFPB, CFPP2000, CLAPI, ESLO, TCOF, LesVocaux, PFC (French oral corpora, all NC),
SLUE-TED (NC-ND), Vigogne/French-Alpaca (NC), gruhit-patel/alpaca_speech_instruct (likely NC),
Stress-17K (NC), FCaps (NC-SA, captions only), MagnaTagATune (NC-SA), MTG-Jamendo (per-track CC,
NC research), CLEAR (NC), Clotho / Clotho-AQA / TACOS (per-file Freesound, mostly CC-BY(-NC)).

## ✅ Clear for redistribution (attribution / share-alike as noted)

Mozilla Common Voice (CC0), VoxPopuli *dataset* (CC0), CoVoST/CoVoST2 (CC0), FLEURS (CC-BY),
Multilingual LibriSpeech (CC-BY), African Accented French (Apache-2.0), SimSamu (MIT),
PxSLU (CC-BY), CohereLabs/aya_collection (Apache-2.0), Menlo/instruction-speech-encodec (MIT),
UltraChat-300K-SLAM-Omni (MIT), AMI (CC-BY), JamendoMaxCaps (CC-BY-SA), FMA (per-track CC).

---

## Speech — ASR & speech translation corpora

| Dataset | License (SPDX) | Source | Redistribution? | Confidence | Notes |
|---|---|---|---|---|---|
| FLEURS | CC-BY-4.0 | https://huggingface.co/datasets/google/fleurs | Attribution | High | Attribution required. |
| Multilingual LibriSpeech (MLS) | CC-BY-4.0 | https://www.openslr.org/94/ | Attribution | High | Source audio public-domain (LibriVox); corpus is CC-BY. |
| Mozilla Common Voice | CC0-1.0 | https://commonvoice.mozilla.org | Yes | High | Public domain; no restrictions. |
| YODAS | CC-BY-3.0 | https://huggingface.co/datasets/espnet/yodas | Attribution | Medium | Per-video CC YouTube content; takedown process exists. |
| VoxForge | GPL-3.0 | https://www.voxforge.org/ | Yes (copyleft) | Medium | GPL copyleft imposes downstream obligations. |
| Multilingual TEDx (mTEDx) | CC-BY-NC-ND-4.0 | https://www.openslr.org/100 | Research-only | High | NonCommercial **and** NoDerivatives — re-segmentation likely violates ND. |
| VoxPopuli | CC0-1.0 (dataset) | https://huggingface.co/datasets/facebook/voxpopuli | Yes | High | Dataset CC0 (EU Parliament). Code/models are CC-BY-NC-4.0 — keep distinct. |
| CoVoST / CoVoST2 | CC0-1.0 | https://github.com/facebookresearch/covost | Yes | High | Built on Common Voice; unrestricted. |
| African Accented French (SLR57) | Apache-2.0 | https://www.openslr.org/57/ | Yes (keep NOTICE) | High | Retain license/attribution notice. |
| PxSLU / PxCorpus | CC-BY-4.0 | https://zenodo.org/records/6482587 | Attribution | High | French prescription speech (arXiv:2207.08292). |
| ACSYNT | CC-BY-SA-4.0 | http://www.llf.cnrs.fr/fr/acsynt | Attribution + SA | Medium | Via ORTOLANG/SLDR; derivatives keep same license. |
| PFC | Research-only (ORTOLANG subset CC-BY-NC-SA-4.0) | https://www.projet-pfc.net/ | No / Unclear | Medium | Authors retain rights, non-commercial; CC only for anonymized subset. Verify or exclude. |
| SimSamu | MIT | https://huggingface.co/datasets/medkit/simsamu | Yes | High | French medical dispatch corpus. |
| CFPB | CC-BY-NC-SA (version unclear) | http://cfpp2000.univ-paris3.fr/ | Research-only | Low–Med | No standalone license; inherits CFPP2000 NC terms. |
| CFPP2000 | CC-BY-NC-SA-3.0 | https://www.ortolang.fr/market/corpora/cfpp2000 | Research-only | High | NonCommercial. |
| CLAPI | CC-BY-NC-SA-4.0 | http://clapi.ish-lyon.cnrs.fr/ | Research-only | High | Teaching/research only, citation required. |
| ESLO | CC-BY-NC-SA-4.0 | http://eslo.huma-num.fr/ | Research-only | High | Audio + transcripts NC-SA. |
| LesVocaux | CC-BY-NC-SA-4.0 | https://huggingface.co/datasets/datasets-CNRS/lesvocaux | Research-only | High | CNRS spontaneous French voice messages. |
| TCOF | CC-BY-NC-SA-2.0-FR | https://www.cnrtl.fr/corpus/tcof/ | Research-only | High | NonCommercial (2.0 France). |
| LINAGORA Meetings | Unknown (internal) | https://huggingface.co/linagora/linto_stt_fr_fastconformer | **No** (treat as) | Low | Internal corpus, no public license. |
| LeVoiceLab CTFAR | Unknown (commercial) | https://speech-data-hub.levoicelab.org/ | **No** (likely) | Low–Med | Commercial marketplace; confirm directly. |
| LeVoiceLab CTFNN1 | Unknown (commercial) | https://speech-data-hub.levoicelab.org/ | **No** (likely) | Low–Med | Same marketplace; no open license found. |
| YouTubeFr | CC0-1.0 (self-asserted) | internal (Linagora) | Unclear / risky | Low–Med | Per-video provenance unverifiable; YouTube ToS + music rights. |

> **Note on the French oral corpora.** The Claire paper (arXiv:2311.16840) lists CFPB/CFPP2000/etc.
> as "CC-BY-SA", but every primary source shows **NC**. Claire only redistributed *text*, not audio.
> Use the **NonCommercial** reading for the audio.

## Speech — spoken QA, instruction & summarization

| Dataset | License (SPDX) | Source | Redistribution? | Confidence | Notes |
|---|---|---|---|---|---|
| SLUE phase-2 — SLUE-SQA-5 | Mixed: Apache-2.0 + CC-BY-SA-4.0 | https://huggingface.co/datasets/asapp/slue-phase-2 | Attribution (per-item) | High | ShareAlike + attribution for most items (arXiv:2212.10525). |
| SLUE phase-2 — SLUE-TED | CC-BY-NC-ND-4.0 | https://huggingface.co/datasets/asapp/slue-phase-2 | Research-only | High | No commercial use, no derivatives. |
| HeySQuAD (human) | CC-BY-4.0 | https://huggingface.co/datasets/yijingwu/HeySQuAD_human | Attribution | Medium | Derived from SQuAD (CC-BY-SA-4.0); upstream SA may bind text. |
| amuvarma/10k-filtered-tune-audio | Unknown | https://huggingface.co/datasets/amuvarma/10k-filtered-tune-audio | Unclear | Low | Gated/private; no license retrievable. |
| gruhit-patel/alpaca_speech_instruct | "cc" (ambiguous) | https://huggingface.co/datasets/gruhit-patel/alpaca_speech_instruct | Unclear | Low | From Stanford Alpaca (CC-BY-NC-4.0 + OpenAI ToS) → treat as NC. |
| liangtianle/science-question | Unknown | https://huggingface.co/datasets/liangtianle/science-question | Unclear | Low | Gated/private; no license. |
| liangtianle/unsafety-question | Unknown | https://huggingface.co/datasets/liangtianle/unsafety-question | Unclear | Low | Public but no license tag. |
| Menlo/instruction-speech-encodec-v1.5 | MIT | https://huggingface.co/datasets/Menlo/instruction-speech-encodec-v1.5 | Yes | High | Synthetic TTS; verify upstream prompt-voice/generator terms. |
| worstchan/UltraChat-300K-SLAM-Omni | MIT | https://huggingface.co/datasets/worstchan/UltraChat-300K-SLAM-Omni | Yes | Medium | From UltraChat (MIT); synthetic speech via CosyVoice. |
| CohereLabs/aya_collection | Apache-2.0 | https://huggingface.co/datasets/CohereLabs/aya_collection | Yes | High | Text-only; curated to be permissive. |
| Vigogne / French Alpaca | CC-BY-NC-4.0 | https://github.com/bofenghuang/vigogne | Research-only | High | Non-commercial; OpenAI ToS also apply. Text-only. |
| Compar:IA (comparia-conversations) | etalab-2.0 | https://huggingface.co/datasets/ministere-culture/comparia-conversations | Unclear | Medium | Contains proprietary-model outputs; filter by provider. Text-only. |
| SIFT-50M | CDLA-Sharing-1.0 | https://huggingface.co/datasets/amazon-agi/SIFT-50M | **No** (audio not shipped) | High | IDs only; audio governed by MLS (CC-BY), Common Voice 15 (CC0), VCTK (CC-BY). |
| VoxPopuli-QA | Unknown (base audio CC0-1.0) | https://huggingface.co/datasets/facebook/voxpopuli | Unclear | Low | QA annotation layer license unidentified; base audio CC0. |
| AMI Meeting Corpus | CC-BY-4.0 | https://groups.inf.ed.ac.uk/ami/corpus/ | Attribution | High | Credit the AMI project. |
| ICSI Meeting Corpus | CC-BY-4.0 (AMI free release) / LDC agreement | https://groups.inf.ed.ac.uk/ami/icsi/ | Attribution (free) / Research-only (LDC) | Medium | Confirm you have the free CC-BY copy, not LDC. |
| Nutshell | CC-BY-4.0 | https://aclanthology.org/2025.iwslt-1.2/ | Attribution | High | Source *ACL talk videos also CC-BY-4.0. |

## Sound & music — captioning & question answering

| Dataset | License (SPDX) | Source | Redistribution? | Confidence | Notes |
|---|---|---|---|---|---|
| CLEAR v1.0.0 & v2.0.0 | CC-BY-NC-4.0 (audio, inherited) | https://ieee-dataport.org/open-access/clear-dataset-compositional-language-and-elementary-acoustic-reasoning | Research-only | Medium | Audio = Good-Sounds (MTG-UPF), CC-BY-NC-4.0. Gen code BSD-3-Clause. |
| Clotho-AQA | MIT (Q&A); audio per-file Freesound | https://zenodo.org/records/6473207 | Attribution (per-file) | Medium | Tampere portal labels CC-BY-NC-SA-4.0; check per-file metadata. |
| mispeech/MECAT-QA | CC-BY-3.0 (tag) | https://huggingface.co/datasets/mispeech/MECAT-QA | Unclear | Low | Audio from ACAV100M (YouTube); tag likely covers annotations only. |
| mispeech/MECAT-Caption | CC-BY-3.0 (tag); code Apache-2.0 | https://huggingface.co/datasets/mispeech/MECAT-Caption | Unclear | Low | Same ACAV100M (YouTube) concern. |
| inclusionAI/AudioMCQ | Apache-2.0 | https://huggingface.co/datasets/inclusionAI/AudioMCQ | **No** (annotations only) | High | Ships MCQ/CoT + paths, no audio; fetch from sources below. |
| ↳ Clotho (AudioMCQ) | Captions CC-BY-NC-4.0; audio mostly CC-BY-4.0 | https://zenodo.org/records/3490684 | Research-only | High | Freesound audio; non-commercial w/ attribution. |
| ↳ CompA-R (AudioMCQ) | Unknown; audio = AudioSet-Strong | https://sreyan88.github.io/gamaaudio/ | **No** | Medium | Audio from AudioSet-Strong (YouTube), copyrighted. |
| ↳ TACOS (AudioMCQ) | Captions CC-BY-4.0; audio per-file Freesound | https://zenodo.org/records/15379789 | Attribution (per-file) | High | ~12k Freesound recordings. |
| ↳ SpeechCraft (AudioMCQ) | Annotations per repo; audio NC EULA | https://github.com/thuhcsi/SpeechCraft | Research-only | Medium | No audio distributed; AudioMCQ uses LibriTTS-R split (CC-BY-4.0). |
| ↳ LP-MusicCaps-MTT (AudioMCQ) | Captions MIT (HF); audio = MagnaTagATune | https://huggingface.co/datasets/seungheondoh/LP-MusicCaps-MTT | **No** (captions only) | Medium | Audio = MagnaTagATune (CC-BY-NC-SA-3.0). |
| MusicCaps | CC-BY-SA-4.0 (captions only) | https://huggingface.co/datasets/google/MusicCaps | **No** (audio) | High | YouTube IDs + timestamps only; audio copyrighted. |
| JamendoMaxCaps | CC-BY-SA-3.0 | https://huggingface.co/datasets/amaai-lab/JamendoMaxCaps | Attribution + SA | Medium | ~362k instrumental CC tracks; audio + captions shipped. |
| FMA (Free Music Archive) | CC-BY-4.0 (metadata); audio per-track CC/PD | https://github.com/mdeff/fma | Attribution (per-track) | High | Check per-track license; non-redistributable tracks excluded upstream. |
| MusicQA | Unknown; code GPL-3.0 | https://github.com/shansongliu/MU-LLaMA | Research-only / Unclear | Low | Generated from MusicCaps + MTT; inherits restrictive upstream terms. |
| LP-MusicCaps | CC-BY-NC-4.0 (GitHub) | https://github.com/seungheondoh/lp-music-caps | Captions only (NC) | Medium | Audio sourced separately; MIT(HF) vs CC-BY-NC-4.0(GitHub) discrepancy. |
| MagnaTagATune (MTT) | CC-BY-NC-SA-3.0 | https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset | Attribution + NC + SA | High | ~25.9k 30s clips; research only. |
| MTG-Jamendo | Audio per-track CC (mixed); metadata CC-BY-NC-SA-4.0 | https://github.com/MTG/mtg-jamendo-dataset | Attribution + NC (research) | High | Commercial use needs Jamendo S.A. authorization. |

## Speech — speaker, language, emotion & paralinguistic

| Dataset | License (SPDX) | Source | Redistribution? | Confidence | Notes |
|---|---|---|---|---|---|
| VoxLingua107 | CC-BY-4.0 | https://bark.phon.ioc.ee/voxlingua107/ | Attribution | Medium | YouTube-sourced; audio redistribution legally murky despite CC-BY. |
| VoxCeleb (1/2) | CC-BY-4.0 (Vox2 metadata CC-BY-SA-4.0) | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ | Research-only | Medium | YouTube-sourced; "research purposes" terms. |
| MELD | GPL-3.0 (repo/annotations) | https://github.com/declare-lab/MELD | **No** | High | Audio = *Friends* TV (Warner Bros). GPL covers code/annotations only. |
| ssi-speech | MIT (declared) | https://huggingface.co/datasets/stapesai/ssi-speech-emotion-recognition | Unclear | Low | Aggregates CREMA-D/RAVDESS/TESS/SAVEE (own licenses); MIT may not override. |
| OreauFR_02 | CC-BY-4.0 | https://zenodo.org/records/4405783 | Attribution | Medium | Some sources say NC; confirm on Zenodo. |
| emotional-speech-audio-dataset-4languages | Unknown | https://huggingface.co/datasets/yukat237/emotional-speech-audio-dataset-4languages | Unclear | Low | No license declared; treat as restricted. |
| Stress-17K | CC-BY-NC-4.0 | https://huggingface.co/datasets/slprl/Stress-17K-raw | Research-only | High | Synthetic (StressTest paper); non-commercial. |
| FCaps | CC-BY-NC-SA-4.0 | https://huggingface.co/datasets/yfyeung/FCaps | Captions only (NC-SA) | High | Captions/metadata only; audio from Emilia/EARS/Expresso/VoxCeleb. |
| CommonVoice speaker attributes | CC0-1.0 | https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0 | Yes | High | Audio CC0. Terms forbid re-identification; attributes are self-reported. |

---

*Compiled automatically from public web sources. Corrections welcome — if you are a rights holder
and something is misattributed or should be removed, please open an issue.*
