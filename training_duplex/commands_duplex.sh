# Duplex STT launch commands (run on JZ, from train_duplex/). See README.md.
cd ~/speechlm/Luciole-Audio-Training/train_duplex

# Two heads (agent + streaming ASR), ASR/AST/QA test mix, 4 GPUs, 2 h on the dev QoS.
# 10k steps will likely not fit in 2 h: the run stops at max_time_per_run (1h45) with a -last.ckpt;
# continue it with --resume on t3.
python duplex_launcher.py --job-name duplex_2heads_4gpu --gpus 4 --qos dev --time 02:00:00 \
    --set ++trainer.max_steps=10000

# ASR head only (diagnostic): same model and settings, ASR manifests only, agent loss off everywhere.
# Compare val asr WER (asr_fr/asr_en, ASR head, silent agent) with duplex_2heads_4gpu (0.89 fr / 0.81 en at
# 10k): much better -> the two heads compete in the LoRA; about the same -> streaming without delay is the
# limit (try an ASR delay). The QA validation is meaningless here (agent untrained).
python duplex_launcher.py --job-name duplex_asr_only_4gpu --gpus 4 --qos dev --time 02:00:00 \
    --input-cfg conf/data/input_cfg_qa_asr_only.yaml \
    --set ++trainer.max_steps=10000

# Same ASR-only diagnostic with a fully trainable ASR channel input embedding (a copy of embed_tokens,
# ~262M params, not tied to the frozen head) instead of the frozen shared one + channel vector + special rows.
python duplex_launcher.py --job-name duplex_asr_only_embtrain_4gpu --gpus 4 --qos dev --time 02:00:00 \
    --input-cfg conf/data/input_cfg_qa_asr_only.yaml \
    --set ++trainer.max_steps=10000 --set model.share_asr_embeddings=false

# Teacher-forcing diagnostic of the ASR head (inference/asr_teacher_forcing.py, ~10 min on 1 dev GPU):
# teacher forcing vs teacher forcing without transcript history vs free running, on asr_fr / asr_en dev.
cd inference && mkdir -p $SCRATCH/speechlm/duplex_diag && sbatch asr_teacher_forcing.slurm duplex_asr_only_4gpu step-010000

# ASR-only diagnostic with ASR input dropout: half of the text tokens of the ASR channel input replaced by <pad>,
# so that the head must find the words in the audio instead of timing them from its transcript history
# (teacher-forcing diagnostic of duplex_asr_only_4gpu: 0 word started without history, 82 % empty in free running).
python duplex_launcher.py --job-name duplex_asr_only_drop50_4gpu --gpus 4 --qos dev --time 02:00:00 \
    --input-cfg conf/data/input_cfg_qa_asr_only.yaml \
    --set ++trainer.max_steps=10000 --set model.asr_input_dropout=0.5
# then: cd inference && sbatch asr_teacher_forcing.slurm duplex_asr_only_drop50_4gpu step-010000
# success: tf_pad starts words (start acc > 0) and free running is no longer mostly empty.

# ASR-only on the SLUE SQA5 documents (en, 168 h) with SLUE's own word timestamps (alignment/slue_sqa5_asr.py,
# manifests in $SCRATCH/speechlm/duplex_data/asr/en/SLUE-SQA5): our CTC alignment puts the first word of most
# CommonVoice en lines at 0 s. Validation set asr_slue (in conf/duplex_qa.yaml).
# Without dropout: compare with duplex_asr_only_4gpu (only the data changes).
python duplex_launcher.py --job-name duplex_asr_slue_4gpu --gpus 4 --qos dev --time 02:00:00 \
    --input-cfg conf/data/input_cfg_duplex_asr_slue.yaml \
    --set ++trainer.max_steps=10000
# With ASR input dropout 0.5: compare with duplex_asr_only_drop50_4gpu.
python duplex_launcher.py --job-name duplex_asr_slue_drop50_4gpu --gpus 4 --qos dev --time 02:00:00 \
    --input-cfg conf/data/input_cfg_duplex_asr_slue.yaml \
    --set ++trainer.max_steps=10000 --set model.asr_input_dropout=0.5
# delay_source_text_by=1 (now the default in duplex_qa.yaml): <pad> as ASR input of frame 0 and the user
# <bos> is a label, as in NeMo's design. Compare with duplex_asr_slue_4gpu evaluated with <bos> at frame 0
# (asr_teacher_forcing free_bos: SLUE WER 0.245); here the regular validation is meaningful.
python duplex_launcher.py --job-name duplex_asr_slue_delay1_4gpu --gpus 4 --qos dev --time 02:00:00 \
    --input-cfg conf/data/input_cfg_duplex_asr_slue.yaml \
    --set ++trainer.max_steps=10000
# Same on the regular ASR mix (fr+en, first-word timestamps fixed in place: fix_first_word_inplace.slurm).
# Compare with duplex_asr_only_4gpu (same data before the fix, <bos> fed at frame 0).
python duplex_launcher.py --job-name duplex_asr_only_delay1_4gpu --gpus 4 --qos dev --time 02:00:00 \
    --input-cfg conf/data/input_cfg_qa_asr_only.yaml \
    --set ++trainer.max_steps=10000
# then: cd inference && sbatch asr_teacher_forcing.slurm <xp> step-010000 "asr_slue asr_en"

# Two heads (agent + streaming ASR) again, now with delay_source_text_by=1 (duplex_asr_only_delay1_4gpu at 10k:
# WER 0.56 fr / 0.60 en / 0.67 slue, 1-4 % empty), the first-word fix and the SLUE gold QA (both turns
# transcribed). Same mix and steps as duplex_2heads_4gpu (0.89 fr / 0.81 en at 10k, 1h48 for 10k steps):
# QA 40 % / ASR 40 % / AST 20 %. Validation sets asr_slue and qa_slue (in conf/duplex_qa.yaml).
# If 10k steps don't fit (max_time_per_run 1h45): rerun the same command with --resume.
python duplex_launcher.py --job-name duplex_2heads_delay1_4gpu --gpus 4 --qos dev --time 02:00:00 \
    --set ++trainer.max_steps=10000

# Validation only of existing checkpoints (duplex_train.py ++validate_ckpts), logged at their step in a new
# version_N of the same run. duplex_2heads_delay1_4gpu: no validation after its resume (ValidationSchedule
# off by one step, fixed in duplex_train.py: ResumeSafeValidationSchedule), 7500 and 10k caught up here (~30 min).
C=$ALL_CCFRSCRATCH/audio/training/duplex/duplex_2heads_delay1_4gpu/checkpoints
python duplex_launcher.py --job-name duplex_2heads_delay1_4gpu --gpus 4 --qos dev --time 00:45:00 \
    --resume \
    --set "++validate_ckpts=[$C/step-007500.ckpt,$C/step-010000.ckpt]"

# Two heads, long run: 25k steps on t3 (14 h, max_time_per_run 13h45), default mix (input_cfg_qa.yaml)
# and validation sets (val_cfg_qa.yaml), batch_duration 480. duplex_2heads_delay1_4gpu at 360 s: ~0.66 s/step.
# If it stops before 25k: same command with --resume (dataloader state is restored).
python duplex_launcher.py --job-name duplex_2heads_25k_4gpu --gpus 4 --qos t3 --time 14:00:00 \
    --set ++trainer.max_steps=25000

# Two heads, long run: max_steps 100k (cosine over 100k) with a 20 h limit (t3 maximum); we see how far it gets
# (duplex_2heads_25k_4gpu: 4h55 for 25k with validations -> ~75-80k in 20 h, LR then still ~15-25 % of its peak:
# the last checkpoint is not annealed). Mix reweighted for 50k (input_cfg_qa.yaml: QA 21 % / ASR 53 % /
# AST 26 % of the examples, <= ~3 passes at 50k, so ~4.5 at 75k; duplex_2heads_25k_4gpu saw ComparIA 35x, HeySQuAD
# 8.6x; ~33 % of the French ASR is spontaneous verbatim speech: ESLO, LVL-Atril; CFPP2000 and its validation
# asr_fr_spont out for now, see input_cfg_qa.yaml).
# To continue after 20 h: same command with --resume.
python duplex_launcher.py --job-name duplex_2heads_100k_4gpu --gpus 4 --qos t3 --time 20:00:00 \
    --set ++trainer.max_steps=100000

# Same run from scratch with explicit bucket bins (conf/duplex_qa.yaml: bucket_duration_bins). duplex_2heads_100k_4gpu
# hit an OOM at 11.3k steps (padded batches up to 3.6x batch_duration with the default 40 s wide buckets) and cannot
# be resumed with other bins (lhotse restores the bucket buffers and max_cuts from the checkpoint).
python duplex_launcher.py --job-name duplex_2heads_100k_v2_4gpu --gpus 4 --qos t3 --time 20:00:00 \
    --set ++trainer.max_steps=100000

# v2 OOM at step ~267 too: the system prompt (HeySQuAD text context, up to ~500 tokens) is prepended to the frame
# sequence but was not in cut.duration, so one HeySQuAD in a short bucket stretched the whole padded batch (63 x 216
# frames for a 7.6 s bucket). NeMo_Duplex fix: conversation_as_duplex stores custom["prompt_duration"] and
# PromptAwareTimeConstraint (lhotse/sampling.py) measures duration + prompt_duration.
python duplex_launcher.py --job-name duplex_2heads_100k_v3_4gpu --gpus 4 --qos t3 --time 20:00:00 \
    --set ++trainer.max_steps=100000

# v3 diagnostic (inference/asr_teacher_forcing.py, step 100k): Nemotron's own RNNT on the same encoder gets WER
# 0.08 fr / 0.12 en, the duplex ASR head 0.52 / 0.49; in teacher forcing only 64 % of the word starts are right
# (P(non-pad) spread over the 4 frames before the word end): the head lacks the evidence that the word ended.
# More lookahead: encoder [56, 6] (480 ms, a trained Nemotron lookahead) + ASR channel delayed by 2 frames.
# Agent latency in real streaming: 0.48 s instead of 0.24 s. Delay <= 4 keeps the user EOS within `tail` (0.4 s).
# 2 h on dev with a 100k cosine, to see how far it gets; continue with --resume (--qos t3 for longer).
python duplex_launcher.py --job-name duplex_2heads_100k_ctx6_delay2_4gpu --gpus 4 --qos dev --time 02:00:00 \
    --set ++trainer.max_steps=100000 \
    --set model.delay_source_text_by=2 --set "model.perception.encoder.att_context_size=[56,6]"

# ctx6 + delay 2 diagnostic at 40k (inference/asr_teacher_forcing.py): teacher forcing improved (P(non-pad) 0.82 at
# the word start) but free running got worse than at 18.7k (normalized WER fr 0.274 -> 0.313, en 0.293 -> 0.362,
# more substitutions/insertions) and the head is silent without its transcript history: exposure bias.
# Same run from scratch with ASR input dropout 0.25 (5 h on t3).
python duplex_launcher.py --job-name duplex_2heads_100k_ctx6_delay2_drop25_4gpu --gpus 4 --qos t3 --time 05:00:00 \
    --set ++trainer.max_steps=100000 \
    --set model.delay_source_text_by=2 --set "model.perception.encoder.att_context_size=[56,6]" \
    --set model.asr_input_dropout=0.25
