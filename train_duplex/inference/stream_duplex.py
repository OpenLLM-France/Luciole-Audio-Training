"""Vibe test of a DuplexSTT checkpoint in true streaming: an audio file is fed chunk by chunk, at
real-time pace, and the agent's replies are printed as the model emits them.

Unlike DuplexSTTStreamingInference.offline_inference (whole audio encoded at once, then the LLM
stepped frame by frame), this is the real serving loop:
  - the encoder runs cache-aware, one chunk at a time (encoder.cache_aware_stream_step; 320 ms
    chunks with att_context_size [56, 3]);
  - the LLM advances one 80 ms frame at a time with its KV cache, the input of a frame being
    user audio frame + embedding of the token it emitted at the previous frame.
The chunk is processed only once the wall clock has reached the end of its audio, so the
printed timings are what a live user would get. Mel features are computed upfront for the
whole file: they are local (25 ms windows, no normalization), so this changes nothing.

--rnnt-lang also prints the user transcript decoded by the Nemotron RNNT (prompt, decoder and joint, loaded from
the .nemo) on the same encoder frames as the LLM. rnnt_stream_wer.py measures its WER.

The input file is followed by --tail-silence seconds of silence: the model answers ~0.4 s after
the user stops, one token per 80 ms, so it needs room after the last question.

--check also runs offline_inference on the same audio and compares the tokens: in float32 (the
default) they match exactly, which validates the streaming loop itself. With --bf16 the chunked
encoder differs by rounding and greedy decoding drifts after a while in long answers.

On dgx2 (docker, NeMo fork branch luciole_duplexstt_3.1 copied to ~/NeMo_Duplex):

  docker run --rm -it --gpus all --user $(id -u):$(id -g) -v /home:/home -e HOME=$HOME \\
      -e PYTHONPATH=$HOME/NeMo_Duplex -w $HOME audiollm_benchmark:automodel \\
      python stream_duplex.py --ckpt ~/models/duplex/Luciole-1B/duplex_test_4gpu/hf_checkpoints/step-010000 \\
          --wav question.wav --prompt "Réponds à la question." --out out_dir
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omegaconf import open_dict
from transformers import DynamicCache

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.speechlm2.models.duplex_stt_model import DuplexSTTModel
from nemo.collections.speechlm2.modules.perception import IdentityConnector
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_nemo
from nemo.collections.speechlm2.parts.special_rows_head import SpecialRowsHead

FRAME = 0.08


class FeatureBuffer(CacheAwareStreamingAudioBuffer):
    """CacheAwareStreamingAudioBuffer over the duplex perception module (it expects an ASR model)."""

    def extract_preprocessor(self):
        self.model_normalize_type = None
        return self.model.preprocessor

    def get_model_device(self):
        return next(self.model.parameters()).device


class EncoderStream:
    """Cache-aware encoder state of one audio stream."""

    def __init__(self, enc):
        self.enc = enc
        if enc.streaming_cfg is None:
            enc.setup_streaming_params()
        param = next(enc.parameters())
        self.cache = enc.get_initial_cache_state(batch_size=1, dtype=param.dtype, device=param.device)
        self.steps = 0

    def step(self, chunk: torch.Tensor, chunk_len: torch.Tensor, last: bool):
        enc_out, enc_len, *self.cache = self.enc.cache_aware_stream_step(
            processed_signal=chunk,
            processed_signal_length=chunk_len,
            cache_last_channel=self.cache[0],
            cache_last_time=self.cache[1],
            cache_last_channel_len=self.cache[2],
            keep_all_outputs=last,
            drop_extra_pre_encoded=0 if self.steps == 0 else self.enc.streaming_cfg.drop_extra_pre_encoded,
        )
        self.steps += 1
        return enc_out, enc_len


class RnntStreamer:
    """Greedy RNNT transcript of one stream, extended at each chunk of encoder frames."""

    def __init__(self, asr):
        self.asr = asr
        self.hyps = None
        self.text = ""

    def step(self, enc_out: torch.Tensor, enc_len: torch.Tensor) -> str:
        if enc_len[0] > 0:
            encoded = self.asr._apply_prompt_to_encoded(enc_out.float())
            self.hyps = self.asr.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded, encoded_lengths=enc_len, return_hypotheses=True, partial_hypotheses=self.hyps
            )
            self.text = self.hyps[0].text
        return self.text


def load_rnnt(path: str, lang: str, device: str, keep_encoder: bool = False):
    """Nemotron .nemo with its language prompt set. The duplex encoder is its frozen encoder: no need for a second one."""
    asr = load_pretrained_nemo(ASRModel, path).to(device).eval()
    asr.set_inference_prompt(lang)
    if not keep_encoder:
        del asr.encoder
    return asr


class DuplexStreamer:
    """Mirrors DuplexSTTStreamingInference step for step, but fed one encoder chunk at a time."""

    def __init__(self, model: DuplexSTTModel, prompt: str | None, rnnt: RnntStreamer | None = None):
        self.model = model
        self.rnnt = rnnt
        perception = model.perception
        assert isinstance(perception.modality_adapter, IdentityConnector) and perception.rote is None, (
            "only an identity modality adapter streams frame by frame as is"
        )
        assert "Nemotron" not in model.cfg.pretrained_llm, "Nemotron-H cache path is not handled here"
        self.enc_stream = EncoderStream(perception.encoder)
        self.w_user = model.cfg.get("duplex_user_channel_weight", 1.0)
        self.w_text = model.cfg.get("duplex_text_channel_weight", 1.0)
        self.llm_cache = DynamicCache()
        self.pos = 0
        self.last = None

        self.prompt_len = 0
        if prompt:
            tok = model.tokenizer
            ids = torch.tensor([tok.bos] + tok.text_to_ids(prompt) + [tok.eos], device=model.device)
            for emb in model.embed_tokens(ids):
                self._llm_step(emb, is_prompt=True)
            self.prompt_len = len(ids)

    def _llm_step(self, user_emb: torch.Tensor, is_prompt: bool) -> int:
        # Position 0 gets the "BOS" embedding (offline_inference uses the pad row), then every
        # position gets the token emitted at the previous one; prompt positions emit <pad>.
        prev = self.model._get_bos_embedding()[0] if self.pos == 0 else self.model.embed_tokens(self.last)[0]
        x = user_emb * self.w_user + prev * self.w_text
        ans = self.model(x.view(1, 1, -1), cache=self.llm_cache)
        self.llm_cache = ans["cache"]
        tok = self.model.text_pad_id if is_prompt else int(ans["text_logits"][0, -1].argmax())
        self.last = torch.tensor([tok], device=x.device)
        self.pos += 1
        return tok

    def feed(self, chunk: torch.Tensor, chunk_len: torch.Tensor, last: bool) -> list[int]:
        enc_out, enc_len = self.enc_stream.step(chunk, chunk_len, last)
        if self.rnnt is not None:
            self.rnnt.step(enc_out, enc_len)
        frames = self.model.perception.proj(enc_out[:, :, : enc_len[0]].transpose(1, 2))[0]
        return [self._llm_step(f, is_prompt=False) for f in frames]


class Display:
    """Prints user speech onsets (energy VAD) and the agent's text as it is emitted."""

    def __init__(self, tokenizer, pad, bos, eos, speech: np.ndarray):
        self.tok, self.pad, self.bos, self.eos = tokenizer, pad, bos, eos
        self.speech = speech  # per-frame bool
        self.user_on = False
        self.turn = None  # {"start", "ids", "printed"}
        self.turns = []
        self.tokens = []

    def _close(self, t):
        self.turn["end"] = t
        self.turn["text"] = self.tok.ids_to_text(self.turn["ids"]) if self.turn["ids"] else ""
        self.turns.append(self.turn)
        self.turn = None

    def push(self, i: int, tok: int, wall: float):
        t = i * FRAME
        self.tokens.append(tok)
        if i < len(self.speech) and self.speech[i] != self.user_on:
            self.user_on = bool(self.speech[i])
            if self.user_on:
                print(f"\n[{t:6.2f}s | wall {wall:6.2f}s] 🗣  (l'utilisateur parle)", flush=True)
        if tok == self.pad:
            return
        if tok == self.bos or self.turn is None:
            if self.turn is not None:
                self._close(t)
            print(f"\n[{t:6.2f}s | wall {wall:6.2f}s] 🤖 ", end="", flush=True)
            self.turn = {"start": t, "ids": [], "printed": ""}
            if tok == self.bos:
                return
        if tok == self.eos:
            print(" ⏹", flush=True)
            self._close(t + FRAME)
            return
        self.turn["ids"].append(tok)
        text = self.tok.ids_to_text(self.turn["ids"])
        print(text[len(self.turn["printed"]) :], end="", flush=True)
        self.turn["printed"] = text

    def finish(self, t_end: float):
        if self.turn is not None:
            print(" … (pas d'EOS)", flush=True)
            self._close(t_end)


def frame_speech(audio: np.ndarray, sr: int, n_frames: int, rel_db: float = -35.0) -> np.ndarray:
    """Crude per-80ms-frame speech flag: RMS above (max RMS + rel_db)."""
    hop = int(FRAME * sr)
    frames = np.zeros(n_frames)
    for i in range(n_frames):
        seg = audio[i * hop : (i + 1) * hop]
        frames[i] = np.sqrt(np.mean(seg**2)) if len(seg) else 0.0
    db = 20 * np.log10(frames + 1e-8)
    return db > db.max() + rel_db


def show_timeline(speech, tokens, pad, bos, eos, width=100):
    sym = {pad: ".", bos: "B", eos: "E"}
    agent = "".join(sym.get(t, "x") for t in tokens)
    user = "".join("█" if s else "·" for s in speech[: len(agent)])
    print(f"\nTimeline (1 caractère = {FRAME * 1000:.0f} ms ; user: █ parole · silence ; agent: . pad, B, x texte, E)")
    for k in range(0, len(agent), width):
        print(f"  {k * FRAME:6.1f}s user  {user[k : k + width]}")
        print(f"          agent {agent[k : k + width]}")


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def load_audio(path: str, sr: int, tail_silence: float) -> np.ndarray:
    audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio[:, 0].copy())
    if file_sr != sr:
        audio = resample(audio[None], file_sr, sr)[0]
    audio = audio.numpy()
    return np.concatenate([audio, np.zeros(int(tail_silence * sr), dtype=np.float32)])


def load_model(ckpt: str, device: str, boosts: dict, bf16: bool) -> DuplexSTTModel:
    t = time.perf_counter()
    model = DuplexSTTModel.from_pretrained(ckpt).to(device).eval()
    if not bf16:
        # float32 is the reference (streaming == offline token for token) and, at batch 1 on
        # the GB10, faster than bf16 autocast, which re-casts the weights at every op.
        model = model.float()
    with open_dict(model.cfg):
        for k, v in boosts.items():
            if v is not None:
                model.cfg[f"inference_{k}_boost"] = v
    print(f"[load] {ckpt} in {time.perf_counter() - t:.0f}s on {device}")

    # from_pretrained loads with strict=False: make sure what training learned is really there,
    # a head without its deltas never emits <pad> and talks non-stop.
    head = model.lm_head
    if not isinstance(head, SpecialRowsHead) or head.delta.abs().max() == 0:
        raise RuntimeError("lm_head has no trained pad/bos/eos deltas: wrong NeMo branch or checkpoint?")
    lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
    if not lora_b or all(p.abs().max() == 0 for p in lora_b):
        raise RuntimeError("LoRA weights are missing or all zero")
    tok = model.tokenizer
    print(
        f"[load] head deltas on ids {head.ids.tolist()} (|delta| {head.delta.float().norm():.3f}), "
        f"{len(lora_b)} LoRA B matrices; pad={model.text_pad_id} bos={tok.bos}:{tok.ids_to_tokens([tok.bos])} "
        f"eos={tok.eos}:{tok.ids_to_tokens([tok.eos])}"
    )
    return model


@torch.inference_mode()
def run(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    boosts = {"pad": args.pad_boost, "bos": args.bos_boost, "eos": args.eos_boost}
    model = load_model(args.ckpt, device, boosts, args.bf16)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda" and args.bf16)
    sr = model.source_sample_rate
    audio = load_audio(args.wav, sr, args.tail_silence)
    duration = len(audio) / sr
    speech = frame_speech(audio, sr, int(np.ceil(duration / FRAME)))

    rnnt = None
    if args.rnnt_lang:
        rnnt = RnntStreamer(load_rnnt(args.asr or model.cfg.pretrained_asr, args.rnnt_lang, device))
    with autocast:
        streamer = DuplexStreamer(model, args.prompt, rnnt)
    buf = FeatureBuffer(model.perception, online_normalization=False, pad_and_drop_preencoded=False)
    buf.append_audio(audio)
    scfg = buf.streaming_cfg
    first_size, next_size = (scfg.chunk_size if isinstance(scfg.chunk_size, list) else [scfg.chunk_size] * 2)
    hop = model.cfg.perception.preprocessor.window_stride
    n_mel = int(buf.streams_length[0])

    pad, bos, eos = model.text_pad_id, model.tokenizer.bos, model.tokenizer.eos
    display = Display(model.tokenizer, pad, bos, eos, speech)
    mode = "temps réel" if args.realtime else "aussi vite que possible"
    print(f"[stream] {args.wav}: {duration:.1f}s dont {args.tail_silence:.0f}s de silence ajouté, "
          f"chunks de {next_size * hop * 1000:.0f} ms, {mode}; prompt: {args.prompt!r}")

    chunk_times, behind, n_frames = [], [], 0
    transcript = ""
    chunks = iter(buf)
    t0 = time.perf_counter()
    while True:
        start = buf.buffer_idx
        audio_ready = min(start + (first_size if start == 0 else next_size), n_mel) * hop
        if args.realtime and (wait := t0 + audio_ready - time.perf_counter()) > 0:
            time.sleep(wait)
        try:
            chunk, chunk_len = next(chunks)
        except StopIteration:
            break
        tc = time.perf_counter()
        with autocast:
            toks = streamer.feed(chunk, chunk_len, last=buf.is_buffer_empty())
        if device == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        chunk_times.append(now - tc)
        behind.append(now - t0 - audio_ready)
        for tok in toks:
            display.push(n_frames, tok, now - t0)
            n_frames += 1
        if rnnt is not None and rnnt.text != transcript:
            print(f"\n[{n_frames * FRAME:6.2f}s | wall {now - t0:6.2f}s] 📝 {rnnt.text[len(transcript) :]}", flush=True)
            transcript = rnnt.text
    display.finish(n_frames * FRAME)

    ct = np.array(chunk_times) * 1000
    print(f"\n[perf] {len(ct)} chunks, calcul par chunk p50 {np.median(ct):.0f} ms / p99 {np.percentile(ct, 99):.0f} ms "
          f"/ max {ct.max():.0f} ms pour {next_size * hop * 1000:.0f} ms d'audio "
          f"(RTF {sum(chunk_times) / duration:.2f}); retard max sur le temps réel {max(behind) * 1000:.0f} ms")
    show_timeline(speech, display.tokens, pad, bos, eos)
    print("\nRéponses :")
    for t in display.turns:
        print(f"  [{t['start']:6.2f}s → {t['end']:6.2f}s] {t['text']}")

    if rnnt is not None:
        print(f"\nTranscription RNNT ({args.rnnt_lang}) : {transcript}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(args.wav).stem
        sf.write(out / f"{stem}.wav", audio, sr)
        with open(out / f"{stem}.srt", "w") as f:
            for k, t in enumerate(display.turns, 1):
                f.write(f"{k}\n{srt_time(t['start'])} --> {srt_time(max(t['end'], t['start'] + 1))}\n{t['text']}\n\n")
        print(f"\n[out] {out}/{stem}.wav + {stem}.srt (ex. mpv {stem}.wav --sub-file={stem}.srt --force-window)")

    if args.check:
        check_offline(model, audio, args.prompt, display.tokens, autocast)


def check_offline(model, audio, prompt, stream_tokens, autocast):
    dev = model.device
    signal = torch.from_numpy(audio)[None].to(dev)
    kwargs = {}
    if prompt:
        tok = model.tokenizer
        ids = torch.tensor([[tok.bos] + tok.text_to_ids(prompt) + [tok.eos]], device=dev)
        kwargs = dict(prompt_tokens=ids, prompt_token_lens=torch.tensor([ids.shape[1]], device=dev))
    with autocast:
        ans = model.streaming_inference.offline_inference(signal, torch.tensor([signal.shape[1]], device=dev), **kwargs)
    offline = ans["tokens_text"][0, : int(ans["tokens_len"][0])].tolist()
    n = min(len(offline), len(stream_tokens))
    diff = [i for i in range(n) if offline[i] != stream_tokens[i]]
    print(f"\n[check] offline_inference: {len(offline)} trames, streaming: {len(stream_tokens)} ; "
          f"{len(diff)} tokens différents sur {n}" + (f", premier à {diff[0] * FRAME:.2f}s" if diff else " → identiques"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="hf_checkpoints/step-N directory")
    p.add_argument("--wav", required=True)
    p.add_argument("--prompt", default=None, help="system prompt (the instruction at training time)")
    p.add_argument("--rnnt-lang", default=None, help="also decode the user transcript with the Nemotron RNNT: fr, en, auto...")
    p.add_argument("--asr", default=None, help="Nemotron .nemo for --rnnt-lang (default: the checkpoint's pretrained_asr)")
    p.add_argument("--tail-silence", type=float, default=10.0, help="seconds of silence appended to the file")
    p.add_argument("--no-realtime", dest="realtime", action="store_false", help="process chunks as fast as possible")
    p.add_argument("--check", action="store_true", help="compare with offline_inference on the same audio")
    p.add_argument("--out", default=None, help="write <wav>.wav (with the silence) + <wav>.srt of the replies here")
    p.add_argument("--device", default=None)
    p.add_argument("--bf16", action="store_true", help="bf16 autocast instead of float32 (greedy then drifts from offline)")
    p.add_argument("--pad-boost", type=float, default=None, help="logit bonus on <pad> (more silent)")
    p.add_argument("--bos-boost", type=float, default=None, help="logit bonus on BOS (answers sooner)")
    p.add_argument("--eos-boost", type=float, default=None, help="logit bonus on EOS (shorter answers)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
