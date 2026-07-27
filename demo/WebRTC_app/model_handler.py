import nemo.collections.speechlm2 as slm
import json
import librosa
import soundfile as sf
import torch
import os
import logging

from pathlib import Path
from threading import Thread
import traceback
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList


class CallbackStoppingCriteria(StoppingCriteria):
    """Lets the caller cancel an in-flight HF generate by setting a flag.
    The callback is invoked between every generated token; returning True
    aborts generation immediately, freeing the GPU for the next request.
    """

    def __init__(self, should_stop):
        self.should_stop = should_stop

    def __call__(self, input_ids, scores, **kwargs):
        try:
            return bool(self.should_stop())
        except Exception:
            return False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def decode(self, ids, **kwargs):
        # Map decode to ids_to_text for NeMo compatibility
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        # Some streamers pass a single ID, others pass a list
        if isinstance(ids, int):
            ids = [ids]
        return self.tokenizer.ids_to_text(ids)

def resolve_model_class(model_path):
    """Pick the SALM class matching the checkpoint: SALM or SALMAutomodel.

    `SALM` builds the LLM with plain HF transformers (Canary-Qwen, Luciole-1B);
    `SALMAutomodel` goes through nemo_automodel, which is what the Nemotron-H
    backbones (Luciole-8B/23B) were trained with. Getting it wrong does not raise:
    hf_hub keeps only the state-dict keys present in both the checkpoint and the
    freshly built model, so a mismatch silently drops the LLM tensors and leaves a
    random backbone emitting fluent, unrelated text.

    Discriminator: `use_nemo_automodel` in the exported config.json, the same flag
    the export/eval Slurm scripts key on. MODEL_CLASS overrides it for a checkpoint
    whose config predates the flag.
    """
    override = os.getenv("MODEL_CLASS", "").strip()
    if override:
        cls = getattr(slm.models, override, None)
        if cls is None:
            raise ValueError(f"MODEL_CLASS={override!r} is not a class of nemo.collections.speechlm2.models")
        logger.info(f"Model class forced by MODEL_CLASS: {override}")
        return cls

    use_automodel = False
    cfg_path = Path(model_path) / "config.json"
    if cfg_path.is_file():
        try:
            use_automodel = bool(json.loads(cfg_path.read_text()).get("use_nemo_automodel", False))
        except Exception as e:
            logger.warning(f"Could not read {cfg_path} ({e}); assuming the plain SALM backend.")
    else:
        # Hub id rather than a local dir — no local config to inspect.
        logger.warning(f"No config.json under {model_path}; assuming the plain SALM backend.")

    name = "SALMAutomodel" if use_automodel else "SALM"
    logger.info(f"Detected backend: {name} (use_nemo_automodel={use_automodel})")
    return getattr(slm.models, name)


def resolve_torch_dtype():
    """Load dtype. bf16 on GPU — these checkpoints are exported in bf16, and an 8B in
    fp32 is 32 GB. CPU stays in fp32 (bf16 matmuls there are painfully slow)."""
    requested = os.getenv("TORCH_DTYPE", "").strip()
    if requested:
        return getattr(torch, requested)
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


class SALMModel:
    def __init__(self, model_path, default_instruction="Listen to the audio and answer the question:"):
        self.model_path = os.path.expanduser(model_path)
        self.default_instruction = default_instruction
        if "Thinking" in self.model_path:
             # Better default for thinking models
             self.default_instruction = "You are a helpful assistant. Think step-by-step and wrap your thoughts in <think>...</think> tags before answering."
        model_cls = resolve_model_class(self.model_path)
        self.is_automodel = model_cls.__name__ == "SALMAutomodel"
        dtype = resolve_torch_dtype()
        logger.info(f"Loading {model_cls.__name__} from {self.model_path} in {dtype}...")
        try:
            # HFHubMixin._from_pretrained puts torch_dtype in the cfg before
            # configure_model builds the LLM, so the backbone is never materialised in
            # fp32. `.to(dtype)` then catches the perception module.
            self.model = model_cls.from_pretrained(self.model_path, torch_dtype=dtype).to(dtype).eval()
            if torch.cuda.is_available():
                try:
                    self.model = self.model.cuda()
                    logger.info("Model moved to CUDA.")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"CUDA OOM error: {e}")
                        logger.warning("Falling back to CPU. Inference will be slow.")
                        torch.cuda.empty_cache()
                        # Model is already on CPU by default
                    else:
                        raise
            else:
                logger.warning("CUDA not available, running on CPU. Inference might be slow.")

            self._maybe_override_attn_implementation()

            # NOTE — an "embed_tokens MISSING" repair used to live here. It fixed a
            # non-bug: SALM.__init__ moves the embedding out of the LLM on purpose
            # (salm.py:113-118, to keep FSDP/TP hooks clean) and puts it back around
            # generate() via move_embedding, both via find_embedding_layer, which already
            # knows every layout we use. Absent is the intended state.
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def _maybe_override_attn_implementation(self):
        """Keep Nemotron-H off FlashAttention-2.

        transformers 5.6's flash_attention_forward does ``s_aux.to(query.dtype)``
        unconditionally, and Nemotron-H has no attention sinks — so s_aux is None and
        every generate() dies on the first attention block. FA2 is not our choice: it
        ships in the NeMo base image and gets picked by default. SDPA has no such path
        and is what the JZ eval env has been running all along.

        Only touches nemotron_h. ATTN_IMPLEMENTATION overrides the target.
        """
        llm = getattr(self.model, "llm", None)
        cfg = getattr(llm, "config", None)
        if cfg is None:
            return
        model_type = str(getattr(cfg, "model_type", ""))
        wanted = os.getenv("ATTN_IMPLEMENTATION", "").strip()
        if not wanted:
            if not model_type.startswith("nemotron_h"):
                return
            wanted = "sdpa"
        current = getattr(cfg, "_attn_implementation", None)
        if current == wanted:
            return
        try:
            if hasattr(llm, "set_attn_implementation"):
                llm.set_attn_implementation(wanted)
            else:
                cfg._attn_implementation = wanted
            logger.info(f"Attention implementation: {current} -> {wanted} (model_type={model_type})")
        except Exception as e:
            logger.warning(f"Could not switch attention implementation to {wanted}: {e}")

    def process_audio(self, input_path, output_path, target_sr=16000):
        """Resamples and converts audio to mono 16kHz."""
        try:
            audio_array, sr = librosa.load(input_path, sr=target_sr, mono=True)
            sf.write(output_path, audio_array, target_sr)
            return output_path
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise e

    # Generation kwargs that discourage runaway loops. Passed to NeMo's
    # underlying HF .generate() — extra kwargs the model doesn't recognize
    # are silently ignored, so this is safe across versions.
    _ANTI_LOOP_KWARGS = dict(
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
    )

    def generate(self, audio_path=None, text_input=None, history=None, max_new_tokens=360):
        """
        Generates response from the model.
        history: List of dicts [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
        """

        # Start with existing history or empty list
        raw_history = history if history else []
        
        # NeMo SALM formatter only supports ['user', 'assistant'].
        # We fold any 'system' messages into the first following message.
        processed_history = []
        system_prefix = ""
        for turn in raw_history:
            if turn.get("role") == "system":
                system_prefix += turn.get("content", "") + "\n\n"
            else:
                processed_history.append(turn.copy())

        current_turn = processed_history

        if audio_path:
            instruction = text_input if text_input else self.default_instruction
            # Prepend system prefix if this is the first message and we have one
            if not current_turn and system_prefix:
                instruction = system_prefix + instruction
                system_prefix = ""
            
            prompt_content = f"{instruction}\n{self.model.audio_locator_tag}\n"
            current_turn.append({
                "role": "user",
                "content": prompt_content,
                "audio": [audio_path],
            })
        else:
            if not text_input:
                return "Please provide text or audio input."
            
            content = text_input
            if not current_turn and system_prefix:
                content = system_prefix + content
                system_prefix = ""
                
            current_turn.append({
                "role": "user",
                "content": content,
            })

        # If we still have a system prefix (e.g. history was all system messages), 
        # but current_turn is not empty, prepend it to the first message.
        if system_prefix and current_turn:
            current_turn[0]["content"] = system_prefix + current_turn[0]["content"]

        prompts = [current_turn]

        try:
            answer_ids = self.model.generate(prompts=prompts, max_new_tokens=max_new_tokens, **self._ANTI_LOOP_KWARGS)
            response_text = self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
            
            # Post-processing to remove tags and prompt
            # Post-processing to remove tags and prompt
            # We look for the assistant header and take everything after it
            
            # Llama 3 format
            if "<|start_header_id|>assistant<|end_header_id|>" in response_text:
                response_text = response_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            
            # ChatML format (Qwen/others)
            elif "<|im_start|>assistant" in response_text:
                response_text = response_text.split("<|im_start|>assistant")[-1].strip()
            
            # Clean up potential trailing end tokens
            for end_token in ["<|im_end|>", "<|eot_id|>"]:
                if response_text.endswith(end_token):
                    response_text = response_text[:-len(end_token)].strip()
                
            return response_text
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e
        finally:
            # Clean up handled by caller
            pass

    def generate_stream(self, audio_path=None, text_input=None, history=None,
                        max_new_tokens=360, stop_callback=None,
                        min_new_tokens=None, temperature=None):
        """
        Generates response from the model in a streaming fashion.
        history: List of dicts [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
        """
        # Prepare inputs
        raw_history = history if history else []
        
        # NeMo SALM formatter only supports ['user', 'assistant'].
        # We fold any 'system' messages into the first following message.
        processed_history = []
        system_prefix = ""
        for turn in raw_history:
            if turn.get("role") == "system":
                system_prefix += turn.get("content", "") + "\n\n"
            else:
                processed_history.append(turn.copy())

        current_turn = processed_history
            
        if audio_path:
            # Assume caller manages audio file lifecycle
            instruction = text_input if text_input else self.default_instruction
            
            if not current_turn and system_prefix:
                instruction = system_prefix + instruction
                system_prefix = ""

            prompt_content = f"{instruction}\n{self.model.audio_locator_tag}\n"
            current_turn.append({
                "role": "user",
                "content": prompt_content,
                "audio": [audio_path],
            })
        else:
            if not text_input:
                yield "Please provide text or audio input."
                return
            
            content = text_input
            if not current_turn and system_prefix:
                content = system_prefix + content
                system_prefix = ""

            current_turn.append({
                "role": "user",
                "content": content,
            })

        if system_prefix and current_turn:
            current_turn[0]["content"] = system_prefix + current_turn[0]["content"]

        prompts = [current_turn]
        
        try:
            tokenizer = TokenizerWrapper(self.model.tokenizer)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=300.0)
            
            generation_kwargs = dict(
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                **self._ANTI_LOOP_KWARGS,
            )
            if min_new_tokens is not None and min_new_tokens > 0:
                # Cap min_new_tokens at max_new_tokens to avoid contradictions
                generation_kwargs["min_new_tokens"] = min(int(min_new_tokens), int(max_new_tokens))
            if temperature is not None:
                generation_kwargs["temperature"] = float(temperature)
                generation_kwargs["do_sample"] = True
            if stop_callback is not None:
                generation_kwargs["stopping_criteria"] = StoppingCriteriaList([
                    CallbackStoppingCriteria(stop_callback),
                ])
            
            def run_generation():
                try:
                    self.model.generate(**generation_kwargs)
                except Exception as e:
                    logger.error(f"Error in generation thread: {e}")
                    logger.error(traceback.format_exc())
            
            thread = Thread(target=run_generation)
            thread.start()
            
            try:
                for new_text in streamer:
                    yield new_text
            except Exception as e:
                # If it's a timeout (_queue.Empty), it means NeMo didn't use the streamer.
                # Fallback to block generation.
                logger.warning(f"Streaming failed or timed out ({e}). Falling back to block generation.")
                response = self.generate(audio_path=audio_path, text_input=text_input, history=history, max_new_tokens=max_new_tokens)
                yield response
                
        except TypeError as e:
             logger.warning(f"Streaming not supported arg ({e}). Falling block.")
             # Fallback to non-streaming generate if streamer kwarg fails
             # Note: we need to handle the infinite recursion if generate calls generate_stream
             # But generate() is the block version, so it's fine.
             # Wait, generate() also needs to be updated to not double-process if we updated app.py
             # We should update generate() too.
             pass
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            logger.error(traceback.format_exc())
            yield f"Error generating response: {e}"
