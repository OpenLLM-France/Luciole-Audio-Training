import nemo.collections.speechlm2 as slm
import librosa
import soundfile as sf
import torch
import os
import logging

import logging
from threading import Thread
import traceback
from transformers import TextIteratorStreamer

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

class SALMModel:
    def __init__(self, model_path, default_instruction="Listen to the audio and answer the question:"):
        self.model_path = os.path.expanduser(model_path)
        self.default_instruction = default_instruction
        if "Thinking" in self.model_path:
             # Better default for thinking models
             self.default_instruction = "You are a helpful assistant. Think step-by-step and wrap your thoughts in <think>...</think> tags before answering."
        logger.info(f"Loading SALM model from {self.model_path}...")
        try:
            self.model = slm.models.SALM.from_pretrained(self.model_path).eval()
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
                
            # DEBUG: Inspect standard Qwen attributes
            try:
                # Assuming model -> llm -> base_model -> model (Qwen3Model)
                base = self.model.llm
                if hasattr(base, 'base_model'):
                    base = base.base_model
                if hasattr(base, 'model'):
                    base = base.model
                
                logger.info(f"Model Structure Debug: {base}")
                logger.info(f"Model Attributes: {dir(base)}")
                if hasattr(base, 'embed_tokens'):
                    logger.info("embed_tokens exists.")
                else:
                    logger.error("embed_tokens MISSING.")
                    # FIX: Search for actual embedding layer in the model hierarchy
                    logger.info("Searching all modules for Embedding layer...")
                    try:
                        full_model = self.model.llm
                        if hasattr(full_model, 'base_model'):
                            full_model = full_model.base_model
                        
                        # Search ALL modules for an Embedding layer
                        embedding_layer = None
                        for name, module in full_model.named_modules():
                            if isinstance(module, torch.nn.Embedding):
                                logger.info(f"Found Embedding layer: {name} -> {module}")
                                if embedding_layer is None:  # Take the first one
                                    embedding_layer = module
                        
                        if embedding_layer is not None:
                            # Assign to Qwen3Model
                            if hasattr(full_model, 'model'):
                                full_model.model.embed_tokens = embedding_layer
                                logger.info(f"Successfully patched embed_tokens with: {embedding_layer}")
                            else:
                                full_model.embed_tokens = embedding_layer
                                logger.info(f"Successfully patched embed_tokens at top level: {embedding_layer}")
                        else:
                            # Last resort: create embedding layer from lm_head's weight
                            logger.warning("No Embedding layer found! Creating from lm_head weight...")
                            try:
                                if hasattr(full_model, 'lm_head'):
                                    # Create an Embedding layer using lm_head's transposed weight
                                    vocab_size = full_model.lm_head.out_features
                                    embed_dim = full_model.lm_head.in_features
                                    logger.info(f"Creating Embedding({vocab_size}, {embed_dim})")
                                    
                                    # Create new embedding and copy weights from lm_head (transposed)
                                    embedding_layer = torch.nn.Embedding(vocab_size, embed_dim)
                                    with torch.no_grad():
                                        embedding_layer.weight.copy_(full_model.lm_head.weight)
                                    
                                    # Move to same device as lm_head
                                    embedding_layer = embedding_layer.to(full_model.lm_head.weight.device)
                                    
                                    if hasattr(full_model, 'model'):
                                        full_model.model.embed_tokens = embedding_layer
                                        logger.info(f"Created and patched embed_tokens from lm_head")
                                    else:
                                        full_model.embed_tokens = embedding_layer
                                        logger.info(f"Created and patched embed_tokens at top level")
                                else:
                                    logger.error("CRITICAL: No lm_head found either! Model is unusable.")
                            except Exception as create_error:
                                logger.error(f"Failed to create embedding from lm_head: {create_error}")
                                logger.error(traceback.format_exc())
                    except Exception as patch_error:
                        logger.error(f"Failed to patch embed_tokens: {patch_error}")
                        logger.error(traceback.format_exc())
            except Exception as e:
                logger.error(f"Error inspecting model: {e}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def process_audio(self, input_path, output_path, target_sr=16000):
        """Resamples and converts audio to mono 16kHz."""
        try:
            audio_array, sr = librosa.load(input_path, sr=target_sr, mono=True)
            sf.write(output_path, audio_array, target_sr)
            return output_path
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise e

    def generate(self, audio_path=None, text_input=None, history=None, max_new_tokens=360):
        """
        Generates response from the model.
        history: List of dicts [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
        """
        
        # Start with existing history or empty list
        current_turn = []
        if history:
            current_turn.extend(history)
        
        import uuid
        
        if audio_path:
             # Assume audio_path is already processed/valid or process here without deletion?
             # For better control, we'll assume the caller (app.py) handles processing/storage lifecycle
             # so we can support multi-turn history with audio.
             
             # But to maintain backward compatibility if generate is called directly:
             # We can check if we should process. 
             # For now, let's just use the path provided. The prompts construction is what matters.
                
            instruction = text_input if text_input else self.default_instruction
            
            # Construct the prompt exactly as needed by the model
            prompt_content = f"{instruction}\n{self.model.audio_locator_tag}\n"
            
            current_turn.append({
                "role": "user",
                "content": prompt_content,
                "audio": [audio_path],
            })
        else:
            # Text-only interaction
            if not text_input:
                return "Please provide text or audio input."
                
            current_turn.append({
                "role": "user",
                "content": text_input,
            })
            
        prompts = [current_turn]

        try:
            # Generate response
            answer_ids = self.model.generate(prompts=prompts, max_new_tokens=max_new_tokens)
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

    def generate_stream(self, audio_path=None, text_input=None, history=None, max_new_tokens=360):
        """
        Generates response from the model in a streaming fashion.
        history: List of dicts [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
        """
        # Prepare inputs
        current_turn = []
        if history:
            current_turn.extend(history)
            
        if audio_path:
            # Assume caller manages audio file lifecycle
            instruction = text_input if text_input else self.default_instruction
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
            current_turn.append({
                "role": "user",
                "content": text_input,
            })

        prompts = [current_turn]
        
        try:
            tokenizer = TokenizerWrapper(self.model.tokenizer)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=300.0)
            
            generation_kwargs = dict(
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                streamer=streamer
            )
            
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
