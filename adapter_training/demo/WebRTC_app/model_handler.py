import nemo.collections.speechlm2 as slm
import librosa
import soundfile as sf
import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SALMModel:
    def __init__(self, model_path, default_instruction="Listen to the audio and answer the question:"):
        self.model_path = os.path.expanduser(model_path)
        self.default_instruction = default_instruction
        logger.info(f"Loading SALM model from {self.model_path}...")
        try:
            self.model = slm.models.SALM.from_pretrained(self.model_path).eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model moved to CUDA.")
            else:
                logger.warning("CUDA not available, running on CPU. Inference might be slow.")
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
        
        if audio_path:
            # Ensure audio is processed
            clean_audio_path = "/tmp/temp_inference_audio.wav"
            self.process_audio(audio_path, clean_audio_path)
            
            # Construct prompt with audio
            # Default instruction if text_input is empty
            instruction = text_input if text_input else self.default_instruction
            
            prompt_content = (
                f"{instruction}\n"
                f"{self.model.audio_locator_tag}\n"
            )
            
            current_turn.append({
                "role": "user",
                "content": prompt_content,
                "audio": [clean_audio_path],
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
            # We look for the assistant header and take everything after it
            separator = "<|start_header_id|>assistant<|end_header_id|>\n"
            if separator in response_text:
                response_text = response_text.split(separator)[-1].strip()
            # Also handle case without newline
            separator_inline = "<|start_header_id|>assistant<|end_header_id|>"
            if separator_inline in response_text:
                response_text = response_text.split(separator_inline)[-1].strip()
                
            return response_text
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return f"Error generating response: {str(e)}"
