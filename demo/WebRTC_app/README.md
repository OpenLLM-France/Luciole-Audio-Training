# WebRTC SALM App

This is a web application that interfaces with the SALM (SpeechLM2) model to provide multimodal interaction capabilities. It supports real-time audio streaming via WebRTC, audio file uploads, and text-based chat.

## Features

- **Real-time Audio Streaming**: Stream audio directly from your microphone to the model using WebRTC.
- **Text Interaction**: Chat with the model using text.
- **Audio Upload**: Upload `.wav` files for the model to process.
- **Interactive UI**: A modern, chat-like interface for easy interaction.

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (highly recommended for model inference)
- [NeMo Toolkit](https://github.com/NVIDIA/NeMo)
- FFmpeg (for audio processing libraries)

## Installation

1.  Clone the repository or navigate to the project directory.
2.  Install the Python dependencies:

    ```bash
    pip install --no-cache-dir -r requirements.txt
    pip install --no-cache-dir nemo_toolkit[all]
    ```

## Configuration

The application uses a `.env` file for configuration. A default `.env` file is provided, but you can modify it to suit your needs.

Create or modify the `.env` file in the project root:

```bash
MODEL_PATH=/path/to/your/model
PORT=9009
MAX_NEW_TOKENS=64
DEFAULT_INSTRUCTION="Listen to the audio and answer the question:"
HF_TOKEN=your_huggingface_token
BASE_MODEL=Qwen/Qwen3-4B-Thinking-2507
```

The demo does not load any weights. Models are served by vLLM servers, one per model,
and the app is an HTTP client:

| Variable | Purpose |
| --- | --- |
| `MODEL_ENDPOINTS` | `name=http://host:port,name=http://host:port`. More than one name turns on the model selector and the "compare all models" checkbox in the chat. |
| `MODEL_PATH` | Only read by `/model-config`, to display the checkpoint's `config.json`. Nothing loads from it. |
| `MODEL_ENDPOINT_READY_TIMEOUT` | How long to wait for each server at startup (default 300 s; an 8B takes ~2 min to load). |

Start a server with `serve_salm.py`, not `vllm serve` — the NeMo plugin registers through
the `vllm.general_plugins` entry point, which vLLM only loads in the engine process, while
the API front-end validates the model config *before* that, in the main process.

```bash
python serve_salm.py serve <checkpoint> --served-model-name Luciole-1B --port 9010 \
    --enforce-eager --dtype bfloat16 --max-model-len 16384 \
    --limit-mm-per-prompt '{"audio":1}' \
    --gpu-memory-utilization 0.35 --kv-cache-memory-bytes 6442450944
```

`--kv-cache-memory-bytes` is not optional on a DGX Spark: GPU memory there *is* system RAM,
so the free-memory reading climbs during vLLM's profiling as soon as another container moves,
and the startup assert fires. Pinning the KV cache skips profiling entirely.

## Backbones: Luciole-1B vs Luciole-8B

One image, one runtime, both families — the vLLM plugin picks its path from the
`architectures` field of the checkpoint's `llm_backbone/`.

| | Luciole-1B | Luciole-8B |
| --- | --- | --- |
| LLM backbone | Nemotron (dense transformer) | Nemotron-H (hybrid Mamba2 + attention) |
| vLLM inner model | `NemotronForCausalLM` | `NemotronHForCausalLM` |
| LoRA | HF PEFT naming, 48 pairs | NeMo naming, 56 pairs |
| Image | `salm-demo:vllm` | `salm-demo:vllm` |

LoRA adapters are merged into the base weights at load time, in float32, by the plugin
itself (`backends.py::_merge_lora_weights`) — no PEFT involved. Both spellings of the LoRA
config are handled (`lora_alpha`/`r` for the 1B, `alpha`/`dim` for the 8B). The merge is
silent: vLLM only configures logging for its own `vllm.*` namespace, so the plugin's
"Merged N LoRA weight pairs" line never reaches the log.

`transformers` must stay at 5.6: the built-in `nemotron_h` only parses the dense
`hybrid_override_pattern` (`'-'` = MLP) from 5.6 on, and vLLM 0.14 ships no fallback config
class for that `model_type`. The 1B alone would run on 4.57.

## Usage

1.  Start the application server:

    ```bash
    python app.py
    ```

2.  Open your web browser and navigate to:

    ```
    http://localhost:9009
    ```

3.  Allow microphone access when prompted to use the streaming feature.

## Docker Usage

Alternatively, you can run the application using Docker.

### 1. Build the Docker Image

You need to provide your Hugging Face token to download the model during the build.
Assuming your `HF_TOKEN` is in your `.env` file, you can run:

```bash
# Export the Hugging Face token & base model (llm) from .env
export $(grep -E '^HF_TOKEN=' .env)
export $(grep -E '^BASE_MODEL=' .env)

# Build the image
docker build \
  --build-arg HF_TOKEN="$HF_TOKEN" \
  --build-arg BASE_MODEL="$BASE_MODEL" \
  -t webrtc-app .
```

### 2. Run it

`run_docker.sh` starts one vLLM server per model, waits for each to answer, then starts
the app pointed at them:

```bash
cd <PATH-TO-REPO>/demo/WebRTC_app
./run_docker.sh /path/ckpt1:Luciole-1B /path/ckpt2:Luciole-8B
```

Two models or more turn on the model selector and the "compare all models" checkbox.
`IMAGE`, `HOST_PORT`, `KV_GIB` and `GPU_FRAC` override the defaults.

On the DGX the supported path is the Airflow DAG `demo_on_dgx`, which additionally
resolves the checkpoint, rsyncs it from the data-server if missing, and arms a TTL
watchdog. `run_docker.sh` is for a machine without Airflow, or for debugging.

### Note
 * `--network host` is required for the TEXT chat, not just for the microphone: without a
   media permission the browser only advertises obfuscated `.local` mDNS ICE candidates,
   and the multicast query aiortc uses to resolve them never leaves a bridged container.
 * the app writes `sessions.json` and `uploads/` into the mounted project directory.

## Project Structure

- `app.py`: Main application server (aiohttp) handling WebRTC signaling and HTTP requests.
- `remote_model.py`: vLLM client (OpenAI chat completions + SSE), the app's only model interface.
- `serve_salm.py`: `vllm serve` launcher that registers the NeMo SpeechLM plugin first.
- `static/`: Contains frontend assets (`index.html`, `client.js`, `styles.css`).
- `requirements.txt`: Python dependency list.

## Troubleshooting

- **Model Loading Error**: Ensure the model path is correct and you have sufficient GPU memory.
- **WebRTC Connection Failed**: Check your browser console for errors. Ensure you are not behind a restrictive firewall blocking WebRTC ports.
- **Audio Issues**: Verify that your microphone is working and permissions are granted in the browser.
