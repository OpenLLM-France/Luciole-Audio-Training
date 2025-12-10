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
    pip install -r requirements.txt
    ```

## Configuration

The application uses a `.env` file for configuration. A default `.env` file is provided, but you can modify it to suit your needs.

Create or modify the `.env` file in the project root:

```bash
MODEL_PATH=/path/to/your/model
PORT=8080
MAX_NEW_TOKENS=360
DEFAULT_INSTRUCTION="Listen to the audio and answer the question:"

```

The application expects the SALM model to be located at the path specified in `MODEL_PATH`.


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

### 2. Run the Container

The model (`meta-llama/Llama-3.2-1B-Instruct`) is baked into the image. If you wish to use a different model, you can just re-build the image with the `BASE_MODEL` argument or just download it inside the container:

```bash
docker run -d \
  -p 9009:9009 \
  -v /path/to/SALM/model:/app/model \
  -e MODEL_PATH=/app/model \
  webrtc-app
```

## Project Structure

- `app.py`: Main application server (aiohttp) handling WebRTC signaling and HTTP requests.
- `model_handler.py`: Wrapper class for loading and interacting with the SALM model.
- `static/`: Contains frontend assets (`index.html`, `client.js`, `styles.css`).
- `requirements.txt`: Python dependency list.

## Troubleshooting

- **Model Loading Error**: Ensure the model path is correct and you have sufficient GPU memory.
- **WebRTC Connection Failed**: Check your browser console for errors. Ensure you are not behind a restrictive firewall blocking WebRTC ports.
- **Audio Issues**: Verify that your microphone is working and permissions are granted in the browser.
