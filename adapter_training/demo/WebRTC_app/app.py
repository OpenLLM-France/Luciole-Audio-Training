#!/usr/bin/env python3

import json
import os
import asyncio
import logging
import uuid
import wave
from pathlib import Path
from aiohttp import web
from aiortc import RTCSessionDescription, RTCPeerConnection
from av.audio.resampler import AudioResampler
from model_handler import SALMModel

# Configuration
ROOT = Path(__file__).parent
MODEL_PATH = "/home/usertn2/MODELS/SpeechLM2/Canary-Llama-2.3B"
PORT = 8080

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebRTC-App")

# Initialize Model
# We initialize it globally for now. In production, might want lazy loading or a separate worker.
try:
    salm_model = SALMModel(MODEL_PATH)
except Exception as e:
    logger.error(f"Could not load model: {e}")
    salm_model = None

class AudioTrackHandler:
    def __init__(self, track):
        self.track = track
        self.resampler = AudioResampler(format='s16', layout='mono', rate=16000)
        self.frames = []
        self.task = None
        self.is_recording = False
        self.saved_path = None

    async def start_recording(self):
        self.is_recording = True
        self.frames = []
        self.saved_path = None
        self.task = asyncio.create_task(self.process_track())

    async def stop_recording(self):
        self.is_recording = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        return self.save_audio()

    async def process_track(self):
        while self.is_recording:
            try:
                frame = await self.track.recv()
                # Resample immediately to save memory and processing time later
                for resampled_frame in self.resampler.resample(frame):
                    self.frames.append(resampled_frame.to_ndarray().tobytes())
            except Exception as e:
                logger.error(f"Error receiving frame: {e}")
                break

    def save_audio(self):
        if self.saved_path:
            return self.saved_path
            
        if not self.frames:
            return None
        
        filename = f"/tmp/webrtc_{uuid.uuid4().hex}.wav"
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"".join(self.frames))
        
        self.saved_path = filename
        return filename

# Global dictionary to store handlers associated with PCs
pcs = set()
handlers = {}
# Global session store: sessionId -> { history: [] }
SESSIONS = {}

async def index(request):
    content = open(str(ROOT / 'static' / 'index.html')).read()
    return web.Response(content_type='text/html', text=content)

async def offer(request):
    # Legacy endpoint, redirect to new logic if needed or just keep as is
    return await offer_with_datachannel(request)

async def offer_with_datachannel(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
    session_id = params.get('sessionId', 'default')
    
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"history": []}

    pc = RTCPeerConnection()
    pcs.add(pc)
    
    # Store state for this connection
    state = {
        "handler": None,
        "channel": None,
        "text_input": "",
        "session_id": session_id # Reference to session ID
    }

    @pc.on("datachannel")
    def on_datachannel(channel):
        state["channel"] = channel
        
        @channel.on("message")
        async def on_message(message):
            # Message can be JSON: { "type": "stop", "text": "optional prompt" }
            # or just text prompt
            logger.info(f"Received message: {message}")
            try:
                data = json.loads(message)
            except:
                data = {"type": "text", "text": message}

            current_history = SESSIONS[state["session_id"]]["history"]

            if data.get("type") == "stop":
                # Stop recording and infer
                if state["handler"]:
                    audio_path = await state["handler"].stop_recording()
                    prompt = data.get("text", "")
                    
                    if salm_model:
                        response = salm_model.generate(audio_path=audio_path, text_input=prompt, history=current_history)
                        
                        # Update history
                        SESSIONS[state["session_id"]]["history"].append({"role": "user", "content": prompt if prompt else "Audio Message"})
                        SESSIONS[state["session_id"]]["history"].append({"role": "assistant", "content": response})
                        
                    else:
                        response = "Model not loaded."
                    
                    channel.send(json.dumps({"type": "response", "text": response}))
                    
                    # Clean up audio file
                    if audio_path and os.path.exists(audio_path):
                        os.remove(audio_path)
            
            elif data.get("type") == "text_only":
                 prompt = data.get("text", "")
                 if salm_model:
                     response = salm_model.generate(text_input=prompt, history=current_history)
                     SESSIONS[state["session_id"]]["history"].append({"role": "user", "content": prompt})
                     SESSIONS[state["session_id"]]["history"].append({"role": "assistant", "content": response})
                 else:
                     response = "Model not loaded."
                 channel.send(json.dumps({"type": "response", "text": response}))

    @pc.on("track")
    async def on_track(track):
        if track.kind == "audio":
            logger.info("Audio track received")
            handler = AudioTrackHandler(track)
            state["handler"] = handler
            await handler.start_recording()

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }),
    )

async def upload_audio(request):
    reader = await request.multipart()
    field = await reader.next()
    
    if field.name != 'audio':
        return web.Response(status=400, text="Invalid file field")
    
    filename = f"/tmp/upload_{uuid.uuid4().hex}.wav"
    size = 0
    with open(filename, 'wb') as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            size += len(chunk)
            f.write(chunk)
            
    # Get optional text prompt and sessionId
    text_prompt = ""
    session_id = "default"
    
    while True:
        field = await reader.next()
        if field is None:
            break
        if field.name == 'text':
            text_prompt = await field.read(decode=True)
            text_prompt = text_prompt.decode('utf-8')
        elif field.name == 'sessionId':
            session_id_bytes = await field.read(decode=True)
            session_id = session_id_bytes.decode('utf-8')

    logger.info(f"Received file {filename} ({size} bytes) with prompt: {text_prompt} for session {session_id}")

    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"history": []}
    current_history = SESSIONS[session_id]["history"]

    if salm_model:
        response = salm_model.generate(audio_path=filename, text_input=text_prompt, history=current_history)
        SESSIONS[session_id]["history"].append({"role": "user", "content": text_prompt if text_prompt else f"Uploaded Audio: {os.path.basename(filename)}"})
        SESSIONS[session_id]["history"].append({"role": "assistant", "content": response})
    else:
        response = "Model not loaded."

    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)

    return web.json_response({"text": response})

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer_with_datachannel)
    app.router.add_post("/upload", upload_audio)
    app.router.add_static("/static/", path=ROOT / "static", name="static")
    
    logger.info(f"Starting server on port {PORT}")
    web.run_app(app, host='localhost', port=PORT)
