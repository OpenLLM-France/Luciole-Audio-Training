#!/usr/bin/env python3

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
import wave
from pathlib import Path
from dotenv import load_dotenv
import threading

load_dotenv()

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from av.audio.resampler import AudioResampler

from model_handler import SALMModel

# Configuration
ROOT = Path(__file__).parent
MODEL_PATH = os.getenv("MODEL_PATH", "/home/usertn2/MODELS/SpeechLM2/Canary-Llama-2.3B")
PORT = int(os.getenv("PORT", 7860))  # HuggingFace Spaces uses port 7860
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 64))
DEFAULT_INSTRUCTION = os.getenv("DEFAULT_INSTRUCTION", "Listen to the audio and answer the question:")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebRTC-App")

async def stream_generator_in_thread(generator_func, *args, **kwargs):
    """
    Runs a synchronous generator in a separate thread and yields items asynchronously.
    """
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    
    def producer():
        try:
            for item in generator_func(*args, **kwargs):
                loop.call_soon_threadsafe(queue.put_nowait, item)
            loop.call_soon_threadsafe(queue.put_nowait, None) # Sentinel
        except Exception as e:
            logger.error(f"Producer thread error: {e}")
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # Start the producer thread
    t = threading.Thread(target=producer)
    t.start()
    
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item

# Initialize Model
# We initialize it globally for now. In production, might want lazy loading or a separate worker.
try:
    salm_model = SALMModel(MODEL_PATH, default_instruction=DEFAULT_INSTRUCTION)
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

# ---------------------------------------------------------------------------
# Session store with JSON-file persistence
# ---------------------------------------------------------------------------
SESSIONS_FILE = ROOT / "sessions.json"
SESSIONS = {}
_save_task = None
_save_lock = asyncio.Lock()


def _load_sessions_from_disk():
    if not SESSIONS_FILE.exists():
        return {}
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as e:
        logger.warning(f"Could not load sessions.json: {e}")
        return {}


def _serializable_sessions():
    """Strip transient fields (audio paths point to /tmp files we delete)."""
    out = {}
    for sid, sess in SESSIONS.items():
        history = []
        for turn in sess.get("history", []):
            t = {k: v for k, v in turn.items() if k != "audio"}
            history.append(t)
        out[sid] = {"history": history}
    return out


async def _flush_sessions():
    async with _save_lock:
        try:
            tmp = SESSIONS_FILE.with_suffix(".tmp")
            data = _serializable_sessions()
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(SESSIONS_FILE)
        except Exception as e:
            logger.warning(f"Could not write sessions.json: {e}")


def schedule_save():
    """Debounced save: coalesces rapid updates into one write."""
    global _save_task
    if _save_task and not _save_task.done():
        return
    loop = asyncio.get_event_loop()

    async def runner():
        await asyncio.sleep(1.0)
        await _flush_sessions()

    _save_task = loop.create_task(runner())


SESSIONS.update(_load_sessions_from_disk())


# Per-session latest-generation tracking. Lets a stale generation skip its
# token emission and history append once a newer request has arrived.
_SESSION_LATEST_GEN: dict = {}


def _mark_generation(session_id: str, gen_id: int):
    _SESSION_LATEST_GEN[session_id] = gen_id


def _is_stale(session_id: str, gen_id: int) -> bool:
    return _SESSION_LATEST_GEN.get(session_id, gen_id) != gen_id


def _safe_send(channel, payload: dict) -> bool:
    """Send a JSON payload only if the channel is still open. Returns
    whether the send actually happened so callers can short-circuit the
    rest of their loop when the connection has dropped."""
    if channel is None or channel.readyState != "open":
        return False
    try:
        channel.send(json.dumps(payload))
        return True
    except Exception as e:
        logger.warning(f"DataChannel send failed: {e}")
        return False


def _effort_overrides(effort_mode: str, max_tokens: int) -> dict:
    """Map a client-supplied effort label to concrete generate kwargs.
    'max' forces the Thinking model to spend its full budget on reasoning;
    anything else (incl. 'normal' or unset) keeps the user's settings.
    """
    if (effort_mode or "").lower() == "max":
        return {
            "max_new_tokens": max(int(max_tokens), 1024),
            "min_new_tokens": 128,
            "temperature": 0.3,
        }
    return {"max_new_tokens": int(max_tokens)}


def _strip_chatml_assistant(text: str) -> str:
    """Strip the trailing assistant header that some chat-templated models
    leak through into the raw output."""
    if "<|im_start|>assistant" in text:
        return text.split("<|im_start|>assistant")[-1].strip()
    return text


# ---------------------------------------------------------------------------
# Generation queue
# ---------------------------------------------------------------------------
# The model has shared internal state, so only one `generate` call may run at
# a time. Instead of an opaque asyncio.Lock, we use an explicit FIFO queue:
# clients see their position, the server can reject when overloaded, and a
# new request from the same session evicts that session's older queued entry
# (no point waiting to discard an answer the client already replaced).

QUEUE_MAX_DEPTH = int(os.getenv("QUEUE_MAX_DEPTH", 5))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", 20))


class QueueFullError(Exception):
    pass


class RateLimitedError(Exception):
    pass


class _QueueEntry:
    __slots__ = ("id", "session_id", "gen_id", "channel", "queued_at", "active_event")

    def __init__(self, session_id, gen_id, channel):
        self.id = uuid.uuid4().hex
        self.session_id = session_id
        self.gen_id = gen_id
        self.channel = channel  # may be None for HTTP requests
        self.queued_at = time.monotonic()
        self.active_event = asyncio.Event()


class GenerationQueue:
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.entries: list = []      # waiting (FIFO)
        self.active = None           # currently-running entry

    def _depth(self) -> int:
        return len(self.entries) + (1 if self.active else 0)

    def enqueue(self, session_id, gen_id, channel):
        # Drop earlier waiting entries from the same session — a newer
        # request supersedes them, no point keeping a slot for a generation
        # the client has already abandoned.
        evicted = [e for e in self.entries if e.session_id == session_id]
        self.entries = [e for e in self.entries if e.session_id != session_id]
        for e in evicted:
            e.active_event.set()  # unblock awaiters; they'll see is_stale
        if self._depth() >= self.max_depth:
            return None
        entry = _QueueEntry(session_id, gen_id, channel)
        self.entries.append(entry)
        self._maybe_activate()
        self._broadcast()
        return entry

    def release(self, entry):
        if self.active is entry:
            self.active = None
        elif entry in self.entries:
            self.entries.remove(entry)
        self._maybe_activate()
        self._broadcast()

    def _maybe_activate(self):
        if self.active is not None or not self.entries:
            return
        next_entry = self.entries.pop(0)
        self.active = next_entry
        next_entry.active_event.set()

    def _broadcast(self):
        for i, e in enumerate(self.entries):
            if e.channel is None:
                continue
            _safe_send(e.channel, {
                "type": "queue_position",
                "position": i + 1,
                "depth": self._depth(),
                "generationId": e.gen_id,
            })

    def metrics(self) -> dict:
        oldest_wait = 0.0
        if self.entries:
            oldest_wait = time.monotonic() - self.entries[0].queued_at
        return {
            "active": self.active is not None,
            "queued": len(self.entries),
            "depth": self._depth(),
            "max_depth": self.max_depth,
            "oldest_wait_s": round(oldest_wait, 2),
        }


_GENERATION_QUEUE = GenerationQueue(max_depth=QUEUE_MAX_DEPTH)


@contextlib.asynccontextmanager
async def acquire_model_slot(session_id: str, gen_id: int, channel=None):
    """Wait in the generation queue. Yields when it's our turn. Raises
    QueueFullError if the queue is at capacity."""
    entry = _GENERATION_QUEUE.enqueue(session_id, gen_id, channel)
    if entry is None:
        raise QueueFullError()
    try:
        await entry.active_event.wait()
        yield
    finally:
        _GENERATION_QUEUE.release(entry)


# Sliding-window per-session rate limit
_RATE_HITS: dict = {}


def check_rate_limit(session_id: str) -> bool:
    now = time.monotonic()
    cutoff = now - 60.0
    window = _RATE_HITS.setdefault(session_id, [])
    window[:] = [t for t in window if t > cutoff]
    if len(window) >= RATE_LIMIT_PER_MIN:
        return False
    window.append(now)
    return True


def _looks_runaway(text: str) -> bool:
    """
    Heuristic: detect whether `text` is a degenerate/looping output we
    should NOT persist into history. False positives would just mean the
    next turn starts fresh, which is a much smaller harm than poisoning
    the conversation forever.
    """
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < 80:
        return False
    # 1) Whole response collapsed to a single repeated character/short string
    if len(set(stripped)) <= 3:
        return True
    # 2) The same 4-word n-gram repeated many times
    words = stripped.split()
    if len(words) >= 40:
        ngram_counts = {}
        for i in range(len(words) - 4):
            key = " ".join(words[i:i + 4])
            ngram_counts[key] = ngram_counts.get(key, 0) + 1
            if ngram_counts[key] >= 6:
                return True
    # 3) Lexical diversity collapse — barely any unique tokens
    if len(words) >= 50:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.15:
            return True
    return False

async def _handle_audio_stop(channel, state, data):
    """Process the 'stop' (mic-recording-finished) message from the client."""
    raw_audio_path = await state["handler"].stop_recording()
    prompt = data.get("text", "")
    generation_id = data.get("generationId", 0)
    max_tokens = int(data.get("maxTokens", MAX_NEW_TOKENS))
    custom_instruction = data.get("instruction") or DEFAULT_INSTRUCTION
    effort_mode = data.get("effortMode", "normal")
    session_id = state["session_id"]

    _mark_generation(session_id, generation_id)

    if not salm_model:
        _safe_send(channel, {"type": "response", "text": "Model not loaded."})
        if raw_audio_path and os.path.exists(raw_audio_path):
            os.remove(raw_audio_path)
        return

    if not check_rate_limit(session_id):
        _safe_send(channel, {"type": "rejected", "reason": "rate_limit", "generationId": generation_id})
        if raw_audio_path and os.path.exists(raw_audio_path):
            os.remove(raw_audio_path)
        return

    try:
        async with acquire_model_slot(session_id, generation_id, channel=channel):
            if _is_stale(session_id, generation_id):
                logger.info(f"Skipping stale audio gen {generation_id} for session {session_id}")
                if raw_audio_path and os.path.exists(raw_audio_path):
                    os.remove(raw_audio_path)
                return

            clean_audio_path = None
            try:
                uploads_dir = ROOT / "uploads"
                uploads_dir.mkdir(exist_ok=True)
                clean_audio_path = str(uploads_dir / f"processed_{uuid.uuid4().hex}.wav")
                salm_model.process_audio(raw_audio_path, clean_audio_path)

                try:
                    transcript = await _transcribe_path(clean_audio_path, already_in_slot=True)
                    if transcript:
                        _safe_send(channel, {
                            "type": "audio_transcript",
                            "text": transcript.strip(),
                            "generationId": generation_id,
                        })
                except Exception as te:
                    logger.warning(f"Transcript pass failed: {te}")

                t0 = time.perf_counter()
                token_count = 0
                full_response = ""
                current_history = SESSIONS[session_id]["history"]
                stop_check = lambda: _is_stale(session_id, generation_id)
                async for token in stream_generator_in_thread(
                    salm_model.generate_stream,
                    audio_path=clean_audio_path,
                    text_input=prompt or custom_instruction,
                    history=current_history,
                    stop_callback=stop_check,
                    **_effort_overrides(effort_mode, max_tokens),
                ):
                    if _is_stale(session_id, generation_id):
                        logger.info(f"Stale during stream; aborting audio gen {generation_id}")
                        return
                    if not _safe_send(channel, {"type": "token", "text": token, "generationId": generation_id}):
                        logger.warning("DataChannel closed during streaming, stopping.")
                        break
                    full_response += token
                    token_count += 1

                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                _safe_send(channel, {
                    "type": "done", "text": "", "generationId": generation_id,
                    "tokenCount": token_count, "elapsedMs": elapsed_ms,
                })

                clean_response = _strip_chatml_assistant(full_response)
                if not clean_response.strip():
                    logger.warning(f"Empty assistant response for session {session_id}; skipping history append.")
                elif _looks_runaway(clean_response):
                    logger.warning(f"Runaway response detected for session {session_id}; skipping history append.")
                else:
                    SESSIONS[session_id]["history"].append({"role": "user", "content": prompt or "Audio Message"})
                    SESSIONS[session_id]["history"].append({"role": "assistant", "content": clean_response})
                    schedule_save()

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                _safe_send(channel, {"type": "response", "text": f"Error: {str(e)}"})
            finally:
                if clean_audio_path and os.path.exists(clean_audio_path):
                    try:
                        os.remove(clean_audio_path)
                    except Exception:
                        pass
                if raw_audio_path and os.path.exists(raw_audio_path):
                    try:
                        os.remove(raw_audio_path)
                    except Exception:
                        pass
    except QueueFullError:
        _safe_send(channel, {"type": "rejected", "reason": "queue_full", "generationId": generation_id})
        if raw_audio_path and os.path.exists(raw_audio_path):
            try:
                os.remove(raw_audio_path)
            except Exception:
                pass


async def _handle_text_only(channel, state, data):
    """Process a text-only generation request."""
    prompt = data.get("text", "")
    generation_id = data.get("generationId", 0)
    max_tokens = int(data.get("maxTokens", MAX_NEW_TOKENS))
    effort_mode = data.get("effortMode", "normal")
    regenerate = bool(data.get("regenerate", False))
    session_id = state["session_id"]

    _mark_generation(session_id, generation_id)

    if regenerate:
        # `dropPairs` tells us how many user+assistant pairs to roll back
        # from the end of history before re-running. Defaults to 1 so
        # legacy clients regenerate just the last turn.
        drop_pairs = max(1, int(data.get("dropPairs", 1)))
        hist = SESSIONS[session_id]["history"]
        for _ in range(drop_pairs):
            if hist and hist[-1].get("role") == "assistant":
                hist.pop()
            if hist and hist[-1].get("role") == "user":
                hist.pop()
        schedule_save()

    if not salm_model:
        _safe_send(channel, {"type": "response", "text": "Model not loaded."})
        return

    if not check_rate_limit(session_id):
        _safe_send(channel, {"type": "rejected", "reason": "rate_limit", "generationId": generation_id})
        return

    try:
        async with acquire_model_slot(session_id, generation_id, channel=channel):
            if _is_stale(session_id, generation_id):
                logger.info(f"Skipping stale text gen {generation_id} for session {session_id}")
                return

            try:
                t0 = time.perf_counter()
                token_count = 0
                full_response = ""
                current_history = SESSIONS[session_id]["history"]
                stop_check = lambda: _is_stale(session_id, generation_id)
                async for token in stream_generator_in_thread(
                    salm_model.generate_stream,
                    text_input=prompt,
                    history=current_history,
                    stop_callback=stop_check,
                    **_effort_overrides(effort_mode, max_tokens),
                ):
                    if _is_stale(session_id, generation_id):
                        logger.info(f"Stale during stream; aborting text gen {generation_id}")
                        return
                    if not _safe_send(channel, {"type": "token", "text": token, "generationId": generation_id}):
                        logger.warning("DataChannel closed during streaming, stopping.")
                        break
                    full_response += token
                    token_count += 1

                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                _safe_send(channel, {
                    "type": "done", "text": "", "generationId": generation_id,
                    "tokenCount": token_count, "elapsedMs": elapsed_ms,
                })

                clean_response = _strip_chatml_assistant(full_response)
                if not clean_response.strip():
                    logger.warning(f"Empty assistant response for session {session_id}; skipping history append.")
                elif _looks_runaway(clean_response):
                    logger.warning(f"Runaway response detected for session {session_id}; skipping history append.")
                else:
                    SESSIONS[session_id]["history"].append({"role": "user", "content": prompt})
                    SESSIONS[session_id]["history"].append({"role": "assistant", "content": clean_response})
                    schedule_save()
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                _safe_send(channel, {"type": "response", "text": f"Error: {str(e)}"})
    except QueueFullError:
        _safe_send(channel, {"type": "rejected", "reason": "queue_full", "generationId": generation_id})


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
            logger.info(f"Received message: {message}")
            try:
                data = json.loads(message)
            except Exception:
                data = {"type": "text", "text": message}

            mtype = data.get("type")
            if mtype == "stop" and state["handler"]:
                await _handle_audio_stop(channel, state, data)
            elif mtype == "text_only":
                await _handle_text_only(channel, state, data)

    @pc.on("track")
    async def on_track(track):
        if track.kind == "audio":
            logger.info("Audio track received")
            handler = AudioTrackHandler(track)
            state["handler"] = handler
            await handler.start_recording()

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        if pc.connectionState in ("closed", "failed"):
            pcs.discard(pc)
            _SESSION_LATEST_GEN.pop(state["session_id"], None)
            _RATE_HITS.pop(state["session_id"], None)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Wait for ICE gathering — non-trickle clients need the candidates
    # included in the answer rather than negotiated separately.
    retry_count = 0
    while pc.iceGatheringState != "complete" and retry_count < 10:
        await asyncio.sleep(0.2)
        retry_count += 1

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }),
    )

async def upload_audio(request):
    reader = await request.multipart()
    file_written = False
    
    # Use a local uploads directory for visibility
    uploads_dir = ROOT / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    
    filename = str(uploads_dir / f"upload_{uuid.uuid4().hex}.wav")
    size = 0
    text_prompt = ""
    session_id = "default"
    max_tokens = MAX_NEW_TOKENS
    custom_instruction = None
    effort_mode = "normal"

    while True:
        field = await reader.next()
        if field is None:
            break

        if field.name == 'audio':
            with open(filename, 'wb') as f:
                while True:
                    chunk = await field.read_chunk()
                    if not chunk:
                        break
                    size += len(chunk)
                    f.write(chunk)
            file_written = True
        elif field.name == 'text':
            text_prompt = (await field.read(decode=True)).decode('utf-8')
        elif field.name == 'sessionId':
            session_id = (await field.read(decode=True)).decode('utf-8')
        elif field.name == 'maxTokens':
            try:
                max_tokens = int((await field.read(decode=True)).decode('utf-8'))
            except (ValueError, TypeError):
                pass
        elif field.name == 'instruction':
            custom_instruction = (await field.read(decode=True)).decode('utf-8')
        elif field.name == 'effortMode':
            effort_mode = (await field.read(decode=True)).decode('utf-8')

    if not file_written:
        return web.Response(status=400, text="No audio file received")

    logger.info(f"Received file {filename} ({size} bytes) with prompt: {text_prompt} for session {session_id}")

    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"history": []}
    current_history = SESSIONS[session_id]["history"]

    if not check_rate_limit(session_id):
        return web.Response(status=429, text="Rate limit exceeded")

    if salm_model:
        clean_audio_path = None
        try:
            async with acquire_model_slot(session_id, 0, channel=None):
                try:
                    logger.info(f"Starting generation for session {session_id}...")

                    processed_filename = f"processed_{uuid.uuid4().hex}.wav"
                    clean_audio_path = str(uploads_dir / processed_filename)
                    salm_model.process_audio(filename, clean_audio_path)

                    full_response = ""
                    async for token in stream_generator_in_thread(salm_model.generate_stream, audio_path=clean_audio_path, text_input=text_prompt, history=current_history, **_effort_overrides(effort_mode, max_tokens)):
                        full_response += token

                    clean_response = _strip_chatml_assistant(full_response)
                    logger.info(f"Generation complete for session {session_id}")

                    instruction = text_prompt or custom_instruction or salm_model.default_instruction
                    prompt_content = f"{instruction}\n{salm_model.model.audio_locator_tag}\n"

                    user_turn = {
                        "role": "user",
                        "content": prompt_content,
                        "audio": [clean_audio_path]
                    }

                    if not clean_response.strip():
                        logger.warning(f"Empty response for upload session {session_id}; skipping history append.")
                    elif _looks_runaway(clean_response):
                        logger.warning(f"Detected runaway response for upload session {session_id}; skipping history append.")
                    else:
                        SESSIONS[session_id]["history"].append(user_turn)
                        SESSIONS[session_id]["history"].append({"role": "assistant", "content": clean_response})
                        schedule_save()

                    response = clean_response

                except Exception as e:
                    logger.error(f"Error during generation: {e}")
                    if clean_audio_path and os.path.exists(clean_audio_path):
                        os.remove(clean_audio_path)
                    if os.path.exists(filename):
                        os.remove(filename)
                    return web.Response(status=500, text=f"Error processing audio: {str(e)}")
        except QueueFullError:
            if os.path.exists(filename):
                os.remove(filename)
            return web.Response(status=503, text="Server busy, queue full")
    else:
        response = "Model not loaded."

    # Cleanup RAW file
    if os.path.exists(filename):
        os.remove(filename)

    return web.json_response({"text": response})

TRANSCRIBE_INSTRUCTION = (
    "Transcribe the audio verbatim. Output only the transcription, "
    "with no additional commentary, prefix, or quotation marks."
)


async def _transcribe_path(clean_audio_path: str, *, session_id: str = "transcribe", already_in_slot: bool = False) -> str:
    """Run a quick verbatim transcription pass.

    Touches the model. If `already_in_slot=True` the caller is already inside
    `acquire_model_slot()`; otherwise this function acquires its own slot.
    """
    if not salm_model:
        return ""

    def _run():
        try:
            return salm_model.generate(
                audio_path=clean_audio_path,
                text_input=TRANSCRIBE_INSTRUCTION,
                history=None,
                max_new_tokens=128,
            )
        except Exception as e:
            logger.warning(f"Transcription failed: {e}")
            return ""

    loop = asyncio.get_event_loop()
    if already_in_slot:
        return await loop.run_in_executor(None, _run)
    async with acquire_model_slot(session_id, 0, channel=None):
        return await loop.run_in_executor(None, _run)


async def transcribe_audio(request):
    """Standalone transcription endpoint for client-uploaded audio files."""
    if not salm_model:
        return web.json_response({"text": "", "error": "Model not loaded."}, status=503)

    reader = await request.multipart()
    uploads_dir = ROOT / "uploads"
    uploads_dir.mkdir(exist_ok=True)

    raw_path = str(uploads_dir / f"transcribe_raw_{uuid.uuid4().hex}.wav")
    clean_path = str(uploads_dir / f"transcribe_clean_{uuid.uuid4().hex}.wav")
    file_written = False

    try:
        while True:
            field = await reader.next()
            if field is None:
                break
            if field.name == "audio":
                with open(raw_path, "wb") as f:
                    while True:
                        chunk = await field.read_chunk()
                        if not chunk:
                            break
                        f.write(chunk)
                file_written = True

        if not file_written:
            return web.json_response({"text": "", "error": "No audio provided"}, status=400)

        salm_model.process_audio(raw_path, clean_path)
        transcript = await _transcribe_path(clean_path)
        return web.json_response({"text": transcript.strip()})
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return web.json_response({"text": "", "error": str(e)}, status=500)
    finally:
        for p in (raw_path, clean_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def _model_display_name() -> str:
    """Best-effort human label for the loaded model. Prefers the MODEL_PATH
    basename (e.g. 'Canary-Qwen3.5B-Thinking'), falls back to BASE_MODEL."""
    path = (os.getenv("MODEL_PATH") or "").rstrip("/")
    if path:
        name = os.path.basename(path)
        if name and name not in {".", ""}:
            return name
    base = os.getenv("BASE_MODEL")
    if base:
        return base
    return "unknown"


async def model_config(request):
    """Return the model's config.json (commonly the HF model config), or an
    error payload describing why it's unavailable."""
    model_path = os.getenv("MODEL_PATH") or ""
    if not model_path:
        return web.json_response(
            {"error": "MODEL_PATH not set", "model_path": None},
            status=404,
        )
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        return web.json_response(
            {"error": f"config.json not found in {model_path}", "model_path": model_path},
            status=404,
        )
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return web.json_response({"model_path": str(cfg_path), "config": data})
    except json.JSONDecodeError as e:
        return web.json_response(
            {"error": f"config.json is not valid JSON: {e}", "model_path": str(cfg_path)},
            status=500,
        )
    except Exception as e:
        return web.json_response(
            {"error": str(e), "model_path": str(cfg_path)},
            status=500,
        )


async def metrics(request):
    """Lightweight metrics for monitoring queue health + session count."""
    return web.json_response({
        "model_loaded": salm_model is not None,
        "model_name": _model_display_name(),
        "queue": _GENERATION_QUEUE.metrics(),
        "sessions": len(SESSIONS),
        "active_pcs": len(pcs),
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
        "queue_max_depth": QUEUE_MAX_DEPTH,
    })


async def reset_session(request):
    """Wipe the server-side history for a given session id."""
    try:
        params = await request.json()
    except Exception:
        params = {}
    session_id = params.get("sessionId", "default")
    if session_id in SESSIONS:
        SESSIONS[session_id]["history"] = []
        schedule_save()
        logger.info(f"Reset session history for {session_id}")
    return web.json_response({"ok": True, "sessionId": session_id})


async def on_shutdown(app):
    await _flush_sessions()
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer_with_datachannel)
    app.router.add_post("/upload", upload_audio)
    app.router.add_post("/reset-session", reset_session)
    app.router.add_post("/transcribe", transcribe_audio)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/model-config", model_config)
    app.router.add_get("/healthz", lambda r: web.json_response({
        "ok": True,
        "model_loaded": salm_model is not None,
        "model_name": _model_display_name(),
    }))
    app.router.add_static("/static/", path=ROOT / "static", name="static")
    
    logger.info(f"Starting server on port {PORT}")
    web.run_app(app, host=os.getenv("HOST", '0.0.0.0'), port=PORT)
