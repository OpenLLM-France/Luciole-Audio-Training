#!/usr/bin/env python3

import asyncio
import contextlib
import json
import logging
import os
import re
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

from remote_model import RemoteSALMModel

# Configuration
ROOT = Path(__file__).parent
# Les modèles sont servis par des serveurs vLLM : "nom=http://host:port,nom=http://host:port",
# un serveur par modèle (vLLM n'en sert qu'un par processus). La démo ne charge aucun poids,
# elle n'est qu'un client HTTP — c'est ce qui lui permet d'en exposer plusieurs, avec le
# sélecteur et la comparaison, sans rien payer en mémoire.
#
# Le chargement des poids DANS ce processus (model_handler.SALMModel, NeMo en direct) a été
# retiré le 2026-07-28. Il vit dans l'historique git jusqu'au commit 7db1bbf, avec l'image
# salm-demo:8b qui allait avec. Motifs : décodage x1,2 à x1,8 plus lent, aucun batching, et
# une image entière de contraintes croisées (nemo_automodel, PEFT épinglé par torchao) dont
# le chemin vLLM n'a pas besoin.
MODEL_ENDPOINTS = os.getenv("MODEL_ENDPOINTS", "")
# Chemin du checkpoint, monté en lecture seule et utilisé UNIQUEMENT par /model-config pour
# afficher le config.json dans l'interface. Plus rien ne charge de poids depuis ici.
MODEL_PATH = os.getenv("MODEL_PATH", "")
PORT = int(os.getenv("PORT", 7860))  # HuggingFace Spaces uses port 7860
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 1024))
DEFAULT_INSTRUCTION = os.getenv("DEFAULT_INSTRUCTION", "Listen to the audio and answer the question:")
# Optional system prompt seeded into every new session. Empty (the default) = no system
# turn at all, i.e. the plain demo behaviour. SALM's formatter only knows user/assistant,
# so remote_model folds any system turn into the first user message.
# SYSTEM_PROMPT_FILE wins over SYSTEM_PROMPT (easier to pass a long prompt via a mount).
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")
_sys_prompt_file = os.getenv("SYSTEM_PROMPT_FILE", "")
if _sys_prompt_file:
    SYSTEM_PROMPT = Path(_sys_prompt_file).read_text()


def _parse_model_specs():
    """Découpe MODEL_ENDPOINTS ("nom=url,nom=url") en [(nom, url)].

    Les noms viennent du client (sélecteur) et reviennent dans chaque message de token pour
    router l'affichage : ils doivent être stables et lisibles. Sans nom explicite on prend le
    dernier segment de l'URL, ce qui donne un port — utilisable, mais autant les nommer.
    """
    specs = []
    for chunk in MODEL_ENDPOINTS.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, sep, url = chunk.partition("=")
        if not sep:
            name, url = os.path.basename(chunk.rstrip("/")), chunk
        specs.append((name.strip(), url.strip()))
    return specs


def new_session():
    """Fresh session state.

    L'historique est PAR MODÈLE : en mode comparaison chaque modèle doit voir ses propres
    réponses passées, pas celles de l'autre — sinon on ne compare plus deux modèles mais un
    modèle et un modèle conditionné par son voisin. Les tours utilisateur sont donc dupliqués
    dans chaque historique.
    """
    return {"history": {name: _fresh_history() for name, _ in _parse_model_specs()}}


def _fresh_history():
    return [{"role": "system", "content": SYSTEM_PROMPT}] if SYSTEM_PROMPT.strip() else []

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

# Mise en place des modèles au démarrage : on vérifie que chaque serveur répond, et on retient
# l'identifiant qu'il sert. Un serveur injoignable ne condamne pas les autres — la démo reste
# utilisable avec ceux qui ont répondu, et /healthz dit lesquels.
MODELS = {}
for _name, _url in _parse_model_specs():
    try:
        MODELS[_name] = RemoteSALMModel(_url, default_instruction=DEFAULT_INSTRUCTION, name=_name)
        logger.info(f"Modèle '{_name}' servi par {_url}")
    except Exception as e:
        logger.error(f"Could not reach model '{_name}' at {_url}: {e}")

# Modèle par défaut : le premier déclaré. C'est celui qui répond quand le client ne précise
# rien — donc aussi tout client antérieur au sélecteur.
DEFAULT_MODEL = next(iter(MODELS), None)
# Compat : le code (et les intégrations) qui parlaient d'un modèle unique.
salm_model = MODELS.get(DEFAULT_MODEL) if DEFAULT_MODEL else None


def get_model(name=None):
    """Le modèle demandé, ou celui par défaut. None si aucun serveur n'a répondu."""
    if name and name in MODELS:
        return MODELS[name]
    return MODELS.get(DEFAULT_MODEL) if DEFAULT_MODEL else None


def selected_models(data):
    """Rend [(nom, modèle)] pour une requête : un seul modèle, ou un sous-ensemble (voire tous)
    si comparaison.

    `compare: true` l'emporte sur `model`. En comparaison, le client peut restreindre à un
    sous-ensemble via `compareModels` (liste de noms, ou chaîne "a,b,c") ; sinon tous les
    modèles sont comparés. Les générations se feront en séquence dans l'ordre de MODELS (le GPU
    est unique), et chaque message émis porte son nom de modèle pour que le client sache dans
    quelle colonne l'écrire.
    """
    if data.get("compare") and len(MODELS) > 1:
        # Sous-ensemble optionnel à comparer. On garde l'ordre de MODELS pour un placement de
        # colonnes stable, et on exige au moins deux noms valides — sinon (client antérieur au
        # sélecteur multi-modèles, ou sélection incohérente) on compare TOUS les modèles,
        # l'ancien comportement.
        wanted = data.get("compareModels")
        if isinstance(wanted, str):
            wanted = [w.strip() for w in wanted.split(",") if w.strip()]
        if wanted:
            wanted = set(wanted)
            chosen = [(n, m) for n, m in MODELS.items() if n in wanted]
            if len(chosen) >= 2:
                return chosen
        return list(MODELS.items())
    name = data.get("model")
    model = get_model(name)
    if model is None:
        return []
    return [(name if name in MODELS else DEFAULT_MODEL, model)]


def session_history(session_id: str, model_name: str):
    """Historique d'UN modèle dans une session, créé à la volée.

    À la volée parce que les sessions relues de sessions.json peuvent dater d'une
    configuration où ce modèle n'existait pas — ou de l'époque où l'historique était une
    simple liste (voir _load_sessions_from_disk).
    """
    hist = SESSIONS[session_id].setdefault("history", {})
    return hist.setdefault(model_name, _fresh_history())

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

# Une RTCPeerConnection par session, réutilisée d'un enregistrement à l'autre au lieu d'en
# recréer une à chaque /offer : recréer forçait une négociation ICE/DTLS complète (~10s) pour
# CHAQUE question posée au micro, alors qu'une connexion déjà établie peut juste être
# renégociée (rapide : même transport ICE, cf. bundlePolicy côté client) pour recevoir la
# piste audio suivante.
PC_BY_SESSION = {}

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
        # Migration : `history` était une liste (un seul modèle), c'est maintenant un dict
        # {nom_de_modèle: liste}. Une session écrite par l'ancienne version est rattachée au
        # modèle par défaut plutôt que jetée.
        for sess in data.values():
            if isinstance(sess, dict) and isinstance(sess.get("history"), list):
                sess["history"] = {DEFAULT_MODEL: sess["history"]} if DEFAULT_MODEL else {}
        return data
    except Exception as e:
        logger.warning(f"Could not load sessions.json: {e}")
        return {}


# Durée de vie des audios de conversation. Ils sont conservés au-delà de la requête
# qui les a produits pour que les tours de suivi puissent les rejouer (voir
# remote_model._build_messages), donc quelque chose doit finir par les effacer.
# Au-delà de ce délai, un tour repasse en texte seul : dégradé, jamais une erreur.
UPLOAD_TTL_S = float(os.getenv("UPLOAD_TTL_SECONDS", 6 * 3600))


def _prune_uploads():
    """Efface les audios de conversation périmés.

    Purge par ÂGE et non par référence : les historiques vivent en mémoire, sont
    rechargés amputés de leurs chemins au redémarrage, et une session n'est jamais
    explicitement close — un balayage des références laisserait donc fuir les
    fichiers des sessions abandonnées, c'est-à-dire la majorité.
    """
    uploads_dir = ROOT / "uploads"
    if not uploads_dir.is_dir():
        return
    cutoff = time.time() - UPLOAD_TTL_S
    for path in uploads_dir.glob("processed_*.wav"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def _serializable_sessions():
    """Strip transient fields (audio paths point to files we prune by age)."""
    out = {}
    for sid, sess in SESSIONS.items():
        histories = {}
        for model_name, turns in (sess.get("history") or {}).items():
            histories[model_name] = [
                {k: v for k, v in turn.items() if k != "audio"} for turn in turns
            ]
        out[sid] = {"history": histories}
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


# Annulation explicite, par session. Avant, une requête plus récente rendait automatiquement
# la précédente « stale » et l'avortait : impossible d'empiler deux questions dans une même
# conversation, la seconde tuait la première. Désormais tout s'empile (une génération à la
# fois, cf. GenerationQueue) et RIEN ne s'annule sans que l'utilisateur le demande — bouton
# Stop (toute la conversation) ou suppression d'un message (cette requête-là).
_CANCELLED: dict = {}   # session_id -> {gen_id annulés}


def _cancel_generation(session_id: str, gen_id: int):
    _CANCELLED.setdefault(session_id, set()).add(gen_id)


def _is_cancelled(session_id: str, gen_id: int) -> bool:
    return gen_id in _CANCELLED.get(session_id, ())


def _forget_cancelled(session_id: str, gen_id: int):
    """Une génération terminée n'a plus à figurer dans l'ensemble des annulées — sinon il
    grossit indéfiniment sur une conversation longue."""
    ids = _CANCELLED.get(session_id)
    if ids:
        ids.discard(gen_id)
        if not ids:
            _CANCELLED.pop(session_id, None)


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
        # Plus d'éviction des requêtes d'une même session : elles font la queue, comme celles
        # des autres sessions. Une seule génération tourne à la fois (le GPU est unique), le
        # reste attend son tour, et seule une annulation explicite retire une entrée.
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

    def cancel(self, session_id, gen_id=None) -> list:
        """Annule les requêtes d'une session : une précise, ou toutes si gen_id est None.

        Les entrées EN ATTENTE sont retirées de la file (leur `active_event` est armé pour
        débloquer l'awaiter, qui verra l'annulation et s'arrêtera là). L'entrée ACTIVE, elle,
        ne peut pas être retirée — sa génération tourne dans un thread ; on rend son gen_id
        pour que l'appelant le marque annulé, et le stop_callback l'arrêtera au prochain token.
        """
        def matches(e):
            return e.session_id == session_id and (gen_id is None or e.gen_id == gen_id)

        cancelled = []
        waiting = [e for e in self.entries if matches(e)]
        self.entries = [e for e in self.entries if not matches(e)]
        for e in waiting:
            cancelled.append(e.gen_id)
            e.active_event.set()
        if self.active is not None and matches(self.active):
            cancelled.append(self.active.gen_id)
        self._maybe_activate()
        self._broadcast()
        return cancelled

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
        # La génération est finie (menée à terme ou annulée) : son drapeau d'annulation n'a
        # plus d'utilité et ne doit pas s'accumuler sur une conversation longue.
        _forget_cancelled(entry.session_id, entry.gen_id)


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
    # Le client envoie sa conversation courante à CHAQUE message. Se fier au seul
    # session_id figé au /offer liait tout l'historique à la conversation ouverte au
    # moment de la connexion : changer de conversation ne changeait rien côté serveur, et
    # le modèle continuait de voir le contexte de la précédente. Repli sur la valeur du
    # /offer pour les clients qui ne l'envoient pas.
    session_id = data.get("sessionId") or state["session_id"]
    if session_id not in SESSIONS:
        SESSIONS[session_id] = new_session()

    # Plus de marquage « la dernière gagne » : cette requête s'ajoute simplement à la file.
    # Elle ne s'arrêtera que sur annulation explicite (Stop, ou suppression du message).

    targets = selected_models(data)
    if not targets:
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
            if _is_cancelled(session_id, generation_id):
                logger.info(f"Skipping stale audio gen {generation_id} for session {session_id}")
                if raw_audio_path and os.path.exists(raw_audio_path):
                    os.remove(raw_audio_path)
                return

            clean_audio_path = None
            audio_referenced = False
            try:
                uploads_dir = ROOT / "uploads"
                uploads_dir.mkdir(exist_ok=True)
                _prune_uploads()
                clean_audio_path = str(uploads_dir / f"processed_{uuid.uuid4().hex}.wav")
                # Le rééchantillonnage ne dépend pas du modèle : fait une fois, le fichier
                # nettoyé sert ensuite aux deux générations en mode comparaison.
                targets[0][1].process_audio(raw_audio_path, clean_audio_path)

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

                for model_name, model in targets:
                    t0 = time.perf_counter()
                    token_count = 0
                    full_response = ""
                    current_history = session_history(session_id, model_name)
                    stop_check = lambda: _is_cancelled(session_id, generation_id)
                    async for token in stream_generator_in_thread(
                        model.generate_stream,
                        audio_path=clean_audio_path,
                        text_input=prompt or custom_instruction,
                        history=current_history,
                        stop_callback=stop_check,
                        **_effort_overrides(effort_mode, max_tokens),
                    ):
                        if _is_cancelled(session_id, generation_id):
                            logger.info(f"Stale during stream; aborting audio gen {generation_id}")
                            return
                        if not _safe_send(channel, {"type": "token", "text": token,
                                                    "generationId": generation_id,
                                                    "model": model_name}):
                            logger.warning("DataChannel closed during streaming, stopping.")
                            break
                        full_response += token
                        token_count += 1

                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    _safe_send(channel, {
                        "type": "done", "text": "", "generationId": generation_id,
                        "tokenCount": token_count, "elapsedMs": elapsed_ms,
                        "model": model_name,
                    })

                    clean_response = _strip_chatml_assistant(full_response)
                    if not clean_response.strip():
                        logger.warning(f"Empty assistant response for session {session_id}; skipping history append.")
                    elif _looks_runaway(clean_response):
                        logger.warning(f"Runaway response detected for session {session_id}; skipping history append.")
                    else:
                        # `audio` fait tenir le tour de suivi : sans lui, une question
                        # posée au tour d'après (« et de quoi ça parle ? ») n'a plus
                        # aucun audio dans le contexte. Même convention que le chemin
                        # upload. Le fichier doit donc SURVIVRE à cette requête.
                        current_history.append({
                            "role": "user",
                            "content": prompt or custom_instruction,
                            "audio": [clean_audio_path],
                        })
                        current_history.append({"role": "assistant", "content": clean_response})
                        audio_referenced = True
                        schedule_save()

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                _safe_send(channel, {"type": "response", "text": f"Error: {str(e)}"})
            finally:
                # Conservé seulement s'il est référencé par un historique ; sinon il ne
                # servira jamais et part tout de suite. La purge par âge (_prune_uploads)
                # se charge des fichiers référencés, une fois périmés.
                if not audio_referenced and clean_audio_path and os.path.exists(clean_audio_path):
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
    # Le client envoie sa conversation courante à CHAQUE message. Se fier au seul
    # session_id figé au /offer liait tout l'historique à la conversation ouverte au
    # moment de la connexion : changer de conversation ne changeait rien côté serveur, et
    # le modèle continuait de voir le contexte de la précédente. Repli sur la valeur du
    # /offer pour les clients qui ne l'envoient pas.
    session_id = data.get("sessionId") or state["session_id"]
    if session_id not in SESSIONS:
        SESSIONS[session_id] = new_session()


    targets = selected_models(data)
    if not targets:
        _safe_send(channel, {"type": "response", "text": "Model not loaded."})
        return

    if regenerate:
        # `dropPairs` tells us how many user+assistant pairs to roll back
        # from the end of history before re-running. Defaults to 1 so
        # legacy clients regenerate just the last turn.
        # Chaque modèle a son propre historique : on déroule ceux qui sont concernés.
        drop_pairs = max(1, int(data.get("dropPairs", 1)))
        for model_name, _ in targets:
            hist = session_history(session_id, model_name)
            for _ in range(drop_pairs):
                if hist and hist[-1].get("role") == "assistant":
                    hist.pop()
                if hist and hist[-1].get("role") == "user":
                    hist.pop()
        schedule_save()

    if not check_rate_limit(session_id):
        _safe_send(channel, {"type": "rejected", "reason": "rate_limit", "generationId": generation_id})
        return

    try:
        async with acquire_model_slot(session_id, generation_id, channel=channel):
            if _is_cancelled(session_id, generation_id):
                logger.info(f"Skipping stale text gen {generation_id} for session {session_id}")
                return

            # En comparaison, les modèles passent l'un après l'autre : le GPU est unique,
            # les paralléliser ne ferait que les ralentir mutuellement.
            for model_name, model in targets:
                try:
                    t0 = time.perf_counter()
                    token_count = 0
                    full_response = ""
                    current_history = session_history(session_id, model_name)
                    stop_check = lambda: _is_cancelled(session_id, generation_id)
                    async for token in stream_generator_in_thread(
                        model.generate_stream,
                        text_input=prompt,
                        history=current_history,
                        stop_callback=stop_check,
                        **_effort_overrides(effort_mode, max_tokens),
                    ):
                        if _is_cancelled(session_id, generation_id):
                            logger.info(f"Stale during stream; aborting text gen {generation_id}")
                            return
                        if not _safe_send(channel, {"type": "token", "text": token,
                                                    "generationId": generation_id,
                                                    "model": model_name}):
                            logger.warning("DataChannel closed during streaming, stopping.")
                            break
                        full_response += token
                        token_count += 1

                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    _safe_send(channel, {
                        "type": "done", "text": "", "generationId": generation_id,
                        "tokenCount": token_count, "elapsedMs": elapsed_ms,
                        "model": model_name,
                    })

                    clean_response = _strip_chatml_assistant(full_response)
                    if not clean_response.strip():
                        logger.warning(f"Empty assistant response for session {session_id}; skipping history append.")
                    elif _looks_runaway(clean_response):
                        logger.warning(f"Runaway response detected for session {session_id}; skipping history append.")
                    else:
                        current_history.append({"role": "user", "content": prompt})
                        current_history.append({"role": "assistant", "content": clean_response})
                        schedule_save()
                except Exception as e:
                    logger.error(f"Streaming error ({model_name}): {e}")
                    _safe_send(channel, {"type": "response", "text": f"Error: {str(e)}",
                                         "model": model_name})
    except QueueFullError:
        _safe_send(channel, {"type": "rejected", "reason": "queue_full", "generationId": generation_id})


# Empreinte automatique des assets. aiohttp sert /static/ SANS Cache-Control : le navigateur
# applique alors sa fraîcheur heuristique et peut resservir un fichier périmé pendant des
# heures. Le `?v=` écrit à la main dans index.html ne protège que ce qu'on a pensé à bumper —
# ça a coûté deux faux diagnostics (un client.js caché qui masquait le sélecteur de modèle,
# puis un styles.css caché qui empilait les réponses au lieu de les mettre en colonnes).
# On estampille donc à la volée, à partir du mtime et de la taille du fichier : plus rien à
# bumper, et l'URL change exactement quand le contenu change.
_STATIC_ASSET_RE = re.compile(
    r'(?P<attr>href|src)="(?P<path>/static/[^"?]+\.(?:css|js))(?:\?[^"]*)?"')


def _stamp_static_assets(html: str) -> str:
    def repl(m):
        rel = m.group("path")
        try:
            st = (ROOT / rel.lstrip("/")).stat()
        except OSError:
            # Asset absent : on laisse l'URL telle quelle plutôt que de casser la page.
            return m.group(0)
        return f'{m.group("attr")}="{rel}?v={int(st.st_mtime)}-{st.st_size}"'
    return _STATIC_ASSET_RE.sub(repl, html)


async def index(request):
    content = _stamp_static_assets(open(str(ROOT / 'static' / 'index.html')).read())
    # L'index porte les empreintes : le mettre en cache reviendrait à cacher les versions
    # d'assets, et on retomberait exactement dans le problème qu'on vient de corriger.
    return web.Response(content_type='text/html', text=content,
                        headers={"Cache-Control": "no-cache"})

async def offer(request):
    # Legacy endpoint, redirect to new logic if needed or just keep as is
    return await offer_with_datachannel(request)

async def offer_with_datachannel(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
    session_id = params.get('sessionId', 'default')
    
    if session_id not in SESSIONS:
        SESSIONS[session_id] = new_session()

    existing = PC_BY_SESSION.get(session_id)
    if existing is not None and existing["pc"].connectionState in ("closed", "failed"):
        pcs.discard(existing["pc"])
        PC_BY_SESSION.pop(session_id, None)
        existing = None

    if existing is not None:
        # Renégociation sur la connexion déjà ouverte : ne PAS ré-attacher les handlers
        # (déjà branchés la première fois) ni recréer state, sous peine de doublons.
        pc = existing["pc"]
        state = existing["state"]
    else:
        pc = RTCPeerConnection()
        pcs.add(pc)

        # Store state for this connection
        state = {
            "handler": None,
            "channel": None,
            "text_input": "",
            "session_id": session_id # Reference to session ID
        }
        PC_BY_SESSION[session_id] = {"pc": pc, "state": state}

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
                PC_BY_SESSION.pop(state["session_id"], None)
                _CANCELLED.pop(state["session_id"], None)
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
    model_choice = None
    compare = False
    compare_models = None
    regenerate = False
    drop_pairs = 1
    # 0 = client antérieur, qui n'en envoie pas : la requête reste alors non annulable, comme
    # avant. C'est le chemin qu'emprunte tout accès distant (le micro WebRTC ne traverse pas un
    # tunnel ssh), donc celui où Stop compte le plus.
    generation_id = 0

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
        elif field.name == 'model':
            model_choice = (await field.read(decode=True)).decode('utf-8')
        elif field.name == 'compare':
            compare = (await field.read(decode=True)).decode('utf-8').lower() in ("1", "true", "yes", "on")
        elif field.name == 'compareModels':
            # Sous-ensemble à comparer, "a,b,c" — selected_models le découpe et retombe sur
            # « tous » s'il reste moins de deux noms valides.
            compare_models = (await field.read(decode=True)).decode('utf-8')
        elif field.name == 'regenerate':
            regenerate = (await field.read(decode=True)).decode('utf-8').lower() in ("1", "true", "yes", "on")
        elif field.name == 'generationId':
            try:
                generation_id = int((await field.read(decode=True)).decode('utf-8'))
            except (ValueError, TypeError):
                pass
        elif field.name == 'dropPairs':
            try:
                drop_pairs = int((await field.read(decode=True)).decode('utf-8'))
            except (ValueError, TypeError):
                pass

    if not file_written:
        return web.Response(status=400, text="No audio file received")

    logger.info(f"Received file {filename} ({size} bytes) with prompt: {text_prompt} for session {session_id}")

    if session_id not in SESSIONS:
        SESSIONS[session_id] = new_session()

    if not check_rate_limit(session_id):
        return web.Response(status=429, text="Rate limit exceeded")

    targets = selected_models({"model": model_choice, "compare": compare,
                               "compareModels": compare_models})

    # Renvoi d'un tour audio déjà joué (bouton Retry du client) : on déroule l'historique
    # d'autant de paires user+assistant que le client vient d'en jeter, sinon le modèle
    # reverrait l'ancien tour EN PLUS du nouveau. Même logique que _handle_text_only, appliquée
    # à chaque modèle visé puisque les historiques sont séparés.
    if regenerate:
        for model_name, _ in targets:
            hist = session_history(session_id, model_name)
            for _ in range(max(1, drop_pairs)):
                if hist and hist[-1].get("role") == "assistant":
                    hist.pop()
                if hist and hist[-1].get("role") == "user":
                    hist.pop()
        schedule_save()

    # Réponses par modèle. L'upload n'est pas streamé : on rend les deux d'un coup, ce qui
    # rend la comparaison plus simple à afficher que deux flux entrelacés.
    responses = {}
    # Chronométrage PAR MODÈLE. Sans lui, la comparaison est trompeuse : les deux bulles
    # apparaissent ensemble à la fin, ce qui donne l'impression que les modèles ont mis le
    # même temps alors qu'on a attendu la somme des deux. Même forme que le message `done`
    # du flux temps réel, pour que le client n'ait qu'un seul rendu à écrire.
    stats = {}
    if targets:
        clean_audio_path = None
        try:
            async with acquire_model_slot(session_id, generation_id, channel=None):
                if generation_id and _is_cancelled(session_id, generation_id):
                    # Annulée pendant l'attente en file : on ne génère rien du tout.
                    logger.info(f"Upload {generation_id} annulé avant génération.")
                    if os.path.exists(filename):
                        os.remove(filename)
                    return web.json_response({"text": "", "responses": {}, "cancelled": True})
                try:
                    logger.info(f"Starting generation for session {session_id}...")

                    _prune_uploads()
                    processed_filename = f"processed_{uuid.uuid4().hex}.wav"
                    clean_audio_path = str(uploads_dir / processed_filename)
                    targets[0][1].process_audio(filename, clean_audio_path)

                    stop_check = (lambda: _is_cancelled(session_id, generation_id)) \
                        if generation_id else None
                    for model_name, model in targets:
                        current_history = session_history(session_id, model_name)
                        full_response = ""
                        t0 = time.perf_counter()
                        token_count = 0
                        async for token in stream_generator_in_thread(
                            model.generate_stream,
                            audio_path=clean_audio_path,
                            text_input=text_prompt or custom_instruction,
                            history=current_history,
                            stop_callback=stop_check,
                            **_effort_overrides(effort_mode, max_tokens),
                        ):
                            full_response += token
                            token_count += 1
                        # Un fragment SSE = un token côté vLLM ; c'est le même comptage que
                        # dans le flux temps réel, donc les deux chemins sont comparables.
                        stats[model_name] = {
                            "tokenCount": token_count,
                            "elapsedMs": int((time.perf_counter() - t0) * 1000),
                        }

                        clean_response = _strip_chatml_assistant(full_response)
                        logger.info(f"Generation complete for session {session_id} ({model_name})")

                        instruction = text_prompt or custom_instruction or model.default_instruction
                        prompt_content = f"{instruction}\n{model.audio_locator_tag}\n"

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
                            current_history.append(user_turn)
                            current_history.append({"role": "assistant", "content": clean_response})
                            schedule_save()

                        responses[model_name] = clean_response

                    response = responses.get(targets[0][0], "")

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

    # `text` reste la réponse du premier modèle : les clients d'avant le sélecteur
    # continuent de marcher sans rien savoir de `responses`.
    return web.json_response({"text": response, "responses": responses, "stats": stats})

TRANSCRIBE_INSTRUCTION = (
    "Transcribe the audio verbatim. Output only the transcription, "
    "with no additional commentary, prefix, or quotation marks."
)


async def _transcribe_path(clean_audio_path: str, *, session_id: str = "transcribe", already_in_slot: bool = False) -> str:
    """Run a quick verbatim transcription pass.

    Touches the model. If `already_in_slot=True` the caller is already inside
    `acquire_model_slot()`; otherwise this function acquires its own slot.

    Toujours le modèle par défaut, même en comparaison : c'est la transcription de CE QUE
    L'UTILISATEUR A DIT, affichée une fois au-dessus des réponses. La faire varier selon le
    modèle comparé n'aurait pas de sens, et la faire deux fois doublerait l'attente.
    """
    model = get_model()
    if model is None:
        return ""

    def _run():
        try:
            return model.generate(
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
    if get_model() is None:
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

        get_model().process_audio(raw_path, clean_path)
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
    basename (e.g. 'Canary-Qwen3.5B-Thinking'), falls back to BASE_MODEL.

    Avec plusieurs modèles, les noms du registre l'emportent : le basename de MODEL_PATH
    vaudrait "model" (le point de montage figé du Dockerfile) pour tout le monde.
    """
    if MODELS:
        return " + ".join(MODELS)
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
        "model_loaded": bool(MODELS),
        "model_name": _model_display_name(),
        "models": list(MODELS),
        "default_model": DEFAULT_MODEL,
        "backend": "vllm",
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
        # Repart d'un dict vide plutôt que d'une liste : l'historique est par modèle depuis
        # l'ajout du sélecteur, et session_history() recréera ce qu'il faut à la volée.
        SESSIONS[session_id]["history"] = {}
        schedule_save()
        logger.info(f"Reset session history for {session_id}")
    return web.json_response({"ok": True, "sessionId": session_id})


async def cancel_generation(request):
    """Annule des générations d'UNE conversation.

    `generationId` absent = tout ce que cette conversation a en cours ou en attente (bouton
    Stop). Présent = cette requête-là seulement (suppression d'un message), les autres restent
    dans la file. C'est cette distinction qui rend l'empilement utilisable : on retire une
    question de la file sans renoncer aux suivantes.
    """
    try:
        params = await request.json()
    except Exception:
        params = {}
    session_id = params.get("sessionId", "default")
    gen_id = params.get("generationId")

    # Les entrées en attente sortent de la file ; l'active ne peut qu'être marquée, sa boucle
    # verra le drapeau au prochain token (stop_callback de model_handler).
    cancelled = _GENERATION_QUEUE.cancel(session_id, gen_id)
    if gen_id is not None and gen_id not in cancelled:
        # Pas encore dans la file : la requête peut être en vol côté client, ou déjà finie. On
        # marque quand même — si elle arrive, elle s'arrêtera aussitôt.
        cancelled.append(gen_id)
    for cid in cancelled:
        _cancel_generation(session_id, cid)

    logger.info(f"Annulation session {session_id} : {cancelled or 'rien à annuler'}")
    return web.json_response({"ok": True, "cancelled": cancelled})


async def delete_turn(request):
    """Retire UN tour (user + assistant) de l'historique serveur d'une conversation.

    Le client sait supprimer un message au milieu de sa conversation ; sans ce pendant côté
    serveur, le modèle continuerait de voir dans son contexte un tour que l'utilisateur a
    effacé. Chaque historique de modèle est une suite de paires (user, assistant) : une réponse
    vide ou en boucle n'est jamais ajoutée, donc jamais de paire dépareillée.

    `pairIndex` est l'index du tour côté client. Les deux historiques peuvent avoir divergé
    (une génération annulée n'ajoute rien ici alors que le message existe là-bas), on vérifie
    donc le texte avant de supprimer : `userText` doit s'y retrouver. Sinon on ne touche à rien
    et on le dit — mieux vaut un tour de trop dans le contexte qu'un tour innocent supprimé.
    """
    try:
        params = await request.json()
    except Exception:
        params = {}
    session_id = params.get("sessionId", "default")
    pair_index = params.get("pairIndex")
    user_text = (params.get("userText") or "").strip()

    if session_id not in SESSIONS or not isinstance(pair_index, int) or pair_index < 0:
        return web.json_response({"ok": False, "reason": "bad_request"}, status=400)

    result = {}
    for model_name, hist in (SESSIONS[session_id].get("history") or {}).items():
        user_positions = [i for i, turn in enumerate(hist) if turn.get("role") == "user"]
        if pair_index >= len(user_positions):
            result[model_name] = "hors_limites"
            continue
        pos = user_positions[pair_index]
        stored = (hist[pos].get("content") or "").strip()
        # Un tour audio stocke l'instruction, pas le texte tapé : on ne compare que si le
        # client nous a donné quelque chose à comparer.
        if user_text and user_text not in stored:
            result[model_name] = "texte_different"
            continue
        end = pos + 1
        if end < len(hist) and hist[end].get("role") == "assistant":
            end += 1
        del hist[pos:end]
        result[model_name] = "supprime"

    schedule_save()
    logger.info(f"Suppression du tour {pair_index} de {session_id} : {result}")
    return web.json_response({"ok": True, "models": result})


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
    app.router.add_post("/cancel", cancel_generation)
    app.router.add_post("/delete-turn", delete_turn)
    app.router.add_post("/transcribe", transcribe_audio)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/model-config", model_config)
    # model_loaded : vrai dès qu'AU MOINS un modèle a chargé. C'est ce que sonde le DAG
    # (DemoReadySensor) ; une démo qui a perdu un modèle sur deux reste utilisable, et
    # `models` dit lesquels ont réellement chargé.
    app.router.add_get("/healthz", lambda r: web.json_response({
        "ok": True,
        "model_loaded": bool(MODELS),
        "model_name": _model_display_name(),
        "models": list(MODELS),
        "default_model": DEFAULT_MODEL,
        "backend": "vllm",
    }))
    app.router.add_static("/static/", path=ROOT / "static", name="static")
    
    logger.info(f"Starting server on port {PORT}")
    web.run_app(app, host=os.getenv("HOST", '0.0.0.0'), port=PORT)
