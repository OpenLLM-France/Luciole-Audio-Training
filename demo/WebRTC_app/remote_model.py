"""Client d'un serveur vLLM, avec la même interface que SALMModel.

La démo peut servir ses modèles de deux façons :

  * en processus — `SALMModel` charge les poids et appelle NeMo directement ;
  * à distance — `RemoteSALMModel` parle à un serveur vLLM par HTTP.

`app.py` ne fait pas la différence : les deux exposent `process_audio`,
`generate`, `generate_stream`, `default_instruction` et `audio_locator_tag`.
Le choix se fait par variable d'environnement (`MODEL_ENDPOINTS` contre
`MODEL_PATHS`), donc au lancement du conteneur, pas dans le code.

Le serveur en face est un `vllm serve` sur un checkpoint SALM exporté, lancé
via `serve_salm.py` (le plugin doit être enregistré avant que le front-end de
l'API ne valide la config). Il expose l'API OpenAI : l'audio passe en
`input_audio` base64 dans les blocs de contenu du message utilisateur.

DEUX DIFFÉRENCES DE COMPORTEMENT, assumées :

1. `no_repeat_ngram_size` (=4 en local) n'existe pas dans les SamplingParams de
   vLLM 0.14 — seule `repetition_penalty` est transmise. Les sorties bouclent un
   peu plus qu'en local. Il faudrait un logits processor côté serveur pour
   retrouver le comportement exact.

2. Le nombre d'audios rejoués est plafonné (`MODEL_MAX_AUDIOS_PER_PROMPT`, 4 par
   défaut) et doit rester ≤ au `--limit-mm-per-prompt` du serveur. Au-delà, les
   tours les plus anciens repassent en texte seul. En local le modèle réentend
   tout l'historique, sans plafond.

L'AUDIO DES TOURS PASSÉS EST RENVOYÉ (voir `_build_messages`). C'est ce qui
permet à un tour de suivi (« et de quoi ça parle ? », sans nouvel audio) de
porter réellement sur l'audio du tour précédent. Renvoyer le base64 ne fait pas
recalculer : vLLM retrouve l'audio dans son cache processeur multimodal (haché
sur le contenu) et le span de tokens correspondant dans son cache de préfixe, du
moment que la conversation ne fait que s'allonger par la fin. Mesuré sur la démo
Luciole-1B : un tour de suivi ne préremplit que ~26 tokens sur 330, l'encodeur
n'est pas rejoué. Ne « simplifiez » donc pas ceci en retirant l'audio.
"""

import base64
import json
import logging
import os
import time

import librosa
import requests
import soundfile as sf

logger = logging.getLogger(__name__)

# Le locator que le plugin vLLM impose (config.py::_AUDIO_PLACEHOLDER). Il est
# validé contre le config.json du checkpoint au démarrage du serveur, donc un
# désaccord se voit là-bas, pas ici.
AUDIO_LOCATOR_TAG = "<|audio|>"

# Réglage anti-boucle de la démo, dans ce que l'API OpenAI de vLLM accepte.
# Voir la note 1 de l'en-tête pour ce qui manque.
_REPETITION_PENALTY = 1.15

# Plafond d'audios par requête. DOIT rester ≤ au `--limit-mm-per-prompt` du serveur
# (voir run_docker.sh) : au-delà, vLLM rejette la requête entière. Le plugin, lui,
# ne limite pas (`get_supported_mm_limits` renvoie {"audio": None}), c'est donc un
# réglage de déploiement des deux côtés, pas une contrainte du modèle.
_MAX_AUDIOS_PER_PROMPT = int(os.getenv("MODEL_MAX_AUDIOS_PER_PROMPT", 4))


class RemoteSALMModel:
    """Un modèle servi par un serveur vLLM, vu comme un SALMModel."""

    def __init__(
        self,
        base_url,
        default_instruction="Listen to the audio and answer the question:",
        name=None,
        ready_timeout=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_instruction = default_instruction
        self.name = name or self.base_url
        self.audio_locator_tag = AUDIO_LOCATOR_TAG
        # Lecture longue : une génération de 128 tokens sur un 8B prend ~9 s, et la
        # toute première d'un serveur frais en prend ~100 de plus (compilation des
        # noyaux Triton). Connexion courte : un serveur absent doit se voir tout de suite.
        self.read_timeout = float(os.getenv("MODEL_ENDPOINT_READ_TIMEOUT", 600))
        self.connect_timeout = float(os.getenv("MODEL_ENDPOINT_CONNECT_TIMEOUT", 10))
        if ready_timeout is None:
            ready_timeout = float(os.getenv("MODEL_ENDPOINT_READY_TIMEOUT", 300))
        self.served_model = self._wait_ready(ready_timeout)
        self._warm_up()

    # ── disponibilité ──

    def _wait_ready(self, timeout):
        """Attend /v1/models et retient l'identifiant servi.

        Le DAG attend déjà que chaque serveur réponde avant de lancer la démo, donc
        en pratique le premier essai passe. L'attente reste utile quand on lance les
        conteneurs à la main, dans le désordre : un 8B met ~2 min à charger.
        """
        deadline = time.monotonic() + timeout
        last_error = None
        while True:
            try:
                r = requests.get(f"{self.base_url}/v1/models", timeout=(self.connect_timeout, 10))
                r.raise_for_status()
                data = r.json().get("data") or []
                if data:
                    served = data[0]["id"]
                    logger.info(f"Serveur vLLM prêt sur {self.base_url} (modèle servi : {served})")
                    return served
                last_error = "aucun modèle déclaré par /v1/models"
            except Exception as e:
                last_error = e
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Serveur vLLM injoignable sur {self.base_url} : {last_error}")
            time.sleep(3)

    def _warm_up(self):
        """Force la compilation des noyaux Triton avant de rendre la main, pour que ce soit le
        démarrage du serveur qui la paie (voir la note sur read_timeout ci-dessus : ~100 s de
        plus pour un gros modèle) et non la première question d'un vrai utilisateur.

        Bloquant par choix : `__init__` ne rend la main qu'une fois cette génération terminée,
        donc `self.served_model` n'est considéré prêt qu'après. Un échec ne doit pas empêcher de
        démarrer — au pire, la compilation aura lieu au premier vrai appel, comme avant.
        """
        logger.info(f"Warmup de {self.name} : génération factice pour compiler les noyaux…")
        started = time.monotonic()
        try:
            self.generate(text_input="Bonjour", max_new_tokens=4)
            logger.info(f"Warmup de {self.name} terminé en {time.monotonic() - started:.1f}s")
        except Exception as e:
            logger.warning(f"Warmup de {self.name} échoué ({e}) — compilation reportée au premier appel réel.")

    # ── audio ──

    def process_audio(self, input_path, output_path, target_sr=16000):
        """Rééchantillonne en mono 16 kHz. Identique à SALMModel : le nettoyage a lieu
        côté démo dans les deux modes, le serveur reçoit déjà du 16 kHz."""
        try:
            audio_array, _ = librosa.load(input_path, sr=target_sr, mono=True)
            sf.write(output_path, audio_array, target_sr)
            return output_path
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise e

    # ── construction de la requête ──

    def _strip_locator(self, content):
        """Retire la balise audio d'un contenu d'historique.

        Les tours passés ont été enregistrés sous la forme `{instruction}\\n<|audio|>\\n`.
        La balise est réinsérée par le template de chat à l'emplacement du bloc audio :
        la garder ici en donnerait DEUX pour un seul audio, et le plugin lève alors
        « Prompt has N placeholders but M audios ». On la retire donc toujours, que le
        tour soit rejoué avec son audio ou non.
        """
        return (content or "").replace(self.audio_locator_tag, "").strip()

    @staticmethod
    def _history_audio_paths(turn):
        """Chemins audio encore lisibles d'un tour d'historique.

        `app.py` enregistre `{"audio": [chemin, ...]}` sur les tours utilisateur. Les
        fichiers sont éphémères (purge de `uploads/`, redémarrage du conteneur, et
        `_serializable_sessions` retire la clé avant d'écrire sessions.json), donc un
        chemin absent est normal : le tour repasse simplement en texte seul.
        """
        raw = turn.get("audio") or []
        if isinstance(raw, str):
            raw = [raw]
        return [p for p in raw if p and os.path.exists(p)]

    def _audio_block(self, path):
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}

    def _build_messages(self, audio_paths, text_input, history):
        raw_history = history if history else []
        # Un seul chemin encore accepté par confort d'appel (transcription, tests) ; le tour
        # courant peut désormais porter plusieurs audios (upload multiple + micro), comme les
        # tours d'historique le font déjà via `_history_audio_paths`.
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
        # Plafonné au budget serveur : le tour courant ne doit pas à lui seul l'épuiser au
        # point de faire déborder `budget` en négatif plus bas (rejeu d'historique alors privé
        # de tout audio sans raison).
        audio_paths = [p for p in (audio_paths or []) if p][:_MAX_AUDIOS_PER_PROMPT]

        # Même repli que SALMModel : les tours système sont fondus dans le premier
        # message utilisateur, pour que les deux modes voient exactement le même prompt.
        messages = []
        system_prefix = ""
        # Tours d'historique rejouables, du plus ancien au plus récent :
        # (indice dans `messages`, chemins audio).
        replayable = []
        for turn in raw_history:
            if turn.get("role") == "system":
                system_prefix += turn.get("content", "") + "\n\n"
                continue
            messages.append({"role": turn["role"], "content": self._strip_locator(turn.get("content"))})
            if turn.get("role") == "user":
                paths = self._history_audio_paths(turn)
                if paths:
                    replayable.append((len(messages) - 1, paths))

        if audio_paths:
            instruction = text_input if text_input else self.default_instruction
        else:
            if not text_input:
                return None
            instruction = text_input

        if not messages and system_prefix:
            instruction = system_prefix + instruction
            system_prefix = ""
        if system_prefix and messages:
            messages[0]["content"] = system_prefix + messages[0]["content"]

        if audio_paths:
            # Texte d'abord, audio ensuite : le rendu du template place la balise à
            # l'endroit du bloc audio, ce qui reproduit le `{instruction}\n<|audio|>\n`
            # de SALMModel. L'ordre inverse fait boucler les deux modèles (mesuré). Plusieurs
            # blocs audio dans un même tour (comparer 2 clips, ou question orale + fichier) :
            # même forme qu'un tour d'historique rejoué avec plusieurs audios.
            content = [{"type": "text", "text": instruction}] + [
                self._audio_block(p) for p in audio_paths
            ]
        else:
            content = instruction

        messages.append({"role": "user", "content": content})

        # Rejeu de l'historique audio, sous le plafond du serveur. On garde les tours
        # les PLUS RÉCENTS : le budget doit d'abord servir l'audio du tour courant,
        # puis remonter le fil. Les tours évincés restent en texte seul — dégradé,
        # jamais une erreur.
        budget = _MAX_AUDIOS_PER_PROMPT - len(audio_paths)
        for idx, paths in reversed(replayable):
            if budget <= 0:
                break
            kept = paths[-budget:]
            budget -= len(kept)
            text = messages[idx]["content"]
            messages[idx]["content"] = [{"type": "text", "text": text}] + [
                self._audio_block(p) for p in kept
            ]
        return messages

    def _payload(self, messages, max_new_tokens, min_new_tokens, temperature, stream):
        payload = {
            "model": self.served_model,
            "messages": messages,
            "max_tokens": int(max_new_tokens),
            "repetition_penalty": _REPETITION_PENALTY,
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        else:
            payload["temperature"] = 0.0
        if min_new_tokens:
            # `min_tokens` est l'extension vLLM ; l'API OpenAI n'a pas d'équivalent.
            payload["min_tokens"] = int(min(int(min_new_tokens), int(max_new_tokens)))
        return payload

    # ── génération ──

    def generate(self, audio_paths=None, text_input=None, history=None, max_new_tokens=360):
        messages = self._build_messages(audio_paths, text_input, history)
        if messages is None:
            return "Please provide text or audio input."
        payload = self._payload(messages, max_new_tokens, None, None, stream=False)
        try:
            r = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=(self.connect_timeout, self.read_timeout),
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:
            logger.error(f"Inference error ({self.name}): {e}")
            raise e

    def generate_stream(self, audio_paths=None, text_input=None, history=None,
                        max_new_tokens=360, stop_callback=None,
                        min_new_tokens=None, temperature=None):
        """Rend les deltas de tokens, comme le TextIteratorStreamer local.

        `stop_callback` est consulté entre deux fragments ; on ferme alors la réponse,
        ce qui annule la requête côté serveur et libère le GPU pour la suivante.
        """
        messages = self._build_messages(audio_paths, text_input, history)
        if messages is None:
            yield "Please provide text or audio input."
            return
        payload = self._payload(messages, max_new_tokens, min_new_tokens, temperature, stream=True)

        try:
            with requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=(self.connect_timeout, self.read_timeout),
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if stop_callback is not None:
                        try:
                            if stop_callback():
                                logger.info(f"Génération annulée ({self.name}), fermeture du flux.")
                                return
                        except Exception:
                            pass
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[len("data:") :].strip()
                    if chunk == "[DONE]":
                        return
                    try:
                        delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                    except (KeyError, IndexError, ValueError) as e:
                        logger.warning(f"Fragment SSE illisible ({self.name}) : {e}")
                        continue
                    if delta:
                        yield delta
        except Exception as e:
            logger.error(f"Streaming error ({self.name}): {e}")
            yield f"Error generating response: {e}"
