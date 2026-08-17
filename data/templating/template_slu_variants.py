"""SLU-variants templater: generate (question, audio, answer) triples for a
fixed set of timestamp-grounded tasks (word→time, time→sentence, …).

This module owns the task-specific prompt/answer templates (in EN and FR) and
the `build_variant` / `build_negative_variant` dispatch. Generic helpers
(sentence/span extraction, multilingual joiners, the streaming `process_split`
driver, JSON answer schemas, etc.) live in `template_qa_common.py`.
"""
import argparse
import random
from pathlib import Path

from template_qa_common import (
    ANSWER_JSON_SCHEMAS,
    PURE_NEG_INVALID_TIME,
    PURE_NEG_NOT_FOUND,
    RANGE_MODE_PROB,
    TRAILING_GAP_TOLERANCE_S,
    audio_duration,
    cased_answer_phrase,
    choose_template,
    dedup_sentences_for_occurrences,
    extract_qa_text,
    extract_sentence,
    find_all_occurrences,
    get_document_audio,
    is_alnum_token,
    locate_answer_span,
    phrase_end_time,
    pick_oov_word,
    process_split,
    pure_times,
    substitute_multi_sentence,
    substitute_multi_time,
    wrap_record,
)


TASKS = (
    "word2time", "time2sentence", "time2word", "word2sentence",
    "answer_with_source", "answer_with_time", "format_json_answer",
)

# Probability that a `word2time` sample with multiple occurrences asks only for
# the FIRST occurrence (the former standalone `word2time_first` task, now merged
# in as a sub-variant) instead of reporting all timestamps.
FIRST_SUBVARIANT_PROB = 0.4


# ---------------------------------------------------------------------------
# Templates: per-language → per-task. Each template is (text, kind). kind="word"
# templates use the noun "word" and are only chosen when the answer is a single
# token. kind="any" works for both single words and multi-word phrases.
# ---------------------------------------------------------------------------

QUESTION_TEMPLATES = {
    "en": {
        "word2time": [
            ("At what time in the audio is the word \"{word}\" spoken?", "word"),
            ("When does the speaker talk about {word}?", "any"),
            ("Find the word {word} in the audio and tell me its timestamp.", "word"),
            ("Locate \"{word}\" in the recording. When does it occur?", "any"),
            ("At which second is {word} pronounced?", "any"),
            ("Give me the timestamp where '{word}' appears.", "any"),
            ("Where in the clip can I hear \"{word}\"?", "any"),
            ("Tell me when they talk about {word} in the audio.", "any"),
            ("I want the start time of the word {word}. What is it?", "word"),
            ("Pinpoint the moment when ({word}) is uttered.", "any"),
            ("Can you tell me at what second \"{word}\" is said?", "any"),
            ("Target word: {word}. When is it spoken?", "word"),
        ],
        "word2time_first": [
            ("What is the first timestamp at which the word \"{word}\" is spoken?", "word"),
            ("When is {word} first said in the audio?", "any"),
            ("Find the first occurrence of {word} in the recording.", "any"),
            ("At what time does {word} first appear?", "any"),
            ("Give me the first timestamp at which '{word}' appears.", "any"),
            ("When is the word {word} first pronounced?", "word"),
            ("What is the earliest second at which \"{word}\" is uttered?", "any"),
            ("Pinpoint the first moment {word} is spoken.", "any"),
            ("At which second does the speaker first say {word}?", "any"),
            ("Target word: {word}. When does it first appear?", "word"),
        ],
        "time2sentence": [
            ("What sentence is being spoken around {time:.1f} seconds?", "any"),
            ("Around {time:.1f}s, what sentence do you hear?", "any"),
            ("Transcribe the full sentence containing the {time:.1f}s mark.", "any"),
            ("Give me the sentence that spans {time:.1f} seconds.", "any"),
            ("Listen near {time:.1f}s, what full sentence is the speaker saying?", "any"),
            ("Report the sentence that includes the {time:.1f}-second mark.", "any"),
            ("At roughly {time:.1f}s, give me the entire sentence being spoken.", "any"),
            ("Which sentence contains the {time:.1f}s moment?", "any"),
            ("Around {time:.1f}s, can you transcribe the full sentence?", "any"),
            ("What sentence spans time {time:.1f}s in the recording?", "any"),
            ("The {time:.1f}s mark falls within a sentence, transcri it fully.", "any"),
            ("Give me the complete sentence that covers {time:.1f}s.", "any"),
        ],
        "time2word": [
            ("What word is spoken at {time:.1f} seconds?", "word"),
            ("Which word does the speaker say at {time:.1f}s?", "word"),
            ("At {time:.1f} seconds, what word can you hear?", "word"),
            ("Tell me the word uttered at the {time:.1f}-second mark.", "word"),
            ("Which word corresponds to {time:.1f}s in the recording?", "word"),
            ("At around {time:.1f}s, what is the speaker saying (single word)?", "word"),
            ("Give me the word pronounced at {time:.1f} seconds.", "word"),
            ("What word is at timestamp {time:.1f}?", "word"),
            ("Listen at {time:.1f}s and report the word you hear.", "word"),
            ("What is being said at {time:.1f} seconds?", "any"),
            ("What do you hear at {time:.1f}s?", "any"),
            ("Report what the speaker utters at {time:.1f} seconds.", "any"),
            ("What is pronounced at {time:.1f}s?", "any"),
        ],
        "word2sentence": [
            ("Which sentence in the audio contains the word \"{word}\"?", "word"),
            ("Find the sentence where {word} is spoken and transcribe it.", "any"),
            ("Give me the full sentence that includes {word}.", "any"),
            ("In which sentence does the speaker use \"{word}\"?", "any"),
            ("Transcribe the sentence containing '{word}'.", "any"),
            ("What sentence does {word} appear in?", "any"),
            ("Report the sentence in which {word} is said.", "any"),
            ("\"{word}\" is somewhere in the audio, give me the sentence it's part of.", "any"),
            ("Locate {word} and transcribe the surrounding sentence.", "any"),
            ("Target word: {word}. Give me the sentence it appears in.", "word"),
        ],
        "answer_with_source": [
            ("Answer the following question from the audio, and cite the source sentence with its timestamp: \"{question}\"", "any"),
            ("Listen to the audio and answer: {question}. Also give me the sentence it comes from and when it's spoken.", "any"),
            ("\"{question}\" Can you answer this using the recording and tell me the supporting sentence and its start time.", "any"),
            ("From the audio, answer this and attribute your answer to a specific sentence plus timestamp. Question: {question}", "any"),
            ("Give me the answer to \"{question}\" along with the sentence in the audio that contains it and the time at which that sentence begins.", "any"),
            ("Answer the question based on the recording. Include the source sentence and when it is said. Question: {question}", "any"),
            ("Please answer '{question}' and back it up with the source sentence from the audio and its timestamp.", "any"),
            ("Using the audio, answer the question: {question}. Cite the sentence the answer comes from and the time it starts.", "any"),
        ],
        "answer_with_time": [
            ("Answer the following question from the audio and give me the timestamp where the answer is spoken: \"{question}\"", "any"),
            ("Listen to the audio and answer: {question}. When is the answer said?", "any"),
            ("Question: {question}. Answer this and tell me when the answer is spoken in the audio.", "any"),
            ("From the audio, answer \"{question}\" and report the timestamp of the answer.", "any"),
            ("Give me the answer to this question along with the time at which it is spoken: {question}", "any"),
            ("Answer the question based on the recording. Include the time the answer is said. Question: {question}", "any"),
            ("Please answer '{question}' and tell me when the answer appears in the audio.", "any"),
            ("Using the audio, answer this and give the timestamp of the answer: {question}", "any"),
        ],
        # Every prompt embeds {example} — the literal JSON shape picked from
        # ANSWER_JSON_SCHEMAS — and spells out that the answer, the timestamp
        # where it's spoken, and the source sentence must all be filled in (the
        # example alone wouldn't tell the model to actually retrieve them).
        "format_json_answer": [
            ("Answer the following question from the audio and return JSON containing the answer, the timestamp where it's spoken, and the source sentence, in this shape: {example}. Question: \"{question}\"", "any"),
            ("Listen to the audio and answer: {question}. Return a JSON object with the answer, the time it is said, and the sentence it comes from, like: {example}.", "any"),
            ("Answer \"{question}\" based on the audio. Output JSON with the answer, its timestamp, and the source sentence, like: {example}.", "any"),
            ("From the audio, answer this and return a JSON object holding the answer, when it is spoken, and the supporting sentence, like: {example}. Question: {question}", "any"),
            ("Using the recording, answer the question: {question}. Format your response as JSON with the answer, its timestamp, and the source sentence, matching: {example}.", "any"),
            ("Please answer '{question}' from the audio and output JSON including the answer, the time it's spoken, and the sentence it comes from: {example}.", "any"),
            ("Answer this using the recording and return JSON matching this shape: {example}. Question: {question}", "any"),
            ("Question: {question}. Answer this from the audio and return JSON — with the answer, the timestamp it is said, and the full source sentence — in this exact shape: {example}.", "any"),
        ],
    },
    "fr": {
        "word2time": [
            ("À quel moment dans l'audio le mot \"{word}\" est-il prononcé ?", "word"),
            ("Quand le locuteur parle-t-il de {word} ?", "any"),
            ("Trouve le mot '{word}' dans l'audio et donne-moi son timestamp.", "word"),
            ("Localise \"{word}\" dans l'enregistrement. À quel instant apparaît-il ?", "any"),
            ("À quelle seconde {word} est-il prononcé ?", "any"),
            ("Donne-moi le timestamp où '{word}' apparaît.", "any"),
            ("Où dans le clip peut-on entendre \"{word}\" ?", "any"),
            ("Dis-moi quand on parle de {word} dans l'audio.", "any"),
            ("Je veux le temps de début du mot {word}. Quel est-il ?", "word"),
            ("Indique précisément le moment où ({word}) est prononcé.", "any"),
            ("Peux-tu me dire à quelle seconde \"{word}\" est dit ?", "any"),
            ("Mot cible : {word}. Quand est-il prononcé ?", "word"),
        ],
        "word2time_first": [
            ("Quel est le premier timestamp auquel le mot \"{word}\" est prononcé ?", "word"),
            ("Quand '{word}' est-il dit pour la première fois dans l'audio ?", "any"),
            ("Trouve la première occurrence de {word} dans l'enregistrement.", "any"),
            ("À quel moment {word} apparaît-il pour la première fois ?", "any"),
            ("Donne-moi le premier timestamp auquel '{word}' apparaît.", "any"),
            ("Quand le mot {word} est-il prononcé pour la première fois ?", "word"),
            ("À quelle seconde \"{word}\" est-il prononcé pour la première fois ?", "any"),
            ("Indique le premier moment où {word} est prononcé.", "any"),
            ("À quelle seconde le locuteur dit-il: {word} pour la première fois ?", "any"),
            ("Mot cible : {word}. Quand apparaît-il pour la première fois ?", "word"),
        ],
        "time2sentence": [
            ("Quelle phrase est prononcée vers {time:.1f} secondes ?", "any"),
            ("Vers {time:.1f}s, quelle phrase entends-tu ?", "any"),
            ("Transcris la phrase complète qui contient le repère {time:.1f}s.", "any"),
            ("Donne-moi la phrase qui couvre {time:.1f} secondes.", "any"),
            ("Écoute vers {time:.1f}s : quelle phrase complète le locuteur dit-il ?", "any"),
            ("Rapporte la phrase qui inclut le repère {time:.1f} secondes.", "any"),
            ("Vers {time:.1f}s environ, donne-moi la phrase entière prononcée.", "any"),
            ("Quelle phrase contient le moment à {time:.1f}s ?", "any"),
            ("Vers {time:.1f}s, peux-tu transcrire la phrase complète ?", "any"),
            ("Quelle phrase couvre le temps {time:.1f}s dans l'enregistrement ?", "any"),
            ("Le repère {time:.1f}s tombe au milieu d'une phrase : transcris-la entièrement.", "any"),
            ("Donne-moi la phrase complète qui couvre {time:.1f}s.", "any"),
        ],
        "time2word": [
            ("Quel mot est prononcé à {time:.1f} secondes ?", "word"),
            ("Quel mot le locuteur dit-il à {time:.1f}s ?", "word"),
            ("À {time:.1f} secondes, quel mot peut-on entendre ?", "word"),
            ("Dis-moi le mot prononcé au repère {time:.1f} secondes.", "word"),
            ("Quel mot correspond à {time:.1f}s dans l'enregistrement ?", "word"),
            ("Vers {time:.1f}s environ, que dit le locuteur (un seul mot) ?", "word"),
            ("Donne-moi le mot prononcé à {time:.1f} secondes.", "word"),
            ("Quel mot se trouve au timestamp {time:.1f} ?", "word"),
            ("Écoute à {time:.1f}s et rapporte le mot que tu entends.", "word"),
            ("Que dit-on à {time:.1f} secondes ?", "any"),
            ("Qu'entends-tu à {time:.1f}s ?", "any"),
            ("Rapporte ce que le locuteur prononce à {time:.1f} secondes.", "any"),
            ("Qu'est-ce qui est prononcé à {time:.1f}s ?", "any"),
        ],
        "word2sentence": [
            ("Quelle phrase de l'audio contient le mot \"{word}\" ?", "word"),
            ("Trouve la phrase où '{word}' est prononcé et transcris-la.", "any"),
            ("Donne-moi la phrase complète qui contient {word}.", "any"),
            ("Dans quelle phrase le locuteur utilise-t-il \"{word}\" ?", "any"),
            ("Transcris la phrase contenant '{word}'.", "any"),
            ("Dans quelle phrase apparaît {word} ?", "any"),
            ("Rapporte la phrase dans laquelle {word} est dit.", "any"),
            ("\"{word}\" se trouve quelque part dans l'audio : donne-moi la phrase dont il fait partie.", "any"),
            ("Localise {word} et transcris la phrase qui l'entoure.", "any"),
            ("Mot cible : {word}. Donne-moi la phrase dans laquelle il apparaît.", "word"),
        ],
        "answer_with_source": [
            ("Réponds à la question suivante à partir de l'audio et cite la phrase source avec son timestamp : \"{question}\"", "any"),
            ("Écoute l'audio et réponds : {question}. Donne-moi aussi la phrase d'où provient la réponse et quand elle est prononcée.", "any"),
            ("\"{question}\" Peux-tu répondre à partir de l'enregistrement et m'indiquer la phrase qui soutient ta réponse ainsi que son temps de début ?", "any"),
            ("À partir de l'audio, réponds à cette question et attribue ta réponse à une phrase précise avec son timestamp. Question : {question}", "any"),
            ("Donne-moi la réponse à \"{question}\" ainsi que la phrase de l'audio qui la contient et l'instant où cette phrase commence.", "any"),
            ("Réponds à la question à partir de l'enregistrement. Inclus la phrase source et le moment où elle est dite. Question : {question}", "any"),
            ("Réponds s'il te plaît à '{question}' et appuie ta réponse sur la phrase source extraite de l'audio et son timestamp.", "any"),
            ("À l'aide de l'audio, réponds à la question : {question}. Cite la phrase d'où provient la réponse ainsi que son temps de début.", "any"),
        ],
        "answer_with_time": [
            ("Réponds à la question suivante à partir de l'audio et donne-moi le timestamp où la réponse est prononcée : \"{question}\"", "any"),
            ("Écoute l'audio et réponds : {question}. Quand la réponse est-elle prononcée ?", "any"),
            ("Question : {question}. Réponds à cela et dis-moi quand la réponse est prononcée dans l'audio.", "any"),
            ("À partir de l'audio, réponds à \"{question}\" et rapporte le timestamp de la réponse.", "any"),
            ("Donne-moi la réponse à cette question ainsi que l'instant auquel elle est prononcée : {question}", "any"),
            ("Réponds à la question à partir de l'enregistrement. Inclus l'instant auquel la réponse est dite. Question : {question}", "any"),
            ("Réponds s'il te plaît à '{question}' et dis-moi quand la réponse apparaît dans l'audio.", "any"),
            ("À l'aide de l'audio, réponds à cela et donne le timestamp de la réponse : {question}", "any"),
        ],
        "format_json_answer": [
            ("Réponds à la question suivante à partir de l'audio et renvoie un JSON contenant la réponse, le timestamp où elle est prononcée et la phrase source, ainsi : {example}. Question : \"{question}\"", "any"),
            ("Écoute l'audio et réponds : {question}. Renvoie un objet JSON avec la réponse, l'instant où elle est dite et la phrase d'où elle provient, comme : {example}.", "any"),
            ("Réponds à \"{question}\" à partir de l'audio. Produis du JSON avec la réponse, son timestamp et la phrase source, comme : {example}.", "any"),
            ("À partir de l'audio, réponds à cela et renvoie un objet JSON contenant la réponse, le moment où elle est prononcée et la phrase qui la soutient, comme : {example}. Question : {question}", "any"),
            ("À l'aide de l'enregistrement, réponds à la question : {question}. Formate ta réponse en JSON avec la réponse, son timestamp et la phrase source, suivant ce schéma : {example}.", "any"),
            ("Réponds s'il te plaît à '{question}' à partir de l'audio et renvoie un JSON incluant la réponse, l'instant où elle est prononcée et la phrase dont elle provient : {example}.", "any"),
            ("Réponds à cela à l'aide de l'enregistrement et renvoie un JSON avec la réponse, son timestamp et la phrase source, correspondant à ce schéma : {example}. Question : {question}", "any"),
            ("Question : {question}. Réponds à cela à partir de l'audio et renvoie un JSON — avec la réponse, le timestamp où elle est dite et la phrase source complète — dans ce schéma exact : {example}.", "any"),
        ],
    },
}

ANSWER_TEMPLATES = {
    "en": {
        "word2time": [
            ("The word \"{word}\" is spoken at {time:.1f} seconds.", "word"),
            ("You can hear {word} at around {time:.1f}s.", "any"),
            ("\"{word}\" appears at {time:.1f} seconds.", "any"),
            ("The speaker says {word} at {time:.1f}s.", "any"),
            ("At {time:.1f} seconds, the word {word} is pronounced.", "word"),
            ("'{word}' is said at {time:.1f}s in the audio.", "any"),
            ("It occurs at {time:.1f} seconds.", "any"),
            ("{time:.1f}s.", "any"),
            ("Around {time:.1f} seconds.", "any"),
            ("The word starts at {time:.1f} seconds.", "word"),
            ("In the audio, \"{word}\" starts at {time:.1f}s.", "any"),
            ("Word: {word}. Time: {time:.1f}s.", "word"),
        ],
        "word2time_first": [
            ("The word \"{word}\" is first spoken at {time:.1f} seconds.", "word"),
            ("{word} first appears at {time:.1f}s.", "any"),
            ("The first occurrence of {word} is at {time:.1f} seconds.", "any"),
            ("First said at {time:.1f}s.", "any"),
            ("\"{word}\" is first uttered at {time:.1f}s.", "any"),
            ("The first timestamp is {time:.1f}s.", "any"),
            ("{time:.1f}s.", "any"),
            ("Around {time:.1f} seconds.", "any"),
            ("The first time the speaker says {word} is at {time:.1f}s.", "any"),
            ("First occurrence: {time:.1f}s.", "any"),
        ],
        "time2sentence": [
            ("Around that moment, the speaker is saying: {sentence}", "any"),
            ("The full sentence being spoken is: \"{sentence}\".", "any"),
            ("Around {time:.1f}s, you can hear: {sentence}", "any"),
            ("The sentence containing the {time:.1f}s mark is: \"{sentence}\".", "any"),
            ("The speaker utters: {sentence}", "any"),
            ("\"{sentence}\"", "any"),
            ("The sentence that spans {time:.1f}s is: {sentence}", "any"),
            ("At roughly {time:.1f}s, the full sentence being said is: \"{sentence}\".", "any"),
            ("Sentence: {sentence}", "any"),
        ],
        "time2word": [
            ("The word spoken at {time:.1f}s is \"{word}\".", "word"),
            ("At that moment, the speaker says {word}.", "any"),
            ("You can hear the word {word}.", "word"),
            ("It is the word \"{word}\".", "word"),
            ("Around {time:.1f} seconds, the word is {word}.", "word"),
            ("\"{word}\"", "any"),
            ("The speaker pronounces {word} at {time:.1f}s.", "any"),
            ("The speaker says '{word}'.", "any"),
            ("You can hear {word}.", "any"),
            ("Word: {word}.", "word"),
        ],
        "word2sentence": [
            ("The sentence containing \"{word}\" is: \"{sentence}\".", "any"),
            ("{word} appears in this sentence: \"{sentence}\".", "any"),
            ("The speaker uses {word} in: \"{sentence}\".", "any"),
            ("It is found in the following sentence: \"{sentence}\".", "any"),
            ("You can hear \"{word}\" in: \"{sentence}\".", "any"),
            ("\"{sentence}\"", "any"),
            ("'{word}' is part of: \"{sentence}\".", "any"),
            ("Sentence containing {word}: \"{sentence}\".", "any"),
        ],
        "answer_with_time": [
            ("The answer is \"{answer}\", spoken at {time:.1f}s.", "any"),
            ("'{answer}', said at {time:.1f}s.", "any"),
            ("Answer: {answer} (at {time:.1f}s).", "any"),
            ("{answer}, which is said at {time:.1f} seconds.", "any"),
            ("The answer is: {answer}. It is spoken at {time:.1f}s in the audio.", "any"),
            ("{answer}, spoken at {time:.1f}s.", "any"),
            ("You can hear the answer \"{answer}\" at {time:.1f}s.", "any"),
            ("\"{answer}\" is the answer; it appears at {time:.1f} seconds.", "any"),
            ("Answer: {answer}. Time: {time:.1f}s.", "any"),
        ],
        "answer_with_source": [
            ("The answer is {answer}. It comes from the sentence \"{sentence}\", spoken starting at {time:.1f}s.", "any"),
            ("\"{answer}\". This is stated in the sentence \"{sentence}\", which begins at {time:.1f}s.", "any"),
            ("Answer: {answer}. Source: {sentence}. Starts around: {time:.1f}s.", "any"),
            ("The audio says the answer is \"{answer}\", specifically in the sentence \"{sentence}\" beginning at {time:.1f}s.", "any"),
            ("The answer is: {answer}. The supporting sentence \"{sentence}\" is spoken at {time:.1f}s.", "any"),
            ("The response {answer} (found in the sentence: {sentence}), which starts around {time:.1f}s in the audio.", "any"),
            ("\"{answer}\" is the answer, which comes from the sentence \"{sentence}\" at {time:.1f}s.", "any"),
        ],
    },
    "fr": {
        "word2time": [
            ("Le mot \"{word}\" est prononcé à {time:.1f} secondes.", "word"),
            ("On peut entendre {word} vers {time:.1f}s.", "any"),
            ("\"{word}\" apparaît à {time:.1f} secondes.", "any"),
            ("Le locuteur dit {word} à {time:.1f}s.", "any"),
            ("À {time:.1f} secondes, le mot {word} est prononcé.", "word"),
            ("'{word}' est dit à {time:.1f}s dans l'audio.", "any"),
            ("Cela se produit à {time:.1f} secondes.", "any"),
            ("{time:.1f}s.", "any"),
            ("Vers {time:.1f} secondes.", "any"),
            ("Le mot commence à {time:.1f} secondes.", "word"),
            ("Dans l'audio, \"{word}\" commence à {time:.1f}s.", "any"),
            ("Mot : {word}. Temps : {time:.1f}s.", "word"),
        ],
        "word2time_first": [
            ("Le mot \"{word}\" est prononcé pour la première fois à {time:.1f} secondes.", "word"),
            ("{word} apparaît pour la première fois à {time:.1f}s.", "any"),
            ("La première occurrence de {word} est à {time:.1f} secondes.", "any"),
            ("Première fois dit à {time:.1f}s.", "any"),
            ("\"{word}\" est prononcé pour la première fois à {time:.1f}s.", "any"),
            ("Le premier timestamp est {time:.1f}s.", "any"),
            ("{time:.1f}s.", "any"),
            ("Vers {time:.1f} secondes.", "any"),
            ("La première fois que le locuteur dit {word} est à {time:.1f}s.", "any"),
            ("Première occurrence : {time:.1f}s.", "any"),
        ],
        "time2sentence": [
            ("Autour de ce moment, le locuteur dit : {sentence}", "any"),
            ("La phrase complète prononcée est : \"{sentence}\".", "any"),
            ("Vers {time:.1f}s, on peut entendre : {sentence}", "any"),
            ("La phrase qui contient le repère {time:.1f}s est : \"{sentence}\".", "any"),
            ("Le locuteur prononce : {sentence}", "any"),
            ("\"{sentence}\"", "any"),
            ("La phrase qui couvre {time:.1f}s est : {sentence}", "any"),
            ("Vers {time:.1f}s environ, la phrase complète prononcée est : \"{sentence}\".", "any"),
            ("Phrase : {sentence}", "any"),
        ],
        "time2word": [
            ("Le mot prononcé à {time:.1f}s est \"{word}\".", "word"),
            ("À ce moment, le locuteur dit {word}.", "any"),
            ("On peut entendre le mot {word}.", "word"),
            ("Il s'agit du mot \"{word}\".", "word"),
            ("Vers {time:.1f} secondes, le mot est {word}.", "word"),
            ("\"{word}\"", "any"),
            ("Le locuteur prononce {word} à {time:.1f}s.", "any"),
            ("Le locuteur dit '{word}'.", "any"),
            ("On peut entendre {word}.", "any"),
            ("Mot : {word}.", "word"),
        ],
        "word2sentence": [
            ("La phrase contenant \"{word}\" est : \"{sentence}\".", "any"),
            ("{word} apparaît dans cette phrase : \"{sentence}\".", "any"),
            ("Le locuteur utilise {word} dans : \"{sentence}\".", "any"),
            ("On le trouve dans la phrase suivante : \"{sentence}\".", "any"),
            ("On peut entendre \"{word}\" dans : \"{sentence}\".", "any"),
            ("\"{sentence}\"", "any"),
            ("'{word}' fait partie de : \"{sentence}\".", "any"),
            ("Phrase contenant {word} : \"{sentence}\".", "any"),
        ],
        "answer_with_time": [
            ("La réponse est \"{answer}\", prononcée à {time:.1f}s.", "any"),
            ("'{answer}', dit à {time:.1f}s.", "any"),
            ("Réponse : {answer} (à {time:.1f}s).", "any"),
            ("{answer}, qui est dit à {time:.1f} secondes.", "any"),
            ("La réponse est : {answer}. Elle est prononcée à {time:.1f}s dans l'audio.", "any"),
            ("{answer}, prononcé à {time:.1f}s.", "any"),
            ("On peut entendre la réponse \"{answer}\" à {time:.1f}s.", "any"),
            ("\"{answer}\" est la réponse ; elle apparaît à {time:.1f} secondes.", "any"),
            ("Réponse : {answer}. Temps : {time:.1f}s.", "any"),
        ],
        "answer_with_source": [
            ("La réponse est {answer}. Elle provient de la phrase \"{sentence}\", prononcée à partir de {time:.1f}s.", "any"),
            ("\"{answer}\". Cela est exprimé dans la phrase \"{sentence}\", qui commence à {time:.1f}s.", "any"),
            ("Réponse : {answer}. Source : {sentence}. Début vers : {time:.1f}s.", "any"),
            ("L'audio indique que la réponse est \"{answer}\", précisément dans la phrase \"{sentence}\" commençant à {time:.1f}s.", "any"),
            ("La réponse est : {answer}. La phrase qui la soutient \"{sentence}\" est prononcée à {time:.1f}s.", "any"),
            ("La réponse {answer} (trouvée dans la phrase : {sentence}), qui commence vers {time:.1f}s dans l'audio.", "any"),
            ("\"{answer}\" est la réponse, qui provient de la phrase \"{sentence}\" à {time:.1f}s.", "any"),
        ],
    },
}

QUESTION_TEMPLATES_RANGE = {
    "en": {
        "word2time": [
            ("Give me both the start and end timestamps of the word \"{word}\".", "word"),
            ("From what second to what second is {word} spoken?", "any"),
            ("What are the start and end times of \"{word}\" in the audio?", "any"),
            ("Locate {word}: I need its timestamps.", "any"),
            ("When does {word} start and end?", "any"),
            ("Give me the time range during which '{word}' is uttered.", "any"),
            ("Tell me when {word} begins and ends in the recording.", "any"),
            ("What is the start and end second of the word {word}?", "word"),
            ("I need the full interval (start and end) for \"{word}\".", "any"),
        ],
        "word2time_first": [
            ("What are the start and end timestamps of the first occurrence of \"{word}\"?", "word"),
            ("When does {word} first start and end in the audio?", "any"),
            ("Give me the start and end time of the first time {word} is spoken.", "any"),
            ("From what second to what second does {word} first appear?", "any"),
            ("First occurrence of {word}: I need both start and end times.", "any"),
            ("What is the start and end second of the first '{word}'?", "any"),
            ("Tell me when {word} first occurs in the recording.", "any"),
            ("Give me the full interval of the first occurrence of {word}.", "any"),
        ],
    },
    "fr": {
        "word2time": [
            ("Donne-moi à la fois le temps de début et de fin du mot \"{word}\".", "word"),
            ("De quelle seconde à quelle seconde {word} est-il prononcé ?", "any"),
            ("Quels sont les temps de début et de fin de \"{word}\" dans l'audio ?", "any"),
            ("Localise {word} : il me faut ses timestamps.", "any"),
            ("Quand {word} commence-t-il et finit-il ?", "any"),
            ("Donne-moi l'intervalle de temps pendant lequel '{word}' est prononcé.", "any"),
            ("Dis-moi quand {word} commence et finit dans l'enregistrement.", "any"),
            ("Quelle est la seconde de début et de fin du mot {word} ?", "word"),
            ("J'ai besoin de l'intervalle complet (début et fin) pour \"{word}\".", "any"),
        ],
        "word2time_first": [
            ("Quels sont les temps de début et de fin de la première occurrence de \"{word}\" ?", "word"),
            ("Quand {word} commence-t-il et finit-il pour la première fois dans l'audio ?", "any"),
            ("Donne-moi le temps de début et de fin de la première fois où {word} est prononcé.", "any"),
            ("De quelle seconde à quelle seconde {word} apparaît-il pour la première fois ?", "any"),
            ("Première occurrence de {word} : il me faut à la fois le début et la fin.", "any"),
            ("Quelle est la seconde de début et de fin du premier '{word}' ?", "any"),
            ("Dis-moi quand {word} apparaît pour la première fois dans l'enregistrement.", "any"),
            ("Donne-moi l'intervalle complet de la première occurrence de {word}.", "any"),
        ],
    },
}

ANSWER_TEMPLATES_RANGE = {
    "en": {
        "word2time": [
            ("The word \"{word}\" is spoken from {time:.1f}s to {end_time:.1f}s.", "word"),
            ("{word} runs from {time:.1f}s to {end_time:.1f}s.", "any"),
            ("Start: {time:.1f}s. End: {end_time:.1f}s.", "any"),
            ("From {time:.1f}s to {end_time:.1f}s.", "any"),
            ("\"{word}\" begins at {time:.1f}s and ends at {end_time:.1f}s.", "any"),
            ("{time:.1f}s - {end_time:.1f}s.", "any"),
            ("It is uttered between {time:.1f}s and {end_time:.1f}s.", "any"),
            ("The interval is {time:.1f}s to {end_time:.1f}s.", "any"),
        ],
        "word2time_first": [
            ("The word \"{word}\" is first spoken from {time:.1f}s to {end_time:.1f}s.", "word"),
            ("{word} first runs from {time:.1f}s to {end_time:.1f}s.", "any"),
            ("Start: {time:.1f}s. End: {end_time:.1f}s.", "any"),
            ("From {time:.1f}s to {end_time:.1f}s.", "any"),
            ("First occurrence: {time:.1f}s - {end_time:.1f}s.", "any"),
            ("\"{word}\" first begins at {time:.1f}s and ends at {end_time:.1f}s.", "any"),
            ("The first occurrence runs from {time:.1f}s to {end_time:.1f}s.", "any"),
            ("First interval: {time:.1f}s to {end_time:.1f}s.", "any"),
        ],
    },
    "fr": {
        "word2time": [
            ("Le mot \"{word}\" est prononcé de {time:.1f}s à {end_time:.1f}s.", "word"),
            ("{word} s'étend de {time:.1f}s à {end_time:.1f}s.", "any"),
            ("Début : {time:.1f}s. Fin : {end_time:.1f}s.", "any"),
            ("De {time:.1f}s à {end_time:.1f}s.", "any"),
            ("\"{word}\" commence à {time:.1f}s et finit à {end_time:.1f}s.", "any"),
            ("{time:.1f}s - {end_time:.1f}s.", "any"),
            ("Il est prononcé entre {time:.1f}s et {end_time:.1f}s.", "any"),
            ("L'intervalle est {time:.1f}s à {end_time:.1f}s.", "any"),
        ],
        "word2time_first": [
            ("Le mot \"{word}\" est prononcé pour la première fois de {time:.1f}s à {end_time:.1f}s.", "word"),
            ("{word} s'étend pour la première fois de {time:.1f}s à {end_time:.1f}s.", "any"),
            ("Début : {time:.1f}s. Fin : {end_time:.1f}s.", "any"),
            ("De {time:.1f}s à {end_time:.1f}s.", "any"),
            ("Première occurrence : {time:.1f}s - {end_time:.1f}s.", "any"),
            ("\"{word}\" commence pour la première fois à {time:.1f}s et finit à {end_time:.1f}s.", "any"),
            ("La première occurrence s'étend de {time:.1f}s à {end_time:.1f}s.", "any"),
            ("Premier intervalle : {time:.1f}s à {end_time:.1f}s.", "any"),
        ],
    },
}


# Candidate words unlikely to appear in speech transcripts — filtered at
# runtime against the actual word list to guarantee a true OOV for the clip.
NEGATIVE_WORD_CANDIDATES = {
    "en": [
        "xylophone", "quokka", "zeppelin", "kumquat", "platypus", "obsidian",
        "saxophone", "marshmallow", "parsnip", "jellyfish", "labyrinth",
        "hippopotamus", "kaleidoscope", "rhubarb", "tangerine", "walrus",
        "narwhal", "avocado", "cinnamon", "dandelion", "eucalyptus",
        "flamingo", "gazelle", "igloo", "mongoose", "nutmeg", "ostrich",
        "pineapple", "raccoon", "trombone", "vulcanize", "zucchini",
        "aardvark", "bumblebee", "chinchilla", "geyser", "pomegranate",
        "document", "time", "restaurant", "meeting", "appointment",
        "file", "text", "email", "message", "plane", "taxi", "doctor",
        "weekday", "important", "voicemail",
        "armadillo", "python", "pelican", "octopus", "barnacle",
        "wombat", "puffin", "cassowary", "manatee", "lemur",
        "saffron", "paprika", "cardamom", "lavender", "dish",
        "didgeridoo", "harpsichord", "accordion", "ukulele", "tambourine",
        "stalactite", "tundra", "fjord", "monsoon", "station",
        "telescope", "code", "helicopter", "bulldozer", "lighthouse",
        "passport", "umbrella", "treadmill", "cathedral", "scaffolding",
        "yesterday", "tomorrow", "midnight", "weekend", "holiday",
        "hospital", "library", "stadium", "warehouse", "embassy",
        "lawyer", "plumber", "astronaut", "engineer", "journalist",
        "spreadsheet", "podcast", "newsletter", "invoice", "receipt",
        "bicycle", "motorcycle", "scooter", "ferry", "subway",
        "semester", "solstice", "urgent", "malfunction",
        "webcam", "boardroom", "ledger", "voucher",
    ],
    "fr": [
        "xylophone", "ornithorynque", "zeppelin", "kumquat", "obsidienne",
        "saxophone", "guimauve", "panais", "méduse", "labyrinthe",
        "hippopotame", "kaléidoscope", "rhubarbe", "mandarine", "morse",
        "narval", "avocat", "cannelle", "pissenlit", "eucalyptus",
        "flamant", "gazelle", "igloo", "mangouste", "muscade", "autruche",
        "ananas", "chat", "trombone", "courgette",
        "tatou", "python", "pélican", "pieuvre", "bernacle",
        "wombat", "macareux", "casoar", "lamantin", "lémurien",
        "safran", "paprika", "cardamome", "lavande",
        "didgeridoo", "clavecin", "accordéon", "ukulélé", "tambourin",
        "stalactite", "toundra", "fjord", "mousson", "station",
        "télescope", "code", "hélicoptère", "bulldozer", "phare",
        "passeport", "parapluie", "trampoline", "cathédrale", "échafaudage",
        "hier", "demain", "minuit", "week-end", "vacances",
        "hôpital", "bibliothèque", "stade", "entrepôt", "ambassade",
        "avocate", "plombier", "astronaute", "ingénieur", "journaliste",
        "tableur", "podcast", "newsletter", "facture", "reçu",
        "bicyclette", "moto", "trottinette", "ferry", "métro",
        "semestre", "solstice", "urgent", "défectueux",
        "visioconférence", "comité", "bilan", "chéquier",
        "document", "courriel", "smartphone", "ordinateur",
        "rendez-vous", "réunion", "fichier", "texte", "message",
        "avion", "taxi", "médecin", "important",
    ],
}

_NEGATIVE_TIME_ANSWERS = {
    "en": [
        "{time:.1f}s is past the end of the audio, so it is not a valid timestamp.",
        "The timestamp {time:.1f}s is beyond the duration of the recording. Not a valid timestamp.",
        "The audio ends before {time:.1f}s, so this is not a valid timestamp.",
        "That is not a valid timestamp: the audio is shorter than {time:.1f} seconds.",
        "Not a valid timestamp ({time:.1f}s is after the end of the audio).",
        "There is nothing at {time:.1f}s because the recording ends earlier.",
        "{time:.1f} seconds falls outside the audio, so the timestamp is invalid.",
        "Invalid timestamp: {time:.1f}s.",
    ],
    "fr": [
        "{time:.1f}s dépasse la fin de l'audio : ce n'est pas un timestamp valide.",
        "Le timestamp {time:.1f}s est au-delà de la durée de l'enregistrement. Ce n'est pas un timestamp valide.",
        "L'audio se termine avant {time:.1f}s, donc ce n'est pas un timestamp valide.",
        "Ce n'est pas un timestamp valide : l'audio dure moins de {time:.1f} secondes.",
        "Timestamp non valide ({time:.1f}s se situe après la fin de l'audio).",
        "Il n'y a rien à {time:.1f}s car l'enregistrement se termine avant.",
        "{time:.1f} secondes tombe en dehors de l'audio, donc le timestamp est invalide.",
        "Timestamp invalide : {time:.1f}s.",
    ],
}

NEGATIVE_QUESTION_TEMPLATES = {
    "en": {
        "word2time": [
            "At what time in the audio is the word \"{word}\" spoken?",
            "When does the speaker say {word}?",
            "Find the word {word} in the audio and tell me its timestamp.",
            "At which second is {word} pronounced?",
            "Where in the clip can I hear '{word}'?",
            "Give me the timestamp where {word} appears.",
            "Tell me when \"{word}\" is said in the audio.",
            "Can you locate {word} in the recording?",
            "Pinpoint the moment when ({word}) is uttered.",
            "Target word: {word}. When is it spoken?",
        ],
        "word2time_first": [
            "What is the first timestamp at which the word \"{word}\" is spoken?",
            "When is {word} first said in the audio?",
            "Find the first occurrence of {word} in the recording.",
            "At what time does {word} first appear?",
            "Give me the first timestamp at which '{word}' appears.",
            "When is the word {word} first pronounced?",
            "What is the earliest second at which \"{word}\" is uttered?",
            "Pinpoint the first moment {word} is spoken.",
            "Target word: {word}. When does it first appear?",
        ],
        "word2sentence": [
            "Which sentence in the audio contains the word \"{word}\"?",
            "Find the sentence where {word} is spoken and transcribe it.",
            "Give me the full sentence that includes {word}.",
            "In which sentence does the speaker use \"{word}\"?",
            "Transcribe the sentence containing '{word}'.",
            "What sentence does {word} appear in?",
            "Report the sentence in which {word} is said.",
            "Locate \"{word}\" and transcribe the surrounding sentence.",
            "Target word: {word}. Give me the sentence it appears in.",
        ],
        "time2word": [
            "What word is spoken at {time:.1f} seconds?",
            "Which word does the speaker say at {time:.1f}s?",
            "At {time:.1f} seconds, what word can you hear?",
            "Tell me the word uttered at the {time:.1f}-second mark.",
            "Give me the word pronounced at {time:.1f} seconds.",
            "What is said at {time:.1f}s?",
            "Report the word at timestamp {time:.1f}.",
        ],
        "time2sentence": [
            "What sentence is being spoken around {time:.1f} seconds?",
            "Around {time:.1f}s, what sentence do you hear?",
            "Transcribe the sentence at {time:.1f} seconds.",
            "Which sentence contains the {time:.1f}s moment?",
            "Give me the sentence that spans {time:.1f}s.",
            "Listen near {time:.1f}s. What full sentence is being said?",
        ],
    },
    "fr": {
        "word2time": [
            "À quel moment dans l'audio le mot \"{word}\" est-il prononcé ?",
            "Quand le locuteur dit-il {word} ?",
            "Trouve le mot {word} dans l'audio et donne-moi son timestamp.",
            "À quelle seconde {word} est-il prononcé ?",
            "Où dans le clip peut-on entendre '{word}' ?",
            "Donne-moi le timestamp où {word} apparaît.",
            "Dis-moi quand \"{word}\" est dit dans l'audio.",
            "Peux-tu localiser {word} dans l'enregistrement ?",
            "Indique précisément le moment où ({word}) est prononcé.",
            "Mot cible : {word}. Quand est-il prononcé ?",
        ],
        "word2time_first": [
            "Quel est le premier timestamp auquel le mot \"{word}\" est prononcé ?",
            "Quand {word} est-il dit pour la première fois dans l'audio ?",
            "Trouve la première occurrence de {word} dans l'enregistrement.",
            "À quel moment {word} apparaît-il pour la première fois ?",
            "Donne-moi le premier timestamp auquel '{word}' apparaît.",
            "Quand le mot {word} est-il prononcé pour la première fois ?",
            "À quelle seconde \"{word}\" est-il prononcé pour la première fois ?",
            "Indique le premier moment où {word} est prononcé.",
            "Mot cible : {word}. Quand apparaît-il pour la première fois ?",
        ],
        "word2sentence": [
            "Quelle phrase de l'audio contient le mot \"{word}\" ?",
            "Trouve la phrase où {word} est prononcé et transcris-la.",
            "Donne-moi la phrase complète qui contient {word}.",
            "Dans quelle phrase le locuteur utilise-t-il \"{word}\" ?",
            "Transcris la phrase contenant '{word}'.",
            "Dans quelle phrase apparaît {word} ?",
            "Rapporte la phrase dans laquelle {word} est dit.",
            "Localise \"{word}\" et transcris la phrase qui l'entoure.",
            "Mot cible : {word}. Donne-moi la phrase dans laquelle il apparaît.",
        ],
        "time2word": [
            "Quel mot est prononcé à {time:.1f} secondes ?",
            "Quel mot le locuteur dit-il à {time:.1f}s ?",
            "À {time:.1f} secondes, quel mot peut-on entendre ?",
            "Dis-moi le mot prononcé au repère {time:.1f} secondes.",
            "Donne-moi le mot prononcé à {time:.1f} secondes.",
            "Que dit-on à {time:.1f}s ?",
            "Rapporte le mot au timestamp {time:.1f}.",
        ],
        "time2sentence": [
            "Quelle phrase est prononcée vers {time:.1f} secondes ?",
            "Vers {time:.1f}s, quelle phrase entends-tu ?",
            "Transcris la phrase à {time:.1f} secondes.",
            "Quelle phrase contient le moment à {time:.1f}s ?",
            "Donne-moi la phrase qui couvre {time:.1f}s.",
            "Écoute vers {time:.1f}s. Quelle phrase complète est dite ?",
        ],
    },
}

NEGATIVE_ANSWER_TEMPLATES = {
    "en": {
        "word2time": [
            "The word \"{word}\" is not spoken in the audio.",
            "I cannot find {word} in the recording.",
            "'{word}' does not appear in the audio.",
            "The word {word} is not present in this clip.",
            "\"{word}\" is not said anywhere in the audio.",
            "The speaker never says {word} in this recording.",
            "There is no occurrence of {word} in the audio.",
            "Not found: {word}.",
        ],
        "word2time_first": [
            "The word \"{word}\" is never spoken in the audio, so there is no first timestamp.",
            "I cannot find any occurrence of {word} in the recording.",
            "'{word}' does not appear in the audio at all.",
            "There is no first occurrence of {word} -- the word is not in this clip.",
            "\"{word}\" is not said anywhere in the audio, so it has no first timestamp.",
            "The speaker never says {word}; no first occurrence to report.",
            "{word} is absent from the recording.",
            "Not found: {word}. No first occurrence.",
        ],
        "word2sentence": [
            "The word \"{word}\" is not spoken in the audio, so there is no sentence containing it.",
            "I cannot find {word} in the recording; no sentence contains it.",
            "'{word}' does not appear in the audio, so no sentence can be reported.",
            "The word {word} is not present in this clip, so there is no matching sentence.",
            "\"{word}\" is not said anywhere in the audio; no sentence contains it.",
            "The speaker never says {word}, so no sentence in the audio includes it.",
            "There is no occurrence of {word} in the audio, so no sentence can be returned.",
            "Not found in the audio: {word}. No sentence to report.",
        ],
        "time2word": _NEGATIVE_TIME_ANSWERS["en"],
        "time2sentence": _NEGATIVE_TIME_ANSWERS["en"],
    },
    "fr": {
        "word2time": [
            "Le mot \"{word}\" n'est pas prononcé dans l'audio.",
            "Je ne trouve pas {word} dans l'enregistrement.",
            "'{word}' n'apparaît pas dans l'audio.",
            "Le mot {word} n'est pas présent dans ce clip.",
            "\"{word}\" n'est dit nulle part dans l'audio.",
            "Le locuteur ne dit jamais {word} dans cet enregistrement.",
            "Il n'y a aucune occurrence de {word} dans l'audio.",
            "Introuvable : {word}.",
        ],
        "word2time_first": [
            "Le mot \"{word}\" n'est jamais prononcé dans l'audio, il n'y a donc pas de premier timestamp.",
            "Je ne trouve aucune occurrence de {word} dans l'enregistrement.",
            "'{word}' n'apparaît pas du tout dans l'audio.",
            "Il n'y a pas de première occurrence de {word} -- ce mot n'est pas dans ce clip.",
            "\"{word}\" n'est dit nulle part dans l'audio, il n'a donc pas de premier timestamp.",
            "Le locuteur ne dit jamais {word} ; pas de première occurrence à rapporter.",
            "{word} est absent de l'enregistrement.",
            "Introuvable : {word}. Pas de première occurrence.",
        ],
        "word2sentence": [
            "Le mot \"{word}\" n'est pas prononcé dans l'audio, il n'y a donc pas de phrase qui le contient.",
            "Je ne trouve pas {word} dans l'enregistrement ; aucune phrase ne le contient.",
            "'{word}' n'apparaît pas dans l'audio, donc aucune phrase ne peut être rapportée.",
            "Le mot {word} n'est pas présent dans ce clip, il n'y a donc pas de phrase correspondante.",
            "\"{word}\" n'est dit nulle part dans l'audio ; aucune phrase ne le contient.",
            "Le locuteur ne dit jamais {word}, donc aucune phrase de l'audio ne l'inclut.",
            "Il n'y a aucune occurrence de {word} dans l'audio, donc aucune phrase ne peut être renvoyée.",
            "Introuvable dans l'audio : {word}. Pas de phrase à rapporter.",
        ],
        "time2word": _NEGATIVE_TIME_ANSWERS["fr"],
        "time2sentence": _NEGATIVE_TIME_ANSWERS["fr"],
    },
}


def build_negative_variant(record, task, lang="en", style="chat"):
    """Return a `_neg` record for `task`, or None if a negative cannot be built."""
    if task not in NEGATIVE_QUESTION_TEMPLATES[lang]:
        return None
    cm = record.get("custom_metadata", {})
    w2t = cm.get("word2time", {})
    document_audio, duration = get_document_audio(record.get("conversations", []))
    if document_audio is None:
        return None
    if task in ("word2time", "word2sentence"):
        oov = pick_oov_word(w2t, NEGATIVE_WORD_CANDIDATES[lang])
        if oov is None:
            return None
        fmt = {"word": oov}
        pure_answer = PURE_NEG_NOT_FOUND
    else:  # time2word, time2sentence
        dur = audio_duration(w2t, duration)
        if dur is None:
            return None
        fmt = {"time": dur + random.uniform(0.5, 10.0)}
        pure_answer = PURE_NEG_INVALID_TIME
    question = random.choice(NEGATIVE_QUESTION_TEMPLATES[lang][task]).format(**fmt)
    answer = random.choice(NEGATIVE_ANSWER_TEMPLATES[lang][task]).format(**fmt)
    if style == "pure":
        answer = pure_answer
        suffix = "_neg_pure"
    else:
        suffix = "_neg"
    return wrap_record(record, task, document_audio, duration, question, answer, suffix=suffix)


def _span_representative_word(w2t, start_idx, end_idx):
    """Reduce a multi-word answer span [start_idx..end_idx] to a single
    representative word and its timing, for the single-WORD tasks
    (`time2word`/`word2time`). Prefers the first distinctive token (>= 4 alnum
    chars) so we don't end up querying a stopword like "the"; falls back to the
    first alnum token. Returns (word, start_second, end_second) or None."""
    words = w2t["word"]
    starts = w2t["start_second"]
    ends = w2t.get("end_second") or starts
    cands = [
        i for i in range(start_idx, end_idx + 1)
        if i < len(words) and is_alnum_token(words[i])
        and i < len(starts) and starts[i] is not None and starts[i] >= 0
    ]
    if not cands:
        return None
    distinctive = [i for i in cands if sum(c.isalnum() for c in words[i]) >= 4]
    j = (distinctive or cands)[0]
    word = words[j].strip(" \t\n\"'.,;:!?()[]")
    # Guarantee a single token: an alignment entry that bundled several words
    # (contains internal whitespace) would otherwise leak a phrase into the
    # single-WORD tasks. Keep the first whitespace-delimited piece.
    word = word.split()[0] if word.split() else ""
    if not word:
        return None
    end = ends[j] if j < len(ends) and ends[j] is not None else starts[j]
    return word, starts[j], end


def build_variant(record, task, lang="en", style="chat"):
    """Return a new jsonl record for `task`, or None if the source record can't support it."""
    qt = QUESTION_TEMPLATES[lang]
    at = ANSWER_TEMPLATES[lang]
    qtr = QUESTION_TEMPLATES_RANGE[lang]
    atr = ANSWER_TEMPLATES_RANGE[lang]

    cm = record.get("custom_metadata", {})
    spans = cm.get("answer_spans", {})
    w2t = cm.get("word2time", {})
    if not spans.get("answer") or not w2t.get("word"):
        return None

    answer_start = spans["start_second"][0]
    answer_end = spans.get("end_second", [answer_start])[0]
    span = locate_answer_span(w2t, answer_start, answer_end)
    if span is None:
        return None
    start_idx, end_idx = span

    phrase = cased_answer_phrase(w2t, start_idx, end_idx) or spans["answer"][0]
    single_token = " " not in phrase

    # `time2word`/`word2time` are single-WORD tasks: a multi-word answer span
    # would ask for a word ("what word is at Ts?" / "when is <word> said?") but
    # answer with a phrase. Reduce the span to one representative word and its
    # timing so these stay well-formed; the phrase still flows untouched into the
    # sentence / answer_with_* / json tasks.
    if not single_token and task in ("time2word", "word2time"):
        reduced = _span_representative_word(w2t, start_idx, end_idx)
        if reduced is None:
            return None
        phrase, answer_start, answer_end = reduced
        single_token = True

    sentence, sent_start_idx = extract_sentence(w2t["word"], start_idx)
    starts = w2t["start_second"]
    sentence_start_time = (
        starts[sent_start_idx] if sent_start_idx < len(starts) else answer_start
    )

    conversations = record.get("conversations", [])
    document_audio, duration = get_document_audio(conversations)
    if document_audio is None:
        return None

    if task == "answer_with_source":
        src_question, src_answer = extract_qa_text(conversations)
        if not src_question or not src_answer:
            return None
        fmt = {
            "question": src_question,
            "answer": src_answer,
            "sentence": sentence,
            "time": sentence_start_time,
        }
        question = choose_template(qt[task], single_token).format(**fmt)
        answer = choose_template(at[task], single_token).format(**fmt)
        pure_answer = f"{src_answer}|{sentence_start_time:.1f}s|{sentence}"
    elif task == "answer_with_time":
        src_question, src_answer = extract_qa_text(conversations)
        if not src_question or not src_answer:
            return None
        lookup_tokens = [t.lower() for t in phrase.split() if is_alnum_token(t)]
        occurrences = find_all_occurrences(lookup_tokens, w2t)
        q_tpl = choose_template(qt[task], single_token)
        a_tpl = choose_template(at[task], single_token)
        question = q_tpl.format(question=src_question)
        if len(occurrences) >= 2:
            times = [s for _, s in occurrences]
            a_tpl = substitute_multi_time(a_tpl, times, lang)
            answer = a_tpl.format(answer=src_answer)
            pure_answer = f"{src_answer}|{pure_times(times)}"
        else:
            answer = a_tpl.format(answer=src_answer, time=answer_start)
            pure_answer = f"{src_answer}|{answer_start:.1f}s"
    elif task == "format_json_answer":
        src_question, src_answer = extract_qa_text(conversations)
        if not src_question or not src_answer:
            return None
        schema_name = random.choice(list(ANSWER_JSON_SCHEMAS))
        schema = ANSWER_JSON_SCHEMAS[schema_name]
        q_tpl = choose_template(qt[task], single_token)
        question = q_tpl.format(
            question=src_question,
            example=schema["example"],
        )
        answer = schema["render"](
            src_question, src_answer, round(answer_start, 1), sentence
        )
        # JSON output is already the canonical short form.
        pure_answer = answer
    elif task == "word2time":
        lookup_tokens = [t.lower() for t in phrase.split() if is_alnum_token(t)]
        occurrences = find_all_occurrences(lookup_tokens, w2t)
        if len(occurrences) >= 2 and random.random() < FIRST_SUBVARIANT_PROB:
            # `first` sub-variant (merged from the former word2time_first task):
            # ask only for the earliest occurrence, with optional range mode.
            first_idx, first_time = min(occurrences, key=lambda io: io[1])
            if random.random() < RANGE_MODE_PROB:
                first_end = phrase_end_time(w2t, first_idx, len(lookup_tokens)) or first_time
                fmt = {"word": phrase, "time": first_time, "end_time": first_end}
                question = choose_template(qtr["word2time_first"], single_token).format(**fmt)
                answer = choose_template(atr["word2time_first"], single_token).format(**fmt)
                pure_answer = f"{first_time:.1f}s-{first_end:.1f}s"
            else:
                fmt = {"word": phrase, "time": first_time}
                question = choose_template(qt["word2time_first"], single_token).format(**fmt)
                answer = choose_template(at["word2time_first"], single_token).format(**fmt)
                pure_answer = f"{first_time:.1f}s"
        elif len(occurrences) <= 1 and random.random() < RANGE_MODE_PROB:
            # Range mode: the question explicitly asks for both start and end
            # timestamps. Limited to single-occurrence cases (rendering multiple
            # ranges per answer would require a separate joiner).
            fmt = {"word": phrase, "time": answer_start, "end_time": answer_end}
            question = choose_template(qtr[task], single_token).format(**fmt)
            answer = choose_template(atr[task], single_token).format(**fmt)
            pure_answer = f"{answer_start:.1f}s-{answer_end:.1f}s"
        else:
            q_tpl = choose_template(qt[task], single_token)
            a_tpl = choose_template(at[task], single_token)
            if len(occurrences) >= 2:
                # Reuse the chosen templates; swap the {time:.1f} placeholder for a
                # joined multi-time string so the answer reads e.g. "at 9.1s, 11.3s,
                # and 24.8s" while the rest of the template wording is unchanged.
                times = [s for _, s in occurrences]
                a_tpl = substitute_multi_time(a_tpl, times, lang)
                question = q_tpl.format(word=phrase)  # q templates don't use {time}
                answer = a_tpl.format(word=phrase)
                pure_answer = pure_times(times)
            else:
                fmt = {"word": phrase, "time": answer_start, "sentence": sentence}
                question = q_tpl.format(**fmt)
                answer = a_tpl.format(**fmt)
                pure_answer = f"{answer_start:.1f}s"
    elif task == "word2sentence":
        lookup_tokens = [t.lower() for t in phrase.split() if is_alnum_token(t)]
        occurrences = find_all_occurrences(lookup_tokens, w2t)
        q_tpl = choose_template(qt[task], single_token)
        unique_sentences = dedup_sentences_for_occurrences(w2t, occurrences) if len(occurrences) >= 2 else []
        if len(unique_sentences) >= 2:
            # Multi-substitution relies on the surrounding `"..."` to render
            # `"S1", "S2", and "S3"`, so restrict to answer templates that wrap
            # `{sentence}` in double quotes.
            multi_safe = [t for t in at[task] if '"{sentence}"' in t[0]]
            a_tpl = choose_template(multi_safe, single_token)
            a_tpl = substitute_multi_sentence(a_tpl, unique_sentences, lang)
            question = q_tpl.format(word=phrase)  # q templates don't use {sentence}
            answer = a_tpl.format(word=phrase)
            pure_answer = "|".join(unique_sentences)
        else:
            a_tpl = choose_template(at[task], single_token)
            # Single occurrence OR multiple hits inside one sentence -> normal flow.
            fmt = {"word": phrase, "sentence": sentence}
            question = q_tpl.format(**fmt)
            answer = a_tpl.format(**fmt)
            pure_answer = sentence
    elif task == "time2word":
        fmt = {"word": phrase, "time": answer_start, "sentence": sentence}
        question = choose_template(qt[task], single_token).format(**fmt)
        answer = choose_template(at[task], single_token).format(**fmt)
        pure_answer = phrase
    elif task == "time2sentence":
        fmt = {"word": phrase, "time": answer_start, "sentence": sentence}
        question = choose_template(qt[task], single_token).format(**fmt)
        answer = choose_template(at[task], single_token).format(**fmt)
        pure_answer = sentence
    else:
        fmt = {"word": phrase, "time": answer_start, "sentence": sentence}
        question = choose_template(qt[task], single_token).format(**fmt)
        answer = choose_template(at[task], single_token).format(**fmt)
        pure_answer = answer

    if style == "pure":
        return wrap_record(record, task, document_audio, duration, question, pure_answer, suffix="_pure")
    return wrap_record(record, task, document_audio, duration, question, answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Directory with train/validation/test jsonl files")
    parser.add_argument("--output_dir", type=str, default="output_slu_variants")
    parser.add_argument(
        "--tasks", type=str, nargs="+", choices=TASKS, default=list(TASKS),
    )
    # SLU is an English-only dataset, so prompts are always English here. The
    # bilingual QUESTION_TEMPLATES / ANSWER_TEMPLATES (incl. their "fr" entries)
    # are kept because template_timestamp_tasks.py imports them for the
    # French-audio MLS generator, which does expose a --language flag.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pure_splits", type=str, nargs="*", default=["all"],
        help="Split stems that also get a pure-answer version in the `<task>` "
             "folder (chat-style answers always go to `<task>_chat`). Use 'all' "
             "(the default) for every detected split, or list specific stems "
             "(e.g. 'verified_test'). Each pure split is a full extra generation pass.",
    )
    parser.add_argument(
        "--negative_rate", type=float, default=0.1,
        help="For word2time/word2sentence/time2word/time2sentence, fraction of "
             "kept positive samples that also get an additional negative sample "
             "appended (OOV word or out-of-range timestamp). Negatives are added, "
             "not substituted.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.data_path)
    split_files = sorted(input_dir.glob("*.jsonl"))
    if not split_files:
        raise SystemExit(f"No *.jsonl files found in {input_dir}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Splits detected: {[f.stem for f in split_files]}")
    print(f"Tasks:           {args.tasks}")

    lang = "en"
    negative_eligible = tuple(NEGATIVE_QUESTION_TEMPLATES[lang].keys())
    pure_all = "all" in args.pure_splits

    total_trailing_skipped = 0
    for task in args.tasks:
        # format_json_answer's answer IS the JSON object, so its chat and pure
        # forms are identical -- emit it only once, into the canonical `<task>`
        # folder, with no `<task>_chat` counterpart.
        json_only = task == "format_json_answer"
        for split_file in split_files:
            split = split_file.stem
            # Chat-style (natural-language) answers live in a `<task>_chat`
            # folder; pure short-form answers live in `<task>`.
            if not json_only:
                chat_file = output_root / f"{task}_chat" / f"{split}.jsonl"
                kept, dropped, neg, trailing_skipped = process_split(
                    split_file, chat_file, task,
                    build_positive=build_variant,
                    build_negative=build_negative_variant,
                    negative_eligible_tasks=negative_eligible,
                    negative_rate=args.negative_rate,
                    lang=lang,
                )
                total_trailing_skipped += trailing_skipped
                print(
                    f"[{task}_chat/{split}] kept={kept} negatives={neg} dropped={dropped} "
                    f"trailing_skipped={trailing_skipped} -> {chat_file}"
                )
            # Selected splits also get a pure version (literal short-form answers
            # like "10.2s,25.6s") for canonical eval comparison. format_json is
            # always emitted here regardless of --pure_splits.
            if json_only or pure_all or split in args.pure_splits:
                pure_file = output_root / task / f"{split}.jsonl"
                kept_p, dropped_p, neg_p, trailing_p = process_split(
                    split_file, pure_file, task,
                    build_positive=build_variant,
                    build_negative=build_negative_variant,
                    negative_eligible_tasks=negative_eligible,
                    negative_rate=args.negative_rate,
                    lang=lang, style="pure",
                )
                print(
                    f"[{task}/{split}] kept={kept_p} negatives={neg_p} dropped={dropped_p} "
                    f"trailing_skipped={trailing_p} -> {pure_file}"
                )

    print(
        f"Total rows discarded due to trailing gap > {TRAILING_GAP_TOLERANCE_S}s: "
        f"{total_trailing_skipped}"
    )
    (output_root / "completed.txt").touch()
