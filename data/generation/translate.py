from transformers import MarianMTModel, MarianTokenizer

_translation_models = {
    "fr-en": "Helsinki-NLP/opus-mt-tc-big-fr-en",
    "en-fr": "Helsinki-NLP/opus-mt-tc-big-en-fr",
}

_loaded_models = {}

def translate_text(text, lan_from, lan_to):
    """
    Translate text from lan_from to lan_to

    Example:
    translate_text("Please transcribe this", "en", "fr") -> "Veuillez transcrire ceci"

    :param text: text to translate
    :param lan_from: source language
    :param lan_to: target language
    :return: translated text
    """

    if isinstance(text, str):
        return translate_text([text], lan_from, lan_to)[0]

    key = f"{lan_from}-{lan_to}"
    if key not in _translation_models:
        raise ValueError(f"Translation model for {key} not found")
    model_name = _translation_models[key]
    if model_name not in _loaded_models:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _loaded_models[model_name] = (tokenizer, model)
    else:
        tokenizer, model = _loaded_models[model_name]

    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return [
        tokenizer.decode(t, skip_special_tokens=True) for t in translated
    ]
