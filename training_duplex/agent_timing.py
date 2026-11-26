"""`data.agent_timing` (+ `data.lag`, `data.wait_words`): one timing for the whole model, applied to every
`conversation_as_duplex` entry of the train and validation input_cfgs.

The timing is a property of the model, not of a dataset: with `simultaneous` the duplex dataset reads the agent
text as timestamped (`<|i|> word <|j|>`), which an `after` answer is not, so the two cannot be mixed in one run.
`data.agent_text_timestamps` is derived from it (true for `simultaneous`).
"""
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

TIMINGS = ("after", "simultaneous")
# Keys of conversation_as_duplex (NeMo fork, cutset.py) set once for the model, under `data`.
MODEL_KEYS = ("agent_timing", "lag", "wait_words")


def _inject(input_cfg, values: dict, where: str):
    """`values` into every conversation_as_duplex entry of `input_cfg` (a list, or a YAML path), groups
    included. Returns the list (loaded from its path if needed)."""
    if isinstance(input_cfg, str):
        input_cfg = OmegaConf.load(input_cfg)
    assert isinstance(input_cfg, (list, ListConfig)), f"{where}: input_cfg is not a list: {input_cfg!r}"
    for i, entry in enumerate(input_cfg):
        name = f"{where}[{i}]"
        if entry.get("input_cfg") is not None:
            with open_dict(entry):
                entry.input_cfg = _inject(entry.input_cfg, values, name)
        if entry.get("type") != "conversation_as_duplex":
            continue
        with open_dict(entry):
            for k, v in values.items():
                if k in entry and entry[k] != v:
                    raise ValueError(f"{name} ({entry.get('manifest_filepath')}): {k}={entry[k]} but data.{k}={v}; "
                                     f"set it once under `data` (conf/duplex_<model>.yaml)")
                entry[k] = v
    return input_cfg


def apply_agent_timing(cfg: DictConfig) -> None:
    """Resolve `data.agent_timing` into the train / validation input_cfgs and `data.agent_text_timestamps`.

    Call before OmegaConf.resolve: the input_cfgs given as YAML paths are loaded into the config, so their
    ${oc.env:...} interpolations are resolved with the rest. `data.validation_ds.datasets` may itself be a
    YAML path (conf/data/val_cfg_<model>.yaml).
    """
    data = cfg.data
    timing = data.get("agent_timing", "after")
    assert timing in TIMINGS, f"data.agent_timing={timing!r}, expected one of {TIMINGS}"
    values = {k: data[k] for k in MODEL_KEYS if data.get(k) is not None}
    values["agent_timing"] = timing
    timestamps = timing == "simultaneous"
    if data.get("agent_text_timestamps") is not None and bool(data.agent_text_timestamps) != timestamps:
        raise ValueError(f"data.agent_text_timestamps={data.agent_text_timestamps} contradicts "
                         f"data.agent_timing={timing}: drop it, it is derived")
    with open_dict(cfg):
        data.agent_text_timestamps = timestamps
        if data.get("train_ds") is not None and data.train_ds.get("input_cfg") is not None:
            data.train_ds.input_cfg = _inject(data.train_ds.input_cfg, values, "data.train_ds.input_cfg")
        val = data.get("validation_ds")
        if val is not None and val.get("datasets") is not None:
            if isinstance(val.datasets, str):
                val.datasets = OmegaConf.load(val.datasets)
            for name, item in val.datasets.items():
                if item.get("input_cfg") is not None:
                    item.input_cfg = _inject(item.input_cfg, values, f"data.validation_ds.datasets.{name}.input_cfg")
