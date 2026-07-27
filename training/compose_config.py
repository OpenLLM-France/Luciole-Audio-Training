#!/usr/bin/env python
"""Compose a Hydra config into a single flat, resolved YAML file.

Tools like NeMo's scripts/speechlm2/oomptimizer.py load their config with a plain
``OmegaConf.load(path)``, which ignores Hydra's ``defaults:`` list — so pointing them at
a composition file (e.g. ``conf/run/encoder_xp.yaml``) fails with ``Missing key model``.
This composes it the way ``salm_train.py`` does and writes the result out flat.

Usage:
    python compose_config.py <conf_dir> <config_name> <output_yaml>
"""
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(f"usage: {sys.argv[0]} <conf_dir> <config_name> <output_yaml>")

    conf_dir, config_name, output_yaml = sys.argv[1:4]

    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name=config_name)

    # Batch sizing only needs the model *shape*, so drop the warm start — otherwise
    # ${oc.env:CHECKPOINT} would have to be set just to satisfy resolve(). `del`, not
    # `pop()`: pop() resolves the value before removing it.
    OmegaConf.set_struct(cfg, False)
    if "init_from_checkpoint" in cfg.model.keys():
        del cfg.model["init_from_checkpoint"]

    OmegaConf.resolve(cfg)

    # `lora: null` means "disable LoRA", but Hydra deep-merges and can only null the key,
    # not delete it — and maybe_install_lora keys off its PRESENCE, so it would call
    # LoraConfig(**None). Drop it.
    if cfg.model.get("lora", "x") is None:
        OmegaConf.set_struct(cfg, False)
        del cfg.model["lora"]

    OmegaConf.save(cfg, output_yaml)
    print(f"Wrote composed config to {output_yaml}")


if __name__ == "__main__":
    main()
