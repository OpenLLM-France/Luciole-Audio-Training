"""Lanceur `vllm serve` pour les checkpoints SALM.

`vllm serve` seul échoue sur nos checkpoints :

    Value error, The checkpoint you are trying to load has model type
    `nemo_speechlm` but Transformers does not recognize this architecture.

Le plugin s'enregistre par l'entry point ``vllm.general_plugins``, que vLLM ne
charge que dans le processus moteur — alors que le front-end de l'API valide la
config du modèle AVANT, dans le processus principal. On appelle donc ``register()``
à la main avant de rendre la main à la CLI de vLLM. Le processus moteur, lui,
continue de passer par l'entry point : rien à changer de ce côté.

Usage : python serve_salm.py serve <ckpt_dir> [options vllm serve...]
"""

import sys

from nemo.collections.speechlm2.vllm.salm import register

register()

from vllm.entrypoints.cli.main import main  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = "vllm"
    main()
