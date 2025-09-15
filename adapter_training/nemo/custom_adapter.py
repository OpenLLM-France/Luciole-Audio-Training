from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Set, Union

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

from nemo.collections.asr.parts.submodules.jasper import (
    JasperBlock,
    MaskedConv1d,
    ParallelBlock,
    SqueezeExcite,
    init_weights,
    jasper_activations,
)
from nemo.collections.asr.parts.submodules.conformer_modules import (
    ConformerLayer,
)
from nemo.collections.asr.parts.submodules.tdnn_attention import (
    AttentivePoolLayer,
    StatsPoolLayer,
    TDNNModule,
    TDNNSEModule,
)
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin, adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    LengthsType,
    LogitsType,
    LogprobsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging
from nemo.collections.asr.modules import ConvASRDecoder

class CustomAdapter(ConvASRDecoder):

    def __init__(self, feat_in, num_classes, init_mode="xavier_uniform", vocabulary=None, add_blank=True):
        super().__init__(feat_in, num_classes, init_mode, vocabulary, add_blank)
        
        self.decoder_layers = nn.Sequential(
            # torch.nn.AvgPool1d(kernel_size=2, stride=2), 
            # torch.nn.Linear(self._feat_in, self._num_classes, bias=True)
            torch.nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True)
        )
        self.apply(lambda x: init_weights(x, mode=init_mode))

        accepted_adapters = [adapter_utils.LINEAR_ADAPTER_CLASSPATH]
        self.set_accepted_adapter_types(accepted_adapters)

        # to change, requires running ``model.temperature = T`` explicitly
        self.temperature = 1.0

    @typecheck()
    def forward(self, encoder_output):
        # Adapter module forward step
        # if self.is_adapter_available():
        #     encoder_output = encoder_output.transpose(1, 2)  # [B, T, C]
        #     encoder_output = self.forward_enabled_adapters(encoder_output)
        #     encoder_output = encoder_output.transpose(1, 2)  # [B, C, T]

        # if self.temperature != 1.0:
        #     return torch.nn.functional.log_softmax(
        #         self.decoder_layers(encoder_output).transpose(1, 2) / self.temperature, dim=-1
        #     )
        return torch.nn.functional.log_softmax(self.decoder_layers(encoder_output).transpose(1, 2), dim=-1)