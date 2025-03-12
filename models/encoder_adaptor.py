import torch
import torch.nn as nn
import torch.nn.functional as F


# class EncoderAdaptor(nn.Module):
#     def __init__(self, model_conf):
#         super().__init__()
#         self.k = model_conf.encoder_projector_ds_rate
#         assert self.k > 0, "Downsampling rate (k) must be positive."
#         self.hidden_dim = getattr(model_conf, 'encoder_projector_hidden_dim', 2048)
        
#         self.encoder_dim = model_conf.encoder_dim 
#         self.llm_dim = model_conf.llm_dim
#         self.linear1 = nn.Linear(self.encoder_dim * self.k, self.hidden_dim)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(self.hidden_dim, self.llm_dim)

#     def forward(self, x):
#         batch_size, seq_len, dim = x.size()
#         num_frames_to_discard = seq_len % self.k 
#         if num_frames_to_discard > 0:
#             x = x[:, :-num_frames_to_discard, :]
#         seq_len = x.size(1)
        
#         x = x.contiguous()
#         x = x.view(batch_size, seq_len // self.k, dim * self.k)
#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.linear2(x)
#         return x

class EncoderAdaptor(nn.Module):
    def __init__(self, model_conf):
        super().__init__()
        self.downsampling_rate = model_conf.encoder_projector_ds_rate
        assert self.downsampling_rate > 0, "Downsampling rate must be positive."
        
        self.encoder_dim = model_conf.encoder_dim
        self.llm_dim = model_conf.llm_dim
        self.hidden_dim = getattr(model_conf, 'encoder_projector_hidden_dim', 2048)
        self.activation = getattr(model_conf, 'encoder_projector_activation', 'relu')
        
        self.linear1 = nn.Linear(self.encoder_dim * self.downsampling_rate, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(getattr(model_conf, 'encoder_projector_dropout', 0.1))
        self.linear2 = nn.Linear(self.hidden_dim, self.llm_dim)
        
        if self.activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif self.activation == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        k = self.downsampling_rate
        target_seq_len = ((seq_len + k - 1) // k) * k  # Round up to nearest multiple of k
        padding = target_seq_len - seq_len
        if padding > 0:
            x = F.pad(x, (0, 0, 0, padding), mode='constant', value=0)  # Pad on seq_len dimension
        
        x = x.view(batch_size, target_seq_len // k, dim * k)
        x = self.linear1(x)
        x = self.norm(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
def set_encoder_adaptor(model_conf):
    encoder_adaptor = EncoderAdaptor(model_conf)
    return encoder_adaptor