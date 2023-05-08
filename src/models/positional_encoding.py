import torch
import torch.nn.functional as F
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size, maxlen):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        return token_embedding + self.pos_embedding[:, :token_embedding.shape[1], :]
