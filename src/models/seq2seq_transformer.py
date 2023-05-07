import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

import typing as tp

import src.metrics as metrics
from src.models.positional_encoding import PositionalEncoding


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    MAX_LEN = 64

    def __init__(
        self,
        model_config: tp.Dict[str, tp.Any],
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.device = model_config["device"]
        self.transformer = nn.Transformer(
            d_model=model_config["emb_size"],
            nhead=model_config["nhead"],
            num_encoder_layers=model_config["num_encoder_layers"],
            num_decoder_layers=model_config["num_decoder_layers"],
            dim_feedforward=model_config["dim_feedforward"],
            dropout=model_config["dropout"],
            batch_first=True,
        )
        self.generator = nn.Linear(
            model_config["emb_size"], model_config["tgt_vocab_size"]
        )
        self.src_tok_emb = TokenEmbedding(
            model_config["src_vocab_size"], model_config["emb_size"]
        )
        self.tgt_tok_emb = TokenEmbedding(
            model_config["tgt_vocab_size"], model_config["emb_size"]
        )
        self.positional_encoding = PositionalEncoding(
            model_config["emb_size"], maxlen=128
        )
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=model_config["lr"],
            weight_decay=model_config["weight_decay"],
        )
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[10, 20, 30, 40, 50, 60, 70, 80, 90], gamma=0.65
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)

    def forward(
        self,
        src: Tensor,
        trg: Tensor = None,
        src_mask: Tensor = None,
        tgt_mask: Tensor = None,
        src_padding_mask: Tensor = None,
        tgt_padding_mask: Tensor = None,
        memory_key_padding_mask: Tensor = None,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(
        self,
        src: Tensor,
        src_mask: Tensor,
        src_padding_mask: tp.Optional[Tensor] = None,
    ):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask, src_padding_mask
        )

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: tp.Optional[Tensor] = None,
        memory_mask: tp.Optional[Tensor] = None,
        tgt_key_padding_mask: tp.Optional[Tensor] = None,
        memory_key_padding_mask: tp.Optional[Tensor] = None,
    ):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
        )

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(
            0, 1
        )
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _create_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(
            torch.bool
        )
        src_padding_mask = (src == self.PAD_IDX).type(torch.bool)
        tgt_padding_mask = (tgt == self.PAD_IDX).type(torch.bool)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def training_step(self, batch):
        src, trg = batch
        trg_in = trg[:, :-1]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self._create_mask(
            src, trg_in
        )

        logits = self.forward(
            src,
            trg_in,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )
        self.optimizer.zero_grad()

        trg_out = trg[:, 1:]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(self, batch):
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self._create_mask(
            src, tgt_input
        )

        logits = self.forward(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        tgt_out = tgt[:, 1:]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        return loss.item()

    def generate(self, src):
        ys = (
            torch.ones(src.shape[0], 1)
            .fill_(self.BOS_IDX)
            .type(torch.long)
            .to(self.device)
        )
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self._create_mask(
            src, ys
        )
        memory = self.encode(src, src_mask, src_padding_mask)
        ended = torch.zeros(src.shape[0], dtype=torch.int32)
        for i in range(self.MAX_LEN - 1):
            out = self.decode(
                ys,
                memory,
                tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask,
            )
            prob = self.generator(out[:, -1, :])
            next_words = torch.argmax(prob, dim=1)
            next_words[ended == 1] = self.PAD_IDX
            ended[next_words == self.EOS_IDX] = 1
            ys = torch.cat([ys, next_words.unsqueeze(dim=1)], dim=1)
            if torch.all(ended == 1).item():
                break
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self._create_mask(
                src, ys
            )
        return ys

    def eval_bleu(self, predicted_ids, target_tensor, target_tokenizer):
        # predicted = torch.stack(predicted_ids_list)
        predicted = predicted_ids.squeeze().detach().cpu().numpy()[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences
