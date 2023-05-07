import torch

import metrics
from models.attention import Seq2seqAttention
from models.decoder_rnn import DecoderRNN
from models.encoder_rnn import EncoderRNN


class Seq2SeqRNN(torch.nn.Module):
    def __init__(
        self,
        encoder_vocab_size: int,
        encoder_embedding_size: int,
        decoder_embedding_size: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        decoder_vocab_size: int,
        lr: float,
        device: str,
        target_tokenizer,
    ):
        super(Seq2SeqRNN, self).__init__()
        self.device = device
        self.encoder = EncoderRNN(
            encoder_vocab_size=encoder_vocab_size,
            embedding_size=encoder_embedding_size,
            hidden_size=encoder_hidden_size,
        ).to(self.device)
        self.attention_module = Seq2seqAttention().to(self.device)
        self.decoder = DecoderRNN(
            embedding_size=decoder_embedding_size,
            decoder_vocab_size=decoder_vocab_size,
            hidden_size=decoder_hidden_size,
        ).to(self.device)

        self.vocab_projection_layer = torch.nn.Linear(
            decoder_hidden_size + encoder_hidden_size, decoder_vocab_size
        ).to(self.device)
        self.softmax = torch.nn.LogSoftmax(dim=1).to(self.device)
        self.criterion = torch.nn.NLLLoss()

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.encoder.parameters()},
                {"params": self.decoder.parameters()},
            ],
            lr=lr,
        )
        self.target_tokenizer = target_tokenizer

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        encoder_states, encoder_last_hidden = self.encoder(input_tensor.squeeze(dim=0))
        decoder_hidden = encoder_last_hidden
        decoder_input = torch.tensor(
            [[0] * batch_size], dtype=torch.long, device=self.device
        ).view(batch_size, 1)
        predicted = []
        each_step_distributions = []
        for _ in range(input_tensor.shape[1]):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            weighted_decoder_output = self.attention_module(
                decoder_hidden.squeeze(dim=0), encoder_states
            )
            decoder_output = decoder_output.squeeze(dim=1)
            decoder_output = torch.cat([decoder_output, weighted_decoder_output], dim=1)
            linear_vocab_proj = self.vocab_projection_layer(decoder_output)
            target_vocab_distribution = self.softmax(linear_vocab_proj)
            _, topi = target_vocab_distribution.topk(1)
            predicted.append(topi.clone().detach().cpu())
            decoder_input = topi
            each_step_distributions.append(target_vocab_distribution)
        return predicted, each_step_distributions

    def training_step(self, batch):
        self.optimizer.zero_grad()
        input_tensor, target_tensor = batch
        predicted, decoder_outputs = self.forward(input_tensor)
        target_tensor = target_tensor[:, :, None]
        target_length = target_tensor.shape[1]
        loss = 0.0
        for di in range(target_length):
            loss += self.criterion(
                decoder_outputs[di].squeeze(), target_tensor[:, di, :].squeeze()
            )
        loss = loss / target_length
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation_step(self, batch):
        input_tensor, target_tensor = batch
        predicted, decoder_outputs = self.forward(input_tensor)
        target_tensor = target_tensor[:, :, None]
        with torch.no_grad():
            target_length = target_tensor.shape[1]
            loss = 0
            for di in range(target_length):
                loss += self.criterion(
                    decoder_outputs[di].squeeze(), target_tensor[:, di, :].squeeze()
                )
            loss = loss / target_length

        return loss.item()

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = torch.stack(predicted_ids_list)
        predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences
