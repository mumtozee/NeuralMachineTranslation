import torch

class DecoderRNN(torch.nn.Module):
    def __init__(self, embedding_size: int, decoder_vocab_size: int, hidden_size: int) -> None:
        super(DecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = torch.nn.Embedding(decoder_vocab_size, embedding_size)
        self.gru = torch.nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output, hidden = self.gru(output, hidden)
        return output, hidden