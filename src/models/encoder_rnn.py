import torch

class EncoderRNN(torch.nn.Module):
    def __init__(self, encoder_vocab_size: int, embedding_size: int, hidden_size: int) -> None:
        super(EncoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = torch.nn.Embedding(encoder_vocab_size, embedding_size)
        self.gru = torch.nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input).squeeze()
        output, hidden = self.gru(embedded)
        return output, hidden