import torch

class Seq2seqAttention(torch.nn.Module):
    def __init__(self):
        super(Seq2seqAttention, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, decoder_hidden, encoder_states):
        decoder_hidden = decoder_hidden[:, :, None]
        decoder_to_encoder_states = torch.matmul(encoder_states, decoder_hidden)
        sm_decoder_to_encoder_states = self.softmax(decoder_to_encoder_states)
        weighted_encoder_states = torch.matmul(sm_decoder_to_encoder_states.transpose(2,1), encoder_states).squeeze(dim=1)
        return weighted_encoder_states
