import torch

import metrics


class Seq2SeqT5(torch.nn.Module):
    def __init__(self, device):
        super(Seq2SeqT5, self).__init__()
        self.device = device
       # TODO: Реализуйте конструктор seq2seq t5 - https://huggingface.co/docs/transformers/model_doc/t5

    def forward(self, input_tensor: torch.Tensor):
        # TODO: Реализуйте forward pass для модели, при необходимости реализуйте другие функции для обучения
        pass


    def training_step(self, batch):
        # TODO: Реализуйте обучение на 1 батче данных по примеру seq2seq_rnn.py
        pass

    def validation_step(self, batch):
        # TODO: Реализуйте оценку на 1 батче данных по примеру seq2seq_rnn.py
        pass

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = torch.stack(predicted_ids_list)
        predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences




