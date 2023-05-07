import typing as tp

from transformers import PreTrainedTokenizerFast

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BPETokenizer:
    UNK = "<unk>"
    PAD = "<pad>"
    EOS = "<eos>"
    BOS = "<bos>"

    def __init__(self, sentence_list, to_train: bool = True):
        """
        sentence_list - список предложений для обучения
        """
        self.is_trained = False
        self.tokenizer = Tokenizer(BPE(unk_token=self.UNK))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.special_tokens_set = {self.UNK, self.PAD, self.BOS, self.EOS}
        self.trainer = BpeTrainer(special_tokens=[self.UNK, self.PAD, self.BOS, self.EOS])
        if to_train:
            self._train(sentence_list)
        self.word2index = dict()
        self.index2word = dict()
        self.build_token_mapping()

    def build_token_mapping(self):
        self.word2index = self.tokenizer.get_vocab()
        self.index2word = {v: k for k, v in self.word2index.items()}

    def save(self, file: str):
        self.tokenizer.save(file)

    @classmethod
    def load(cls, file: str) -> "BPETokenizer":
        toker = BPETokenizer([], to_train=False)
        toker.tokenizer = PreTrainedTokenizerFast(tokenizer_file=file)
        toker.is_trained = True
        return toker

    def _train(self, corpus: tp.List[str]):
        self.tokenizer.train_from_iterator(iterator=corpus, trainer=self.trainer)
        self.is_trained = True

    def __call__(self, sentence, pad=True, truncate=True, max_len=64):
        """
        sentence - входное предложение
        """
        ids = self.tokenizer.encode(sentence).ids
        eos_idx = self.word2index[self.EOS]
        bos_idx = self.word2index[self.BOS]
        ids = [bos_idx, *ids, eos_idx]
        if len(ids) > max_len and truncate:
            ids = ids[:max_len]
            ids[-1] = self.word2index[self.EOS]
        elif len(ids) < max_len and pad:
            pad_idx = self.word2index[self.PAD]
            len_diff = max_len - len(ids)
            ids = ids + [pad_idx] * len_diff
        return ids

    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        predicted_tokens = []
        for token_id in token_list:
            predicted_token = self.index2word[token_id]
            if predicted_token not in self.special_tokens_set:
                predicted_tokens.append(predicted_token)
        return predicted_tokens
