from typing import Tuple, List

import numpy as np
from torchtext.data.metrics import bleu_score


def bleu_scorer(predicted: np.ndarray, actual: np.ndarray, target_tokenizer):
    """Convert predictions to sentences and calculate
    BLEU score.

    Args:
        predicted (np.ndarray): batch of indices of predicted words
        actual (np.ndarray): batch of indices of ground truth words

    Returns:
        Tuple[float, List[str], List[str]]: tuple of
            (
                bleu score,
                ground truth sentences,
                predicted sentences
            )
    """
    batch_bleu = []
    predicted_sentences = []
    actual_sentences = []
    for a, b in zip(predicted, actual):
        words_predicted = target_tokenizer.decode(a)
        words_actual = target_tokenizer.decode(b)
        batch_bleu.append(bleu_score([words_predicted], [[words_actual]], max_n=4, weights=[0.25, 0.25, 0.25, 0.25]))
        predicted_sentences.append(" ".join(words_predicted))
        actual_sentences.append(" ".join(words_actual))
    batch_bleu = np.mean(batch_bleu)
    return batch_bleu, actual_sentences, predicted_sentences