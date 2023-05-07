import re
from typing import List, Tuple, Optional
import typing as tp
import json
import pandas as pd
from dataclasses import dataclass
import torch

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

EXTRA_ID_0 = "<extra_id_0>"


class TextUtils:
    @staticmethod
    def normalize_text(s: str) -> str:
        """Normalizes string, removes punctuation and
        non alphabet symbols

        Args:
            s (str): string to mormalize

        Returns:
            str: normalized string
        """
        s = s.lower()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Zа-яйёьъА-Яй]+", r" ", s)
        s = s.strip()
        return s

    @staticmethod
    def read_langs_pairs_from_file(filename: str):
        """Read lang from file

        Args:
            filename (str): path to dataset
            lang1 (str): name of first lang
            lang2 (str): name of second lang
            reverse (Optional[bool]): revers inputs (eng->ru of ru->eng)

        Returns:
            Tuple[Lang, Lang, List[Tuple[str, str]]]: tuple of
                (input lang class, out lang class, string pairs)
        """
        with open(filename, mode="r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        lang_pairs = []
        for line in tqdm(lines, desc="Reading from file"):
            lang_pair = tuple(map(TextUtils.normalize_text, line.split("\t")[:2]))
            lang_pairs.append(lang_pair)

        return lang_pairs


@dataclass
class T2TDataCollator:
    def __call__(self, batch: tp.List) -> tp.Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example["input_ids"] for example in batch])
        lm_labels = torch.stack([example["labels"] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        decoder_attention_mask = torch.stack(
            [example["decoder_attention_mask"] for example in batch]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_attention_mask": decoder_attention_mask,
        }


def short_text_filter_function(x, max_length, prefix_filter=None):
    len_filter = (
        lambda x: len(x[0].split(" ")) <= max_length
        and len(x[1].split(" ")) <= max_length
    )
    if prefix_filter:
        prefix_filter_func = lambda x: x[0].startswith(prefix_filter)
    else:
        prefix_filter_func = lambda x: True
    return len_filter(x) and prefix_filter_func(x)


def plot_results(fname: str = None):
    lines = None
    with open(fname, "r") as f:
        lines = f.readlines()

    tr_loss = []
    val_loss = []
    metric = []
    epoch = []
    for line in lines[1:]:
        data_dict = json.loads(line)
        if "epoch" in data_dict:
            epoch.append(data_dict["epoch"])
        tr_loss.append(data_dict["train_loss"])
        val_loss.append(data_dict["val_loss"])
        metric.append(data_dict["bleu_score"])

    tr_loss = np.array(tr_loss, dtype=np.float64)
    val_loss = np.array(val_loss, dtype=np.float64)
    metric = np.array(metric, dtype=np.float64)
    if len(epoch) == 0:
        epoch = np.arange(len(val_loss))
    else:
        epoch = np.array(epoch, dtype=np.float64)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    sns.lineplot(ax=ax[0], x=epoch, y=val_loss, color="orange", label="val_loss")
    sns.lineplot(ax=ax[0], x=epoch, y=tr_loss, color="blue", label="train_loss")
    sns.lineplot(ax=ax[1], x=epoch, y=metric, color="blue")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[1].set_title("BLEU")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("score")
    plt.show()


def to_csv(src_list: List[str], tgt_list: List[str], outf: str):
    df_data = {"en": [], "ru": []}
    for i in range(len(src_list)):
        df_data["en"].append(f"English: {src_list[i]}. Russian: {EXTRA_ID_0}")
        df_data["ru"].append(f"{EXTRA_ID_0} {tgt_list[i]}")
    pd.DataFrame(df_data).to_csv(outf, index=False)
