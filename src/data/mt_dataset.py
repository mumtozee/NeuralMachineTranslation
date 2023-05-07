import torch
from torch.utils.data import Dataset


class MTDataset(Dataset):
    def __init__(self, tokenized_source_list, tokenized_target_list, dev):
        self.tokenized_source_list = tokenized_source_list
        self.tokenized_target_list = tokenized_target_list
        self.device = dev

    def __len__(self):
        return len(self.tokenized_source_list)

    def __getitem__(self, idx):
        source_ids, target_ids = torch.tensor(self.tokenized_source_list[idx]).to(self.device), \
                                 torch.tensor(self.tokenized_target_list[idx]).to(self.device)
        return source_ids, target_ids
