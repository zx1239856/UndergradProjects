import random

import numpy as np
import torch
from torch.utils.data import Dataset


class NicoDataset(Dataset):
    def __init__(self, dataset_path, device):
        # data is sized (n, 514) containing n samples, each with 514 dim,
        # where the first 512 dim are features, dim 513 is context label (0-9), 
        # and dim 514 is class label (0-9)
        data = torch.tensor(np.load(dataset_path), device=device)
        self.features = data[:, :512].type(torch.float32)
        self.ctx_labels = data[:, -2].type(torch.int64)
        self.cls_labels = data[:, -1].type(torch.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.cls_labels[index], self.ctx_labels[index]


class NicoPairedDataset(Dataset):
    def __init__(self, dataset_path, device):
        data = torch.tensor(np.load(dataset_path), device=device)
        self.features = data[:, :512].type(torch.float32)
        self.ctx_labels = data[:, -2].type(torch.int64)
        self.cls_labels = data[:, -1].type(torch.int64)
        max_class = int(torch.max(self.cls_labels).cpu().numpy())
        self._indices = []
        for clz in range(max_class + 1):
            clz_indices = torch.where(self.cls_labels == clz)[0]
            sub_ctx_labels = self.ctx_labels[clz_indices]
            max_ctx = int(torch.max(sub_ctx_labels))
            ctx_indices = []
            for k in range(max_ctx + 1):
                ctx_indices.append(clz_indices[sub_ctx_labels == k].cpu().numpy())
            clz_indices = clz_indices.cpu().numpy()
            for i in clz_indices:
                diff = 0
                cur_ctx = int(self.ctx_labels[i])
                for ctx, arr in enumerate(ctx_indices):
                    if ctx == cur_ctx:
                        continue
                    if len(arr) == 0:
                        continue
                    self._indices.append((i, np.random.choice(arr)))
                    diff += 1
                ind = ctx_indices[cur_ctx]
                j_list = np.random.choice(ind[ind != i], diff)
                for j in j_list:
                    self._indices.append((i, j))

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        i, j = self._indices[index]
        return (self.features[i], self.cls_labels[i], self.ctx_labels[i]), (self.features[j], self.cls_labels[j], self.ctx_labels[j])
