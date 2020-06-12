import torch
from torch import nn
from utils import GradientReversal


class Classifier(nn.Module):
    def __init__(self, feature_size=512, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc_shared = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.fc_cls = nn.Linear(hidden_size, num_classes)
        self.fc_ctx = nn.Linear(hidden_size, num_classes)

    def forward(self, features):
        out = self.fc_shared(features)
        cls_pred = self.fc_cls(out)
        ctx_pred = self.fc_ctx(out)
        return cls_pred, ctx_pred


class AttentionResidualBlock(nn.Module):
    def __init__(self, in_size, out_size, proc_fn, hidden_size=[]):
        super().__init__()
        mods = nn.ModuleList()
        size_list = [in_size]
        size_list.extend(hidden_size)
        size_list.append(out_size)
        for i in range(1, len(size_list)):
            mods.append(nn.Linear(size_list[i-1], size_list[i], bias=False))
            if i + 1 != len(size_list):
                mods.append(nn.ReLU(inplace=True))
        self.attention = mods
        self.proc_fn = proc_fn

    def forward(self, x):
        for mod in self.attention:
            x = mod(x)
        mask = torch.sigmoid(x)
        return (1 + mask) * self.proc_fn(x)


class AttentionClassifier(nn.Module):
    def __init__(self, feature_size=512, hidden_size=512, num_classes=10):
        super().__init__()
        self.feature_size = feature_size
        proc_branch = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.attention = AttentionResidualBlock(
            feature_size, hidden_size, proc_branch)
        self.fc_cls = nn.Sequential(
            nn.Linear(hidden_size, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(50, num_classes)
        )
        self.fc_ctx = nn.Sequential(
            GradientReversal(),
            nn.Linear(hidden_size, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, num_classes)
        )
        self.discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(2 * hidden_size, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 1)
        )

    def forward(self, features):
        out = self.attention(features)
        cls_pred = self.fc_cls(out)
        ctx_pred = self.fc_ctx(out)
        return cls_pred, ctx_pred
