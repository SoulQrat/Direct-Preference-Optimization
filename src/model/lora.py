import torch.nn as nn

class LinearLora(nn.Module):
    def __init__(self, pretrain: nn.Linear, rank=4, alpha=16):
        super().__init__()
        self.pretrain = pretrain
        self.pretrain.weight.requires_grad = False
        if self.pretrain.bias is not None:
            self.pretrain.bias.requires_grad = False

        self.in_features = pretrain.in_features
        self.out_features = pretrain.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Linear(self.in_features, self.rank, bias=False)
        self.B = nn.Linear(self.rank, self.out_features, bias=False)
        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        out = self.pretrain(x)
        return out + self.B(self.A(x)) * self.scaling