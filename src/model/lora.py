import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


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


class LoraModel(nn.Module):
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer, linear_lora=True):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        self.linear_lora = linear_lora

        if self.linear_lora:
            self.lora_(self.model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def lora_(self, module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LinearLora(child))
            else:
                self.lora_(child)

    def params_cnt(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        return {
            "All parameters": all_parameters,
            "Trainable parameters": trainable_parameters,
        }
