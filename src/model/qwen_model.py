import torch
from torch import nn
from src.model.lora import LinearLora
from transformers import AutoTokenizer, AutoModelForCausalLM

class Qwen(nn.Module):
    def __init__(self, linear_lora=True):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        
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
        return {'All parameters': all_parameters, 'Trainable parameters': trainable_parameters}