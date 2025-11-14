from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.lora import LoraModel


class Pythia(LoraModel):
    def __init__(self, linear_lora=True):
        super().__init__(
            AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b"),
            AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b"),
            linear_lora,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
