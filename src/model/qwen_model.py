from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.lora import LoraModel


class Qwen(LoraModel):
    def __init__(self, linear_lora=True):
        super().__init__(
            AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B"),
            AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"),
            linear_lora,
        )
