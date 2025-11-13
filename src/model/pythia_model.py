from src.model.lora import LoraModel
from transformers import AutoTokenizer, AutoModelForCausalLM

class Pythia(LoraModel):
    def __init__(self, linear_lora=True):
        super().__init__(AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b"), AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b"), linear_lora)
