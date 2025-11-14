from torch.utils.data import Dataset
from datasets import load_dataset

class RLHFDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_len=1024):
        self.dataset = load_dataset(
            "Anthropic/hh-rlhf", split=split
        )
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]["chosen"]
        split_text = text.rsplit('Assistant:', 1)
        prompt_len = len(self.tokenizer(split_text[0] + ' Assistant:' , add_special_tokens=False)["input_ids"])

        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        for i in range(1, prompt_len):
            labels[i] = -100

        return input_ids, attention_mask, labels