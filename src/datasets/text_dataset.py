from datasets import load_dataset
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokenizer, split='train[:1%]', max_len=64):
        self.dataset = load_dataset('agentlans/high-quality-english-sentences', split=split)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        tokenized = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return tokenized['input_ids'].squeeze(0), tokenized['attention_mask'].squeeze(0)