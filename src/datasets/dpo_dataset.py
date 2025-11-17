from torch.utils.data import Dataset
from datasets import load_dataset

class DPODataset(Dataset):
    def __init__(self, tokenizer, split='train', max_len=128):
        self.dataset = load_dataset(
            'Anthropic/hh-rlhf', split=split
        )
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        def get_data(text):
            tokenized = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt',
            )

            input_ids = tokenized['input_ids'].squeeze(0)
            attention_mask = tokenized['attention_mask'].squeeze(0)
            labels = input_ids.clone()

            return input_ids, attention_mask, labels
        
        input_ids, attention_mask, labels_chosen = get_data(self.dataset[idx]['chosen'])
        _, _, labels_rejected = get_data(self.dataset[idx]['rejected'])
        
        return input_ids, attention_mask, labels_chosen, labels_rejected