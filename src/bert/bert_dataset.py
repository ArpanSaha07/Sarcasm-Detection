import torch
from torch.utils.data import Dataset

class SarcasmDataset(Dataset):

    def __init__(
          self, 
          texts: list[str], 
          labels: list[int], 
          tokenizer,
          max_len: int = 128
    ):
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length.")

        self.texts = texts                  # Documents
        self.labels = labels                # Labels
        self.tokenizer = tokenizer          # Depends on the LLM
        self.max_len = max_len              # Maximum length of tokens for text inputs

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize and encode the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,        # Adds '[CLS]' and '[SEP]'
            max_length=self.max_len,        # Truncate or pad to max_len
            truncation=True,
            padding='max_length',           # Pad to max_length
            return_attention_mask=True,     # Generate the attention mask
            return_tensors='pt'             # Return PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),           # Tensor of token IDs
            'attention_mask': encoding['attention_mask'].squeeze(), # Tensor of attention masks
            'labels': torch.tensor(label, dtype=torch.long)          # Label tensor
        }