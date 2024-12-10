from typing import Tuple, List, Any
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .dataset import SarcasmDataset

def find_max_len(
    df: pd.DataFrame, 
    text_column: str, 
    percentile: int = 95,
    model_name: str = 'bert-base-uncased'
) -> int:

    """ Find a decent max_len for the model """

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize all texts and count the number of tokens
    token_lengths = df[text_column].apply(lambda text: len(tokenizer.encode(text, add_special_tokens=True))).tolist()

    # Calculate the desired percentile of token lengths
    optimal_length = int(pd.Series(token_lengths).quantile(percentile / 100.0))

    return optimal_length

def convert_dataframe_to_dataloaders(
    model_name: str, 
    train_df: pd.DataFrame, 
    validate_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    """ Create Dataloaders from a Pandas DataFrame """

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Factory function to create a dataset
    def create_dataset(texts: List[str], labels: List[Any]) -> SarcasmDataset:
      return SarcasmDataset(texts, labels, tokenizer)

    train_dataset = create_dataset(
      train_df['text'].tolist(),
      train_df['labels'].tolist()
    )

    validate_dataset = create_dataset(
      validate_df['text'].tolist(),
      validate_df['labels'].tolist()
    )

    test_dataset = create_dataset(
      test_df['text'].tolist(),
      test_df['labels'].tolist()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validate_loader, test_loader