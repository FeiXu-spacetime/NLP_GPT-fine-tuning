import os
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
import pdb
import re
import torch

def load_or_download_wikitext(save_path, dataset_name='wikitext', dataset_version='wikitext-2-raw-v1', split=None):
    expected_file = os.path.join(save_path)
    # Check if the expected dataset file exists
    if os.path.exists(expected_file):
        print(f"Dataset found at {save_path}, loading from file...")
        if split:
            ds = load_from_disk(os.path.join(save_path, split))
        else:
            ds = load_from_disk(os.path.join(save_path))
    else:
        print(f"Dataset not found at {save_path}, downloading from Hugging Face...")
        # Download the dataset from Hugging Face and save it in the specified directory
        if split:
            ds = load_dataset(dataset_name, dataset_version, split=split)
        else:
            ds = load_dataset(dataset_name, dataset_version)
        ds.save_to_disk(save_path)
        print(ds)
    return ds

# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb#scrollTo=iaAJy5Hu3l_B
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])
# Define a preprocessing function to remove specific symbols
def remove_symbols(example):
    pdb.set_trace()
    # Define the symbols you want to skip or remove
    symbols_to_remove = r'[^a-zA-Z0-9 .,:]'  # Replace YOUR_SYMBOLS_HERE with the symbols you want to remove
    # Use regex to remove the symbols from the text field
    example['text'] = re.sub(symbols_to_remove, '', example['text'])
    return example
# Assuming 'data' is your dictionary containing the splits
def convert_to_dataset_dict(data):
    dataset_dict = {}
    for split, examples in data.items():
        # Convert each list of dictionaries into a Dataset
        dataset = Dataset.from_dict(examples)
        pdb.set_trace()
        # Add the Dataset to the DatasetDict under the appropriate split
        dataset_dict[split] = dataset
    # Convert the dictionary of Datasets into a DatasetDict
    return DatasetDict(dataset_dict)
# Concatenate all text and then split based on max_length
def group_texts(examples, block_size):
    # Concatenate all texts.
    
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    pdb.set_trace()
    total_length = len(concatenated_examples[torch.tensor(list(examples.keys()))[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result