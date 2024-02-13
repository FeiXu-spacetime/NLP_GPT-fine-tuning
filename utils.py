import os
from datasets import load_dataset, load_from_disk
import pdb

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
        pdb.set_trace() 
        ds.save_to_disk(save_path)
        print(ds)
    return ds
