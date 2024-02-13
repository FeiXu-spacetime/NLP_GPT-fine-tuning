import torch
from torch.utils.data import Dataset
import json
import pdb

class My_Dataset(Dataset):
    def __init__(self, args, textdata, tokenizer):
        
        self.X = textdata['text'][:int(len(textdata['text'])*args.data_percentage)]
        pdb.set_trace()
        
        #print(self.X[:10])
        #self.X_encoded = tokenizer(self.X, truncation=True, padding=True, return_tensors="pt")
        self.X_encoded = tokenizer(self.X, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids'].cuda() # number of rows X max_length
        self.attention_mask = self.X_encoded['attention_mask'].cuda()
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])