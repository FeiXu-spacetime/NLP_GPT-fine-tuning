import torch
from torch.utils.data import Dataset
import json
import pdb
import utils 
from utils import group_texts

class My_Dataset(Dataset):
    def __init__(self, args, textdata, tokenizer):
        
        #self.X = textdata['text'][:int(len(textdata['text'])*args.data_percentage)]
        self.X = textdata[:int(len(textdata)*args.data_percentage)]
        
        #print(self.X[:10])
        #self.X_encoded = tokenizer(self.X, truncation=True, padding=True, return_tensors="pt")
        #self.X_encoded = tokenizer(self.X, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = torch.tensor(self.X['input_ids']).cuda() # number of rows X max_length
        self.attention_mask = torch.tensor(self.X['attention_mask']).cuda()
        
        

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])