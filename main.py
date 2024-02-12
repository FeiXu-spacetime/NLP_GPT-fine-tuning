import json
import os
from pprint import pprint
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#import transformers
from datasets import load_dataset
from my_dataset import My_Dataset
from utils import *
from torch.optim import Adam
#from huggingface_hub import notebook_login
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer
)

import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def decoder_training(args, model):
    
    num_gpus = torch.cuda.device_count()
    print('number of avilable gpus: %d' % num_gpus)

    # Constrain most sources of randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = 'cuda'
    model = model.to(device)
    
    parameters = list(model.parameters())
    optim = torch.optim.Adam(parameters, args.learning_rate)

    if args.use_data_parallel and num_gpus > 1:
        device_ids_list = list(range(num_gpus))
        model = torch.nn.DataParallel(model, device_ids=device_ids_list)
        #model = DDP(model, device_ids=device_ids_list)
       
    # Create the dataset
    #dir = os.path.join(args.dataset_path, args.mesh.name)
    dataset = wikitext_dataset_train_tokenized

    # Initialize variables
    start_epoch = 0
    start_batch = 0
    save_path = os.path.join(args.save_dir, args.model_name)
    losses = []
    all_indices = list(range(len(dataset)))

    if args.continue_train:
        checkpoint = torch.load(save_path, map_location=device)
        if 'attention_state_dict' in checkpoint.keys():
            attention.load_state_dict(checkpoint['attention_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch_idx']
        all_indices = checkpoint['shuffled_indices']
        losses = checkpoint['losses']

    
    for epoch in range(start_epoch, args.num_epochs):
         # If this is a new epoch, shuffle the indices
        if epoch != start_epoch or start_batch == 0:
            random.shuffle(all_indices)

        # SubsetRandomSampler handles the batching
        sampler = SubsetRandomSampler(all_indices[start_batch * args.batch_size:])
        
        # create the DataLoader
        dataloader =  DataLoader(wikitext_dataset_train_tokenized, sampler=sampler, batch_size=args.batch_size)
        #dataloader = DataLoader(dataset, batch_size=args.batch_size)
        pdb.set_trace()
        # Sampling data using shuffled indices
        num_batches = len(dataloader)
        
        for batch_idx, (batch_data, batch_attention_mask) in enumerate(dataloader):
            print(batch_data, batch_attention_mask)
            print("epoch_idx: %d" % (epoch))
            print("batch_idx: %d" % (batch_idx + start_batch))
            #batch_size = batch_labels.shape[0]
            #batch_data = batch_data.to(device)
            #batch_attention_mask = batch_attention_mask.to(device)

            # Calculate the loss
            pdb.set_trace()
            loss = model(batch_data, attention_mask=batch_attention_mask, labels=batch_data).loss.sum()
            loss.backward()
            print('loss', loss)
            optim.step()
            optim.zero_grad()
            with torch.no_grad():
                losses.append(loss.item())
            torch.cuda.empty_cache()
            
            # Save checkpoint logic (every args.save_interval batches)
            batch_count = batch_idx + 1
            batch_count_tot = batch_idx + start_batch + 1
            if batch_count == num_batches or batch_count_tot % args.save_interval == 0:

                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir, exist_ok = True) 
                
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'epoch': epoch,
                    'batch_idx': batch_count_tot,
                    'shuffled_indices': all_indices,
                    'losses': losses,
                    'num_batch': round(num_batches)+1
                }, save_path)
                print('checkpoint saved for epoch %d batch count total %d' % (epoch, batch_count_tot))

              
        # save epoch check point
        if num_batches > 0:
            save_dir, fname = os.path.split(save_path)
            fbody, fext = fname.split(".")
            fname_epoch = ".".join(["%s_e%d" % (fbody, epoch + 1), fext])
            save_path_epoch = os.path.join(save_dir, fname_epoch)
            pdb.set_trace()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok = True) 
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'epoch': epoch,
                        'batch_idx': batch_idx + start_batch + 1,
                        'shuffled_indices': all_indices,
                        'losses': losses,
                        'num_batch': round(num_batches)+1
                        }, save_path_epoch)
            print('checkpoint saved for epoch %d' % epoch)
            

        start_batch = 0


def decoder_infer(input_prompts):
    #inp = "<startofstring> "+inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a )
    output = tokenizer.decode(output[0])
    return output

 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--seed', type=int, default=0)

    # directory structure
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--obj_path', type=str, default='./demo_meshes/hammer.obj')
    parser.add_argument('--save_dir', type=str, default='./experiments/')
    parser.add_argument('--model_name', type=str, default='decoder_checkpoint.pth')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--encoder_f_path', type=str, default='./demo_models/hammer/pred_f.pth') # encoder feature file path

    # training data setting

    # data info
    #parser.add_argument('--data_percentage', type=float, default=1.0)
    
    # network
    parser.add_argument('--continue_train', type=int, default=0)
    parser.add_argument('--depth', type=int, default=14)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--n_classes', type=int, default=256) # 256 channels for SAM embedding feature

    # optimization
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--return_original', type=int, default=0)
    

    # parallel, multi-GPU training
    parser.add_argument('--use_data_parallel', type=int, default=1)

    args = parser.parse_args()
    
    # Usage example
    save_path = '/home-nfs/fx2024/textdata'  # Set your custom save path
    wikitext_dataset_train = load_or_download_wikitext(save_path, dataset_name='wikitext', dataset_version='wikitext-2-raw-v1', split='train')

    # Print a sample from the dataset to verify
    #print(wikitext_dataset_train['train']['text'][:5])
    print(wikitext_dataset_train['text'][0])


    # training

    # tokenize the text
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    wikitext_dataset_train_tokenized = My_Dataset(wikitext_dataset_train, tokenizer)

    #model.train()

    #optim = Adam(model.parameters(), lr=1e-3)

    print("training .... ")


    epochs = 12

    decoder_training(args, model)
    input_prompts = 'Valkyria Chronicles'
    decoder_infer(input_prompts)

    


