import json
import os
from pprint import pprint
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
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
import wandb

# training function
def gpt_finetune(args, model):
    
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

    

    # set up wandb configurations
    id = args.wandb_run_id#wandb.util.generate_id()

    #if args.wandb_delete_previous_run == 1:
    #    api = wandb.Api()
    #    # Delete the previous result
    #    try:
    #        run_to_delete = api.run("fayxu/test-project/test")
    #        run_to_delete.delete()
    #    except:
    #        print("id {} doesn't exist".format(id))
    # initialize wandb project configurations
    wandb.init(
        # set the wandb project where this run will be logged
        project="test-project",
        name='test',
        id=id, 
        resume="allow",
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.learning_rate,
        "architecture": "GPT2",
        "dataset": "wikitext",
        "epochs": args.num_epochs,
        }
    )

    # move GPT model to CUDA
    device = 'cuda'
    model = model.to(device)
    
    parameters = list(model.parameters())
    optim = torch.optim.Adam(parameters, args.learning_rate)

    # Check GPU status, assign model to all available GPUs
    if args.use_data_parallel and num_gpus > 1:
        device_ids_list = list(range(num_gpus))
        model = torch.nn.DataParallel(model, device_ids=device_ids_list)
        #model = DDP(model, device_ids=device_ids_list)
       
    # Load tokenized training dataset
    #dir = os.path.join(args.dataset_path)
    dataset = wikitext_dataset_train_tokenized

    # Initialize training-related variables
    start_epoch = 0
    start_batch = 0
    save_path = os.path.join(args.save_dir, args.model_name)
    losses = []
    all_indices = list(range(len(dataset)))

    # Load checkpoint, for continue training 
    if args.continue_train:
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch_idx']
        all_indices = checkpoint['shuffled_indices']
        losses = checkpoint['losses']

    # Start training
    for epoch in range(start_epoch, args.num_epochs):
         # If this is a new epoch, shuffle the indices
        if epoch != start_epoch or start_batch == 0:
            random.shuffle(all_indices)

        # SubsetRandomSampler handles the batching
        sampler = SubsetRandomSampler(all_indices[start_batch * args.batch_size:])
        
        # create the DataLoader
        dataloader =  DataLoader(wikitext_dataset_train_tokenized, sampler=sampler, batch_size=args.batch_size)
        #dataloader = DataLoader(dataset, batch_size=args.batch_size)
        
        # Sampling data using shuffled indices
        num_batches = len(dataloader)

        # Loop over all batches
        for batch_idx, (batch_input_ids, batch_attention_mask) in enumerate(tqdm(dataloader)):
            print("epoch_idx: %d" % (epoch))
            print("batch_idx: %d" % (batch_idx + start_batch))

            # Calculate the loss
            loss = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_input_ids).loss.sum()
            loss.backward()
            print('loss', loss)
            optim.step()
            optim.zero_grad()
            # Track losses
            with torch.no_grad():
                losses.append(loss.item())
                wandb.log({"loss": loss})
            torch.cuda.empty_cache()
            
            # Save checkpoint logic (every args.save_interval batches)
            batch_count = batch_idx + 1
            batch_count_tot = batch_idx + start_batch + 1
            if batch_count == num_batches or batch_count_tot % args.save_interval == 0:

                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir, exist_ok = True) 
                torch.save({
                    'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'epoch': epoch,
                    'batch_idx': batch_count_tot,
                    'shuffled_indices': all_indices,
                    'losses': losses,
                    'num_batch': round(num_batches)+1
                }, save_path)
                print('checkpoint saved for epoch %d batch count total %d' % (epoch, batch_count_tot))
                

              
        # save epoch checkpoint
        if num_batches > 0:
            save_dir, fname = os.path.split(save_path)
            fbody, fext = fname.split(".")
            fname_epoch = ".".join(["%s_e%d" % (fbody, epoch + 1), fext])
            save_path_epoch = os.path.join(save_dir, fname_epoch)
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok = True) 
            torch.save({
                        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'epoch': epoch,
                        'batch_idx': batch_idx + start_batch + 1,
                        'shuffled_indices': all_indices,
                        'losses': losses,
                        'num_batch': round(num_batches)+1
                        }, save_path_epoch)
            print('checkpoint saved for epoch %d' % epoch)
            
        start_batch = 0

    wandb.finish()

def plot_loss(loss, type=''):
    plt.figure()
    plt.plot(loss, label=type)
    plt.legend(True)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(dir, 'loss.jpg'))

# Test checkpoint
def gpt_infer(args, model):
    num_gpus = torch.cuda.device_count()
    print('number of avilable gpus: %d' % num_gpus)

    device = 'cuda'
    model = model.to(device)

    # Read checkpoint model
    checkpoint = torch.load(args.test_model_path, map_location=device)

    if args.use_data_parallel and num_gpus > 1:
        device_ids_list = list(range(num_gpus))
        model = nn.DataParallel(model, device_ids=device_ids_list)
        #model = DDP(model, device_ids=device_ids_list)
        model = model.module    

    
    model.load_state_dict(checkpoint['model_state_dict'])
    #model = AutoModelWithLMHead.from_pretrained(train_path)

    inputs = tokenizer.encode(args.input_prompts, return_tensors = 'pt').to(device)
    # Create an attention mask for the inputs
    attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=device)
    # Set pad_token_id to the pad_token_id of the tokenizer
    pad_token_id = tokenizer.pad_token_id


    print("\ngenerating output")
    outputs = model.generate(
        inputs, 
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        max_length=100, 
        num_beams=5, 
        num_return_sequences=5, 
        early_stopping=True # Stop generating once max_length is reached
    )

    output = tokenizer.decode(outputs[0])
    return output

 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--seed', type=int, default=0)

    # data info
    parser.add_argument('--data_percentage', type=float, default=0.1)

    # directory structure
    parser.add_argument('--save_dir', type=str, default='../experiments/')
    parser.add_argument('--data_path', type=str, default='/home-nfs/fx2024/NLP/textdata')
    parser.add_argument('--model_name', type=str, default='checkpoint.pth')

    # wandb keywords
    parser.add_argument('--wandb_delete_previous_run', type=int, default=0)
    parser.add_argument('--wandb_run_id', type=str, default='test1')
    
    # network
    parser.add_argument('--continue_train', type=int, default=0)
    parser.add_argument('--depth', type=int, default=14)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--n_classes', type=int, default=256) # 256 channels for SAM embedding feature

    # optimization
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--return_original', type=int, default=0)
    

    # parallel, multi-GPU training
    parser.add_argument('--use_data_parallel', type=int, default=1)

    # mode
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--test', type=int, default=1)

    # Inference
    parser.add_argument('--input_prompts', type=str, default='Liu Kang is') # input prompts
    parser.add_argument('--test_model_path', type=str, default='/home-nfs/fx2024/NLP/experiments/checkpoint.pth')

    args = parser.parse_args()
    
    device = 'cuda'

    # Load wikitext data
    wikitext_dataset_train = load_or_download_wikitext(args.data_path, dataset_name='wikitext', dataset_version='wikitext-2-raw-v1')['train']

    # Print a sample from the dataset to verify
    #print(wikitext_dataset_train['train']['text'][:5])
    #print(wikitext_dataset_train['text'][0])

    # tokenize the text
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    wikitext_dataset_train_tokenized = My_Dataset(args, wikitext_dataset_train, tokenizer)

    # Test prompt before training
    inputs = tokenizer.encode(args.input_prompts, return_tensors = 'pt').to(device)
    # Create an attention mask for the inputs
    attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=device)
    # Set pad_token_id to the pad_token_id of the tokenizer
    pad_token_id = tokenizer.pad_token_id
    print("\ngenerating output")
    outputs = model.generate(
        inputs, 
        attention_mask=attention_mask,
        #pad_token_id=pad_token_id,
        max_length=100, 
        num_beams=5, 
        num_return_sequences=5, 
        early_stopping=True # Stop generating once max_length is reached
    )
    output = tokenizer.decode(outputs[0])
    print('Before training', args.input_prompts, output)

    # training
    if args.train == 1:
        print("training .... ")
        gpt_finetune(args, model)
    # testing
    if args.test == 1:
        output = gpt_infer(args, model)
        print(args.input_prompts, output)

    


