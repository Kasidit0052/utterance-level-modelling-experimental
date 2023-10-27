#!/usr/bin/env python3

import os
import sys
import math
import data as utils
import uttapc_extractor as apc
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import dataset
import torch.nn
import torch.optim
import argparse
import json
import rand
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--feat-scp')
    parser.add_argument('--feat-mean-var')
    parser.add_argument('--layers', type=int)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--time-shift', type=int)
    ### Experiment parameters [start]
    parser.add_argument('--config-residual', type=bool)
    parser.add_argument('--use-var', type=bool)
    parser.add_argument('--merge-mode', type=str)
    parser.add_argument('--sq-ratio', type=int)
    parser.add_argument('--extract-layer', type=int)
    ### Experiment parameters [end]
    parser.add_argument('--param')
    
    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)
    
    args = parser.parse_args()

    if args.config:
        f = open(args.config)
        config = json.load(f)
        f.close()
    
        for k, v in config.items():
            if k not in args.__dict__ or args.__dict__[k] is None:
                args.__dict__[k] = v

    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print()

    return args

print(' '.join(sys.argv))
args = parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device:', device)
print()

model = apc.UtteranceLevelAPC(40, args.hidden_size, args.layers, 0.0, 
                              args.config_residual, args.use_var, args.merge_mode, args.sq_ratio, 
                              args.__dict__.get("utterance_level_dropout", None), args.extract_layer)

model.to(device)

checkpoint = torch.load(args.param)
model.load_state_dict(checkpoint['model'])

shift     = args.time_shift
feat_mean, feat_var = utils.load_mean_var(args.feat_mean_var)
rand_eng = rand.Random()

ds = dataset.LibriSpeechDataset(args.feat_scp, feat_mean, feat_var, shuffling=False, rand=rand_eng)
dataloader  = data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)  # deterministic shuffle

model.eval()

list_reps = []
for i, data in enumerate(dataloader, 0):
    key, feat, length = data
    length = torch.Tensor(length).to(device)
    padded_inputs = pad_sequence(feat, batch_first=True).to(device)
    with torch.no_grad():
        mean_preframes, utt_embeddings, mean_postframes = model(padded_inputs, length, device)   #input: Utterance Embeddings[Batchsize x mel_dims], 

    for j in range(0, len(key)):
        sample = {}
        sample["key"] =  key[j]
        sample["mean-preframes"] =  mean_preframes[j].cpu().detach()
        sample["utt_embeddings"] = utt_embeddings[j].cpu().detach()
        sample["mean-postframes"] =  mean_preframes[j].cpu().detach()
        list_reps.append(sample)

model.train()

parentdir = os.getcwd() + os.sep + "/".join(args.config.split("/")[:-1])
exp_name = "layer"+ str(args.extract_layer) +"_"+ args.param.split("/")[-1]
exp_dir = parentdir + os.sep + exp_name

with open('{}.pkl'.format(exp_dir), 'wb') as f:
    pickle.dump(list_reps, f, protocol=pickle.HIGHEST_PROTOCOL)

                                                                        

