#!/usr/bin/env python3

import os
import sys
import math
import data as utils
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import dataset
import torch.nn
import argparse
import json
import rand
import pickle

class UniLstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout,
            bidirectional=False, batch_first=True)
        self.proj = torch.nn.Linear(hidden_size, input_size)

    def forward(self, feat, length):
        seq_len = feat.size(1)
        packed_inputs = pack_padded_sequence(feat, length, batch_first=True, enforce_sorted=False)
        hidden, _ = self.lstm(packed_inputs)
        seq_unpacked, lens_unpacked = pad_packed_sequence(hidden, batch_first=True, total_length=seq_len)
        return seq_unpacked, self.proj(seq_unpacked)
    
class VanillaAPC(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, config_residual=True, extract_layer=3):
        super().__init__()

        input_sizes  = ([input_size] +[hidden_size] * (num_layers - 1))
        output_sizes = [hidden_size] * num_layers
        self.lstms = torch.nn.ModuleList([
            torch.nn.LSTM(in_size, out_size, dropout= dropout if i < len(input_sizes)-1 else 0.0, batch_first=True) 
            for i, (in_size, out_size) in enumerate(zip(input_sizes, output_sizes))
        ])
        self.proj = torch.nn.Linear(hidden_size, input_size)
        self.residual = config_residual

        self.extract_layer = extract_layer
        assert self.extract_layer > 0 and self.extract_layer < len(self.lstms)+1

    def forward(self, feat, length):
        seq_len = feat.size(1)
        for i, layer in enumerate(self.lstms):
            packed_inputs  = pack_padded_sequence(feat, length.cpu(), batch_first=True, enforce_sorted=False) 
            packed_hidden,_ = layer(packed_inputs)
            seq_unpacked,_ = pad_packed_sequence(packed_hidden, batch_first=True, total_length=seq_len)   

            if self.residual and feat.size(-1) == seq_unpacked.size(-1):
                seq_unpacked = seq_unpacked + feat

            feat = seq_unpacked

            if i+1 == self.extract_layer:
                utt_embeddings = torch.sum(seq_unpacked,axis=1)*1/length.unsqueeze(1)            
                return utt_embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--feat-scp')
    parser.add_argument('--feat-mean-var')
    parser.add_argument('--layers', type=int)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--config-residual', type=bool)
    parser.add_argument('--extract-layer', type=int)
    parser.add_argument('--time-shift', type=int)
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

model =  VanillaAPC(40, args.hidden_size, args.layers, 0.0, args.config_residual, args.extract_layer)
model.to(device)

checkpoint = torch.load(args.param)
model.load_state_dict(checkpoint['model'])

shift     = args.time_shift
feat_mean, feat_var = utils.load_mean_var(args.feat_mean_var)
rand_eng = rand.Random()

ds = dataset.LibriSpeechDataset(args.feat_scp, feat_mean, feat_var, shuffling=False, rand=rand_eng)     
dataloader = data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)   # deterministic shuffle

model.eval()

list_reps = []
for i, data in enumerate(dataloader, 0):
    key, feat, length = data
    length = torch.Tensor(length).to(device)
    padded_inputs = pad_sequence(feat, batch_first=True).to(device)
    with torch.no_grad():
        utt_embeddings = model(padded_inputs, length)    #input: list[Batchsize x Tensor(seq_len, mel_dims)], output prediction:[ batch_size, mel_dims]

    for j in range(0, len(key)):
        sample = {}
        sample["key"] =  key[j]
        sample["utt_embeddings"] = utt_embeddings[j].cpu().detach()
        list_reps.append(sample)

model.train()

parentdir = os.getcwd() + os.sep + "/".join(args.config.split("/")[:-1])
exp_name = "layer"+ str(args.extract_layer) +"_"+ args.param.split("/")[-1]
exp_dir = parentdir + os.sep + exp_name

with open('{}.pkl'.format(exp_dir), 'wb') as f:
     pickle.dump(list_reps, f, protocol=pickle.HIGHEST_PROTOCOL)



