#!/usr/bin/env python3

import sys
import math
import data
import torch.nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import torch.optim
import argparse
import json
import rand
import numpy


class ApcLstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout,
            bidirectional=False, batch_first=True)
        self.proj = torch.nn.Linear(hidden_size, input_size)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        return hidden, self.proj(hidden)


class FrmLstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.0, config_residual=True, extract_layer=3):
        super().__init__()
        input_sizes  = ([input_size] +[hidden_size] * (num_layers - 1))
        output_sizes = [hidden_size] * num_layers
        self.lstms = torch.nn.ModuleList([
            torch.nn.LSTM(in_size, out_size, dropout= dropout if i < len(input_sizes)-1 else 0.0, batch_first=True) 
            for i, (in_size, out_size) in enumerate(zip(input_sizes, output_sizes))
        ])

        self.extract_layer = extract_layer
        assert self.extract_layer > 0 and self.extract_layer < len(self.lstms)+1
        
        self.pred = torch.nn.Linear(hidden_size, output_size)
        self.residual = config_residual

    def forward(self, feat, length):
        seq_len = feat.size(1)
        for i, layer in enumerate(self.lstms):
            packed_inputs  = pack_padded_sequence(feat, length, batch_first=True, enforce_sorted=False) 
            packed_hidden,_ = layer(packed_inputs)
            seq_unpacked,_ = pad_packed_sequence(packed_hidden, batch_first=True, total_length=seq_len)   

            if self.residual and feat.size(-1) == seq_unpacked.size(-1):
                seq_unpacked = seq_unpacked + feat

            if i+1 == self.extract_layer:
                return seq_unpacked, self.pred(seq_unpacked)
            
            feat = seq_unpacked
        return seq_unpacked, self.pred(seq_unpacked)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--label-set')
    parser.add_argument('--feat-scp')
    parser.add_argument('--feat-mean-var')
    parser.add_argument('--layers', type=int)
    ### Experiment parameters [start]
    parser.add_argument('--config-residual', type=bool)
    parser.add_argument('--extract-layer', type=int)
    ### Experiment parameters [end]
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--apc-param')
    parser.add_argument('--pred-param')
    
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

f = open(args.label_set)
id_label = []
for line in f:
    id_label.append(line.strip())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FrmLstm(40, args.hidden_size, len(id_label), args.layers, 0.0, args.config_residual, args.extract_layer)

ckpt = torch.load(args.pred_param)
model.pred.load_state_dict(ckpt['pred'])

ckpt = torch.load(args.apc_param)
p = {}
for k in ckpt['model']:
    if k.startswith('lstms'):
        p[k[len('lstms.'):]] = ckpt['model'][k]
model.lstms.load_state_dict(p)

model.to(device)
model.eval()

feat_mean, feat_var = data.load_mean_var(args.feat_mean_var)

dataset = data.WsjFeat(args.feat_scp, feat_mean, feat_var)

for sample, (key, feat) in enumerate(dataset):
    print(key)

    length = torch.Tensor([len(feat)])
    feat = torch.Tensor(feat).to(device)

    feat = pad_sequence([feat], batch_first=True)
    hidden, pred = model(feat,length)

    _, nframes, nclass = pred.shape
    pred = pred.reshape(nframes, nclass)

    labels = torch.argmax(pred, dim=1)

    result = [id_label[int(e)] for e in labels]

    print(' '.join(result))
    print('.')

