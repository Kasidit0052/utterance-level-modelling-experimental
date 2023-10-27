#!/usr/bin/env python3

import sys
import math
import data
import torch.nn
import torch.optim
import argparse
import json
import rand
import dataprep
import numpy


class SpkLstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout,
            bidirectional=False, batch_first=True)
        self.pred = torch.nn.Linear(hidden_size, output_size)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        spk_emb = torch.mean(hidden, 1)
        return self.pred(spk_emb)


class SpkLinear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.pred = torch.nn.Linear(input_size, output_size)

    def forward(self, feat):
        spk_emb = torch.mean(feat, 1)
        return self.pred(spk_emb)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--spk-set')
    parser.add_argument('--feat-scp')
    parser.add_argument('--feat-mean-var')
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--model')
    parser.add_argument('--layers', type=int)
    parser.add_argument('--hidden-size', type=int)
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

    if args.dropout is None:
        args.dropout = 0.0

    return args


args = parse_args()

f = open(args.spk_set)
id_spk = []
for line in f:
    id_spk.append(line.strip())
f.close()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model == 'linear':
    model = SpkLinear(40, len(id_spk))
elif args.model == 'lstm':
    model = SpkLstm(40, args.hidden_size, len(id_spk), args.layers, args.dropout)

if args.param:
    ckpt = torch.load(args.param)
    model.load_state_dict(ckpt['model'])

model.to(device)
model.eval()

feat_mean, feat_var = dataprep.load_mean_var(args.feat_mean_var)

err = 0
utt = 0

dataset = data.Vox1Spkid(args.feat_scp, feat_mean, feat_var, shuffling=False)

for sample, (key, spk, feat) in enumerate(dataset):
    feat = torch.Tensor(feat).to(device)

    nframes, ndim = feat.shape
    feat = feat.reshape(1, nframes, ndim)

    pred = model(feat)

    a, m = max(enumerate(pred[0]), key=lambda t: t[1])

    if id_spk[a] != spk:
        err += 1

    utt += 1
    print('utt: {}, pred: {}, err: {}'.format(key, id_spk[a], 1 if id_spk[a] != spk else 0))


print('total: {}, err: {}, rate: {:.6}'.format(utt, err, err / utt))

