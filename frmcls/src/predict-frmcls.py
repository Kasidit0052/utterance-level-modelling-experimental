#!/usr/bin/env python3

import sys
import math
import data 
import frmapc as apc
import torch.nn
import torch.optim
import argparse
import json
import rand
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--label-set')
    parser.add_argument('--feat-scp')
    parser.add_argument('--feat-mean-var')
    parser.add_argument('--layers', type=int)
    parser.add_argument('--hidden-size', type=int)
    ### Experiment parameters [start]
    parser.add_argument('--config-residual', type=bool)
    parser.add_argument('--use-var', type=bool)
    parser.add_argument('--merge-mode', type=str)
    parser.add_argument('--sq-ratio', type=int)
    parser.add_argument('--use-preframe', type=bool)
    parser.add_argument('--extract-layer', type=int)
    ### Experiment parameters [end]
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

model = apc.UttAPCFrmLstm(40, args.hidden_size, len(id_label), args.layers, 0.0, 
                          args.config_residual, args.use_var, args.merge_mode, args.sq_ratio,
                          args.use_preframe, args.__dict__.get("utterance_level_dropout", None), args.extract_layer)

model.to(device)
model.eval()
model.lstms.requires_grad_(False)
model.statpool.requires_grad_(False)

ckpt = torch.load(args.pred_param)
model.pred.load_state_dict(ckpt['pred'])

ckpt = torch.load(args.apc_param)
p = {}
s = {}
for k in ckpt['model']:
    if k.startswith('lstms'):
        p[k[len('lstms.'):]] = ckpt['model'][k]
    if k.startswith('statpool'):
        s[k[len('statpool.'):]] = ckpt['model'][k]
model.lstms.load_state_dict(p)
model.statpool.load_state_dict(s)


feat_mean, feat_var = data.load_mean_var(args.feat_mean_var)
dataset = data.WsjFeat(args.feat_scp, feat_mean, feat_var)

for sample, (key, feat) in enumerate(dataset):
    print(key)

    feat = torch.Tensor(feat).to(device)

    nframes, ndim = feat.shape
    length = torch.Tensor([nframes]).to(device)
    feat = feat.reshape(1, nframes, ndim)

    hidden, pred,_,_ = model(feat, length, device)

    _, nframes, nclass = pred.shape
    pred = pred.reshape(nframes, nclass)

    labels = torch.argmax(pred, dim=1)

    result = [id_label[int(e)] for e in labels]

    print(' '.join(result))
    print('.')

