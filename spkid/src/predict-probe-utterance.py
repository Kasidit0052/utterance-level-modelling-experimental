#!/usr/bin/env python3

import sys
import math
import spkapc as apc
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import dataset
import torch.nn
import torch.optim
import argparse
import json
import rand
import dataprep
import numpy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--spk-set')
    parser.add_argument('--feat-scp')
    parser.add_argument('--feat-mean-var')
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--batch-size', type=int)
    ### Experiment parameters [start]
    parser.add_argument('--config-residual', type=bool)
    parser.add_argument('--use-var', type=bool)
    parser.add_argument('--merge-mode', type=str)
    parser.add_argument('--sq-ratio', type=int)
    parser.add_argument('--extract-mode', type=str)
    parser.add_argument('--utterance-level-dropout', type=float)
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

model = apc.UttAPCSpkLstm(40, args.hidden_size, len(id_spk), args.layers, 0.0, 
                          args.config_residual, args.use_var, args.merge_mode, args.sq_ratio,
                          args.extract_mode, args.__dict__.get("utterance_level_dropout", None), args.extract_layer)
model.to(device)
model.eval()
model.lstms.requires_grad_(False)
model.statpool.requires_grad_(False)

if args.pred_param:
    ckpt = torch.load(args.pred_param)
    model.pred.load_state_dict(ckpt['pred'])

if args.apc_param:
    ckpt = torch.load(args.apc_param)
    p = {}
    s = {}
    for k in ckpt['model']:
        if k.startswith('lstms'):
            if int(k[6]) < args.layers:
                p[k[len('lstms.'):]] = ckpt['model'][k]
        if k.startswith('statpool'):
            if int(k[9]) < args.layers:
                s[k[len('statpool.'):]] = ckpt['model'][k]
    model.lstms.load_state_dict(p)
    model.statpool.load_state_dict(s)

feat_mean, feat_var = dataprep.load_mean_var(args.feat_mean_var)

err = 0
utt = 0

rand_eng = rand.Random()
ds = dataset.Vox1Dataset(args.feat_scp, feat_mean, feat_var, args.spk_set, shuffling=False, rand=rand_eng)
dataloader = data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=ds.collate_fn)   # deterministic shuffle

for i, data in enumerate(dataloader, 0):
    key, feat, length, labels, one_hots = data
    with torch.no_grad():
        padded_inputs = pad_sequence(feat, batch_first=True).to(device)
        length = torch.Tensor(length).to(device)
        hidden, pred = model(padded_inputs, length, device)                                 # pred.shape [batch_size, num_speaker]
        output = model.post_processing(pred)
        spk_labels = one_hots.to(device)

        output = torch.argmax(output,dim=1)
        spk_labels = torch.argmax(spk_labels,dim=1)
        zero_one_loss = (output!=spk_labels).sum().cpu().detach().numpy()

        print('iter:', i)
        print('pred: {}'.format(output.tolist()))
        print('utt: {}'.format(spk_labels.tolist()))
        print('err: {}, batch-size: {}'.format(zero_one_loss, len(output)))

        err += zero_one_loss
        utt += len(output)
        print('total-err: {}, total-sample: {}'.format(err,utt))
        print('')

print('Summarizing.................\n')
print('total-err: {}, total-sample: {}, rate: {:.6}'.format(err,utt, err / utt))







