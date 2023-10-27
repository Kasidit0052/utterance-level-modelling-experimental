#!/usr/bin/env python3

import sys
import math
import data as utils
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import dataset
import torch.nn
import torch.optim
import argparse
import json
import rand
import numpy as np


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
    parser.add_argument('--label-scp')
    parser.add_argument('--feat-scp')
    parser.add_argument('--feat-mean-var')
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--step-size', type=float)
    parser.add_argument('--grad-clip', type=float)
    parser.add_argument('--reduction', type=str)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--batch-size', type=int)
    ### Experiment parameters [start]
    parser.add_argument('--config-residual', type=bool)
    parser.add_argument('--extract-layer', type=int)
    parser.add_argument('--seed', type=int)
    ### Experiment parameters [end]
    parser.add_argument('--step-accumulate', type=int)
    parser.add_argument('--init', action="store_true")
    parser.add_argument('--apc-param')
    parser.add_argument('--pred-param')
    parser.add_argument('--pred-param-output')
    
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

    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print()

    return args


print(' '.join(sys.argv))
args = parse_args()

label_dict = utils.load_label_dict(args.label_set)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device:', device)
print()

model = FrmLstm(40, args.hidden_size, len(label_dict), args.layers, args.dropout, args.config_residual, args.extract_layer)

loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
total_param = list(model.pred.parameters())

model.to(device)
model.lstms.requires_grad_(False)

assert args.optimizer in ["sgd","adam"], f"optimizer is not supported"
if args.optimizer == "sgd":
    opt = torch.optim.SGD(total_param, lr=0.0)
if args.optimizer == "adam":
    opt = torch.optim.Adam(total_param, lr=0.0)

if args.pred_param:
    ckpt = torch.load(args.pred_param)
    model.pred.load_state_dict(ckpt['pred'])
    opt.load_state_dict(ckpt['opt'])

if args.apc_param:
    ckpt = torch.load(args.apc_param)
    p = {}
    for k in ckpt['model']:
        if k.startswith('lstms'):
            p[k[len('lstms.'):]] = ckpt['model'][k]
    model.lstms.load_state_dict(p)

if args.init:
    torch.save(
        {
            'pred': model.pred.state_dict(),
            'opt': opt.state_dict()
        },
        args.pred_param_output)
    exit()

step_size = args.step_size
grad_clip = args.grad_clip

feat_mean, feat_var = utils.load_mean_var(args.feat_mean_var)
rand_eng = rand.Random(args.seed)

ds = dataset.WSJDataset(args.feat_scp, args.label_scp, feat_mean, feat_var, label_dict, shuffling=True, rand=rand_eng)
dataloader = data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=ds.collate_fn)   # deterministic shuffle

for i, data in enumerate(dataloader, 0):
    key, feat, length, labels, one_hots = data
    padded_inputs = pad_sequence(feat, batch_first=True).to(device)
    hidden, pred = model(padded_inputs, length)
    one_hots = pad_sequence(one_hots, batch_first=True).to(device)

    # removing zero from padded location in predictions
    packed_outputs = pack_padded_sequence(pred, length, batch_first=True, enforce_sorted=False)
    outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=int(length.max()))

    # computing crossentropy over vocabulary dimension
    outputs = torch.permute(outputs, (0,2,1))
    one_hots = torch.permute(one_hots, (0,2,1))
    loss = loss_fn(outputs, one_hots)

    # computing loss over timestep
    length = length.to(device)
    loss_samples = torch.sum(loss,dim=1) * 1/length        

    # computing loss over batch
    if args.reduction == "mean":
        loss_batch  = torch.mean(loss_samples)              # loss per batch: Scalar
    if args.reduction == "sum":
        loss_batch  = torch.sum(loss_samples)               # loss per batch: Scalar

    print('iter:', i)
    print('key:', key[:5])
    print('frames:',length[:5].cpu().detach().numpy())
    print('average loss per batch: {:.6}'.format(loss_batch.item()))

    loss_batch.backward()

    if ((i + 1) % args.step_accumulate == 0) or (i + 1 == len(ds)/args.batch_size):

        total_norm = 0
        for p in total_param:
            n = p.grad.norm(2).item()
            total_norm += n * n
        grad_norm = math.sqrt(total_norm)

        print('grad norm: {:.6}'.format(grad_norm))

        param_0 = total_param[0][0, 0].item()

        if grad_norm > grad_clip:
            opt.param_groups[0]['lr'] = step_size / grad_norm * grad_clip
        else:
            opt.param_groups[0]['lr'] = step_size

        opt.step()
        opt.zero_grad()

        param_0_new = total_param[0][0, 0].item()

        print('param: {:.6}, update: {:.6}, rate: {:.6}'.format(param_0, param_0_new - param_0, (param_0_new - param_0) / param_0))

        print()

torch.save(
{
    'pred': model.pred.state_dict(),
    'opt': opt.state_dict()
},
args.pred_param_output)

