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


def one_hot(spk, spk_dict):
    result = numpy.zeros((1, len(spk_dict)))
    result[0, spk_dict[spk]] = 1
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--spk-set')
    parser.add_argument('--feat-scp')
    parser.add_argument('--feat-mean-var')
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--step-size', type=float)
    parser.add_argument('--grad-clip', type=float)
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--param')
    parser.add_argument('--param-output')
    parser.add_argument('--seed', type=int)
    
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

spk_dict = dataprep.load_label_dict(args.spk_set)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device:', device)
print()

if args.model == 'linear':
    model = SpkLinear(40, len(spk_dict))
elif args.model == 'lstm':
    model = SpkLstm(40, args.hidden_size, len(spk_dict), args.layers, args.dropout)

loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

total_param = list(model.parameters())

opt = torch.optim.SGD(total_param, lr=0.0)

if args.param:
    ckpt = torch.load(args.param)
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['opt'])

if args.init:
    torch.save(
        {
            'model': model.state_dict(),
            'opt': opt.state_dict()
        },
        args.param_output)
    exit()

model.to(device)

step_size = args.step_size
grad_clip = args.grad_clip

feat_mean, feat_var = dataprep.load_mean_var(args.feat_mean_var)

rand_eng = rand.Random(args.seed)

dataset = data.Vox1Spkid(args.feat_scp, feat_mean, feat_var, shuffling=True, rand=rand_eng)

for sample, (key, spk, feat) in enumerate(dataset):
    feat = torch.Tensor(feat).to(device)

    nframes, ndim = feat.shape
    feat = feat.reshape(1, nframes, ndim)

    spk_label = torch.Tensor(one_hot(spk, spk_dict)).to(device)

    opt.zero_grad()

    pred = model(feat)

    loss = loss_fn(pred, spk_label)

    print('sample:', sample)
    print('key:', key)
    print('frames:', nframes)
    print('loss: {:.6}'.format(loss.item()))

    loss = loss * nframes
    loss.backward()

    model_norm = 0
    grad_norm = 0
    for p in total_param:
        n = p.norm(2).item()
        model_norm += n * n

        n = p.grad.norm(2).item()
        grad_norm += n * n
    model_norm = math.sqrt(model_norm)
    grad_norm = math.sqrt(grad_norm)

    print('model norm: {:.6}'.format(model_norm))
    print('grad norm: {:.6}'.format(grad_norm))

    param_0 = total_param[0][0, 0].item()

    if grad_norm > grad_clip:
        opt.param_groups[0]['lr'] = step_size / grad_norm * grad_clip
    else:
        opt.param_groups[0]['lr'] = step_size

    opt.step()

    param_0_new = total_param[0][0, 0].item()

    print('param: {:.6}, update: {:.6}, rate: {:.6}'.format(param_0, param_0_new - param_0, (param_0_new - param_0) / param_0))

    print()


torch.save(
    {
        'model': model.state_dict(),
        'opt': opt.state_dict()
    },
    args.param_output)

