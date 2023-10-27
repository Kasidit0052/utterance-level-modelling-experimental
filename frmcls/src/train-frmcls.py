#!/usr/bin/env python3

import sys
import math
import frmapc as apc
import data as utils
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import dataset
import torch.nn
import torch.optim
import argparse
import json
import rand

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
    parser.add_argument('--step-accumulate', type=int)
    ### Experiment parameters [start]
    parser.add_argument('--config-residual', type=bool)
    parser.add_argument('--use-var', type=bool)
    parser.add_argument('--merge-mode', type=str)
    parser.add_argument('--sq-ratio', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--use-preframe', type=bool)
    parser.add_argument('--extract-layer', type=int)
    ### Experiment parameters [end]
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

model = apc.UttAPCFrmLstm(40, args.hidden_size, len(label_dict), args.layers, args.dropout, 
                          args.config_residual, args.use_var, args.merge_mode, args.sq_ratio, 
                          args.use_preframe, args.__dict__.get("utterance_level_dropout", None), args.extract_layer)
loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
total_param = list(model.pred.parameters())

model.to(device)
model.lstms.requires_grad_(False)
model.statpool.eval()
model.statpool.requires_grad_(False)

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
    s = {}
    for k in ckpt['model']:
        if k.startswith('lstms'):
            p[k[len('lstms.'):]] = ckpt['model'][k]
        if k.startswith('statpool'):
            s[k[len('statpool.'):]] = ckpt['model'][k]
    model.lstms.load_state_dict(p)
    model.statpool.load_state_dict(s)

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
    length = torch.Tensor(length).to(device)
    padded_inputs = pad_sequence(feat, batch_first=True).to(device)
    hidden,pred,_,pre_frame = model(padded_inputs, length, device)
    one_hots = pad_sequence(one_hots, batch_first=True).to(device)

    # removing zero from padded location in predictions
    packed_outputs = pack_padded_sequence(pred, length.cpu(), batch_first=True, enforce_sorted=False)
    outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=int(length.max()))

    # computing crossentropy over vocabulary dimension
    outputs = torch.permute(outputs, (0,2,1))
    one_hots = torch.permute(one_hots, (0,2,1))
    loss = loss_fn(outputs, one_hots)

    # computing loss over timestep
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

