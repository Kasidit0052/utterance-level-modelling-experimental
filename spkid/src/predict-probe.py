#!/usr/bin/env python3

import sys
import math
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import dataset
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

class SpkLstmVanilla(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.0, config_residual=True, extract_layer=3):
        super().__init__()
        input_sizes  = ([input_size] +[hidden_size] * (num_layers - 1))
        output_sizes = [hidden_size] * num_layers
        self.lstms = torch.nn.ModuleList([
            torch.nn.LSTM(in_size, out_size, dropout= dropout if i < len(input_sizes)-1 else 0.0, batch_first=True) 
            for i, (in_size, out_size) in enumerate(zip(input_sizes, output_sizes))
        ])

        self.post_processing = torch.nn.Softmax(dim=1)
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
                length = torch.Tensor(length).to(device).unsqueeze(1)
                spk_emb = torch.sum(seq_unpacked, 1)*1/length
                return self.pred(spk_emb)
            
            feat = seq_unpacked

        length = torch.Tensor(length).to(device).unsqueeze(1)
        spk_emb = torch.sum(seq_unpacked, 1)*1/length
        return self.pred(spk_emb)

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

model = SpkLstmVanilla(40, args.hidden_size, len(id_spk), args.layers, 0.0, args.config_residual, args.extract_layer)
model.to(device)
model.eval()

if args.pred_param:
    ckpt = torch.load(args.pred_param)
    model.pred.load_state_dict(ckpt['pred'])

if args.apc_param:
    ckpt = torch.load(args.apc_param)
    p = {}
    for k in ckpt['model']:
        if k.startswith('lstms') and int(k[-1]) < args.layers:
            p[k[len('lstms.'):]] = ckpt['model'][k]
    model.lstms.load_state_dict(p)

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
        pred = model(padded_inputs, length)                                 # pred.shape [batch_size, num_speaker]
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
