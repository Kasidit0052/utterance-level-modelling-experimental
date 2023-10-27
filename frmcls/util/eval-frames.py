#!/usr/bin/env python3

import sys

pred_f = open(sys.argv[1])
gold_f = open(sys.argv[2])

total_len = 0
total_err = 0

area = 'head'

for line_p, line_g in zip(pred_f, gold_f):

    if area == 'head':
        assert line_p == line_g
        area = 'body'
        key = line_p.strip()
    elif area == 'body' and line_p == '.\n':
        area = 'head' 
    elif area == 'body':
        labels_p = line_p.split()
        labels_g = line_g.split()

        assert len(labels_p) == len(labels_g)

        err = sum([1 if p != g else 0 for p, g in zip(labels_p, labels_g)])

        total_len += len(labels_p)
        total_err += err

        print('{}, len: {}, err: {}, rate: {:.6}'.format(key, len(labels_p), err, err / len(labels_p)))

print('total len: {}, total err: {}, rate: {:.6}'.format(total_len, total_err, total_err / total_len))
