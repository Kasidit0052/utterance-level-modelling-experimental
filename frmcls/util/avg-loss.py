#!/usr/bin/env python3

import sys
import re

total_loss = 0
nsample = 0

for line in sys.stdin:
    m = re.match('.+: (.+)', line)

    if m:
        total_loss += float(m.group(1))
        nsample += 1

print('samples: {}, avg loss: {:.6}'.format(nsample, total_loss / nsample))
