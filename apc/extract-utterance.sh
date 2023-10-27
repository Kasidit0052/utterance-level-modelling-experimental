#!/bin/sh
. /home/htang2/toolchain-06022023/toolchain.rc
. /home/s2426593/utt_apc/apc-env/bin/activate

exp=$1
epoch=$2

./src/extract_utterance.py --config $exp/extract.conf \
    --param $exp/param-$epoch \