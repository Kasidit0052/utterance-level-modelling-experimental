#!/bin/sh
. /home/htang2/toolchain-06022023/toolchain.rc
. /home/s2426593/utt_apc/apc-env/bin/activate

exp=$1
epoch=$2

src/predict-probe-utterance.py \
    --config $exp/predict.conf \
    --pred-param $exp/param-$epoch > $exp/dev-$epoch-eval.log