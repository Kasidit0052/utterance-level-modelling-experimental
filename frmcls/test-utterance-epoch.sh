#!/bin/sh
. /home/htang2/toolchain-06022023/toolchain.rc
. /home/s2426593/utt_apc/apc-env/bin/activate

exp=$1
epoch=$2

./src/predict-frmcls.py \
    --config $exp/test.conf \
    --pred-param $exp/param-$epoch \
    > $exp/dev-$epoch.log

tail -n+18 $exp/dev-$epoch.log > $exp/dev-$epoch-nohead.log
./util/eval-frames.py "dataset/wsj/extra/si284-0.9-dev.bpali" $exp/dev-$epoch-nohead.log > $exp/dev-$epoch-eval.log
rm $exp/dev-$epoch-nohead.log

