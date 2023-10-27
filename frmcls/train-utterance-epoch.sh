#!/bin/sh
. /home/htang2/toolchain-06022023/toolchain.rc
. /home/s2426593/utt_apc/apc-env/bin/activate

exp=$1
epoch=$2

if [[ $epoch == 0 ]]; then 

    ./src/train-frmcls.py --config $exp/train.conf \
        --init \
        --pred-param-output $exp/param-$epoch \

else

    ./src/train-frmcls.py --config $exp/train.conf \
        --pred-param $exp/param-$((epoch-1)) \
        --pred-param-output $exp/param-$epoch \
        > $exp/log-$epoch

fi