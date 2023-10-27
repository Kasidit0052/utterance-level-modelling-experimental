#!/bin/sh
. /home/htang2/toolchain-06022023/toolchain.rc
. /home/s2426593/utt_apc/apc-env/bin/activate

exp_folder=$1
epoch_start=$2
epoch_end=$3

echo  -e "JobName...........\n"
job_name= basename $exp_folder
echo  -e "\n"

echo -e "Evaluating model on speaker classification...........\n"

for ((i = $epoch_start; i <= $epoch_end; i++)); do

    if test -f "${exp_folder}/dev-${i}-eval.log"; then
        echo -e "Epoch ${exp_folder}/dev-${i}-eval.log exists"
    else
        echo -e "Epoch ${i}: Command[srun -J $job_name bash predict-probe-epoch.sh $exp_folder $i]"
        srun -J $job_name bash predict-probe-epoch.sh $exp_folder $i
        wait
    fi

    echo -e "\n"

done