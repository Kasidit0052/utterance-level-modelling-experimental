#!/bin/sh
. /home/htang2/toolchain-06022023/toolchain.rc
. /home/s2426593/utt_apc/apc-env/bin/activate

exp_folder=$1
epoch_start=$2
epoch_end=$3

echo  -e "JobName...........\n"
job_name= basename $exp_folder
echo  -e "\n"

for ((i = $epoch_start; i <= $epoch_end; i++)); do
    if [[ $i == 1 ]]; then 
        echo -e "Initializing the model...........\n"

        if test -f "${exp_folder}/param-0"; then
            echo -e "Epoch 0: ${exp_folder}/param-0 exists"
        else
            echo -e "Epoch 0: Command[srun -J $job_name bash train-utterance-epoch.sh $exp_folder 0]"
            srun -J $job_name bash train-utterance-epoch.sh $exp_folder 0
            wait
        fi
        echo -e "\n"
        echo -e "Training the the model...........\n"
    fi

    if test -f "${exp_folder}/param-${i}"; then
        echo -e "Epoch ${i}: ${exp_folder}/param-${i} exists"
    else
        echo -e "Epoch ${i}: Command[srun -J $job_name bash train-utterance-epoch.sh $exp_folder $i]"
        srun -J $job_name bash train-utterance-epoch.sh $exp_folder $i
        wait
    fi

    echo -e "\n"

done
