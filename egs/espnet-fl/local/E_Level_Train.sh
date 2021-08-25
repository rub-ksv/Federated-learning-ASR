#!/bin/bash

# Copyright 2021 Ruhr University Bochum (Wentao Yu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

backend=pytorch
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
do_delta=false
N=0
verbose=0
nbpe=5000
bpemode=unigram
resume=exp/pretrain/results/model.last10.avg.best	# The initial model dir




. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;


dumpdir=$1
dict=$2

dumpdir=$dumpdir/FL
expdir=exp/FL

train_config=conf/train_fl.yaml
preprocess_config=conf/specaug.yaml



mkdir -p ${expdir}
python3 local/spkuttdict.py data/FL ${expdir}	# creat dict contains how many utterances has a speaker
trainjson=${expdir}/train.json 			#Path to Json-File with size of all speaker training sets

n_average=$(< Speakerlist wc -l)		 #max Value for Allset
n_iter=50   					#predefined value, chosen by IO Dialog

printf "Training Mode configuration"
printf "Choose All or random Subset:\n All: 1 \n Subset: 2\n"
read experiment
if [[ $experiment -eq 1 ]]; then
    printf "Input: Allset \n"
    exp_string="Allset"
elif [[ $experiment -eq 2 ]]; then
    exp_string="Subset"
    printf "Input: ${exp_string} \n"
    printf "Speakers per Round?:"
    read n_average
    printf "Number of Speakers: $n_average \n" 
else 
    exit
fi
printf "mean or weighted Federated Averaging? \n mean: 1 \n weighted:2 \n"
read FL_meth
if [[ $FL_meth -eq 1 ]]; then
    FL_name=mean
    #printf "mean chosen \n"
elif [[ $FL_meth -eq 2 ]]; then
    FL_name=weighted
    #printf "weighted chosen \n"
else 
    exit
fi 

printf "Number of Rounds n_iter?"
read n_iter
printf "Number of Rounds: $n_iter \n"
printf "Outputs epoch interval in exp/FL/central n_iter_show?"
read n_iter_show
if [ $((${n_iter} % n_iter_show)) -eq 0 ]; then
    printf "Summary: ${exp_string} with ${n_average} Speakers per Round using ${FL_name} Averaging for ${n_iter} Rounds\n"
else
    printf "n_iter divide n_iter_show must be integer"
    exit
fi

printf "Is this correct? \n yes: 1 \n no: 0 \n"
read last_chance_to_quit
if [ ${last_chance_to_quit} -eq 0 ]; then
    printf "Terminiating by choice"
    exit
fi

printf "Running now!!!\n"

ls $dumpdir/ > Speakerlist   ## make a speaker list of all speakers

if [ ${experiment} -eq 1 ]; then
    #n_average=1372
    #${expdir}/results/2average_iter_${exp_string}${Nspeaker}_${FL_name}
    mkdir -p ${expdir}/results/2average_iter_${exp_string}_${FL_name} || exit 1
    mkdir -p ${expdir}/results/devices_iter_${exp_string}_${FL_name} || exit 1
    mkdir -p ${expdir}/results/central_iter_${exp_string}_${FL_name} || exit 1
    mkdir -p ${expdir}/results/central || exit 1
    rm -f ${expdir}/results/progress_iter_${exp_string}_${FL_name}
    for iter in $(seq 1 ${n_iter}); # experimental shortening # should be tested!
        do
            for dir in $dumpdir/*
            do                
                set_name=${dir:8}
                train_set=${set_name}/train
                train_dev=${set_name}/dev
                test_set=${set_name}/test
                recog_set="${train_dev} ${test_set}"
                feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}
                feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}
                
                if [ ${iter} -eq 1 ]; then
                    mkdir -p ${expdir}/results/devices_iter_${exp_string}_${FL_name}/${set_name}
                    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train_${exp_string}_${FL_name}.log \
                        asr_train_fl.py \
                        --config ${train_config} \
                        --preprocess-conf ${preprocess_config} \
                        --ngpu ${ngpu} \
                        --backend ${backend} \
                        --outdir ${expdir}/results/devices_iter_${exp_string}_${FL_name}/${set_name}  \
                        --debugmode ${debugmode} \
                        --dict ${dict} \
                        --debugdir ${expdir} \
                        --minibatches ${N} \
                        --verbose ${verbose} \
                        --pretrain-model ${resume} \
                        --num-save-attention 0 \
                        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
                        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json || exit 1

                else
                    last_iter=$((iter-1))
                    last_snap=${expdir}/results/devices_iter_${exp_string}_${FL_name}/${set_name}/snapshot.ep.${last_iter}
                    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train_${exp_string}_${FL_name}.log \
                        asr_train_fl.py \
                        --config ${train_config} \
                        --preprocess-conf ${preprocess_config} \
                        --epochs ${iter} \
                        --ngpu ${ngpu} \
                        --backend ${backend} \
                        --outdir ${expdir}/results/devices_iter_${exp_string}_${FL_name}/${set_name}  \
                        --debugmode ${debugmode} \
                        --dict ${dict} \
                        --debugdir ${expdir} \
                        --minibatches ${N} \
                        --verbose ${verbose} \
                        --resume ${last_snap} \
                        --pretrain-model ${resume} \
                        --num-save-attention 0 \
                        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
                        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json || exit 1
                 
                        rm ${last_snap} || exit 1
                fi    

                newest=$(ls -t ${expdir}/results/devices_iter_${exp_string}_${FL_name}/${set_name}/snapshot.ep* | head -1) || exit 1
                ln -rsf ${newest} ${expdir}/results/2average_iter_${exp_string}_${FL_name}/snapshot.${set_name} || exit 1
                echo ${set_name} >> ${expdir}/results/progress_iter_${exp_string}_${FL_name} || exit 1
                cat ${expdir}/train_${exp_string}_${FL_name}.log >> ${expdir}/full_train_${exp_string}_${FL_name}.log || exit 1
            done
            if [ ${FL_meth} -eq 1 ]; then
                local/FL_functions/average_speaker.py --backend ${backend} \
                --snapshots ${expdir}/results/2average_iter_${exp_string}_${FL_name}/snapshot.* \
                --out ${expdir}/results/central_iter_${exp_string}_${FL_name}/model.${FL_name}.avg.${iter} \
                --snap 1 \
                --num ${n_average}  || exit 1

                resume=${expdir}/results/central_iter_${exp_string}_${FL_name}/model.${FL_name}.avg.${iter} 
            fi

            if [ ${FL_meth} -eq 2 ]; then
                local/FL_functions/fed_avg.py --snapshots ${expdir}/results/2average_iter_${exp_string}_${FL_name}/snapshot.* \
                        --out ${expdir}/results/central_iter_${exp_string}_${FL_name}/model.${FL_name}.avg.${iter} \
                        --num ${n_average} \
                        --backend ${backend} \
                        --jsonlist ${trainjson} \
                        --partlist ${expdir}/results/progress_iter_${exp_string}_${FL_name} \
                        --snap 1  || exit 1
                resume=${expdir}/results/central_iter_${exp_string}_${FL_name}/model.${FL_name}.avg.${iter}
            fi
            if [ $((${iter} % n_iter_show)) -eq 0 ]; then
                    ln -rsf ${resume} ${expdir}/results/central/model.Epoch_${exp_string}${FL_name}${iter}.avg || exit 1
            fi
            rm ${expdir}/results/progress_iter_${exp_string}_${FL_name}
            rm ${expdir}/results/2average_iter_${exp_string}_${FL_name}/snapshot* 
        done
fi


if [ ${experiment} -eq 2 ]; then
    Nspeaker=$n_average
    rm -rf ${expdir}/results/2average_iter_${exp_string}${Nspeaker}_${FL_name}
    rm -rf ${expdir}/results/devices_iter_${exp_string}${Nspeaker}_${FL_name}
    rm -rf ${expdir}/results/central_iter_${exp_string}${Nspeaker}_${FL_name}
    #rm -rf ${expdir}/results/spk_trail
    mkdir -p ${expdir}/results/2average_iter_${exp_string}${Nspeaker}_${FL_name}
    mkdir -p ${expdir}/results/devices_iter_${exp_string}${Nspeaker}_${FL_name}
    mkdir -p ${expdir}/results/central_iter_${exp_string}${Nspeaker}_${FL_name}
    mkdir -p ${expdir}/results/spk_trail${exp_string}${Nspeaker}_${FL_name}
    mkdir -p ${expdir}/results/central || exit 1
    #mkdir -p ${expdir}/results/lr_snap${Nspeaker}
    rm -f ${expdir}/results/progress_fl_${exp_string}${Nspeaker}_${FL_name}
    del_last=0 # just a status val if last snapshot should be removed, initial value must be 0
    for iter in $(seq 1 ${n_iter});
        do
            shuf -n $Nspeaker Speakerlist > exp/FL/results/spk_trail${exp_string}${Nspeaker}_${FL_name}/selectedspeaker${Nspeaker}.${iter}   ## randomly selected Nspeaker speakers for this epoch, the subspeaker set is saved in selectedspeaker.$iter
            cat exp/FL/results/spk_trail${exp_string}${Nspeaker}_${FL_name}/selectedspeaker${Nspeaker}.${iter} | while read line
            do
                dir=$dumpdir/$line
                set_name=${dir:8}
                train_set=${set_name}/train
                train_dev=${set_name}/dev
                test_set=${set_name}/test
                recog_set="${train_dev} ${test_set}"
                feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}
                feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}
                snaps=${expdir}/results/devices_iter_${exp_string}${Nspeaker}_${FL_name}/${set_name}/snapshot.ep.*
                if ls ${snaps} >/dev/null 2>&1 ; then
                    newest=$(ls -t ${expdir}/results/devices_iter_${exp_string}${Nspeaker}_${FL_name}/${set_name}/snapshot.ep* | head -1)
                    last_snap=${newest}
                    del_last=1
                    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train_${exp_string}${Nspeaker}_${FL_name}.log \
                        asr_train_fl.py \
                        --config ${train_config} \
                        --preprocess-conf ${preprocess_config} \
                        --ngpu ${ngpu} \
                        --backend ${backend} \
                        --outdir ${expdir}/results/devices_iter_${exp_string}${Nspeaker}_${FL_name}/${set_name}  \
                        --debugmode ${debugmode} \
                        --dict ${dict} \
                        --debugdir ${expdir} \
                        --minibatches ${N} \
                        --verbose ${verbose} \
                        --resume ${last_snap} \
                        --pretrain-model ${resume} \
                        --num-save-attention 0 \
                        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
                        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
                        rm ${last_snap} || exit 1
                else
                     ${cuda_cmd} --gpu ${ngpu} ${expdir}/train_${exp_string}${Nspeaker}_${FL_name}.log \
                        asr_train_fl.py \
                        --config ${train_config} \
                        --preprocess-conf ${preprocess_config} \
                        --ngpu ${ngpu} \
                        --backend ${backend} \
                        --outdir ${expdir}/results/devices_iter_${exp_string}${Nspeaker}_${FL_name}/${set_name}  \
                        --debugmode ${debugmode} \
                        --dict ${dict} \
                        --debugdir ${expdir} \
                        --minibatches ${N} \
                        --verbose ${verbose} \
                        --pretrain-model ${resume} \
                        --num-save-attention 0 \
                        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
                        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json || exit 1
                fi    
                newest=$(ls -t ${expdir}/results/devices_iter_${exp_string}${Nspeaker}_${FL_name}/${set_name}/snapshot.ep* | head -1)
                ln -rsf ${newest} ${expdir}/results/2average_iter_${exp_string}${Nspeaker}_${FL_name}/snapshot.${set_name}
                echo ${set_name} >> ${expdir}/results/progress_fl_${exp_string}${Nspeaker}_${FL_name}
                cat ${expdir}/train_${exp_string}${Nspeaker}_${FL_name}.log >> ${expdir}/full_train_${exp_string}${Nspeaker}_${FL_name}.log
            done

            if [ ${FL_meth} -eq 1 ]; then
                local/FL_functions/average_speaker.py --backend ${backend} \
                --snapshots ${expdir}/results/2average_iter_${exp_string}_${FL_name}/snapshot.* \
                --out ${expdir}/results/central_iter_${exp_string}${Nspeaker}_${FL_name}/model.${FL_name}.avg.${iter} \
                --snap 1 \
                --num ${n_average}  || exit 1

                resume=${expdir}/results/central_iter_${exp_string}${Nspeaker}_${FL_name}/model.${FL_name}.avg.${iter} 
            fi
            if [ ${FL_meth} -le 2 ] && [ ${FL_meth} -ge 2 ]; then
                local/FL_functions/fed_avg.py --snapshots ${expdir}/results/2average_iter_${exp_string}${Nspeaker}_${FL_name}/snapshot.* \
                        --out ${expdir}/results/central_iter_${exp_string}${Nspeaker}_${FL_name}/model.weighted.avg.${iter} \
                        --num ${n_average} \
                        --backend ${backend} \
                        --jsonlist ${expdir}/train.json \
                        --partlist ${expdir}/results/progress_fl_${exp_string}${Nspeaker}_${FL_name} \
                        --snap 1  || exit 1
                resume=${expdir}/results/central_iter_${exp_string}${Nspeaker}_${FL_name}/model.weighted.avg.${iter}
            fi
            if [ $((${iter} % n_iter_show)) -eq 0 ]; then
                ln -rsf ${resume} ${expdir}/results/central/model.subset${Nspeaker}_${iter}.avg
            fi
            rm ${expdir}/results/progress_fl_${exp_string}${Nspeaker}_${FL_name} || exit 1
            rm ${expdir}/results/2average_iter_${exp_string}${Nspeaker}_${FL_name}/snapshot* || exit 1

        done
fi