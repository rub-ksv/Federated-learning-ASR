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

experiment=0
level=0
stop_level=100


train_config=conf/train_once.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
preprocess_config=conf/specaug.yaml




if [ ${experiment} -le 0 ] && [ ${experiment} -ge 0 ]; then
    #first experiment using all participants one time
    if [ ${level} -le 0 ] && [ ${stop_level} -ge 0 ]; then
        #here only training
        expdir=exp/FL
        n_average=10
        rm -f ${expdir}/results/progress_once
        rm -f ${expdir}/full_train_once.log
        mkdir -p ${expdir}/results/devices_once 
        mkdir -p ${expdir}/results/2average_once
        mkdir -p ${expdir}/results/central_once

        for dir in dump/FL/*
        do
            set_name=${dir:8}
            #set_name=127
            mkdir -p ${expdir}/results/devices_once/${set_name}
            train_set=${set_name}/train
            train_dev=${set_name}/dev
            test_set=${set_name}/test
            recog_set="${train_dev} ${test_set}"
            feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}
            feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}

            ${cuda_cmd} --gpu ${ngpu} ${expdir}/train_once.log \
                    asr_train_fl.py \
                    --config ${train_config} \
                    --preprocess-conf ${preprocess_config} \
                    --ngpu ${ngpu} \
                    --backend ${backend} \
                    --outdir ${expdir}/results/devices_once/${set_name}  \
                    --debugmode ${debugmode} \
                    --dict ${dict} \
                    --debugdir ${expdir} \
                    --minibatches ${N} \
                    --verbose ${verbose} \
                    --pretrain-model ${resume} \
                    --num-save-attention 0 \
                    --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
                    --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
                    #--tensorboard-dir tensorboard/${expname} \

            average_checkpoints.py --backend ${backend} \
                    --snapshots ${expdir}/results/devices_once/${set_name}/snapshot.* \
                    --out ${expdir}/results/devices_once/${set_name}/model.last${n_average}.avg \
                    --num ${n_average}
            rm ${expdir}/results/devices_once/${set_name}/snapshot.ep.*
            ln -rsf ${expdir}/results/devices_once/${set_name}/model.last10.avg ${expdir}/results/2average_once/snapshot.${set_name}
            cat ${expdir}/train_once.log >> ${expdir}/full_train_once.log
            echo ${set_name} >> ${expdir}/results/progress_once
        done
    fi
    if [ ${level} -le 1 ] && [ ${stop_level} -ge 1 ]; then
        expdir=exp/FL
        n_average=$(wc -l ${expdir}/results/progress_once | cut -d' ' -f1)
        local/FL_functions/fed_avg.py --snapshots ${expdir}/results/2average_once/snapshot.* \
                --out ${expdir}/results/central_once/model.weighted_once.avg \
                --num ${n_average} \
                --backend ${backend} \
                --jsonlist ${expdir}/train.json \
                --partlist ${expdir}/results/progress_once \
                --snap 0 

        local/FL_functions/average_speaker.py --backend ${backend} \
                --snapshots ${expdir}/results/2average_once/snapshot.* \
                --out ${expdir}/results/central_once/model.gleich_once.avg \
                --snap 0 \
                --num ${n_average} 

        ln -rsf ${expdir}/results/central_once/model.weighted_once.avg ${expdir}/results/central/model.weighted_once.avg
        ln -rsf ${expdir}/results/central_once/model.gleich_once.avg ${expdir}/results/central/model.gleich_once.avg
    fi

fi
