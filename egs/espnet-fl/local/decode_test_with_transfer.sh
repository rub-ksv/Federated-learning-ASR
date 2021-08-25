#!/bin/bash

# Copyright 2021 Ruhr University Bochum (Wentao Yu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


#this script is used for decoding test dataset

backend=pytorch
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
do_delta=false
nbpe=5000
bpemode=unigram
nj=32

experiment=0

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

dumpdir=$1
dict=$2
bpemodel=$3
model_type=$4



decode_config=conf/decode.yaml


#train_dev=pretrain/dev
test_set=pretrain/test
transf_pre_test=pretrain/transfer_pre_test_org
#transf_pre_dev=pretrain/transfer_pre_dev_org
transf_fl_test=pretrain/transfer_fl_test_org
#transf_fl_dev=pretrain/transfer_fl_dev_org
recog_set="${test_set} ${transf_fl_test} ${transf_pre_test}"

#typically models are saved as model.<name of training>.avg as seen below
#if you want to decode multiple models just use model_type_<name> and add it to model_type_iter list
model_type_iter="${model_type}"

if [ ${experiment} -le 0 ] && [ ${experiment} -ge 0 ]; then
    echo "stage 5: Decoding pretraindecoding"
        
        expdir=exp/FL
        lmexpdir=exp/pretrainedlm
        set_name=pretrain

        for iter in ${model_type_iter}
        do
            model_name=model.${iter}.avg
            recog_model=${expdir}/results/central/${model_name}
            for testset in ${recog_set};do
                pids=() # initialize pids
                for rtask in ${testset}; do
                (
                    decode_dir=${iter}/${rtask}_decode_lm
                    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

                    # split data
                    splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json || exit 1

                    #### use CPU for decoding
                    ngpu=0

                    ${decode_cmd} JOB=1:${nj} ${expdir}/decode/${decode_dir}/log/decode.JOB.log \
                        asr_recog.py \
                        --config ${decode_config} \
                        --ngpu ${ngpu} \
                        --backend ${backend} \
                        --batchsize 0 \
                        --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
                        --result-label ${expdir}/decode/${decode_dir}/data.JOB.json \
                        --model ${recog_model}  \
                        --rnnlm $lmexpdir/rnnlm.model.best \
                        --api v2 || exit 1
                    
                    score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/decode/${decode_dir} ${dict} || exit 1

                ) &
                pids+=($!) # store background pids
                done
                i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
                [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
            

                done
        done

fi
