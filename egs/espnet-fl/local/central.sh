#!/bin/bash

# Copyright 2021 Ruhr University Bochum (Wentao Yu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


do_delta=false
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
backend=pytorch
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
bpemode=unigram
nbpe=5000
nj=32

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

dumpdir=$1
dict=$2
lmexpdir=$3
lang_model=$4



nbpe=5000
stage=0       
stop_stage=100

resume=        # Resume the training from snapshot

preprocess_config=conf/specaug.yaml # TODO ACHTUNG HIER BEI TRAINING ANPASSEN
train_config=conf/train.yaml # current default recipe requires 4 gpus.
                             # if you do not have 4 gpus, please reconfigure the `batch-bins` and `accum-grad` parameters in config.
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml



# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=10                  # the number of ASR models to be averaged
use_valbest_average=false     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=0               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.






pretrain=pretrain
transf_pre_test=${pretrain}/transfer_pre_test_org
transf_pre_dev=${pretrain}/transfer_pre_dev_org
transf_fl_test=${pretrain}/transfer_fl_test_org
transf_fl_dev=${pretrain}/transfer_fl_dev_org

setname_a=whole #reference
setname_b=pretrain # initial model for decentralized training
setname_list="${setname_a} ${setname_b}"

for setname in ${setname_list};do
    train_set=${setname}/train
    train_dev=${setname}/dev
    test_set=${setname}/test

    recog_set="${test_set} ${train_dev} ${test_pre_set} ${train_pre_dev} ${transf_pre_test} ${transf_pre_dev} ${transf_fl_test} ${transf_fl_dev}"
    feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
    feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
    expdir=exp/${setname}
    mkdir -p ${expdir}
    if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
        echo "stage 0: Network Training"
        ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
            asr_train.py \
            --config ${train_config} \
            --preprocess-conf ${preprocess_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --outdir ${expdir}/results \
            --debugmode ${debugmode} \
            --dict ${dict} \
            --debugdir ${expdir} \
            --minibatches ${N} \
            --verbose ${verbose} \
            --resume ${resume} \
            --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
            --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json || exit 1
    fi
done

for setname in ${setname_list};do
    train_set=${setname}/train
    train_dev=${setname}/dev
    test_set=${setname}/test

    #train_pre_set=${pretrain}/train
    #train_pre_dev=${pretrain}/dev
    #test_pre_set=${pretrain}/test
    expdir=exp/${setname}
    recog_set="${test_pre_set} ${train_pre_dev} ${transf_pre_test} ${transf_pre_dev} ${transf_fl_test} ${transf_fl_dev}"
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        echo "stage 1: Decoding"
        if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]]; then
            # Average ASR models
            if ${use_valbest_average}; then
                recog_model=model.val${n_average}.avg.best
                opt="--log ${expdir}/results/log"
            else
                recog_model=model.last${n_average}.avg.best #recog_model=model.acc.best #model.last${n_average}.avg.best
                opt="--log"
            fi
            average_checkpoints.py \
                ${opt} \
                --backend ${backend} \
                --snapshots ${expdir}/results/snapshot.ep.* \
                --out ${expdir}/results/${recog_model} \
                --num ${n_average} || exit 1

            # Average LM models
            if [ ${lm_n_average} -eq 0 ]; then
                lang_model=rnnlm.model.best
            else
                if ${use_lm_valbest_average}; then
                    lang_model=rnnlm.val${lm_n_average}.avg.best
                    opt="--log ${lmexpdir}/log"
                else
                    lang_model=rnnlm.last${lm_n_average}.avg.best
                    opt="--log"
                fi
                average_checkpoints.py \
                    ${opt} \
                    --backend ${backend} \
                    --snapshots ${lmexpdir}/snapshot.ep.* \
                    --out ${lmexpdir}/${lang_model} \
                    --num ${lm_n_average} || exit 1
            fi
        fi
        for testset in ${recog_set};do
            pids=() # initialize pids
                for rtask in ${testset}; do
                (
                    decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
                    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

                    # split data
                    splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json || exit 1

                    #### use CPU for decoding
                    ngpu=0

                    # set batchsize 0 to disable batch decoding
                    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                        asr_recog.py \
                        --config ${decode_config} \
                        --ngpu ${ngpu} \
                        --backend ${backend} \
                        --batchsize 0 \
                        --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
                        --result-label ${expdir}/${decode_dir}/data.JOB.json \
                        --model ${expdir}/results/model.last10.avg.best \
                        --rnnlm ${lmexpdir}/${lang_model} \
                        --api v2 || exit 1

                        #${expdir}/results/${recog_model}  \
                        #--rnnlm ${lmexpdir}/${lang_model} \
                        #--api v2

                    score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict} || exit 1
    
                    ) &
                    pids+=($!) # store background pids
                    done
                    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
                    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
                    echo "Finished"
        done
    fi

done

# Service: copy of federated learning models will be saved in exp/FL/results/central, in order to decode them, a model.json file
# with the correct modelparameters is needed, pretrain file has all needed information
mkdir -p exp/FL || exit 1
mkdir -p exp/FL/results || exit 1
mkdir -p exp/FL/results/central || exit 1
cp exp/pretrain/results/model.json exp/FL/results/central || exit 1
