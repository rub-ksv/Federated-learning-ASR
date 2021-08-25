#!/bin/bash

# Copyright 2021 Ruhr University Bochum (Wentao Yu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)



#This script is searching for models that need to be used for decoding dev dataset in exp/FL/results/central, no need to do anything here just start it regularly and work through the results (no script for that, sorry)


backend=pytorch
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
do_delta=false
nbpe=5000
bpemode=unigram
nj=32

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;


dumpdir=$1
dict=$2
bpemodel=$3


decode_config=conf/decode.yaml


pretrain=pretrain

train_dev=dev
test_set=test
transf_pre_test=transfer_pre_test_org
transf_pre_dev=transfer_pre_dev_org
transf_fl_test=transfer_fl_test_org
transf_fl_dev=transfer_fl_dev_org

recog_set="${train_dev} ${transf_pre_dev} ${transf_fl_dev}"

modeldir=exp/FL/results/central
file=exp/FL/decodelists/reference.txt



expdir=exp/FL
lmexpdir=exp/LM
mkdir -p exp/FL/decodelists
if [ ! -e "${file}" ]; then
    echo model.json >> exp/FL/decodelists/reference.txt || exit 1
fi

if [ ! -e "exp/FL/results/central/model.json" ]; then
    cp exp/pretrain/results/model.json  exp/FL/results/central || exit 1
fi
ls ${modeldir} > exp/FL/decodelists/tocheck.txt || exit 1

sort exp/FL/decodelists/reference.txt -u > exp/FL/decodelists/reference_sorted.txt || exit 1
sort exp/FL/decodelists/tocheck.txt -u > exp/FL/decodelists/tocheck_sorted.txt || exit 1
diff exp/FL/decodelists/reference_sorted.txt exp/FL/decodelists/tocheck_sorted.txt | grep '^>' | sed 's/^>\ //' > exp/FL/decodelists/decode.txt || exit 1

cat exp/FL/decodelists/decode.txt | while read line
do
    recog_model=${modeldir}/${line}
    for testset in ${recog_set};do
            pids=() # initialize pids
                for rtask in ${testset}; do
                (
                    decname=$(echo ${line} | cut -d'.' -f2)
                    decode_dir=decode/${decname}/pretrain/${rtask}/$(basename ${decode_config%.*})_lm
                    #decode_dir=decode_${rtask}_${line}_$(basename ${decode_config%.*})_${lmtag}
                    feat_recog_dir=${dumpdir}/pretrain/${rtask}/delta${do_delta}

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
                        --model ${recog_model}  \
                        --rnnlm exp/pretrainedlm/rnnlm.model.best \
                        --api v2 || exit 1


                    score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict} || exit 1
    
                    ) &
                    pids+=($!) # store background pids
                    done
                    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
                    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
                    echo "Finished"
    done
    echo ${line} >> exp/FL/decodelists/reference.txt || exit 1
    

done 