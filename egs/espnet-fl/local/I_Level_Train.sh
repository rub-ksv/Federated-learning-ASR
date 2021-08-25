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
n_iter=100	# how many epochs
resume=exp/pretrain/results/model.last10.avg.best	# The initial model dir

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

dumpdir=$1
dict=$2


dumpdir=$dumpdir/FL
experiment=0
stage=0
stop_stage=0

train_config=conf/train_chunk.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
preprocess_config=conf/specaug.yaml


if [ ${experiment} -le 0 ] && [ ${experiment} -ge 0 ]; then
    FL_meth=1 # 1: mean Federated Averaging 2: weighted Federated Averaging
    startup=1 #1: neue Iteration beginnen, 40: Beginn nach 40 Epochen
    expdir=exp/FL
    n_average=$(< Speakerlist wc -l)
    
    itval=4 
    accum_fac=2 #muss = accum-grad sein!
    iterate_on=1 # bypassing option for testing averagings
    epstop=0 
    
    if [ ${FL_meth} -le 1 ] && [ ${FL_meth} -ge 1 ]; then
        FL_name=mean
    fi
    if [ ${FL_meth} -le 2 ] && [ ${FL_meth} -ge 2 ]; then
        FL_name=weighted
    fi

    mkdir -p ${expdir}/results/2average_iter_chunk_${FL_name}
    mkdir -p ${expdir}/results/devices_iter_chunk_${FL_name}
    mkdir -p ${expdir}/results/central_iter_chunk_${FL_name}
    rm -rf ${expdir}/results/progress_chunk_${FL_name}
    
    if [ ${iterate_on} -le 1 ] && [ ${iterate_on} -ge 1 ]; then
        if [ ${startup} -le 1 ] && [ ${startup} -ge 1 ]; then
            iter=1
            rm -f ${expdir}/results/2average_iter_chunk_${FL_name}/snapshot*
            for dir in dump/FL/*
            do
                set_name=${dir:8}
                train_set=${set_name}/train
                train_dev=${set_name}/dev
                test_set=${set_name}/test
                recog_set="${train_dev} ${test_set}"
                feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}
                feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}
                mkdir -p ${expdir}/results/devices_iter_chunk_${FL_name}/${set_name}
                size=$(wc -l ${dir}/train/deltafalse/utt2num_frames | cut -d' ' -f1)
                ${cuda_cmd} --gpu ${ngpu} ${expdir}/train_chunk.log \
                    asr_train_fl.py \
                    --config ${train_config} \
                    --preprocess-conf ${preprocess_config} \
                    --ngpu ${ngpu} \
                    --backend ${backend} \
                    --outdir ${expdir}/results/devices_iter_chunk_${FL_name}/${set_name}  \
                    --debugmode ${debugmode} \
                    --dict ${dict} \
                    --debugdir ${expdir} \
                    --minibatches ${N} \
                    --verbose ${verbose} \
                    --pretrain-model ${resume} \
                    --num-save-attention 0 \
                    --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
                    --epstop ${epstop} \
                    --startup 1\
                    --iterval ${itval} \
                    --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
                    #--tensorboard-dir tensorboard/${expname} 
                cat ${expdir}/results/devices_iter_chunk_${FL_name}/${set_name}/itercount.txt >> ${expdir}/results/speakeriters_chunk
                newest=$(ls -t ${expdir}/results/devices_iter_chunk_${FL_name}/${set_name}/snapshot* | head -1)
                ln -rsf ${newest} ${expdir}/results/2average_iter_chunk_${FL_name}/snapshot.${set_name}
                echo ${set_name} >> ${expdir}/results/progress_chunk_${FL_name}
                cat ${expdir}/train_chunk.log >> ${expdir}/full_train_iter_chunk_${FL_name}.log
            done
            num=$(wc -l ${expdir}/results/progress_chunk_${FL_name} | cut -d' ' -f1)
            echo "num" ${num}
            if [ ${FL_meth} -le 1 ] && [ ${FL_meth} -ge 1 ]; then
                local/FL_functions/average_speaker.py --backend ${backend} \
                        --snapshots ${expdir}/results/2average_iter_chunk_${FL_name}/snapshot.* \
                        --out ${expdir}/results/central_iter_chunk_${FL_name}/model.iter.avg.${iter} \
                        --snap 1 \
                        --num ${num}
                resume=$(ls -t ${expdir}/results/central_iter_chunk_${FL_name}/model* | head -1)
                local/FL_functions/maxiters.py --infile ${expdir}/results/speakeriters_chunk \
                        --outfile ${expdir}/results/maxiters.txt
                max_entry=$(cat ${expdir}/results/maxiters.txt)
                factor=$((accum_fac*itval))
                max_its=$(((max_entry+factor-1)/factor))
            fi
            
            if [ ${FL_meth} -le 2 ] && [ ${FL_meth} -ge 2 ]; then
                local/FL_functions/fed_avg.py --snapshots ${expdir}/results/2average_iter_chunk_${FL_name}/snapshot.* \
                    --out ${expdir}/results/central_iter_chunk_${FL_name}/model.iter.avg.${iter} \
                    --num ${num} \
                    --backend ${backend} \
                    --snap 1 \
                    --jsonlist ${expdir}/train.json \
                    --partlist ${expdir}/results/progress_chunk_${FL_name} 
                resume=$(ls -t ${expdir}/results/central_iter_chunk_${FL_name}/model* | head -1)
                local/FL_functions/maxiters.py --infile ${expdir}/results/speakeriters_chunk \
                        --outfile ${expdir}/results/maxiters.txt
                max_entry=$(cat ${expdir}/results/maxiters.txt)
                factor=$((accum_fac*itval))
                max_its=$(((max_entry+factor-1)/factor))        
                #echo ${max_its}
            fi
            rm ${expdir}/results/progress_chunk_${FL_name}
            rm ${expdir}/results/2average_iter_chunk_${FL_name}/snapshot*
            #done
            min_its=2
        fi
        for iter in $(seq 0 ${n_iter-1});
        do
            for its in $(seq ${min_its} ${max_its})
            do
                #for name in ${test_names}
                for dir in dump/FL/*
                do
                    #dir=${name}
                    set_name=${dir:8}
                    train_set=${set_name}/train
                    train_dev=${set_name}/dev
                    test_set=${set_name}/test
                    recog_set="${train_dev} ${test_set}"
                    feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}
                    feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}
                    spkr_its=$(cat ${expdir}/results/devices_iter_chunk_${FL_name}/${set_name}/itercount.txt)
                    akt_its_done=$((its-1))
                    eps_its_done=$((akt_its_done*factor))

                    if [ ${eps_its_done} -lt ${spkr_its} ]; then
                        diff_part=$((spkr_its-eps_its_done))
                        diff_val=$((diff_part))
                        echo "diff-Wert" ${diff_val}
                        if [ ${diff_val} -le ${factor} ]; then
                            epstop=1
                        fi
                        newest=$(ls -t ${expdir}/results/devices_iter_chunk_${FL_name}/${set_name}/snapshot* | head -1)
                        resume=$(ls -t ${expdir}/results/central_iter_chunk_${FL_name}/model* | head -1)
                        echo "epstop" ${epstop}
                        ${cuda_cmd} --gpu ${ngpu} ${expdir}/train_chunk.log \
                            asr_train_fl.py \
                            --config ${train_config} \
                            --preprocess-conf ${preprocess_config} \
                            --ngpu ${ngpu} \
                            --backend ${backend} \
                            --outdir ${expdir}/results/devices_iter_chunk_${FL_name}/${set_name}  \
                            --debugmode ${debugmode} \
                            --dict ${dict} \
                            --debugdir ${expdir} \
                            --minibatches ${N} \
                            --verbose ${verbose} \
                            --resume ${newest} \
                            --epochs ${iter} \
                            --pretrain-model ${resume} \
                            --num-save-attention 0 \
                            --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
                            --epstop ${epstop} \
                            --iterval ${itval} \
                            --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
                            #--preprocess-conf ${preprocess_config} \
                        newest=$(ls -t ${expdir}/results/devices_iter_chunk_${FL_name}/${set_name}/snapshot* | head -1)
                        ln -rsf ${newest} ${expdir}/results/2average_iter_chunk_${FL_name}/snapshot.${set_name}
                        echo ${set_name} >> ${expdir}/results/progress_chunk_${FL_name}
                        cat ${expdir}/train_chunk.log >> ${expdir}/full_train_iter_chunk_${FL_name}.log
                        x=$(ls -1q ${expdir}/results/devices_iter_chunk_${FL_name}/${set_name}/snapshot* | wc -l)
                        if [ ${x} -gt 1 ]; then
                            ls -t1 ${expdir}/results/devices_iter_chunk_${FL_name}/${set_name}/snapshot* | tail -n +2 | xargs rm -r
                        fi
                        #x=$(ls -1q ${expdir}/results/devices_iter_chunk_${FL_name}/${set_name}/snapshot* | wc -l)
                        epstop=0
                    fi
                done
                num=$(wc -l ${expdir}/results/progress_chunk_${FL_name} | cut -d' ' -f1)
                if [ ${FL_meth} -le 1 ] && [ ${FL_meth} -ge 1 ]; then
                    if [ ${its} -eq ${max_its} ]; then
                        outname=${expdir}/results/central_iter_chunk_${FL_name}/model.epoch.avg.${iter}
                    else
                        outname=${expdir}/results/central_iter_chunk_${FL_name}/model.iter.avg.${its}
                    fi

                    local/FL_functions/average_speaker.py --backend ${backend} \
                        --snapshots ${expdir}/results/2average_iter_chunk_${FL_name}/snapshot.* \
                        --out ${outname} \
                        --snap 1 \
                        --num ${num}
                    resume=$(ls -t ${expdir}/results/central_iter_chunk_${FL_name}/model* | head -1)
                fi

                if [ ${FL_meth} -le 2 ] && [ ${FL_meth} -ge 2 ]; then
                    if [ ${its} -eq ${max_its} ]; then
                        outname=${expdir}/results/central_iter_chunk_${FL_name}/model.epoch.avg.${iter}
                    else
                        outname=${expdir}/results/central_iter_chunk_${FL_name}/model.iter.avg.${its}
                    fi
                    local/FL_functions/fed_avg.py --snapshots ${expdir}/results/2average_iter_chunk_${FL_name}/snapshot.* \
                            --out ${outname} \
                            --num ${num} \
                            --backend ${backend} \
                            --jsonlist ${expdir}/train.json \
                            --partlist ${expdir}/results/progress_chunk_${FL_name} \
                            --snap 1
                    resume=$(ls -t ${expdir}/results/central_iter_chunk_${FL_name}/model* | head -1)
                fi
                rm ${expdir}/results/progress_chunk_${FL_name}
                rm ${expdir}/results/2average_iter_chunk_${FL_name}/snapshot*
            done
        min_its=1
        ln -rsf ${resume} ${expdir}/results/central/model.chunk_ep${iter}.avg
        rm ${expdir}/results/central_iter_chunk_${FL_name}/model.iter*      
        done
    fi
fi
