#!/bin/bash

# Copyright 2021 Ruhr University Bochum (Wentao Yu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


nj=32
do_delta=false
bpemode=unigram
nbpe=5000

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

# general configuration
dumpdir=$1
fbankdir=$2
debug=$3
num=$4
dict=$5
bpemodel=$6



stage=1
stop_stage=100
level=-2
stop_level=100
preproc=1
dumpdir=$dumpdir/FL
fbankdir=$fbankdir/FL


if [ "$debug" = true ] ; then
    ## If debug is true, select $num speaker for training
    rm -rf data/temp/FL
    mv data/FL data/temp || exit 1
    mkdir data/FL
    SOURCE=data/temp/FL
    DESTINATION=data/FL/
    find "$SOURCE" -mindepth 1 -maxdepth 1 -type d|shuf -n $num|xargs -d'\n' -I{} cp -r "{}" "$DESTINATION" || exit 1
fi

if [ ${preproc} -le 1 ] && [ ${preproc} -ge 1 ]; then
    ## Generate features
    for dir in data/FL/*
    do
        set_name=${dir:8}
	subfbankdir=$fbankdir/$set_name
	mkdir -p $subfbankdir
        train_set=${set_name}/train
        train_dev=${set_name}/dev
        test_set=${set_name}/test
        recog_set="${train_dev} ${test_set}"
        if [ ${level} -le -1 ] && [ ${stop_level} -ge -1 ]; then
            #PROBABLY NOT NECESSARY, USE ONLY IF fix_data_dir.sh CRASHES FOR MISSING utt2spk
            echo "Fixing" ${set_name}
            ./utils/utt2spk_to_spk2utt.pl ${dir}/dev/utt2spk  > ${dir}/dev/spk2utt || exit 1
            ./utils/utt2spk_to_spk2utt.pl ${dir}/test/utt2spk  > ${dir}/test/spk2utt || exit 1
            ./utils/utt2spk_to_spk2utt.pl ${dir}/train/utt2spk > ${dir}/train/spk2utt || exit 1
            utils/fix_data_dir.sh ${dir}/train || exit 1
            utils/fix_data_dir.sh ${dir}/test || exit 1
            utils/fix_data_dir.sh ${dir}/dev || exit 1
        fi
    
        feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
        feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
        if [ ${level} -le 1 ] && [ ${stop_level} -ge 1 ]; then
            echo "Feature Extraction" ${set_name}
            # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
            for x in ${train_set} ${train_dev} ${recog_set}; do
                max_len=3000
        	remove_longshortdata.sh  --maxframes $max_len --maxchars 400 data/FL/${x} data/temp/FL/${x}_temp || exit 1
        	rm -rf data/FL/${x}
        	mv data/temp/FL/${x}_temp data/FL/${x} || exit 1
                steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 1 --write_utt2num_frames true \
                                          data/FL/${x} exp/make_fbank/FL/${x} ${subfbankdir} || exit 1
                utils/fix_data_dir.sh data/FL/${x} || exit 1
            done

            # compute global CMVN
            compute-cmvn-stats scp:data/FL/${train_set}/feats.scp data/FL/${train_set}/cmvn.ark || exit 1

            dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
                    data/FL/${train_set}/feats.scp data/FL/${train_set}/cmvn.ark exp/dump_feats/FL/train ${feat_tr_dir} || exit 1
            dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
                    data/FL/${train_dev}/feats.scp data/FL/${train_set}/cmvn.ark exp/dump_feats/FL/dev ${feat_dt_dir} || exit 1
            for rtask in ${recog_set}; do
                feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
                dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
                        data/FL/${rtask}/feats.scp data/FL/${train_set}/cmvn.ark exp/dump_feats/recog/FL/${rtask} \
                        ${feat_recog_dir} || exit 1
            done
        fi
        # using same dict for every model
        if [ ${level} -le 2 ] && [ ${stop_level} -ge 2 ]; then
            # Generate JSON files 
            echo "make json files"
            data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
                         data/FL/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json || exit 1
            data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
                         data/FL/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json || exit 1
            for rtask in ${recog_set}; do
                feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
                data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
                             data/FL/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json || exit 1
            done
        fi
    done
fi