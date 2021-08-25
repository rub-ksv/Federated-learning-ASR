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
setname=whole
train_set=${setname}/train
dev_set=${setname}/dev
test_set=${setname}/test
setname=pretrain
train_pre_set=${setname}/train
train_pre_dev=${setname}/dev
test_pre_set=${setname}/test
transf_pre_test=${setname}/transfer_pre_test_org
transf_pre_dev=${setname}/transfer_pre_dev_org
transf_fl_test=${setname}/transfer_fl_test_org
transf_fl_dev=${setname}/transfer_fl_dev_org
recog_set="${train_set} ${dev_set} ${test_set} ${train_pre_set} ${test_pre_set} ${train_pre_dev} ${transf_pre_test} ${transf_pre_dev} ${transf_fl_test} ${transf_fl_dev}"

if [ "$debug" = true ] ; then
    mkdir -p data/temp
    mkdir -p data/temp/whole
    mkdir -p data/temp/pretrain
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dev_dir=${dumpdir}/${dev_set}/delta${do_delta}; mkdir -p ${feat_dev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Generate features
    echo "stage 1: Feature Generation"
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${recog_set}; do
	if [ "$debug" = true ] ; then
	    cp -R data/${x} data/temp/${x} || exit 1
	    utils/subset_data_dir.sh data/temp/${x} $num data/${x} || exit 1
 	fi
        utils/fix_data_dir.sh data/${x}  || exit 1
	if [ "$x" = "$train_set" ] ; then
	    subfbankdir=$fbankdir/whole
	elif [ "$x" = "$dev_set" ] ; then
	    subfbankdir=$fbankdir/whole
	elif [ "$x" = "$test_set" ] ; then
	    subfbankdir=$fbankdir/whole
	else
	    subfbankdir=$fbankdir/pretrain
	fi
        # remove utt having more than 3000 frames
        # remove utt having more than 400 characters
        # Remove features with too long frames in training data
        max_len=3000
        remove_longshortdata.sh  --maxframes $max_len --maxchars 400 data/${x} data/temp/${x}_temp || exit 1
        rm -rf data/${x}
        mv data/temp/${x}_temp data/${x} || exit 1

        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${subfbankdir} || exit 1
        utils/fix_data_dir.sh data/${x} || exit 1
    done
    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark || exit 1

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir} || exit 1
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Creat JSON Files
    echo "stage 2: Dictionary and Json Data Preparation"
    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json || exit 1

    data2json.sh --feat ${feat_dev_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${dev_set} ${dict} > ${feat_dev_dir}/data_${bpemode}${nbpe}.json || exit 1


    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json || exit 1
    done
fi