#!/bin/bash

# Copyright 2021 Ruhr University Bochum (Wentao Yu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


# general configuration
debug=false				# set the debug mode. If debug is true the debugger settings could be configured: how
					# many utts and how many speakers are used for debug
num=0					# if debug is true, only extract $num utts for debug
numspk=0				# if debug is true, only extract $numspk speakers

backend=pytorch
stage=-1      				# start from -1 if you need to start from data download
stop_stage=100
ngpu=1         				# number of gpus ("0" uses cpu, otherwise use gpu)
nj=32					
debugmode=1
dumpdir=dump   				# directory to dump full features
fbankdir=fbank				# folder to save audio features
N=0            				# number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      				# verbose option
resume=        				# Resume the training from snapshot
train_lm=false 				# true: Train own language model, false: use pretrained librispeech LM model

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml 	# TODO ACHTUNG HIER BEI TRAINING ANPASSEN
lm_config=conf/lm.yaml

# rnnlm related
lm_resume= 				# specify a snapshot file to resume LM training
lmtag=     				# tag for managing LMs

# decoding parameter
recog_model=model.acc.best  		# set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best 		# set a language model to be used for decoding

# model average realted (only for transformer)
n_average=10                  		# the number of ASR models to be averaged
use_valbest_average=false     		# if true, the validation `n_average`-best ASR models will be averaged.
                             		# if false, the last `n_average` ASR models will be averaged.
lm_n_average=0               		# the number of languge models to be averaged
use_lm_valbest_average=false 		# if true, the validation `lm_n_average`-best language models will be averaged.
                             		# if false, the last `lm_n_average` language models will be averaged.

datadir=./Source			# The dir where save the LibriSpeech dataset

# base url for downloads.
data_url=www.openslr.org/resources/12	# LibriSpeech dataset download URL

# the function to download the language model
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.
. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    mkdir -p $datadir
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        local/download_and_untar.sh ${datadir} ${data_url} ${part} || exit 1;
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    # this stage creat the Kaldi files, like text wav.scp and utt2spk etc.
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/metadata/${part//-/_} || exit 1;
    done
    utils/combine_data.sh --extra_files utt2num_frames data/metadata/train_960_org \
							data/metadata/train_clean_100 \
							data/metadata/train_clean_360 \
							data/metadata/train_other_500 || exit 1;
    utils/combine_data.sh --extra_files utt2num_frames data/metadata/test_org \
							data/metadata/test_clean \
							data/metadata/test_other || exit 1;
    utils/combine_data.sh --extra_files utt2num_frames data/metadata/dev_org \
							data/metadata/dev_clean \
							data/metadata/dev_other || exit 1;
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Creat the federated learning dataset. 
    echo "stage 1: Split Federated Learning Dataset"
    python3 local/egs_preprocessing_make_librispeech_fl_set.py data/metadata data/splitdata 0 || exit 1;
    cp -R data/splitdata/data/* data/
fi

train_set=train_960_org
dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"

    mkdir -p data/lang_char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " data/metadata/${train_set}/text > data/lang_char/input.txt
    spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    if [ "$train_lm" = false ] ; then		# if train_lm is false, download the language model.
	gdrive_download '1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6&export=download' 'pretrained.tar.gz'	
	tar -xf pretrained.tar.gz
	mv exp/irielm.ep11.last5.avg exp/pretrainedlm
  	rm -rf exp/train*
        echo "Replace dict with librispeech pretrained dict"
        python3 local/remake_dict.py exp/pretrainedlm/model.json ${dict}  # new downloaded language model, $dict should be also changed
    fi
fi


if [ "$train_lm" = false ] ; then
    lmexpname=pretrainedlm
    lmexpdir=exp/${lmexpname}
else
    if [ -z ${lmtag} ]; then
        lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
    lmexpdir=exp/${lmexpname}
    mkdir -p ${lmexpdir}
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    if [ "$train_lm" = false ] ; then
        echo "stage 3: Use pretrained LM"
    else
        echo "stage 3: LM Preparation"
        lmdatadir=data/local/lm_train_${bpemode}${nbpe}
        # use external data
        if [ ! -e data/local/lm_train/librispeech-lm-norm.txt.gz ]; then
            wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/lm_train/
        fi
        if [ ! -e ${lmdatadir} ]; then
            mkdir -p ${lmdatadir}
            cut -f 2- -d" " data/${train_set}/text | gzip -c > data/local/lm_train/${train_set}_text.gz
            # combine external text and transcriptions and shuffle them with seed 777
            zcat data/local/lm_train/librispeech-lm-norm.txt.gz data/local/lm_train/${train_set}_text.gz |\
                spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
            cut -f 2- -d" " data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece \
                                                            > ${lmdatadir}/valid.txt
        fi
        ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
            lm_train.py \
            --config ${lm_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --verbose 1 \
            --outdir ${lmexpdir} \
            --tensorboard-dir tensorboard/${lmexpname} \
            --train-label ${lmdatadir}/train.txt \
            --valid-label ${lmdatadir}/valid.txt \
            --resume ${lm_resume} \
            --dict ${dict} \
            --dump-hdf5-path ${lmdatadir}
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Make initial and federated learning dataset features and dump files"
    local/preprocess_central.sh --nj "$nj" --do_delta "$do_delta" --bpemode "$bpemode" --nbpe "$nbpe"\
		   	   	$dumpdir $fbankdir $debug $num $dict $bpemodel || exit 1
    local/preprocess_FL.sh --nj "$nj" --do_delta "$do_delta"  --bpemode "$bpemode"  --nbpe "$nbpe"\
		   	   	$dumpdir $fbankdir $debug $numspk $dict $bpemodel || exit 1
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Creat symbolic link to ESPnet"
    # we have to mount our code to expnet folder
    rm -f ${MAIN_ROOT}/espnet/asr/asr_utils_fl.py
    ln -rsf  local/espnet/asr/asr_utils_fl.py ${MAIN_ROOT}/espnet/asr/asr_utils_fl.py
    rm -f ${MAIN_ROOT}/espnet/asr/pytorch_backend/asrfl.py
    ln -rsf  local/espnet/asr/pytorch_backend/asrfl.py ${MAIN_ROOT}/espnet/asr/pytorch_backend/asrfl.py
    rm -f ${MAIN_ROOT}/espnet/bin/asr_train_fl.py
    ln -rsf  local/espnet/bin/asr_train_fl.py ${MAIN_ROOT}/espnet/bin/asr_train_fl.py
    rm -f ${MAIN_ROOT}/espnet/nets/pytorch_backend/transformer/optimizer_sgd.py
    ln -rsf  local/espnet/nets/pytorch_backend/transformer/optimizer_sgd.py ${MAIN_ROOT}/espnet/nets/pytorch_backend/transformer/optimizer_sgd.py
    rm -f ${MAIN_ROOT}/espnet/utils/training/train_utils_fl.py
    ln -rsf  local/espnet/utils/training/train_utils_fl.py ${MAIN_ROOT}/espnet/utils/training/train_utils_fl.py
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Train whole and initial model"
    ### train Ref (entir Dataset) and Initial (pretrain model)
    local/central.sh  --backend "$backend" --do_delta "$do_delta" \
		      --ngpu "$ngpu" --debugmode "$debugmode" --N "$N" \
		      --verbose "$verbose" --bpemode "$bpemode" --nbpe "$nbpe"\
                      $dumpdir $dict $lmexpdir $lang_model || exit 1

fi

initialmodel=exp/pretrain/results/model.last10.avg.best		# this model is used to initial the FL server model
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: The update is performed with the selected clients’ models after every epoch"
    local/E_Level_Train.sh  --backend "$backend" --do_delta "$do_delta" \
		      --ngpu "$ngpu" --debugmode "$debugmode" --N "$N" --verbose "$verbose" \
	   	      --bpemode "$bpemode" --nbpe "$nbpe"\
		      --resume "$initialmodel" \
                      $dumpdir $dict || exit 1
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: The update is performed with the selected clients’ final models, after convergence."
    local/C_Level_Train.sh  --backend "$backend" --do_delta "$do_delta" \
		      --ngpu "$ngpu" --debugmode "$debugmode" --N "$N" --verbose "$verbose" \
	   	      --bpemode "$bpemode" --nbpe "$nbpe"\
		      --resume "$initialmodel" \
                      $dumpdir $dict || exit 1
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9: The update is always performed after a mini-batches."
    local/I_Level_Train.sh  --backend "$backend" --do_delta "$do_delta" \
		      --ngpu "$ngpu" --debugmode "$debugmode" --N "$N" --verbose "$verbose" \
	   	      --bpemode "$bpemode" --nbpe "$nbpe" --n_iter 10 \
		      --resume "$initialmodel" \
                      $dumpdir $dict || exit 1
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10: Decode the model on dev set"
    local/decode_dev.sh  --backend "$backend" --do_delta "$do_delta" \
		      --ngpu "$ngpu" --debugmode "$debugmode" \
		      --bpemode "$bpemode" --nbpe "$nbpe" --nj "$nj"\
                      $dumpdir $dict $bpemodel || exit 1
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "stage 11: Decode the model on test set, please specify which model is used \
          to decode according the decoding results on dev set"
    decode_model_type=subset2_2	# Choosing the model with the best decoding result on dev set. 
     				# All the model are saved in 
				# exp/FL/results/central/model.$decode_model_type.avg
    local/decode_test_with_transfer.sh  --backend "$backend" --do_delta "$do_delta" \
		      --ngpu "$ngpu" --debugmode "$debugmode" \
		      --bpemode "$bpemode" --nbpe "$nbpe" --nj "$nj" \
                      $dumpdir $dict $bpemodel $decode_model_type || exit 1
fi

# remove all symbolic links
rm -f ${MAIN_ROOT}/espnet/asr/asr_utils_fl.py || exit 1
rm -f ${MAIN_ROOT}/espnet/asr/pytorch_backend/asrfl.py || exit 1
rm -f ${MAIN_ROOT}/espnet/bin/asr_train_fl.py || exit 1
rm -f ${MAIN_ROOT}/espnet/nets/pytorch_backend/transformer/optimizer_sgd.py || exit 1
rm -f ${MAIN_ROOT}/espnet/utils/training/train_utils_fl.py || exit 1
echo "removed symbolic links"
echo "finished"
exit 0