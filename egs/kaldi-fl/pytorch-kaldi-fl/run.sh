#!/usr/bin/env bash

##########################################################
# Ruhr-University Bochum
# Cognitive Signal Processing Group
# April 2021
##########################################################

# We used the kaldi-librispeech example as a starting point for the develpoment of this code.
# This script does not support the CLSP grid.

# Set this to somewhere where you want to put your data, or where
# someone else has already put it. You'll want to change this.
data=/home/<user>/data/kaldi/librispeech

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11
mfccdir=mfcc

# train "whole" first, then pretrain, since we need some files in the second training
set=whole # pretrain or whole ?
stage=1

# adapt to your machine
nj=24

. ./cmd.sh
. ./path.sh
. parse_options.sh

# you might not want to do this for interactive shells.
set -e

echo "##################################"
echo "#            Setup               #"
echo "##################################"

if [ $stage -le 1 ]; then
  echo "Preparing LM..."
  # download the LM resources
  local/download_lm.sh $lm_url data/local/lm
  
  local/prepare_dict.sh --stage 3 --nj $nj --cmd "$train_cmd" \
   data/local/lm data/local/lm data/local/dict_nosp

  utils/prepare_lang.sh data/local/dict_nosp \
   "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

  local/format_lms.sh --src-dir data/lang_nosp data/local/lm
fi

if [ $stage -le 2 ]; then
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
    data/lang_nosp data/lang_nosp_test_tglarge
  utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
    data/lang_nosp data/lang_nosp_test_fglarge
fi

if [ $stage -le 3 ]; then
  echo "Download and prepare dataset..."
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    echo "Downloading data (${part})..."
    local/download_and_untar.sh $data $data_url ${part}
    echo "Preparing data (${part})..."
    # use underscore-separated names in data directories.
    local/data_prep.sh $data/LibriSpeech/${part} data/$(echo ${part} | sed s/-/_/g)
  done
fi

if [ $stage -le 4 ]; then
  echo "Combining datasets..."
  utils/combine_data.sh data/train_clean_460 data/train_clean_100 data/train_clean_360
  utils/combine_data.sh data/train_960 data/train_clean_460 data/train_other_500
  utils/combine_data.sh data/test data/test_clean data/test_other
  utils/combine_data.sh data/dev data/dev_clean data/dev_other
fi

if [ $stage -le 5 ]; then
  echo "Creating complete, pretrain and FL datasets..."

  # create symlinks to match wentaos naming conventions
  # the symlinks will be removed later to keep everything tidy
  for part in test dev train_960; do
    rm -rf data/${part}_org
    ln -s ${part} data/${part}_org
  done

  # run wentao's script :-)
  python ../preprocessing/make_librispeech_fl_set_3.py data data/fl_set
    
  mv data/fl_set/data/pretrain data/pretrain
  mv data/fl_set/data/whole data/whole
  mv data/fl_set/data/FL data/FL
  
  # delete symlinks
  for part in dev_org test_org train_960_org; do
    rm -fr data/${part}
  done
  
  # more cleanup
  for part in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500 train_clean_460; do
    rm -fr data/${part}
  done
  
  # delete obsolete sets and symlink correct sets
  rm -r data/whole/test
  rm -r data/whole/dev
  cp -r data/pretrain/test data/whole/test
  cp -r data/pretrain/dev data/whole/dev
fi


echo "##################################"
echo "#    Feature Extraction          #"
echo "##################################"

if [ $stage -le 6 ]; then
  for part in `ls data/${set}`; do
    echo "Extracting mfcc and cmvn stats of pretrain set (${part})..."
    #sorting files
    utils/fix_data_dir.sh data/${set}/${part}    
    utils/utt2spk_to_spk2utt.pl data/${set}/${part}/utt2spk > data/${set}/${part}/spk2utt
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/${set}/${part} exp/make_mfcc/${set}/${part} $mfccdir/${set}/${part}
    steps/compute_cmvn_stats.sh data/${set}/${part} exp/make_mfcc/${set}/${part} $mfccdir/${set}/${part}
  done
fi


## GMM Training on pretrain-set

if [ $stage -le 7 ]; then
  echo "Training GMM monophone model..."
  # train monophone 
  steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/${set}/train data/lang_nosp exp/${set}/mono
  # align
  steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/${set}/train data/lang_nosp exp/${set}/mono exp/${set}/mono_ali
fi


if [ $stage -le 8 ]; then
  echo "Training GMM triphone model..."
  steps/train_deltas.sh --cmd "$train_cmd" 2000 10000 data/${set}/train data/lang_nosp exp/${set}/mono_ali exp/${set}/tri1
  steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" data/${set}/train data/lang_nosp exp/${set}/tri1 exp/${set}/tri1_ali
fi

if [ $stage -le 9 ]; then
  echo "Training GMM triphone LDA+MLLT model..."
  steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" \
     2000 10000 data/${set}/train data/lang_nosp exp/${set}/tri1_ali exp/${set}/tri2
  steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/${set}/train data/lang_nosp exp/${set}/tri2 exp/${set}/tri2_ali
fi

if [ $stage -le 10 ]; then
  echo "Training GMM triphone LDA+MLLT+SAT model..."
  steps/train_sat.sh --cmd "$train_cmd" \
     2000 10000 data/${set}/train data/lang_nosp exp/${set}/tri2_ali exp/${set}/tri3
  steps/align_fmllr.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/${set}/train data/lang_nosp exp/${set}/tri3 exp/${set}/tri3_ali
fi

if [ $stage -le 11 ]; then
  echo "Adapting language model..."
  # Now we compute the pronunciation and silence probabilities from training data,
  # and re-create the lang directory.
  steps/get_prons.sh --cmd "$train_cmd" \
                     data/${set}/train data/lang_nosp exp/${set}/tri3
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
                                  data/local/dict_nosp \
                                  exp/${set}/tri3/pron_counts_nowb.txt exp/${set}/tri3/sil_counts_nowb.txt \
                                  exp/${set}/tri3/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict \
                        "<UNK>" data/local/lang_tmp data/lang
  local/format_lms.sh --src-dir data/lang data/local/lm

  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge
fi

echo "##################################"
echo "#     Decode pretrain models     #"
echo "##################################"
if [ $stage -le 12 ]; then

    for model in mono tri1 tri2 tri3; do
      utils/mkgraph.sh data/lang_test_tgsmall exp/${set}/${model} exp/${set}/${model}/graph

      for part in `ls data/${set} | grep 'test\|dev'`; do
        if [ "${model}" -eq "tri3" ]; then
          steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
            exp/${set}/${model}/graph data/${set}/${part} exp/${set}/${model}/decode_${part}
        else
          steps/decode.sh --nj $nj --cmd "$decode_cmd" \
            exp/${set}/${model}/graph data/${set}/${part} exp/${set}/${model}/decode_${part}
        fi
      done

    done

fi


echo "##################################"
echo "#      DNN training              #"
echo "##################################"

if [ $stage -le 13 ]; then
    echo "Alining ${set} train/dev/test..."
    for part in test dev train; do
        steps/align_fmllr.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
            data/${set}/${part} data/lang exp/${set}/tri3 exp/alignments/${set}/${part}
    done
  
fi

if [ $stage -le 14 ]; then
    echo "Training initial pytorch-kaldi model"
    cd ../pytorch-kaldi
    python run_exp.py cfg/fl/${set}_mlp_mfcc.cfg
    cd ../pytorch-kaldi-fl

fi

if [ "${set}" == "whole" ]; then

    if [ $stage -le 15 ]; then
        echo "generate training counts for normalization..."
        alidir=../pytorch-kaldi-fl/exp/whole/tri3_ali
        num_pdf=$(hmm-info $alidir/final.mdl | awk '/pdfs/{print $4}')
        labels_tr_pdf="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"
        analyze-counts --verbose=1 --binary=false --counts-dim=$num_pdf "$labels_tr_pdf" ${alidir}/ali_train_pdf.counts
    fi

    if [ $stage -le 16 ]; then
        echo "decoding whole test with whole model..."
        cd ../pytorch-kaldi
        mkdir -p exp/fl/whole_mlp_mfcc_prod/exp_files/
        cp exp/fl/whole_mlp_mfcc/exp_files/final_architecture1.pkl exp/fl/whole_mlp_mfcc_prod/exp_files/final_architecture1.pkl
        cp ../pytorch-kaldi-fl/exp/whole/tri3/final.mdl ../pytorch-kaldi-fl/exp/whole/tri3/graph/final.mdl
        python run_exp.py cfg/fl/whole_mlp_mfcc_prod.cfg
    fi

    if [ $stage -le 17 ]; then
        echo "scoring whole test with whole model..."
        cd ../pytorch-kaldi-fl
        local/score.sh --cmd run.pl data/pretrain/test exp/whole/tri3/graph ../pytorch-kaldi/exp/fl/whole_mlp_mfcc_prod/decode_whole_test_out_dnn1
    fi
    if [ $stage -le 18 ]; then
        echo "decoding pretrain transfer test with whole model..."
        cd ../pytorch-kaldi
        python run_exp.py cfg/fl/whole_mlp_mfcc_prod.cfg --data_use,forward_with=whole_transfer_test
        python run_exp.py cfg/fl/whole_mlp_mfcc_prod.cfg --data_use,forward_with=whole_transfer_fl_test

    fi

    if [ $stage -le 19 ]; then
        echo "scoring pretrain test with whole model..."
        cd ../pytorch-kaldi-fl
        local/score.sh --cmd run.pl data/pretrain/transfer_pre_test_org exp/whole/tri3/graph ../pytorch-kaldi/exp/fl/whole_mlp_mfcc_prod/decode_whole_transfer_test_out_dnn1
        local/score.sh --cmd run.pl data/pretrain/transfer_fl_test_org exp/whole/tri3/graph ../pytorch-kaldi/exp/fl/whole_mlp_mfcc_prod/decode_whole_transfer_fl_test_out_dnn1
    fi
    exit
fi

if [ $stage -le 15 ]; then
  echo "Generating FL features and alignments..."
  for spk in `ls data/FL | sort -n`; do
    for part in dev train test; do
      echo "Generate features for speaker ${spk} ${part} set..."
      utils/fix_data_dir.sh data/FL/${spk}/${part}  
      utils/utt2spk_to_spk2utt.pl data/FL/${spk}/${part}/utt2spk > data/FL/${spk}/${part}/spk2utt
      
      steps/make_mfcc.sh --cmd "$train_cmd" data/FL/${spk}/${part} exp/make_mfcc/FL/${spk}/${part} $mfccdir/FL/${spk}/${part}
      steps/compute_cmvn_stats.sh data/FL/${spk}/${part} exp/make_mfcc/FL/${spk}/${part} $mfccdir/FL/${spk}/${part}

      steps/align_fmllr.sh --boost-silence 1.25 --nj 1 --cmd "$train_cmd" data/FL/${spk}/${part} data/lang exp/pretrain/tri3 exp/alignments/FL/${spk}/${part}
    done
  done
fi


if [ $stage -le 16 ]; then
  echo ""
fi

if [ $stage -le 17 ]; then
    echo "Generate configs for all fl-speakers"
    for ck in 2 4 8 16; do
        for spk in `ls ../pytorch-kaldi-fl/data/FL`; do
            mkdir -p ../pytorch-kaldi/cfg/fl/${ck}
            local/generate_pytorch_kaldi_cfg.sh ${spk} exp/fl/pretrain_mlp_mfcc/exp_files/final_architecture1.pkl ${ck} > ../pytorch-kaldi/cfg/fl/${ck}/fl_${spk}_mlp_mfcc.cfg
        done
    done
fi

if [ $stage -le 18 ]; then
    echo "Training on the FL set"
    cd ../pytorch-kaldi
    for ck in 4 16 8 2; do
        
        while ! [ -f exp/fl/ck${ck}/clients/fl_${spk}_mlp_mfcc/exp_files/final_architecture1.pkl ]; do
            echo "exp/fl/ck${ck}/clients/fl_${spk}_mlp_mfcc/exp_files/final_architecture1.pkl"
            for spk in `ls ../pytorch-kaldi-fl/data/FL | sort -n` ; do #
                if [ "${spk}" -ne "3559" ]; then
                   mkdir -p exp/fl/ck${ck}/clients
                   # if this crashes on the first try (depends on the system) 
                   # you need to generate this file by hand and restart this stage:
                   # echo "0" > exp/fl/ck${ck}/clients/last_spk
                   touch exp/fl/ck${ck}/clients/last_spk
                   lspk=$(cat exp/fl/ck${ck}/clients/last_spk)
                   if [ "${spk}" -gt "$lspk" ]; then
                   #mkdir -p exp/fl/ck${ck}/clients/fl_${spk}_mlp_mfcc
                   until python run_exp_fl.py cfg/fl/${ck}/fl_${spk}_mlp_mfcc.cfg
                   do
                      echo "some error occured in spk ${spk}... retrying " >> main.log
                      sleep 1
                   done
                    echo "${spk}" > exp/fl/ck${ck}/clients/last_spk
                   fi
                fi
            done
            
            echo "0" > exp/fl/ck${ck}/clients/last_spk
            #wait
            # average here
            cd ../pytorch-kaldi-fl
            python local/weighted_avg.py ../pytorch-kaldi/exp/fl/ck${ck} data/FL
            cd ../pytorch-kaldi
        done
    done

fi

if [ $stage -le 19 ]; then
    echo "generate training counts for normalization..."
    alidir=../pytorch-kaldi-fl/exp/pretrain/tri3_ali
    num_pdf=$(hmm-info $alidir/final.mdl | awk '/pdfs/{print $4}')
    labels_tr_pdf="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"
    analyze-counts --verbose=1 --binary=false --counts-dim=$num_pdf "$labels_tr_pdf" ${alidir}/ali_train_pdf.counts
fi

if [ $stage -le 20 ]; then
    echo "decoding pretrain test with pretrain model..."
    cd ../pytorch-kaldi
    mkdir -p exp/fl/pretrain_mlp_mfcc_prod/exp_files/
    cp exp/fl/pretrain_mlp_mfcc/exp_files/final_architecture1.pkl exp/fl/pretrain_mlp_mfcc_prod/exp_files/final_architecture1.pkl
    cp ../pytorch-kaldi-fl/exp/pretrain/tri3/final.mdl ../pytorch-kaldi-fl/exp/pretrain/tri3/graph/final.mdl
    python run_exp.py cfg/fl/pretrain_mlp_mfcc_prod.cfg
fi

if [ $stage -le 21 ]; then
    echo "scoring pretrain test with pretrain model..."
    cd ../pytorch-kaldi-fl
    local/score.sh --cmd run.pl data/pretrain/test exp/pretrain/tri3/graph ../pytorch-kaldi/exp/fl/pretrain_mlp_mfcc_prod/decode_pretrain_test_out_dnn1
fi

if [ $stage -le 22 ]; then
    echo "decoding pretrain transfer test with pretrain model..."
    cd ../pytorch-kaldi
    python run_exp.py cfg/fl/pretrain_mlp_mfcc_prod.cfg --data_use,forward_with=pretrain_transfer_test
    python run_exp.py cfg/fl/pretrain_mlp_mfcc_prod.cfg --data_use,forward_with=pretrain_transfer_fl_test

fi

if [ $stage -le 23 ]; then
    echo "scoring pretrain test with pretrain model..."
    cd ../pytorch-kaldi-fl
    local/score.sh --cmd run.pl data/pretrain/transfer_pre_test_org exp/pretrain/tri3/graph ../pytorch-kaldi/exp/fl/pretrain_mlp_mfcc_prod/decode_pretrain_transfer_test_out_dnn1
    local/score.sh --cmd run.pl data/pretrain/transfer_fl_test_org exp/pretrain/tri3/graph ../pytorch-kaldi/exp/fl/pretrain_mlp_mfcc_prod/decode_pretrain_transfer_fl_test_out_dnn1
fi

if [ $stage -le 24 ]; then
    echo "decoding with fl models..."
    
    for ck in 2 4 8 16; do
        cd ../pytorch-kaldi
        mkdir -p exp/fl/fl_${ck}_mlp_mfcc_prod/exp_files/
        cp exp/fl/ck${ck}/clients/fl_16_mlp_mfcc/exp_files/final_architecture1.pkl exp/fl/fl_${ck}_mlp_mfcc_prod/exp_files/final_architecture1.pkl
        cp ../pytorch-kaldi-fl/exp/pretrain/tri3/final.mdl ../pytorch-kaldi-fl/exp/pretrain/tri3/graph/final.mdl
        python run_exp.py cfg/fl/fl_${ck}_mlp_mfcc_prod.cfg
        python run_exp.py cfg/fl/fl_${ck}_mlp_mfcc_prod.cfg --data_use,forward_with=pretrain_transfer_test
        python run_exp.py cfg/fl/fl_${ck}_mlp_mfcc_prod.cfg --data_use,forward_with=pretrain_transfer_fl_test
        
    done
fi

if [ $stage -le 25 ]; then
    echo "scoring fl models..."
    
    for ck in 2 4 8 16; do

        cd ../pytorch-kaldi-fl
        local/score.sh --cmd run.pl data/pretrain/test exp/pretrain/tri3/graph ../pytorch-kaldi/exp/fl/fl_${ck}_mlp_mfcc_prod/decode_pretrain_test_out_dnn1
        local/score.sh --cmd run.pl data/pretrain/transfer_pre_test_org exp/pretrain/tri3/graph ../pytorch-kaldi/exp/fl/fl_${ck}_mlp_mfcc_prod/decode_pretrain_transfer_test_out_dnn1
        local/score.sh --cmd run.pl data/pretrain/transfer_fl_test_org exp/pretrain/tri3/graph ../pytorch-kaldi/exp/fl/fl_${ck}_mlp_mfcc_prod/decode_pretrain_transfer_fl_test_out_dnn1
    done
fi

if [ $stage -le 26 ]; then
    echo "(whole) generate training counts for normalization..."
    alidir=../pytorch-kaldi-whole/exp/whole/tri3_ali
    num_pdf=$(hmm-info $alidir/final.mdl | awk '/pdfs/{print $4}')
    labels_tr_pdf="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"
    analyze-counts --verbose=1 --binary=false --counts-dim=$num_pdf "$labels_tr_pdf" ${alidir}/ali_train_pdf.counts
fi

if [ $stage -le 27 ]; then
    echo "decoding pretrain test with whole model..."
    cd ../pytorch-kaldi
    mkdir -p exp/whole/mlp_mfcc_prod/exp_files/
    cp exp/whole/mlp_mfcc/exp_files/final_architecture1.pkl exp/whole/mlp_mfcc_prod/exp_files/final_architecture1.pkl
    
    cp ../pytorch-kaldi-whole/exp/whole/tri3/final.mdl ../pytorch-kaldi-whole/exp/whole/tri3/graph/final.mdl
    python run_exp.py cfg/whole/mlp_mfcc_prod.cfg
fi

if [ $stage -le 28 ]; then
    echo "scoring pretrain test with whole model..."
    cd ../pytorch-kaldi-fl
    local/score.sh --cmd run.pl data/pretrain/test exp/pretrain/tri3/graph ../pytorch-kaldi/exp/whole/mlp_mfcc_prod/decode_pretrain_test_out_dnn1
fi

if [ $stage -le 29 ]; then
    echo "decoding pretrain transfer test with whole model..."
    cd ../pytorch-kaldi
    python run_exp.py cfg/whole/mlp_mfcc_prod.cfg --data_use,forward_with=pretrain_transfer_test
fi

if [ $stage -le 30 ]; then
    echo "scoring pretrain test with whole model..."
    cd ../pytorch-kaldi-fl
    local/score.sh --cmd run.pl data/pretrain/transfer_test exp/pretrain/tri3/graph ../pytorch-kaldi/exp/whole/mlp_mfcc_prod/decode_pretrain_transfer_test_out_dnn1
fi

