#!/usr/bin/env bash
#
# Based mostly on the Switchboard recipe. The training database is TED-LIUM,
# it consists of TED talks with cleaned automatic transcripts:
#
# https://lium.univ-lemans.fr/ted-lium3/
# http://www.openslr.org/resources (Mirror).
#
# The data is distributed under 'Creative Commons BY-NC-ND 3.0' license,
# which allow free non-commercial use, while only a citation is required.
#
# Copyright  2014  Nickolay V. Shmyrev
#            2014  Brno University of Technology (Author: Karel Vesely)
#            2016  Vincent Nguyen
#            2016  Johns Hopkins University (Author: Daniel Povey)
#            2018  François Hernandez
#
# Apache 2.0
#

. ./cmd.sh
. ./path.sh


set -e -o pipefail -u

nj=8
decode_nj=4    # note: should not be >38 which is the number of speakers in the dev set
               # after applying --seconds-per-spk-max 180.  We decode with 4 threads, so
               # this will be too many jobs if you're using run.pl.
stage=16
train_rnnlm=false
train_lm=false

. utils/parse_options.sh # accept options

# Data preparation
if [ $stage -le 0 ]; then
  echo "==== stage 00: start ===="
  local/download_data.sh
  echo "==== stage 00: end ===="
fi

if [ $stage -le 1 ]; then
  echo "==== stage 01: start ===="
  local/prepare_data.sh
  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  # [we chose 3 minutes because that gives us 38 speakers for the dev data, which is
  #  more than our normal 30 jobs.]
  for dset in dev test train; do
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
  done
  echo "==== stage 01: done ===="
fi


if [ $stage -le 2 ]; then
  echo "==== stage 02: start ===="
  local/prepare_dict.sh
  echo "==== stage 02: end ===="
fi


if [ $stage -le 3 ]; then
  echo "==== stage 03: start ===="
  utils/prepare_lang.sh data/local/dict_nosp \
    "<unk>" data/local/lang_nosp data/lang_nosp
  echo "==== stage 03: done ===="
fi

if [ $stage -le 4 ]; then
  echo "==== stage 04: start ===="
  # later on we'll change this script so you have the option to
  # download the pre-built LMs from openslr.org instead of building them
  # locally.
  if $train_lm; then
    local/ted_train_lm.sh
  else
    local/ted_download_lm.sh
  fi
  echo "==== stage 04: done ===="
fi

if [ $stage -le 5 ]; then
  echo "==== stage 05: start ===="
  local/format_lms.sh
  echo "==== stage 05: done ===="
fi

# Feature extraction
if [ $stage -le 6 ]; then
  echo "==== stage 06: start ===="
  for set in test dev train; do
    dir=data/$set
    steps/make_mfcc.sh --nj 30 --cmd "$train_cmd" $dir
    steps/compute_cmvn_stats.sh $dir
  done
  echo "==== stage 06: done ===="
fi

# Now we have 452 hours of training data.
# Well create a subset with 10k short segments to make flat-start training easier:
if [ $stage -le 7 ]; then
  echo "==== stage 07: start ===="
  utils/subset_data_dir.sh --shortest data/train 10000 data/train_10kshort
  utils/data/remove_dup_utts.sh 10 data/train_10kshort data/train_10kshort_nodup
  echo "==== stage 07: done ===="
fi

# Train
if [ $stage -le 8 ]; then
  echo "==== stage 08: start ===="
  steps/train_mono.sh --nj 20 --cmd "$train_cmd" \
    data/train_10kshort_nodup data/lang_nosp exp/mono
  echo "==== stage 08: done ===="
fi

if [ $stage -le 9 ]; then
  echo "==== stage 09: start ===="
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/mono exp/mono_ali
  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 30000 data/train data/lang_nosp exp/mono_ali exp/tri1
  echo "==== stage 09: done ===="
fi

if [ $stage -le 10 ]; then
  echo "==== stage 10: start ===="
  utils/mkgraph.sh data/lang_nosp exp/tri1 exp/tri1/graph_nosp

  # The slowest part about this decoding is the scoring, which we can't really
  # control as the bottleneck is the NIST tools.
#  for dset in dev test; do
#    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
#      exp/tri1/graph_nosp data/${dset} exp/tri1/decode_nosp_${dset}
#    steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
#       data/${dset} exp/tri1/decode_nosp_${dset} exp/tri1/decode_nosp_${dset}_rescore
#  done
  echo "==== stage 10: done ===="
fi

if [ $stage -le 11 ]; then
  echo "==== stage 11: start ===="
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    4000 50000 data/train data/lang_nosp exp/tri1_ali exp/tri2
  echo "==== stage 11: done ===="
fi

if [ $stage -le 12 ]; then
  echo "==== stage 12: start ===="
  utils/mkgraph.sh data/lang_nosp exp/tri2 exp/tri2/graph_nosp
#  for dset in dev test; do
#    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
#      exp/tri2/graph_nosp data/${dset} exp/tri2/decode_nosp_${dset}
#    steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
#       data/${dset} exp/tri2/decode_nosp_${dset} exp/tri2/decode_nosp_${dset}_rescore
#  done
  echo "==== stage 12: done ===="
fi

if [ $stage -le 13 ]; then
  echo "==== stage 13: start ===="
  steps/get_prons.sh --cmd "$train_cmd" data/train data/lang_nosp exp/tri2
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp exp/tri2/pron_counts_nowb.txt \
    exp/tri2/sil_counts_nowb.txt \
    exp/tri2/pron_bigram_counts_nowb.txt data/local/dict
  echo "==== stage 13: done ===="
fi

if [ $stage -le 14 ]; then
  echo "==== stage 14: start ===="
  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
  cp -rT data/lang data/lang_rescore
  cp data/lang_nosp/G.fst data/lang/
  cp data/lang_nosp_rescore/G.carpa data/lang_rescore/

  utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph

#  for dset in dev test; do
#    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
#      exp/tri2/graph data/${dset} exp/tri2/decode_${dset}
#    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
#       data/${dset} exp/tri2/decode_${dset} exp/tri2/decode_${dset}_rescore
#  done
  echo "==== stage 14: done ===="
fi

if [ $stage -le 15 ]; then
  echo "==== stage 15: start ===="
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 data/train data/lang exp/tri2_ali exp/tri3

  utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph

#  for dset in dev test; do
#    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
#      exp/tri3/graph data/${dset} exp/tri3/decode_${dset}
#    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
#       data/${dset} exp/tri3/decode_${dset} exp/tri3/decode_${dset}_rescore
#  done
  echo "==== stage 15: done ===="
fi

if [ $stage -le 16 ]; then
  echo "==== stage 16: start ===="
  # this does some data-cleaning.  It actually degrades the GMM-level results
  # slightly, but the cleaned data should be useful when we add the neural net and chain
  # systems.  If not we'll remove this stage.
  local/run_cleanup_segmentation.sh
  echo "==== stage 16: done ===="
fi

if [ $stage -le 17 ]; then
  echo "==== stage 17: start ===="
  # This will only work if you have GPUs on your system (and note that it requires
  # you to have the queue set up the right way... see kaldi-asr.org/doc/queue.html)
  local/chain/run_tdnn.sh
  echo "==== stage 17: done ===="
fi


exit


if [ $stage -le 18 ]; then
  echo "==== stage 18: start ===="
  # You can either train your own rnnlm or download a pre-trained one
  if $train_rnnlm; then
    local/rnnlm/tuning/run_lstm_tdnn_a.sh
    local/rnnlm/average_rnnlm.sh
  else
    local/ted_download_rnnlm.sh
  fi
  echo "==== stage 18: done ===="
fi

if [ $stage -le 19 ]; then
  echo "==== stage 19: start ===="
  # Here we rescore the lattices generated at stage 17
  rnnlm_dir=exp/rnnlm_lstm_tdnn_a_averaged
  lang_dir=data/lang_chain
  ngram_order=4

  for dset in dev test; do
    data_dir=data/${dset}_hires
    decoding_dir=exp/chain_cleaned/tdnnf_1a/decode_${dset}
    suffix=$(basename $rnnlm_dir)
    output_dir=${decoding_dir}_$suffix

    rnnlm/lmrescore_pruned.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.5 --max-ngram-order $ngram_order \
      $lang_dir $rnnlm_dir \
      $data_dir $decoding_dir \
      $output_dir
  done
  echo "==== stage 19: done ===="
fi


echo "$0: success."
exit 0
