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
decode_nj=38   # note: should not be >38 which is the number of speakers in the dev set
               # after applying --seconds-per-spk-max 180.  We decode with 4 threads, so
               # this will be too many jobs if you're using run.pl.
stage=101
train_rnnlm=true
train_lm=false

nnet3_affix=_cleaned_1d  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned
tdnn_affix=1d  #affix for TDNN directory, e.g. "a" or "b", in case we change the configuration.
dir=exp/chain${nnet3_affix}/tdnn${tdnn_affix}_sp

. utils/parse_options.sh # accept options



if [ -z $1 ]; then
  echo "Need to specify the keyword as the first argument"
  exit 1
fi
keyword=$1
echo $keyword

echo "stage 0"
if [ $stage -le 0 ]; then
  # Format the dictionary
  local/prepare_kws_dict.sh $keyword

  # Creates L.fst
  utils/prepare_lang.sh data/local/kws_dict_nosp \
    "<unk>" data/local/kws_lang_nosp data/kws_lang_nosp

fi



echo "stage 1"
if [ $stage -le 1 ]; then
  echo "copying train data set"
  cp -r data/train data/train_kws
  echo "copying dev data set"
  cp -r data/dev data/dev_kws
fi



echo "stage 2"
if [ $stage -le 2 ]; then

  # Replace the non keyword words from the training and dev text
#   python3 local/replace_non_kw.py --keyword $keyword --text_file data/dev/text --output_file data/dev_kws/text --text
  # python3 local/replace_non_kw.py --keyword $keyword --text_file data/dev/stm --output_file data/dev_kws/stm --stm
  python3 local/replace_non_kw.py --keyword $keyword --text_file data/train/text --output_file data/train_kws/text --text
  # python3 local/replace_non_kw.py --keyword $keyword --text_file data/train/stm --output_file data/train_kws/stm --stm

  python3 local/strip_utterance_id.py --text_file data/train_kws/text --output_file data/train_kws/text_stripped
fi




echo "stage 3"
if [ $stage -le 3 ]; then

  if [ -d data/lang_kws ]; then
    rm -r data/lang_kws
  fi

  cp -r data/kws_lang_nosp data/lang_kws

  ngram-count -text data/train_kws/text_stripped -no-sos -no-eos -order 2 -lm data/lang_kws/lm.ARPA
  arpa2fst --disambig-symbol=#0 --read-symbol-table=data/lang_kws/words.txt \
    data/lang_kws/lm.ARPA data/lang_kws/G.fst
  # ngramread --ARPA data/lang_kws/lm.ARPA data/lang_kws/G.fst

  local/build_const_arpa_kws_lm.sh data/lang_kws/lm.ARPA data/lang_kws data/lang_kws_rescore || exit 1;
  # Train the language model from the new data
  # local/ted_train_kw_lm.sh $keyword
fi


echo "stage 100"
if [ $stage -le 100 ]; then

  # Create data/graph_kws directory and use it for decoding
    cp -r $dir/graph $dir/graph_kws

    for filename in $(ls $dir/graph_kws/phones); do
        cp data/lang_kws/phones/$filename $dir/graph_kws/phones/$filename
    done
    cp data/lang_kws/phones.txt $dir/graph_kws/phones.txt
    cp data/lang_kws/words.txt $dir/graph_kws/words.txt

    cp data/lang_kws/words.txt $dir/graph_kws
    cp data/lang_kws/words.txt $dir/graph_kws

    cp data/lang_kws/phones.txt $dir/phones.txt
    # Replace the non keyword words from the training and dev text
    # python3 local/replace_non_kw.py --keyword $keyword --text_file data/dev/text --output_file data/dev_kws/text --text
    # python3 local/replace_non_kw.py --keyword $keyword --text_file data/dev/stm --output_file data/dev_kws/stm --stm
    # python3 local/replace_non_kw.py --keyword $keyword --text_file $dir/graph/text --output_file $dir/graph_kws/words.txt --text
    # python3 local/replace_non_kw.py --keyword $keyword --text_file $dir/graph/stm --output_file $dir/graph_kws/stm --stm
    # python3 local/replace_non_kw.py --keyword $keyword --text_file data/train/stm --output_file data/train_kws/stm --stm


#  python3 local/strip_utterance_id.py --text_file data/train_kws/text --output_file data/train_kws/text_stripped
fi


echo "stage 101"
if [ $stage -le 101 ]; then

    for dset in dev test; do
        cp -r data/${dset}_hires data/${dset}_kws_hires
        python3 local/replace_non_kw.py --keyword $keyword --text_file data/${dset}_hires/text --output_file data/${dset}_kws_hires/text --text
        python3 local/replace_non_kw.py --keyword $keyword --text_file data/${dset}_hires/stm --output_file data/${dset}_kws_hires/stm --stm
    done
    # Replace the non keyword words from the training and dev text
    # python3 local/replace_non_kw.py --keyword $keyword --text_file data/dev/stm --output_file data/dev_kws/stm --stm
    # python3 local/replace_non_kw.py --keyword $keyword --text_file $dir/graph/text --output_file $dir/graph_kws/words.txt --text
    # python3 local/replace_non_kw.py --keyword $keyword --text_file $dir/graph/stm --output_file $dir/graph_kws/stm --stm
    # python3 local/replace_non_kw.py --keyword $keyword --text_file data/train/stm --output_file data/train_kws/stm --stm


#  python3 local/strip_utterance_id.py --text_file data/train_kws/text --output_file data/train_kws/text_stripped
fi



echo "stage 198"
if [ $stage -le 198 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  if [ -f data/graph_kws/HCLG.fst ]; then
      rm data/graph_kws/HCLG.fst
  fi
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_kws $dir $dir/graph_kws
fi




echo "stage 200"
if [ $stage -le 200 ]; then
  rm $dir/.error 2>/dev/null || true
  for dset in dev test; do
      (
      steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $dir/graph_kws data/${dset}_kws_hires $dir/decode_kws_${dset} || exit 1;
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_kws data/lang_kws_rescore \
        data/${dset}_kws_hires ${dir}/decode_kws_${dset} ${dir}/decode_kws_${dset}_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

exit


# echo "stage 3"
# if [ $stage -le 4 ]; then

#   # Train the language model from the new data
#   local/ted_train_kw_lm.sh $keyword
# fi


exit



#echo "stage 17"
#if [ $stage -le 17 ]; then
#  # This will only work if you have GPUs on your system (and note that it requires
#  # you to have the queue set up the right way... see kaldi-asr.org/doc/queue.html)
#  local/chain/run_tdnn.sh
#fi
#echo "stage 18"
#if [ $stage -le 18 ]; then
#  # You can either train your own rnnlm or download a pre-trained one
#  if $train_rnnlm; then
#    local/rnnlm/tuning/run_lstm_tdnn_a.sh
#    local/rnnlm/average_rnnlm.sh
#  else
#    local/ted_download_rnnlm.sh
#  fi
#fi
#
#echo "stage 19"
#if [ $stage -le 19 ]; then
#  # Here we rescore the lattices generated at stage 17
#  rnnlm_dir=exp/rnnlm_lstm_tdnn_a_averaged
#  lang_dir=data/lang_chain
#  ngram_order=4
#
#  for dset in dev test; do
#    data_dir=data/${dset}_hires
#    decoding_dir=exp/chain_cleaned/tdnnf_1a/decode_${dset}
#    suffix=$(basename $rnnlm_dir)
#    output_dir=${decoding_dir}_$suffix
#
#    rnnlm/lmrescore_pruned.sh \
#      --cmd "$decode_cmd --mem 4G" \
#      --weight 0.5 --max-ngram-order $ngram_order \
#      $lang_dir $rnnlm_dir \
#      $data_dir $decoding_dir \
#      $output_dir
#  done
#fi


echo "$0: success."
exit 0