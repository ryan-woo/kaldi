#!/usr/bin/env bash
#
# rsw2148
# I wrote this whole thing. See the full paper for a full understanding of 
# everything that has been done
#

. ./cmd.sh
. ./path.sh


set -e -o pipefail -u

nj=8
decode_nj=8   # note: should not be >38 which is the number of speakers in the dev set
               # after applying --seconds-per-spk-max 180.  We decode with 4 threads, so
               # this will be too many jobs if you're using run.pl.
stage=358
train_rnnlm=true
train_lm=false

nnet3_affix=_cleaned_1d  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned
tdnn_affix=1d  #affix for TDNN directory, e.g. "a" or "b", in case we change the configuration.
dir=exp/chain${nnet3_affix}/tdnn${tdnn_affix}_sp
kws_dir=${dir}_kws



# Args for training an epoch
train_stage=0
train_set=train_cleaned
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'
remove_egs=false
tree_affix=  # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
common_egs_dir=  # you can set this to use previously dumped egs.
tree_dir=exp/chain${nnet3_affix}/tree_bi${tree_affix}
gmm=tri3_cleaned  # the gmm for the target data
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats

orig_train_data_dir=data/${train_set}_sp_hires
kws_train_data_dir=data/${train_set}_kws_sp_hires

# Setting 'online_cmvn' to true replaces 'apply-cmvn' by
# 'apply-cmvn-online' both for i-vector extraction and TDNN input.
# The i-vector extractor uses the config 'conf/online_cmvn.conf' for
# both the UBM and the i-extractor. The TDNN input is configured via
# '--feat.cmvn-opts' that is set to the same config, so we use the
# same cmvn for i-extractor and the TDNN input.
online_cmvn=true


. utils/parse_options.sh # accept options



if [ -z $1 ]; then
  echo "Need to specify the keyword as the first argument"
  exit 1
fi
keyword=$1
echo $keyword

if [ $stage -le -2 ]; then
  echo "Stage -2 start"
  local/download_data.sh
  echo "Stage -2 end"
fi

if [ $stage -le -1 ]; then
  echo "Stage -1 start"
  echo "Stage -1: Preparing data start"
  local/prepare_data.sh
  echo "Stage -1: Preparing data end"
  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  # [we chose 3 minutes because that gives us 38 speakers for the dev data, which is
  #  more than our normal 30 jobs.]
  echo "Stage -1: Modifying speaker info start"
  for dset in dev test train; do
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
  done
    echo "Stage -1: Modifying speaker info end"
    echo "Stage -1 end"

fi


echo "stage 0"
if [ $stage -le 0 ]; then
  # Here we create a new dictionary by removing all non-keywords from the existing tedlium
  # dictionary. Then we re-generate the dictionary FST.

  # Format the dictionary
  local/prepare_kws_dict.sh $keyword

  # Creates L.fst
  utils/prepare_lang.sh data/local/kws_dict_nosp \
    "<unk>" data/local/kws_lang_nosp data/kws_lang_nosp

fi



echo "stage 1"
if [ $stage -le 1 ]; then
  # Here we simply copy over the datasets to not corrupt the original

  for set in train dev test; do

    if [ -d data/${set}_kws ]; then
      echo "Removing data/${set}_kws to stay clean"
      rm -r data/${set}_kws
    fi

    echo "copying $set data set to data/${set}_kws"
    cp -r data/$set data/${set}_kws
  done

fi


echo "stage 2"
if [ $stage -le 2 ]; then

  # Replace the non keyword words from the training, test, and dev text
  python3 local/replace_non_kw.py --keyword $keyword --text_file data/dev/text --output_file data/dev_kws/text --text
  python3 local/replace_non_kw.py --keyword $keyword --text_file data/dev/stm --output_file data/dev_kws/stm --stm
  python3 local/replace_non_kw.py --keyword $keyword --text_file data/train/text --output_file data/train_kws/text --text
  python3 local/replace_non_kw.py --keyword $keyword --text_file data/train/stm --output_file data/train_kws/stm --stm
  python3 local/replace_non_kw.py --keyword $keyword --text_file data/test/text --output_file data/test_kws/text --text
  python3 local/replace_non_kw.py --keyword $keyword --text_file data/test/stm --output_file data/test_kws/stm --stm

  python3 local/strip_utterance_id.py --text_file data/train_kws/text --output_file data/train_kws/text_stripped
fi


echo "stage 3"
if [ $stage -le 3 ]; then

  # In this stage, the new language model is generated using SRILM.

  if [ -d data/lang_kws ]; then
    rm -r data/lang_kws
  fi

  cp -r data/kws_lang_nosp data/lang_kws

  # These next two lines make G.fst. We train a bigram model, though the order probably doesn't matter that much.
  # -unk allows the unknown token in the lm. This uses SRILM
  ngram-count -text data/train_kws/text_stripped -unk -no-sos -no-eos -order 2 -lm data/lang_kws/lm.ARPA
  arpa2fst --disambig-symbol=#0 --read-symbol-table=data/lang_kws/words.txt \
    data/lang_kws/lm.ARPA data/lang_kws/G.fst

  # This is my own slightly modified copy of the build_const_arpa_lm script
  local/build_const_arpa_kws_lm.sh data/lang_kws/lm.ARPA data/lang_kws data/lang_kws_rescore || exit 1;

fi

echo "stage 100"
if [ $stage -le 100 ]; then
  # This stage creates a new working directory for the graphs and decoding.

  # Create data/graph_kws directory and use it for decoding
    if [ -d $dir/graph_kws ]; then
      rm -r $dir/graph_kws
    fi
    cp -r $dir/graph $dir/graph_kws

    for filename in $(ls $dir/graph_kws/phones); do
        cp data/lang_kws/phones/$filename $dir/graph_kws/phones/$filename
    done
    cp data/lang_kws/phones.txt $dir/graph_kws/phones.txt
    cp data/lang_kws/words.txt $dir/graph_kws/words.txt

    cp data/lang_kws/words.txt $dir/graph_kws

    cp data/lang_kws/phones.txt $dir/phones.txt
fi


echo $dir

echo "stage 101"
if [ $stage -le 101 ]; then

    # In this stage, the non-keywords are removed from the hires text and stm files.
    # We also copy the original model into these hires data directories.

    for dset in dev test; do
    # for dset in dev; do
        if [ -d data/${dset}_kws_hires ]; then
          rm -r data/${dset}_kws_hires
        fi


        cp -r data/${dset}_hires data/${dset}_kws_hires
        rm -r data/${dset}_kws_hires/split* data/${dset}_kws_hires/.backup
        rm -r data/${dset}_kws_hires/data
        python3 local/replace_non_kw.py --keyword $keyword --text_file data/${dset}_hires/text --output_file data/${dset}_kws_hires/text --text
        python3 local/replace_non_kw.py --keyword $keyword --text_file data/${dset}_hires/stm --output_file data/${dset}_kws_hires/stm --stm
        cp exp/chain_cleaned_1d/tdnn1d_sp/final.mdl data/${dset}_kws_hires/        
    done
fi


# Feature extraction
echo "stage 102"
if [ $stage -le 102 ]; then
  for set in dev_kws_hires test_kws_hires; do
    datadir=data/$set
    steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc_hires.conf --nj 30 --cmd "$train_cmd" $datadir
    steps/compute_cmvn_stats.sh $datadir

    # The matrix-sum was required because the old global_cmvn.scp file was not working with some error
    matrix-sum --binary=false scp:$datadir/cmvn.scp - > $datadir/global_cmvn.stats 2>/dev/null;
  done

fi


echo "stage 198"
if [ $stage -le 198 ]; then
  
  # Recompose the graph with the new L and G
  if [ -f $dir/graph_kws/HCLG.fst ]; then
      rm $dir/graph_kws/HCLG.fst
  fi
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_kws $dir $dir/graph_kws
fi

echo $dir


# # nnet3-latgen-faster-parallel --num-threads=4 --online-ivectors=scp:exp/nnet3_cleaned_1d/ivectors_test_hires/ivector_online.scp --online-ivector-period=10 --frame-subsampling-factor=3 --frames-per-chunk=50 --extra-left-context=0 --extra-right-context=0 --extra-left-context-initial=-1 --extra-right-context-final=-1 --minimize=false --max-active=7000 --min-active=200 --beam=15.0 --lattice-beam=8.0 --acoustic-scale=1.0 --allow-partial=true --word-symbol-table=exp/chain_cleaned_1d/tdnn1d_sp/graph_kws/words.txt exp/chain_cleaned_1d/tdnn1d_sp/final.mdl exp/chain_cleaned_1d/tdnn1d_sp/graph_kws/HCLG.fst "ark,s,cs:apply-cmvn-online --config=conf/online_cmvn.conf --spk2utt=ark:data/test_kws_hires/split8/1/spk2utt exp/chain_cleaned_1d/tdnn1d_sp/global_cmvn.stats scp:data/test_kws_hires/split8/1/feats.scp ark:- |" "ark:|lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_test/lat.1.gz" 
# Started at Wed Apr 20 20:04:00 UTC 2022


echo "stage 200"
if [ $stage -le 200 ]; then

  # Decode

  rm $dir/.error 2>/dev/null || true
  for dset in dev test; do
  # for dset in test; do
      (
      cp data/${dset}_kws_hires/global_cmvn.stats $dir/global_cmvn.stats;
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

results_dir=$dir/results

echo "stage 250"
if [ $stage -le 250 ]; then
  
  # Evaluate beyond the WER. 
  # Here a ctm is also extracted

  cur_results_dir=${results_dir}/no_retrain_12_layer
  mkdir -p $cur_results_dir

  for dset in dev test; do

    echo $cur_results_dir
    lattice-to-nbest --n=10 "ark:gunzip -c  $dir/decode_kws_${dset}/lat.*.gz|" ark,t:${cur_results_dir}/${dset}-10.best
    nbest-to-linear ark:${cur_results_dir}/${dset}-10.best ark,t:${cur_results_dir}/${dset}-10.ali \
      ark,t:${cur_results_dir}/${dset}-10.words ark,t:${cur_results_dir}/${dset}-10.lmscore \
      ark,t:${cur_results_dir}/${dset}-10.acscore
    utils/int2sym.pl -f 2- exp/chain_cleaned_1d/tdnn1d_sp/graph_kws/words.txt ${cur_results_dir}/${dset}-10.words > ${cur_results_dir}/${dset}-10-decoded.txt
    echo "Placed decoded words in ${cur_results_dir}/${dset}-10-decoded.txt"

    python3 local/confusion.py --reference ${dir}/decode_kws_${dset}/scoring/test_filt.txt \
      --hypothesis ${results_dir}/no_retrain_12_layer/${dset}-10-decoded.txt --keyword $keyword \
      --alignment-tol 3 --best-n 5 | tee ${cur_results_dir}/${dset}-10-confusion.txt
    echo "Also placed the confusion matrix into ${cur_results_dir}/${dset}-10-confusion.txt"

    ./steps/get_ctm.sh data/${dset}_kws/ data/lang_kws exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_${dset}

  done
fi


echo "stage 251"
if [ $stage -le 251 ]; then

  # Perform a computation of the kws_results. We do this
  # by comparing the start and stop times of results from a ground truth.

  for dset in dev test; do
    if [ -f $dir/decode_kws_${dset}/kws_results.csv ]; then
      rm $dir/decode_kws_${dset}/kws_results.csv
    fi
    echo "Scoring the ${dset} set in $dir/decode_kws_${dset}"

    for i in {1..20}; do
      python3 local/analyze.py --keyword $keyword \
        --hypothesis $dir/decode_kws_${dset}/score_${i}/${dset}_kws.ctm \
        --reference $dir/decode_${dset}/score_${i}/${dset}.ctm \
        --lm-level $i --num-layers 12 --tolerance 1 | tee -a $dir/decode_kws_${dset}/kws_results.csv
    done
    echo "Wrote results to ${dir}/decode_kws_${dset}/kws_results.txt"
  done
fi



echo "stage 300"
if [ $stage -le 300 ]; then

  # Replace words in the train_data_dir
  if [ ! -d $kws_train_data_dir ]; then
    echo "Copying $orig_train_data_dir to $kws_train_data_dir"
    cp -r $orig_train_data_dir $kws_train_data_dir
    rm -r $kws_train_data_dir/split*
  fi
          
  
  echo "Replacing non-keywords in $kws_train_data_dir/text"
  python3 local/replace_non_kw.py --keyword $keyword --text_file $orig_train_data_dir/text --output_file $kws_train_data_dir/text --text

fi

echo "stage 301"
if [ $stage -le 301 ]; then
    # Regenerate mfccs for the training data

    steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc_hires.conf --nj 8 --cmd "$train_cmd" $kws_train_data_dir
    steps/compute_cmvn_stats.sh $kws_train_data_dir
    matrix-sum --binary=false scp:$kws_train_data_dir/cmvn.scp - > $kws_train_data_dir/global_cmvn.stats 2>/dev/null;
fi

echo "stage 302"
if [ $stage -le 302 ]; then
  # Save the existing final.mdl to final_orig.mdl

  cp $dir/configs/ref.raw $dir/configs/ref_orig.raw
  cp $dir/final.mdl $dir/final_orig.mdl
fi

mkdir -p $dir/../kws_tdnn1d_sp/

echo "stage 310"
if [ $stage -le 310 ]; then

  # Train for one epoch

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --use-gpu=wait \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts="--config=conf/online_cmvn.conf" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.stage=5 \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false --online-cmvn $online_cmvn" \
    --egs.chunk-width 150,110,100 \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 5000000 \
    --trainer.num-epochs 1 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 2 \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --trainer.input-model $dir/final.mdl \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $kws_train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir
fi


echo "stage 311"
if [ $stage -le 311 ]; then
  if [ -d $dir/graph_kws_retrain_12 ]; then
    rm -r $dir/graph_kws_retrain_12
  fi

  cp -r $dir/graph_kws $dir/graph_kws_retrain_12
  cp data/lang_kws/phones.txt $dir/phones.txt
fi


echo "stage 312"
if [ $stage -le 312 ]; then

  # Recompose graph
  if [ -f $dir/graph_kws_retrain_12/HCLG.fst ]; then
      rm $dir/graph_kws_retrain_12/HCLG.fst
  fi
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_kws $dir $dir/graph_kws_retrain_12
fi



echo "stage 315"
if [ $stage -le 315 ]; then
  # Decode

  rm $dir/.error 2>/dev/null || true
  for dset in dev test; do
      (      
      cp data/${dset}_kws_hires/global_cmvn.stats $dir/global_cmvn.stats;
      steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $dir/graph_kws_retrain_12 data/${dset}_kws_hires $dir/decode_kws_${dset}_retrain_12 || exit 1;

      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_kws data/lang_kws_rescore \
        data/${dset}_kws_hires ${dir}/decode_kws_${dset}_retrain_12 ${dir}/decode_kws_${dset}_retrain_12_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi


echo "stage 316"
if [ $stage -le 316 ]; then

  # Score
  cur_results_dir=${results_dir}/retrain_12_layer
  mkdir -p $cur_results_dir

  for dset in dev test; do

    echo $cur_results_dir
    lattice-to-nbest --n=10 "ark:gunzip -c  $dir/decode_kws_${dset}/lat.*.gz|" ark,t:${cur_results_dir}/${dset}-10.best
    nbest-to-linear ark:${cur_results_dir}/${dset}-10.best ark,t:${cur_results_dir}/${dset}-10.ali \
      ark,t:${cur_results_dir}/${dset}-10.words ark,t:${cur_results_dir}/${dset}-10.lmscore \
      ark,t:${cur_results_dir}/${dset}-10.acscore
    utils/int2sym.pl -f 2- exp/chain_cleaned_1d/tdnn1d_sp/graph_kws_retrain_12/words.txt ${cur_results_dir}/${dset}-10.words > ${cur_results_dir}/${dset}-10-decoded.txt
    echo "Placed decoded words in ${cur_results_dir}/${dset}-10-decoded.txt"

    python3 local/confusion.py --reference ${dir}/decode_kws_${dset}/scoring/test_filt.txt \
      --hypothesis ${results_dir}/no_retrain_12_layer/${dset}-10-decoded.txt --keyword $keyword \
      --alignment-tol 3 --best-n 5 | tee ${cur_results_dir}/${dset}-10-confusion.txt
    echo "Also placed the confusion matrix into ${cur_results_dir}/${dset}-10-confusion.txt"

  done
fi

echo "stage 317"
if [ $stage -le 317 ]; then
  # get ctms

  ./steps/get_ctm.sh data/dev_kws_hires/ data/lang_kws exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_dev_retrain_12
  ./steps/get_ctm.sh data/test_kws_hires/ data/lang_kws exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_test_retrain_12
fi


echo "stage 318"
if [ $stage -le 318 ]; then
  # Score

  for dset in dev test; do
    if [ -f $dir/decode_kws_${dset}_retrain_12/kws_results.csv ]; then
      rm $dir/decode_kws_${dset}_retrain_12/kws_results.csv
    fi
    echo "Scoring the ${dset} set in $dir/decode_kws_${dset}_retrain_12/"

    for i in {1..20}; do
      python3 local/analyze.py --keyword $keyword \
        --hypothesis $dir/decode_kws_${dset}_retrain_12/score_${i}/${dset}_kws_hires.ctm \
        --reference $dir/decode_${dset}/score_${i}/${dset}.ctm \
        --lm-level $i --num-layers 12 --retrained --tolerance 1 | tee -a $dir/decode_kws_${dset}_retrain_12/kws_results.csv
    done
    echo "Wrote results to $dir/decode_kws_${dset}_retrain_12/kws_results.csv"
  done
fi


echo "stage 320"
if [ $stage -le 320 ]; then
  # Create a new model with one layer removed (only 11 hidden layers).

  echo "Configuring model with only 11 hidden tdnnf layers"

  echo "component-node name=prefinal-l component=prefinal-l input=tdnnf12.noop" > $dir/config.11
  nnet3-am-copy --nnet-config=${dir}/config.11 --edits=remove-orphans \
    $dir/final_orig.mdl $dir/final_11_input.mdl
fi


echo "stage 321"
if [ $stage -le 321 ]; then
  # Retrain it

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --use-gpu=wait \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts="--config=conf/online_cmvn.conf" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.stage=5 \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false --online-cmvn $online_cmvn" \
    --egs.chunk-width 150,110,100 \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 5000000 \
    --trainer.num-epochs 1 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 2 \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --trainer.input-model $dir/final_11_input.mdl \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $kws_train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir
fi


 
echo "stage 322"
if [ $stage -le 322 ]; then
  if [ -d $dir/graph_kws_retrain_11 ]; then
    rm -r $dir/graph_kws_retrain_11
  fi

  cp -r $dir/graph_kws $dir/graph_kws_retrain_11
  cp data/lang_kws/phones.txt $dir/phones.txt
fi


echo "stage 323"
if [ $stage -le 323 ]; then

  if [ -f $dir/graph_kws_retrain_11/HCLG.fst ]; then
      rm $dir/graph_kws_retrain_11/HCLG.fst
  fi
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_kws $dir $dir/graph_kws_retrain_11
fi


echo "stage 325"
if [ $stage -le 325 ]; then
  rm $dir/.error 2>/dev/null || true
  for dset in dev test; do
      (      
      cp data/${dset}_kws_hires/global_cmvn.stats $dir/global_cmvn.stats;
      steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $dir/graph_kws_retrain_11 data/${dset}_kws_hires $dir/decode_kws_${dset}_retrain_11 || exit 1;

      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_kws data/lang_kws_rescore \
        data/${dset}_kws_hires ${dir}/decode_kws_${dset}_retrain_11 ${dir}/decode_kws_${dset}_retrain_11_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi


echo "stage 326"
if [ $stage -le 326 ]; then
  cur_results_dir=${results_dir}/retrain_11_layer
  mkdir -p $cur_results_dir

  for dset in dev test; do

    echo $cur_results_dir
    lattice-to-nbest --n=10 "ark:gunzip -c  $dir/decode_kws_${dset}/lat.*.gz|" ark,t:${cur_results_dir}/${dset}-10.best
    nbest-to-linear ark:${cur_results_dir}/${dset}-10.best ark,t:${cur_results_dir}/${dset}-10.ali \
      ark,t:${cur_results_dir}/${dset}-10.words ark,t:${cur_results_dir}/${dset}-10.lmscore \
      ark,t:${cur_results_dir}/${dset}-10.acscore
    utils/int2sym.pl -f 2- exp/chain_cleaned_1d/tdnn1d_sp/graph_kws_retrain_11/words.txt ${cur_results_dir}/${dset}-10.words > ${cur_results_dir}/${dset}-10-decoded.txt
    echo "Placed decoded words in ${cur_results_dir}/${dset}-10-decoded.txt"

    python3 local/confusion.py --reference ${dir}/decode_kws_${dset}/scoring/test_filt.txt \
      --hypothesis ${results_dir}/retrain_11_layer/${dset}-10-decoded.txt --keyword $keyword \
      --alignment-tol 3 --best-n 5 | tee ${cur_results_dir}/${dset}-10-confusion.txt
    echo "Also placed the confusion matrix into ${cur_results_dir}/${dset}-10-confusion.txt"

  done
fi

echo "stage 327"
if [ $stage -le 327 ]; then


  ./steps/get_ctm.sh data/dev_kws_hires/ data/lang_kws exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_dev_retrain_11
  ./steps/get_ctm.sh data/test_kws_hires/ data/lang_kws exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_test_retrain_11
fi


echo "stage 328"
if [ $stage -le 328 ]; then
  for dset in dev test; do
    if [ -f $dir/decode_kws_${dset}_retrain_11/kws_results.csv ]; then
      rm $dir/decode_kws_${dset}_retrain_11/kws_results.csv
    fi
    echo "Scoring the ${dset} set in $dir/decode_kws_${dset}_retrain_11/"

    for i in {1..20}; do
      python3 local/analyze.py --keyword $keyword \
        --hypothesis $dir/decode_kws_${dset}_retrain_11/score_${i}/${dset}_kws_hires.ctm \
        --reference $dir/decode_${dset}/score_${i}/${dset}.ctm \
        --lm-level $i --num-layers 11 --retrained --tolerance 1 | tee -a $dir/decode_kws_${dset}_retrain_11/kws_results.csv
    done
    echo "Wrote results to $dir/decode_kws_${dset}_retrain_11/kws_results.csv"
  done
fi



echo "stage 330"
if [ $stage -le 330 ]; then
  # Create a new model with one more layer removed (only 10 hidden layers).

  echo "Configuring model with only 10 hidden tdnnf layers"

  echo "component-node name=prefinal-l component=prefinal-l input=tdnnf11.noop" > $dir/config.10
  nnet3-am-copy --nnet-config=${dir}/config.10 --edits=remove-orphans \
    $dir/final_orig.mdl $dir/final_10_input.mdl
fi


echo "stage 331"
if [ $stage -le 331 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --use-gpu=wait \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts="--config=conf/online_cmvn.conf" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.stage=5 \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false --online-cmvn $online_cmvn" \
    --egs.chunk-width 150,110,100 \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 5000000 \
    --trainer.num-epochs 1 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 2 \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --trainer.input-model $dir/final_10_input.mdl \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $kws_train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir
fi

echo "stage 332"
if [ $stage -le 332 ]; then
  if [ -d $dir/graph_kws_retrain_10 ]; then
    rm -r $dir/graph_kws_retrain_10
  fi

  cp -r $dir/graph_kws $dir/graph_kws_retrain_10
  cp data/lang_kws/phones.txt $dir/phones.txt
fi

echo "stage 333"
if [ $stage -le 333 ]; then

  # TODO I think my phones file is coming from the wrong place in the one-epoch re-training.
  # I should be able to remove this line once I figure that out.
  # cp data/lang_kws/phones.txt $dir/phones.txt

  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  if [ -f $dir/graph_kws_retrain_10/HCLG.fst ]; then
      rm $dir/graph_kws_retrain_10/HCLG.fst
  fi
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_kws $dir $dir/graph_kws_retrain_10
fi


echo "stage 335"
if [ $stage -le 335 ]; then
  rm $dir/.error 2>/dev/null || true
  for dset in dev test; do
      (      
      cp data/${dset}_kws_hires/global_cmvn.stats $dir/global_cmvn.stats;
      steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $dir/graph_kws_retrain_10 data/${dset}_kws_hires $dir/decode_kws_${dset}_retrain_10 || exit 1;

      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_kws data/lang_kws_rescore \
        data/${dset}_kws_hires ${dir}/decode_kws_${dset}_retrain_10 ${dir}/decode_kws_${dset}_retrain_10_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi


echo "stage 336"
if [ $stage -le 336 ]; then
  cur_results_dir=${results_dir}/retrain_10_layer
  mkdir -p $cur_results_dir

  for dset in dev test; do

    echo $cur_results_dir
    lattice-to-nbest --n=10 "ark:gunzip -c  $dir/decode_kws_${dset}/lat.*.gz|" ark,t:${cur_results_dir}/${dset}-10.best
    nbest-to-linear ark:${cur_results_dir}/${dset}-10.best ark,t:${cur_results_dir}/${dset}-10.ali \
      ark,t:${cur_results_dir}/${dset}-10.words ark,t:${cur_results_dir}/${dset}-10.lmscore \
      ark,t:${cur_results_dir}/${dset}-10.acscore
    utils/int2sym.pl -f 2- exp/chain_cleaned_1d/tdnn1d_sp/graph_kws_retrain_10/words.txt ${cur_results_dir}/${dset}-10.words > ${cur_results_dir}/${dset}-10-decoded.txt
    echo "Placed decoded words in ${cur_results_dir}/${dset}-10-decoded.txt"

    python3 local/confusion.py --reference ${dir}/decode_kws_${dset}/scoring/test_filt.txt \
      --hypothesis ${results_dir}/retrain_10_layer/${dset}-10-decoded.txt --keyword $keyword \
      --alignment-tol 3 --best-n 5 | tee ${cur_results_dir}/${dset}-10-confusion.txt
    echo "Also placed the confusion matrix into ${cur_results_dir}/${dset}-10-confusion.txt"

  done
fi

echo "stage 337"
if [ $stage -le 337 ]; then


  ./steps/get_ctm.sh data/dev_kws_hires/ data/lang_kws exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_dev_retrain_10
  ./steps/get_ctm.sh data/test_kws_hires/ data/lang_kws exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_test_retrain_10
fi


echo "stage 338"
if [ $stage -le 338 ]; then
  for dset in dev test; do
    if [ -f $dir/decode_kws_${dset}_retrain_10/kws_results.csv ]; then
      rm $dir/decode_kws_${dset}_retrain_10/kws_results.csv
    fi
    echo "Scoring the ${dset} set in $dir/decode_kws_${dset}_retrain_10/"

    for i in {1..20}; do
      python3 local/analyze.py --keyword $keyword \
        --hypothesis $dir/decode_kws_${dset}_retrain_10/score_${i}/${dset}_kws_hires.ctm \
        --reference $dir/decode_${dset}/score_${i}/${dset}.ctm \
        --lm-level $i --num-layers 10 --retrained --tolerance 1 | tee -a $dir/decode_kws_${dset}_retrain_10/kws_results.csv
    done
    echo "Wrote results to $dir/decode_kws_${dset}_retrain_10/kws_results.csv"
  done
fi



#------------------- 09 layers --------------------------

echo "stage 340"
if [ $stage -le 340 ]; then
  # Create a new model with one more layer removed (only 09 hidden layers).

  echo "Configuring model with only 09 hidden tdnnf layers"

  echo "component-node name=prefinal-l component=prefinal-l input=tdnnf10.noop" > $dir/config.09
  nnet3-am-copy --nnet-config=${dir}/config.09 --edits=remove-orphans \
    $dir/final_orig.mdl $dir/final_09_input.mdl
fi


echo "stage 341"
if [ $stage -le 341 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --use-gpu=wait \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts="--config=conf/online_cmvn.conf" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.stage=5 \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false --online-cmvn $online_cmvn" \
    --egs.chunk-width 150,110,100 \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 5000000 \
    --trainer.num-epochs 1 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 2 \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --trainer.input-model $dir/final_09_input.mdl \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $kws_train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir
fi

echo "stage 342"
if [ $stage -le 342 ]; then
  if [ -d $dir/graph_kws_retrain_09 ]; then
    rm -r $dir/graph_kws_retrain_09
  fi

  cp -r $dir/graph_kws $dir/graph_kws_retrain_09
  cp data/lang_kws/phones.txt $dir/phones.txt
fi

echo "stage 343"
if [ $stage -le 343 ]; then

  # TODO I think my phones file is coming from the wrong place in the one-epoch re-training.
  # I should be able to remove this line once I figure that out.
  # cp data/lang_kws/phones.txt $dir/phones.txt

  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  if [ -f $dir/graph_kws_retrain_09/HCLG.fst ]; then
      rm $dir/graph_kws_retrain_09/HCLG.fst
  fi
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_kws $dir $dir/graph_kws_retrain_09
fi


echo "stage 345"
if [ $stage -le 345 ]; then
  rm $dir/.error 2>/dev/null || true
  for dset in dev test; do
      (      
      cp data/${dset}_kws_hires/global_cmvn.stats $dir/global_cmvn.stats;
      steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $dir/graph_kws_retrain_09 data/${dset}_kws_hires $dir/decode_kws_${dset}_retrain_09 || exit 1;

      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_kws data/lang_kws_rescore \
        data/${dset}_kws_hires ${dir}/decode_kws_${dset}_retrain_09 ${dir}/decode_kws_${dset}_retrain_09_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi


echo "stage 346"
if [ $stage -le 346 ]; then
  cur_results_dir=${results_dir}/retrain_09_layer
  mkdir -p $cur_results_dir

  for dset in dev test; do

    echo $cur_results_dir
    lattice-to-nbest --n=10 "ark:gunzip -c  $dir/decode_kws_${dset}/lat.*.gz|" ark,t:${cur_results_dir}/${dset}-10.best
    nbest-to-linear ark:${cur_results_dir}/${dset}-10.best ark,t:${cur_results_dir}/${dset}-10.ali \
      ark,t:${cur_results_dir}/${dset}-10.words ark,t:${cur_results_dir}/${dset}-10.lmscore \
      ark,t:${cur_results_dir}/${dset}-10.acscore
    utils/int2sym.pl -f 2- exp/chain_cleaned_1d/tdnn1d_sp/graph_kws_retrain_09/words.txt ${cur_results_dir}/${dset}-10.words > ${cur_results_dir}/${dset}-10-decoded.txt
    echo "Placed decoded words in ${cur_results_dir}/${dset}-10-decoded.txt"

    python3 local/confusion.py --reference ${dir}/decode_kws_${dset}/scoring/test_filt.txt \
      --hypothesis ${results_dir}/retrain_09_layer/${dset}-10-decoded.txt --keyword $keyword \
      --alignment-tol 3 --best-n 5 | tee ${cur_results_dir}/${dset}-10-confusion.txt
    echo "Also placed the confusion matrix into ${cur_results_dir}/${dset}-10-confusion.txt"

  done
fi

echo "stage 347"
if [ $stage -le 347 ]; then


  ./steps/get_ctm.sh data/dev_kws_hires/ data/lang_kws exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_dev_retrain_09
  ./steps/get_ctm.sh data/test_kws_hires/ data/lang_kws exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_test_retrain_09
fi


echo "stage 348"
if [ $stage -le 348 ]; then
  for dset in dev test; do
    if [ -f $dir/decode_kws_${dset}_retrain_09/kws_results.csv ]; then
      rm $dir/decode_kws_${dset}_retrain_09/kws_results.csv
    fi
    echo "Scoring the ${dset} set in $dir/decode_kws_${dset}_retrain_09/"

    for i in {1..20}; do
      python3 local/analyze.py --keyword $keyword \
        --hypothesis $dir/decode_kws_${dset}_retrain_09/score_${i}/${dset}_kws_hires.ctm \
        --reference $dir/decode_${dset}/score_${i}/${dset}.ctm \
        --lm-level $i --num-layers 9 --retrained --tolerance 1 | tee -a $dir/decode_kws_${dset}_retrain_09/kws_results.csv
    done
    echo "Wrote results to $dir/decode_kws_${dset}_retrain_09/kws_results.csv"
  done
fi


#------------------- 08 layers --------------------------

echo "stage 350"
if [ $stage -le 350 ]; then
  # Create a new model with one more layer removed (only 08 hidden layers).

  echo "Configuring model with only 08 hidden tdnnf layers"

  echo "component-node name=prefinal-l component=prefinal-l input=tdnnf9.noop" > $dir/config.08
  nnet3-am-copy --nnet-config=${dir}/config.08 --edits=remove-orphans \
    $dir/final_orig.mdl $dir/final_08_input.mdl
fi


echo "stage 351"
if [ $stage -le 351 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --use-gpu=wait \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts="--config=conf/online_cmvn.conf" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.stage=5 \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false --online-cmvn $online_cmvn" \
    --egs.chunk-width 150,110,100 \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 5000000 \
    --trainer.num-epochs 1 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 2 \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --trainer.input-model $dir/final_08_input.mdl \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $kws_train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir
fi

echo "stage 352"
if [ $stage -le 352 ]; then
  if [ -d $dir/graph_kws_retrain_08 ]; then
    rm -r $dir/graph_kws_retrain_08
  fi

  cp -r $dir/graph_kws $dir/graph_kws_retrain_08
  cp data/lang_kws/phones.txt $dir/phones.txt
fi

echo "stage 353"
if [ $stage -le 353 ]; then

  # TODO I think my phones file is coming from the wrong place in the one-epoch re-training.
  # I should be able to remove this line once I figure that out.
  # cp data/lang_kws/phones.txt $dir/phones.txt

  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  if [ -f $dir/graph_kws_retrain_08/HCLG.fst ]; then
      rm $dir/graph_kws_retrain_08/HCLG.fst
  fi
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_kws $dir $dir/graph_kws_retrain_08
fi


echo "stage 355"
if [ $stage -le 355 ]; then
  rm $dir/.error 2>/dev/null || true
  for dset in dev test; do
      (      
      cp data/${dset}_kws_hires/global_cmvn.stats $dir/global_cmvn.stats;
      steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $dir/graph_kws_retrain_08 data/${dset}_kws_hires $dir/decode_kws_${dset}_retrain_08 || exit 1;

      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_kws data/lang_kws_rescore \
        data/${dset}_kws_hires ${dir}/decode_kws_${dset}_retrain_08 ${dir}/decode_kws_${dset}_retrain_08_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi


echo "stage 356"
if [ $stage -le 356 ]; then
  cur_results_dir=${results_dir}/retrain_08_layer
  mkdir -p $cur_results_dir

  for dset in dev test; do

    echo $cur_results_dir
    lattice-to-nbest --n=10 "ark:gunzip -c  $dir/decode_kws_${dset}/lat.*.gz|" ark,t:${cur_results_dir}/${dset}-10.best
    nbest-to-linear ark:${cur_results_dir}/${dset}-10.best ark,t:${cur_results_dir}/${dset}-10.ali \
      ark,t:${cur_results_dir}/${dset}-10.words ark,t:${cur_results_dir}/${dset}-10.lmscore \
      ark,t:${cur_results_dir}/${dset}-10.acscore
    utils/int2sym.pl -f 2- exp/chain_cleaned_1d/tdnn1d_sp/graph_kws_retrain_08/words.txt ${cur_results_dir}/${dset}-10.words > ${cur_results_dir}/${dset}-10-decoded.txt
    echo "Placed decoded words in ${cur_results_dir}/${dset}-10-decoded.txt"

    python3 local/confusion.py --reference ${dir}/decode_kws_${dset}/scoring/test_filt.txt \
      --hypothesis ${results_dir}/retrain_08_layer/${dset}-10-decoded.txt --keyword $keyword \
      --alignment-tol 3 --best-n 5 | tee ${cur_results_dir}/${dset}-10-confusion.txt
    echo "Also placed the confusion matrix into ${cur_results_dir}/${dset}-10-confusion.txt"

  done
fi

echo "stage 357"
if [ $stage -le 357 ]; then
  ./steps/get_ctm.sh data/dev_kws_hires/ data/lang_kws exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_dev_retrain_08
  ./steps/get_ctm.sh data/test_kws_hires/ data/lang_kws exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_test_retrain_08
fi


echo "stage 358"
if [ $stage -le 358 ]; then
  for dset in dev test; do
    if [ -f $dir/decode_kws_${dset}_retrain_08/kws_results.csv ]; then
      rm $dir/decode_kws_${dset}_retrain_08/kws_results.csv
    fi
    echo "Scoring the ${dset} set in $dir/decode_kws_${dset}_retrain_08/"

    for i in {1..20}; do
      python3 local/analyze.py --keyword $keyword \
        --hypothesis $dir/decode_kws_${dset}_retrain_08/score_${i}/${dset}_kws_hires.ctm \
        --reference $dir/decode_${dset}/score_${i}/${dset}.ctm \
        --lm-level $i --num-layers 8 --retrained --tolerance 1 | tee -a $dir/decode_kws_${dset}_retrain_08/kws_results.csv
    done
    echo "Wrote results to $dir/decode_kws_${dset}_retrain_08/kws_results.csv"
  done
fi

echo "Complete!"
exit


 # EVERYTHING BELOW HERE IS UNVERIFIED

# echo "stage 400"
# if [ $stage -le 400 ]; then
#   if [ -d $dir/graph_kws_retrain_12 ]; then
#     rm -r $dir/graph_kws_retrain_12
#   fi

#   cp -r $dir/graph_kws $dir/graph_kws_retrain_12
#   cp data/lang_kws/phones.txt $dir/phones.txt
# fi


# echo "stage 401"
# if [ $stage -le 401 ]; then

#   # TODO I think my phones file is coming from the wrong place in the one-epoch re-training.
#   # I should be able to remove this line once I figure that out.
#   # cp data/lang_kws/phones.txt $dir/phones.txt

#   # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
#   # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
#   # the lang directory.
#   if [ -f $dir/graph_kws_retrain_12/HCLG.fst ]; then
#       rm $dir/graph_kws_retrain_12/HCLG.fst
#   fi
#   utils/mkgraph.sh --self-loop-scale 1.0 data/lang_kws $dir $dir/graph_kws_retrain_12
# fi

# echo "stage 410"
# if [ $stage -le 410 ]; then
#   rm $dir/.error 2>/dev/null || true
#   for dset in dev test; do
#       (      
#       cp data/${dset}_kws_hires/global_cmvn.stats $dir/global_cmvn.stats;
#       steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
#           --acwt 1.0 --post-decode-acwt 10.0 \
#           --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
#           --scoring-opts "--min-lmwt 5 " \
#          $dir/graph_kws_retrain_12 data/${dset}_kws_hires $dir/decode_kws_${dset} || exit 1;
#       steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_kws data/lang_kws_rescore \
#         data/${dset}_kws_hires ${dir}/decode_kws_${dset} ${dir}/decode_kws_${dset}_rescore || exit 1
#     ) || touch $dir/.error &
#   done
#   wait
#   if [ -f $dir/.error ]; then
#     echo "$0: something went wrong in decoding"
#     exit 1
#   fi
# fi


# echo "stage 450"
# if [ $stage -le 450 ]; then
  
#   cur_results_dir=${results_dir}/retrain_12_layer
#   mkdir -p $cur_results_dir

#   for dset in dev test; do

#     echo $cur_results_dir
#     lattice-to-nbest --n=10 "ark:gunzip -c  $dir/decode_kws_${dset}/lat.*.gz|" ark,t:${cur_results_dir}/${dset}-10.best
#     nbest-to-linear ark:${cur_results_dir}/${dset}-10.best ark,t:${cur_results_dir}/${dset}-10.ali \
#       ark,t:${cur_results_dir}/${dset}-10.words ark,t:${cur_results_dir}/${dset}-10.lmscore \
#       ark,t:${cur_results_dir}/${dset}-10.acscore
#     utils/int2sym.pl -f 2- exp/chain_cleaned_1d/tdnn1d_sp/graph_kws_retrain_12/words.txt ${cur_results_dir}/${dset}-10.words > ${cur_results_dir}/${dset}-10-decoded.txt
#     echo "Placed decoded words in ${cur_results_dir}/${dset}-10-decoded.txt"

#     python3 local/confusion.py --reference ${dir}/decode_kws_${dset}/scoring/test_filt.txt \
#       --hypothesis ${results_dir}/no_retrain_12_layer/${dset}-10-decoded.txt --keyword $keyword \
#       --alignment-tol 3 --best-n 5 | tee ${cur_results_dir}/${dset}-10-confusion.txt
#     echo "Also placed the confusion matrix into ${cur_results_dir}/${dset}-10-confusion.txt"

#   done
# fi


# What I did

#  dir=exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_test
# lattice-to-nbest --n=10 "ark:gunzip -c  $dir/lat.*.gz|" ark,t:1.nbest

# nbest-to-linear ark:1.nbest ark,t:1.ali ark,t:1.words ark,t:1.lmscore ark,t:1.acscore
# utils/int2sym.pl -f 2- exp/chain_cleaned_1d/tdnn1d_sp/graph_kws/words.txt 1.words





exit



echo "$0: success."
exit 0
