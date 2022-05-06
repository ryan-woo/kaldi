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
decode_nj=2   # note: should not be >38 which is the number of speakers in the dev set
               # after applying --seconds-per-spk-max 180.  We decode with 4 threads, so
               # this will be too many jobs if you're using run.pl.
stage=100
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



keyword=discovery
echo $keyword


# =================
# These next few stages are commented out because they are not necessary for decoding
# =================


# echo "stage 0"
# if [ $stage -le 0 ]; then
#   # Here we create a new dictionary by removing all non-keywords from the existing tedlium
#   # dictionary. Then we re-generate the dictionary FST.

#   # Format the dictionary
#   local/prepare_kws_dict.sh $keyword

#   # Creates L.fst
#   utils/prepare_lang.sh data/local/kws_dict_nosp \
#     "<unk>" data/local/kws_lang_nosp data/kws_lang_nosp

# fi

# echo "stage 0.5"
# if [ $stage -le 1 ]; then

#   # Get the first 129 utterances - the length of Aimee Mullin's talk.
#   if [ -d data/test_tiny ]; then
#     rm -r data/test_tiny
#   fi

#   utils/subset_data_dir.sh --first data/test 129 data/test_tiny
# fi


# echo "stage 1"
# if [ $stage -le 1 ]; then
#   # Here we simply copy over the datasets to not corrupt the original

#   for set in train dev test; do

#     if [ -d data/${set}_kws ]; then
#       echo "Removing data/${set}_kws to stay clean"
#       rm -r data/${set}_kws
#     fi

#     echo "copying $set data set to data/${set}_kws"
#     cp -r data/$set data/${set}_kws
#   done

# fi


# echo "stage 2"
# if [ $stage -le 2 ]; then

  # Replace the non keyword words from the training, test, and dev text
  # python3 local/replace_non_kw.py --keyword $keyword --text_file data/dev/text --output_file data/dev_kws/text --text
  # python3 local/replace_non_kw.py --keyword $keyword --text_file data/dev/stm --output_file data/dev_kws/stm --stm
  # python3 local/replace_non_kw.py --keyword $keyword --text_file data/train/text --output_file data/train_kws/text --text
  # python3 local/replace_non_kw.py --keyword $keyword --text_file data/train/stm --output_file data/train_kws/stm --stm
  # python3 local/replace_non_kw.py --keyword $keyword --text_file data/test/text --output_file data/test_kws/text --text
  # python3 local/replace_non_kw.py --keyword $keyword --text_file data/test/stm --output_file data/test_kws/stm --stm

  # python3 local/strip_utterance_id.py --text_file data/train_kws/text --output_file data/train_kws/text_stripped
# fi


# echo "stage 3"
# if [ $stage -le 3 ]; then

#   # In this stage, the new language model is generated using SRILM.

#   if [ -d data/lang_kws ]; then
#     rm -r data/lang_kws
#   fi

#   cp -r data/kws_lang_nosp data/lang_kws

#   # These next two lines make G.fst. We train a bigram model, though the order probably doesn't matter that much.
#   # -unk allows the unknown token in the lm.
#   ngram-count -text data/train_kws/text_stripped -unk -no-sos -no-eos -order 2 -lm data/lang_kws/lm.ARPA
#   arpa2fst --disambig-symbol=#0 --read-symbol-table=data/lang_kws/words.txt \
#     data/lang_kws/lm.ARPA data/lang_kws/G.fst
#   # ngramread --ARPA data/lang_kws/lm.ARPA data/lang_kws/G.fst

#   local/build_const_arpa_kws_lm.sh data/lang_kws/lm.ARPA data/lang_kws data/lang_kws_rescore || exit 1;

# fi

# echo "stage 100"
# if [ $stage -le 100 ]; then
#   # This stage creates a new working directory for the graphs and decoding.

#   # Create data/graph_kws directory and use it for decoding
#     if [ -d $dir/graph_kws ]; then
#       rm -r $dir/graph_kws
#     fi
#     cp -r $dir/graph $dir/graph_kws

#     for filename in $(ls $dir/graph_kws/phones); do
#         cp data/lang_kws/phones/$filename $dir/graph_kws/phones/$filename
#     done
#     cp data/lang_kws/phones.txt $dir/graph_kws/phones.txt
#     cp data/lang_kws/words.txt $dir/graph_kws/words.txt

#     cp data/lang_kws/words.txt $dir/graph_kws

#     cp data/lang_kws/phones.txt $dir/phones.txt
# fi


echo $dir

echo "stage 100"
if [ $stage -le 100 ]; then
  # Get the first 25 utterances.
  if [ -d data/test_kws_hires_tiny ]; then
    rm -r data/test_kws_hires_tiny
  fi

  utils/subset_data_dir.sh --first data/test_hires 25 data/test_hires_tiny
fi

echo "stage 101"
if [ $stage -le 101 ]; then

    # In this stage, the non-keywords are removed from the hires text and stm files.
    # We also copy the original model into these hires data directories.

    for dset in test; do
        if [ -d data/${dset}_kws_hires_tiny ]; then
          rm -r data/${dset}_kws_hires_tiny
        fi


        cp -r data/${dset}_hires_tiny data/${dset}_kws_hires_tiny
        # rm -r data/${dset}_kws_hires_tiny/split* data/${dset}_kws_hires_tiny/.backup
        # rm -r data/${dset}_kws_hires_tiny/data
        python3 local/replace_non_kw.py --keyword $keyword --text_file data/${dset}_hires_tiny/text --output_file data/${dset}_kws_hires_tiny/text --text
        python3 local/replace_non_kw.py --keyword $keyword --text_file data/${dset}_hires_tiny/stm --output_file data/${dset}_kws_hires_tiny/stm --stm
        cp exp/chain_cleaned_1d/tdnn1d_sp/final.mdl data/${dset}_kws_hires_tiny/        
    done
fi


# Feature extraction
echo "stage 102"
if [ $stage -le 102 ]; then
  for set in test_kws_hires_tiny; do
    datadir=data/$set
    steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc_hires.conf --nj 2 --cmd "$train_cmd" $datadir
    steps/compute_cmvn_stats.sh $datadir

    # The matrix-sum was required because the old global_cmvn.scp file was not working with some error
    matrix-sum --binary=false scp:$datadir/cmvn.scp - > $datadir/global_cmvn.stats 2>/dev/null;
  done

fi


results_dir=$dir/results


#------------------- 08 layers --------------------------

# echo "stage 350"
# if [ $stage -le 350 ]; then
#   # Create a new model with one more layer removed (only 08 hidden layers).

#   echo "Configuring model with only 08 hidden tdnnf layers"

#   echo "component-node name=prefinal-l component=prefinal-l input=tdnnf9.noop" > $dir/config.08
#   nnet3-am-copy --nnet-config=${dir}/config.08 --edits=remove-orphans \
#     $dir/final_orig.mdl $dir/final_08_input.mdl
# fi


# echo "stage 351"
# if [ $stage -le 351 ]; then
#   steps/nnet3/chain/train.py --stage $train_stage \
#     --cmd "$decode_cmd" \
#     --use-gpu=wait \
#     --feat.online-ivector-dir $train_ivector_dir \
#     --feat.cmvn-opts="--config=conf/online_cmvn.conf" \
#     --chain.xent-regularize $xent_regularize \
#     --chain.leaky-hmm-coefficient 0.1 \
#     --chain.l2-regularize 0.0 \
#     --chain.apply-deriv-weights false \
#     --chain.lm-opts="--num-extra-lm-states=2000" \
#     --trainer.dropout-schedule $dropout_schedule \
#     --trainer.add-option="--optimization.memory-compression-level=2" \
#     --egs.stage=5 \
#     --egs.dir "$common_egs_dir" \
#     --egs.opts "--frames-overlap-per-eg 0 --constrained false --online-cmvn $online_cmvn" \
#     --egs.chunk-width 150,110,100 \
#     --trainer.num-chunk-per-minibatch 64 \
#     --trainer.frames-per-iter 5000000 \
#     --trainer.num-epochs 1 \
#     --trainer.optimization.num-jobs-initial 2 \
#     --trainer.optimization.num-jobs-final 2 \
#     --trainer.optimization.initial-effective-lrate 0.00025 \
#     --trainer.optimization.final-effective-lrate 0.000025 \
#     --trainer.max-param-change 2.0 \
#     --trainer.input-model $dir/final_08_input.mdl \
#     --cleanup.remove-egs $remove_egs \
#     --feat-dir $kws_train_data_dir \
#     --tree-dir $tree_dir \
#     --lat-dir $lat_dir \
#     --dir $dir
# fi

# echo "stage 352"
# if [ $stage -le 352 ]; then
#   if [ -d $dir/graph_kws_retrain_08 ]; then
#     rm -r $dir/graph_kws_retrain_08
#   fi

#   cp -r $dir/graph_kws $dir/graph_kws_retrain_08
#   cp data/lang_kws/phones.txt $dir/phones.txt
# fi

echo "stage 353"
if [ $stage -le 353 ]; then

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
  for dset in test; do
      (      
      cp data/${dset}_kws_hires_tiny/global_cmvn.stats $dir/global_cmvn.stats;
      steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $dir/graph_kws_retrain_08 data/${dset}_kws_hires_tiny $dir/decode_kws_${dset}_retrain_08 || exit 1;

      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_kws data/lang_kws_rescore \
        data/${dset}_kws_hires_tiny ${dir}/decode_kws_${dset}_retrain_08 ${dir}/decode_kws_${dset}_retrain_08_rescore || exit 1
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

  for dset in test; do

    echo $cur_results_dir
    lattice-to-nbest --n=10 "ark:gunzip -c  $dir/decode_kws_${dset}_retrain_08/lat.*.gz|" ark,t:${cur_results_dir}/${dset}-10.best
    nbest-to-linear ark:${cur_results_dir}/${dset}-10.best ark,t:${cur_results_dir}/${dset}-10.ali \
      ark,t:${cur_results_dir}/${dset}-10.words ark,t:${cur_results_dir}/${dset}-10.lmscore \
      ark,t:${cur_results_dir}/${dset}-10.acscore
    utils/int2sym.pl -f 2- exp/chain_cleaned_1d/tdnn1d_sp/graph_kws_retrain_08/words.txt ${cur_results_dir}/${dset}-10.words > ${cur_results_dir}/${dset}-10-decoded.txt
    echo "Placed decoded words in ${cur_results_dir}/${dset}-10-decoded.txt"

    python3 local/confusion.py --reference data/test_hires_tiny/text \
      --hypothesis ${results_dir}/retrain_08_layer/${dset}-10-decoded.txt --keyword $keyword \
      --alignment-tol 3 --best-n 5 | tee ${cur_results_dir}/${dset}-10-confusion.txt
    echo "Also placed the confusion matrix into ${cur_results_dir}/${dset}-10-confusion.txt"

  done
fi


echo "stage 357"
if [ $stage -le 357 ]; then
  ./steps/get_ctm.sh data/test_kws_hires_tiny/ data/lang_kws exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_test_retrain_08

  # Generate tiny reference files
  for i in {1..20}; do
    python3 local/tiny_ctm.py --input-ctm $dir/decode_test/score_${i}/test.ctm --output-ctm $dir/decode_test/score_${i}/test_tiny.ctm
    # head -25 $dir/decode_test/score_${i}/test.ctm > $dir/decode_test/score_${i}/test_tiny.ctm
  done
fi

echo "stage 358"
if [ $stage -le 358 ]; then
  for dset in test; do
    if [ -f $dir/decode_kws_${dset}_retrain_08/kws_tiny_results.csv ]; then
      rm $dir/decode_kws_${dset}_retrain_08/kws_tiny_results.csv
    fi
    echo "Scoring the ${dset} set in $dir/decode_kws_${dset}_retrain_08/"

    for i in {1..20}; do
      python3 local/analyze.py --keyword $keyword \
        --hypothesis $dir/decode_kws_${dset}_retrain_08/score_${i}/${dset}_kws_hires_tiny.ctm \
        --reference $dir/decode_${dset}/score_${i}/${dset}_tiny.ctm \
        --lm-level $i --num-layers 8 --retrained --tolerance 1 | tee -a $dir/decode_kws_${dset}_retrain_08/kws_tiny_results.csv
    done
    echo "Wrote results to $dir/decode_kws_${dset}_retrain_08/kws_tiny_results.csv"
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
