#!/usr/bin/env bash

# Copyright 2014  Guoguo Chen
# Apache 2.0

# rsw2148
# This script is a direct copy of utils/build_const_arpa_lm.sh except we don't 
# assume the lm is gzipped. Anottated again later.


# This script reads in an Arpa format language model, and converts it into the
# ConstArpaLm format language model.

# begin configuration section
# end configuration section

[ -f path.sh ] && . ./path.sh;

. utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: "
  echo "  $0 [options] <arpa-lm-path> <old-lang-dir> <new-lang-dir>"
  echo "e.g.:"
  echo "  $0 data/local/lm/3-gram.full.arpa.gz data/lang/ data/lang_test_tgmed"
  echo "Options"
  exit 1;
fi

export LC_ALL=C

arpa_lm=$1
old_lang=$2
new_lang=$3

mkdir -p $new_lang

mkdir -p $new_lang
cp -r $old_lang/* $new_lang

unk=`cat $old_lang/oov.int`
bos=`grep "^<s>\s" $old_lang/words.txt | awk '{print $2}'`
eos=`grep "^</s>\s" $old_lang/words.txt | awk '{print $2}'`
if [[ -z $bos || -z $eos ]]; then
  echo "$0: <s> and </s> symbols are not in $old_lang/words.txt"
  exit 1
fi
if [[ -z $unk ]]; then
  echo "$0: can't find oov symbol id in $old_lang/oov.int"
  exit 1
fi


# rsw2148
# Here, we simply cat the arpa_lm because it is never gzipped in our scripts. Here is the diff
# (venv) rsw2148_columbia_edu@coms6998:~/kaldi/egs/tedlium/s5_r3$ diff local/build_const_arpa_kws_lm.sh utils/build_const_arpa_lm.sh 
# 51c51
# <   "cat $arpa_lm | utils/map_arpa_lm.pl $new_lang/words.txt|"  $new_lang/G.carpa  || exit 1;
# ---
# >   "gunzip -c $arpa_lm | utils/map_arpa_lm.pl $new_lang/words.txt|"  $new_lang/G.carpa  || exit 1;
arpa-to-const-arpa --bos-symbol=$bos \
  --eos-symbol=$eos --unk-symbol=$unk \
  "cat $arpa_lm | utils/map_arpa_lm.pl $new_lang/words.txt|"  $new_lang/G.carpa  || exit 1;

exit 0;
