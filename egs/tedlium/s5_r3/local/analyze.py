#!/usr/bin/env python
"""
rsw2148

Compare ctms from the reference dataset and the hypothesis dataset.
The ctms are compared with timestamps - the tolerance lets you add a "fudge factor" where
the hypothesis dataset can be off from the original dataset by up tolerance number of seconds.
"""
# coding: utf-8


import argparse

import pandas as pd





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--tolerance", type=float, default=2)
    parser.add_argument("--tn-tolerance", type=float, default=0.2)
    parser.add_argument("--lm-level", required=True)
    parser.add_argument("--num-layers", required=True)
    parser.add_argument("--retrained", action="store_true")

    return parser.parse_args()

def main():

    args = parse_args()

    # keyword = "discovery"
    # reference = "kaldi/egs/tedlium/s5_r3/exp/chain_cleaned_1d/tdnn1d_sp/decode_dev/score_1/dev.ctm"
    # hypothesis = "kaldi/egs/tedlium/s5_r3/exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_dev/score_1/dev_kws.ctm"
    # tolerance = 2

    keyword = args.keyword
    reference = args.reference
    hypothesis = args.hypothesis
    tolerance = args.tolerance

    col_names = ["speaker_id", "channel", "start_time", "duration", "word"]

    reference_df = pd.read_csv(reference, sep=r"\s+", names=col_names)
    hypothesis_df = pd.read_csv(hypothesis, sep=r"\s+", names=col_names)

    reference_df.head()


    # Select rows with the keyword

    ref_kw_df = reference_df.loc[reference_df.word == keyword]
    hyp_kw_df = hypothesis_df.loc[hypothesis_df.word == keyword]


    # Select rows from hypothesis within tolerance of the reference words
    tp = 0
    for index, row in ref_kw_df.iterrows():
        start_minus_tol = row.start_time - tolerance
        end_plus_tol = row.start_time + row.duration + tolerance

        # Select correct speaker
        s = hyp_kw_df.loc[hyp_kw_df.speaker_id == row.speaker_id]

        # Select correct times for that speaker
        s = s.loc[s.start_time.between(start_minus_tol, end_plus_tol)]
        if not s.empty:
            tp += 1

            # Remove to prevent double counting
            hyp_kw_df = hyp_kw_df.drop(s.head(1).index)

    tn_ref_kw_df = reference_df.loc[reference_df.word != keyword]
    tn_hyp_kw_df = hypothesis_df.loc[hypothesis_df.word != keyword]

    # Select rows from hypothesis within tolerance of the reference words
    tn = 0
    for index, row in tn_ref_kw_df.iterrows():
        start_minus_tol = row.start_time - args.tn_tolerance
        end_plus_tol = row.start_time + row.duration + args.tn_tolerance

        # Select correct speaker
        s = tn_hyp_kw_df.loc[tn_hyp_kw_df.speaker_id == row.speaker_id]

        # Select correct times for that speaker
        s = s.loc[s.start_time.between(start_minus_tol, end_plus_tol)]
        if not s.empty:
            tn += 1

            # Remove to prevent double counting
            tn_hyp_kw_df = tn_hyp_kw_df.drop(s.head(1).index)
    
    # tp = nearby_hyp_kw_df.shape[0]
    fn = ref_kw_df.shape[0] - tp
    fp = hyp_kw_df.shape[0]

    # Output with the following format:
    # <retrained>,<number_layers>,<lm_level>,<tp>,<fp>,<tn>,<fn>
    print(f"{args.retrained},{args.num_layers},{args.lm_level},{tp},{fp},{tn},{fn}")

    # print(f"lm-level {args.lm_level} True positives:", tp)
    # print(f"lm-level {args.lm_level} False positives:", fp)
    # print(f"lm-level {args.lm_level} False negatives:", fn)

if __name__ == "__main__":
    main()
