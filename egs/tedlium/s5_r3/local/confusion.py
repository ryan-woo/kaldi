
import argparse
from sys import stdin


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--read-stdin")
    parser.add_argument("--hypothesis", required=False)

    # How many words away from the index of the reference we will
    # look to see if the keyword is found
    parser.add_argument("--alignment-tol", default=0, type=int)
    parser.add_argument("--best-n", default=1, type=int)
    parser.add_argument("--keyword", required=True)

    return parser.parse_args()

# Reference is exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_test/scoring/test_filt.txt
# Hypothesis is exp/chain_cleaned_1d/tdnn1d_sp/results/no_retrain_12_layer/test-10-decoded.txt


def lines_to_dict(reference_lines):

    result = {}
    for line in reference_lines:

        split = line.strip().split(" ")
        utt_id = split[0]
        reference_words = split[1:]
        result[utt_id] = reference_words
    return result

"""
If the ref word is the keyword and appears within the tolerance it is a TP
if the ref word is the <unk> and  keyword does not appear within the tolerance it is a TN
if the ref word is the keyword and keyword does appear within the tolerance it is a FP.
if the ref word is the <unk> and the <unk> does not appear within the tolerance, it is a FN


if the ref word is keyword
    if keyword appears in hyp +- tol
        TP
        replace first instance of kw in hyp +- tol with unk to avoid double counting
    else (keyword does not appear in hyp +- tol)
        FN

else (ref word is <unk>)
    
    if all hyp +- tol words are unk:
        TN
    if all hyp +- tol words are keyword:
        FP
    if keyword appears sometimes in hyp +- tol
        TN  # Give benefit of the doubt

"""

UNK = "<unk>"


# def compare_words(ref, hyp, tolerance, keyword):
#     true_positives = 0
#     true_negatives = 0
#     false_positives = 0
#     false_negatives = 0
#
#     # Some checks that may help speed up stuff in general
#     if all(map(lambda x: x==UNK, ref)) and all(map(lambda x: x==UNK, hyp)):
#         true_negatives = len(ref)
#         return true_positives, true_negatives, false_positives, false_negatives
#
#     for hyp_index in range(len(hyp)):
#
#         approx_index = round(hyp_index/len(hyp) * len(ref))
#
#         min_index = max(0, approx_index-tolerance)
#         max_index = min(len(hyp), approx_index+tolerance+1)
#         ref_search_area = ref[min_index: max_index]
#
#         hyp_word = hyp[hyp_index]
#
#         if hyp_word == keyword:
#             if hyp_word in ref_search_area:
#                 import pdb; pdb.set_trace()
#                 true_positives += 1
#
#                 # We now replace the keyword in the ref with <unk> so that it does not get counted twice
#                 ref[ref.index(hyp_word)] = UNK
#             else:
#                 false_positives += 1
#         else:
#             if all(map(lambda x: x == UNK, ref_search_area)):
#                 true_negatives += 1
#             elif all(map(lambda x: x == keyword, ref_search_area)):
#
#                 # import pdb; pdb.set_trace()
#                 false_negatives += 1
#             else:  # The unk token appears in the tolerance, so we will give the benefit of the doubt here
#                 true_negatives += 1
#
#     return true_positives, true_negatives, false_positives, false_negatives


def compare_words(ref, hyp, tolerance, keyword):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # Some checks that may help speed up stuff in general
    if all(map(lambda x: x==UNK, ref)) and all(map(lambda x: x==UNK, hyp)):
        true_negatives = len(ref)
        return true_positives, true_negatives, false_positives, false_negatives

    for ref_index in range(len(ref)):

        approx_index = round(ref_index/len(ref) * len(hyp))

        min_index = max(0, approx_index-tolerance)
        max_index = min(len(hyp), approx_index+tolerance+1)
        hyp_search_area = hyp[min_index: max_index]

        ref_word = ref[ref_index]

        if ref_word == keyword:
            if ref_word in hyp_search_area:
                true_positives += 1

                # We now replace the keyword in the hyp with <unk> so that it does not get counted twice
                index = hyp_search_area.index(ref_word) + min_index
                hyp[index] = UNK
            else:
                false_negatives += 1
        else:
            if all(map(lambda x: x == UNK, hyp_search_area)):
                true_negatives += 1
            elif all(map(lambda x: x == keyword, hyp_search_area)):

                false_positives += 1
            else:  # The unk token appears in the tolerance, so we will give the benefit of the doubt here
                true_negatives += 1

    return true_positives, true_negatives, false_positives, false_negatives


def compute_keyword_or_best_n(hypothesis_dict, best_n, utt_id, keyword):

    or_hyp = []
    for i in range(best_n):
        hyp = hypothesis_dict.get(utt_id + f"-{i+1}")
        if hyp is None:
            break

        orred = []
        for j in range(min(len(or_hyp), len(hyp))):
            if or_hyp[j] == keyword or hyp[j] == keyword:
                orred.append(keyword)
            else:
                orred.append(UNK)

        or_hyp = orred
        if len(hyp) > len(or_hyp):
            or_hyp.extend(hyp[len(or_hyp):])
    return or_hyp


def compare_utterances(reference_dict, hypothesis_dict, best_n, keyword):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for utt_id, ref in reference_dict.items():

        hyp = compute_keyword_or_best_n(hypothesis_dict, best_n, utt_id, keyword)

        if keyword in ref and keyword in hyp:
            tp += 1
        elif keyword in ref and keyword not in hyp:
            fn += 1
        elif keyword not in ref and keyword in hyp:
            fp += 1
        elif keyword not in ref and keyword not in hyp:
            fn += 1

    return tp, tn, fp, fn


def main():

    args = parse_args()

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    with open(args.hypothesis) as f:
        # This file should be 10x as long as the reference
        hypothesis_lines = f.readlines()

    with open(args.reference) as f:
        reference_lines = f.readlines()


    hypothesis_dict = lines_to_dict(hypothesis_lines)
    reference_dict = lines_to_dict(reference_lines)

    for utt_id, ref_line in reference_dict.items():

        # for i in range(args.best_n):
            # hypothesis_key = utt_id + f"-{i+1}"
        hyp_line = compute_keyword_or_best_n(hypothesis_dict, args.best_n, utt_id, args.keyword)
        # hyp_line = hypothesis_dict.get(hypothesis_key)

        tp, tn, fp, fn = compare_words(ref_line, hyp_line, args.alignment_tol, args.keyword)
        true_positives += tp
        true_negatives += tn
        false_positives += fp
        false_negatives += fn

    print("Word-based-TP: ", true_positives)
    print("Word-based-TN: ", true_negatives)
    print("Word-based-FP: ", false_positives)
    print("Word-based-FN: ", false_negatives)

    tp, tn, fp, fn = compare_utterances(reference_dict, hypothesis_dict, args.best_n, args.keyword)
    print("utt-based-TP: ", tp)
    print("utt-based-TN: ", tn)
    print("utt-based-FP: ", fp)
    print("utt-based-FN: ", fn)


if __name__ == "__main__":
    main()