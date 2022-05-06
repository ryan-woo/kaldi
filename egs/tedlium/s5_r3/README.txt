rsw2148 Ryan Woo
May 5, 2022

Small Vocabulary Keyword Spotting from a Large Vocabulary Automatic Speech Recognition System

===== ABSTRACT =====
Keyword Spotting systems have achieved remarkable accuracy on small vocabulary datasets. 
These are often trained directly on large corpora consisting of many examples of the keywords 
in their vocabularies. However, the small vocabulary means that the keyword spotting system 
is not configurable to a user's choice of keyword. In this paper we present a method for 
producing a small vocabulary keyword spotting system from a larger ASR system. In our 
method, the user is able to configure the keyword for the system and produce a new system 
dedicated to their keyword choice very easily. This system is evaluated on several different 
keywords. Then we follow up with tuning the audio model with a range of layers in order to 
reduce the size of the KWS system in order to see if this method is viable on small devices. 


===== Kaldi tools used =====
These tools are used and are not found in the TED-LIUM examples

- arpa2fst - used to convert an ARPA formatted language model to an FST
- matrix-sum - used to create global_cmvn.stats. The equivalent file that 
  came with the pre-trained model had an error with theirs.
- lattice-to-nbest - used to get the n best decodings from a lattice file
- nbest-to-linear - used to convert the n best decodings to transcriptions (of lexicon entries)
- int2sym.pl - used to convert the set of n best transcriptions of lexicon entries to the actual words
- steps/get_ctm.sh - used to get time stamps from each model for each word the model transcribes.
- steps/make_mfcc_pitch - used to create MFCCs with pitch features added.
- nnet3-am-copy - used to create a model with one or more final layers removed
- train.py - the argument for the input-model is added to the example from the tedlium pipeline


===== External tools used =====

** REQUIRED FOR RUNNING DEMO SCRIPT
- SRILM http://www.speech.sri.com/projects/srilm/, http://www.speech.sri.com/projects/srilm/papers/icslp2002-srilm.pdf
    - Can be installed with kaldi using `tools/extra/install_srilm.sh`

- Pandas https://pandas.pydata.org/
    - Can be installed with `pip install pandas`

** NOT REQUIRED FOR RUNNING DEMO SCRIPT (but used for analysis)

- matplotlib https://matplotlib.org/
    - Can be installed with `pip install matplotlib`

- jupyter notebook https://jupyter.org/
    - Not required to run anything - was just used for analysis.


===== Main executables =====
Please run `bash`. This should automatically source the python3 virtual environment. If it does not,
then source the python3 virtual environment with `source ~/.venv/bin/activate`.

Please cd to the `kaldi/egs/tedlium_kws/s5_r3` directory before running the scripts mentioned below.

The main script which runs everything is run_kws.sh. **This is probably not what you want to run 
for grading.** You can run this script with `run_kws.sh <keyword>`. 
For example, to run everything with the keyword `discovery`, it can be done using `run_kws.sh discovery`.
This script takes approximately 7 days to complete and requires a GPU. If you want to run this script,
please first copy `exp/chain_cleaned_1d/tdnn1d_sp/final_orig.mdl` to 
`exp/chain_cleaned_1d/tdnn1d_sp/final.mdl`, as it requires the original model from the pre-trained 
TED-LIUM example to be used to start with.

The second script which just decodes something with a trained model is `run_kws_demo.sh`. This script
does not require any arguments. It is based on run_kws.sh, but skips earlier stages that are not 
necessary to decode the model submitted. A copy of the 8-layer model can be found in 
`exp/chain_cleaned_1d/tdnn1d_sp/final_08.mdl`, which is a duplicate of 
`exp/chain_cleaned_1d/tdnn1d_sp/final.mdl` when I submitted. If `exp/chain_cleaned_1d/tdnn1d_sp/final.mdl` 
is edited in any way (for example, by replacing it with final_orig.mdl), you can safely copy 
`exp/chain_cleaned_1d/tdnn1d_sp/final_08.mdl` to `exp/chain_cleaned_1d/tdnn1d_sp/final.mdl` 
before running run_kws_demo.sh.

`run_kws_demo.sh` only runs decoding and scoring on the first 25 utterances of a single TED talk 
in the test set. This is to ensure decoding and scoring does not take longer than 15 minutes.
The `run_kws_demo.sh` script will run and score this smaller test set using a tuned 8-layer model.
Results are output to both stdout and to a file. When they are output to a file, the stdout will
tell you where they are. The complete decoding takes approximately 8 minutes.

Example output when `run_kws_demo.sh` completes:

```
Scoring the test set in exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_test_retrain_08/
True,8,1,1,169,2375,0
True,8,2,1,169,2312,0
True,8,3,1,169,2262,0
True,8,4,1,169,2221,0
True,8,5,1,169,2167,0
True,8,6,1,167,2118,0
True,8,7,1,165,2077,0
True,8,8,1,161,2028,0
True,8,9,1,146,1995,0
True,8,10,1,127,1971,0
True,8,11,1,110,1964,0
True,8,12,0,104,1929,1
True,8,13,0,100,1896,1
True,8,14,0,97,1872,1
True,8,15,0,96,1827,1
True,8,16,0,95,1797,1
True,8,17,0,95,1762,1
True,8,18,0,95,1735,1
True,8,19,0,95,1716,1
True,8,20,0,95,1691,1
Wrote results to exp/chain_cleaned_1d/tdnn1d_sp/decode_kws_test_retrain_08/kws_tiny_results.csv
```

===== Other directories to note =====

The `/home/rsw2148/kaldi/egs/tedlium_kws/s5_r3/scripts` directory contains links to all the scripts
that I wrote. You can follow them to where they live in the example directory.

I performed results analysis on my personal computer, including making charts. This was performed
by copying the files from my GCP instance to my personal computer. I have thus uploaded 
the jupyter notebook I used and copies of the data used for performing the analysis in an 
`/home/rsw2148/analysis/` directory. The jupyter notebook is contained in
`/home/rsw2148/analysis/det_roc_curve.ipynb`.

The diff of the TED-LIUM example and mine is found in the home directory at 
`/home/rsw2148/egs_tedlium_s5_r3_rsw2148.diff`

===== Data Origins =====
Data comes from the TED-LIUM dataset which is available publicly. In my case, it was downloaded 
directly from the bucket provided to us. This was the pitch-trained TED-LIUM example.


===== Note on why my project is so different from my proposal =====
My original project proposal was to work with the mini-librispeech keyword spotting example, 
add background and babble noise, and re-train the model to see how resilient the model would
be to the enhanced audio. Before and during spring break, I got the example working on my
personal computer, but Professor Beigi and I discovered in office hours that the example has
rather poor accuracy. In particular, the ATWV (in essence, the number of false positives -
the number of true positives) was below zero with the example code. Professor Beigi and I 
decided to come up with a completely new project after spring break. That was this project.
