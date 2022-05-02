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


===== External tools used =====

** REQUIRED FOR RUNNING DEMO SCRIPT
- SRILM http://www.speech.sri.com/projects/srilm/, http://www.speech.sri.com/projects/srilm/papers/icslp2002-srilm.pdf
    - Can be installed with kaldi using `tools/extra/install_srilm.sh`

- Pandas https://pandas.pydata.org/
    - Can be installed with `pip install pandas`

** NOT REQUIRED FOR RUNNING DEMO SCRIPT, BUT USED for analysis

- matplotlib https://matplotlib.org/
    - Can be installed with `pip install matplotlib`


===== Directories and executables =====
You can run everything from `kaldi/egs/tedlium_kws/s5_r3` directory.

The main script which runs everything is run_kws.sh. You can run it with `run_kws.sh <keyword>`. 
For example, to run everything with the keyword `discovery`, it can be done using `run_kws.sh discovery`.
This script takes approximately 7 days to complete and requires a GPU. If you want to run this script,
please first copy `exp/chain_cleaned_1d/tdnn1d_sp/final_orig.mdl` to 
`exp/chain_cleaned_1d/tdnn1d_sp/final.mdl`, as it requires the original model from the pre-trained 
TED-LIUM example to be used to start with.

The second script which just decodes something with a trained model is `run_kws_demo.sh`. This script
does not require any arguments. It is based on run_kws.sh, but skips earlier stages that are not 
necessary to decode the model submitted. A copy of the 8-layer model can be found in 
`exp/chain_cleaned_1d/tdnn1d_sp/final_08.mdl`. If `exp/chain_cleaned_1d/tdnn1d_sp/final.mdl` is edited
in any way (for example, by replacing it with final_orig.mdl), you can safely copy 
`exp/chain_cleaned_1d/tdnn1d_sp/final_08.mdl` to `exp/chain_cleaned_1d/tdnn1d_sp/final.mdl` 
before running run_kws_demo.sh.

`run_kws_demo.sh` only runs decoding and scoring on a single TED talk 
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

===== Data Origins =====
Data comes from the TED-LIUM dataset which is available publicly. In our case, it was downloaded 
directly from the bucket provided to us. This was the pitch-trained TED-LIUM example.


