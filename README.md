-------------------------------------------------------------------------

L101 Project Reproduction Instructions
Jun Yen Leung
jyl45@cam.ac.uk
DOWNLOAD THIS FILE TO READ IT PROPERLY - GitHub README formatting is awful

-------------------------------------------------------------------------

PREREQUISITE DOWNLOADS:
fasttext model trained on English with vector length of 300
    you can get this on the fasttext website
    it is too large for me to upload (9+gb compressed)
my data files 
    download from: https://drive.google.com/file/d/1IsoyvyEEDGhxlO3zI6QCVigRwTcp524J/view?usp=sharing
    extract and place files in the data folder
ukwac corpus
    you can skip this if you're using my data files
    to get it you will need to contact wacky@sslmit.unibo.it and ask them for it (I am ethically bound not to redistribute it)
    
NOTES:
to run these scripts you will need a LOT of disk space and about 16GB of RAM (maybe a little more)
you may have to place files in different drives (according to the space you have available) and correct data paths 
the scripts may seem to hang while loading the fasttext model
    don't worry, it just takes really long
    but if your computer is freezing, it means you don't have enough RAM

-------------------------------------------------------------------------    
    
    
    
    
    
    
-------------------------------------------------------------------------

THE BRIEF INSTRUCTIONS (no configuration, not recommended) (will not get you the same results because parameters will be different from the paper)
(but good if you just want to verify that it works)

-------------------------------------------------------------------------

1. model 4.1
cd data
python3 simplemodel.py
2. model 4.2
python3 genfasttextvectors.py (set fasttext model filepath on line 19)
cd ../nn
python3 dnn.py (set fasttext model filepath on line 399)
3. model 4.2
cd ../wmodels
python3 wmodel.py  (set fasttext model filepath on line 35)
4. experiment 5.1
open wmodel.py and set self.holdout to True (line 31), change test input file (line 26) to "../data/testheld.txt"
python3 wmodel.py
5. experiment 5.2
python3 clustering.py (set fasttext model filepath on line 34)
6. experiment 5.3
python3 latentmodel.py (set fasttext model filepath on line 41)
python3 clusteringlatent.py (set fasttext model filepath on line 29)

-------------------------------------------------------------------------






-------------------------------------------------------------------------

THE FULL INSTRUCTIONS (configuration, full steps)

-------------------------------------------------------------------------

STEP ONE: preparing the data
the outputs of this step (data.txt, datawn.txt, train.txt, valid.txt, test.txt) are PROVIDED. you DO NOT have to run them and to do so would be a waste of time 

FROM THE ROOT FOLDER
cd data
open data/dataparse.py and set the path to the UKWAC corpus on line 25
    also set lines 10 and 11 to do partial corpus extraction (it's really big so you might run out of space otherwise)
    this also allows you to use just a subset of the data
run dataparse.py
open datacombine.py and set the range of files from the previous step you want to combine into your data file (line 3)
run datacombine.py
run datafixer.py
run datasplitter.py

-------------------------------------------------------------------------

STEP TWO: running the closed form counting model (model 4.1)

FROM THE ROOT FOLDER
cd data
open gentrainprobs.py and set the training limit on line 7 (number of inputs to train on)
run gentrainprobs.py - this outputs probs.txt, which is provided (so you can skip up to this line for this step)
open simplemodel.py and set your test input limit on line 5
    if you wish (e.g. something is wrong with data paths, which will only happen if you messed something up), you can change file inputs on lines 60 and 74
run simplemodel.py

-------------------------------------------------------------------------

STEP THREE: running the feedforward neural network model (model 4.2)

FROM THE ROOT FOLDER
cd data
python3 genfasttextvectors.py (set fasttext model filepath on line 19)
cd ../nn
open dnn.py and set the training limit with the DEBUGLIMIT variable on line 13
    you can also configure additional parameters at the end of the file
run dnn.py

-------------------------------------------------------------------------

STEP FOUR: running the interpretable W matrix model (model 4.3)

FROM THE ROOT FOLDER
cd wmodels
open wmodel.py and set parameters in the init function on line 13
    importantly, set the fasttext model path on line 35
    self.mode can be set to "TRAIN" or "TEST" or "BOTH" (self explanatory)
run wmodel.py

-------------------------------------------------------------------------

STEP FIVE: running the heldout experiment (experiment 5.1)

FROM THE ROOT FOLDER
cd data
choose held out words in holdout.txt (already done)
generate held out tests with genheldtest.py (already done)
cd ../wmodels
open wmodel.py and set self.holdout on line 31 to True
uncomment line 27 and comment line 26 (changing test file to the heldout file)
run wmodel.py

-------------------------------------------------------------------------

STEP SIX: running initial clustering experiment (experiment 5.2)
(you may want to do this before STEP FIVE, so you have a model trained on all the data)

FROM THE ROOT FOLDER
cd wmodels
train the interpretable W matrix model (STEP FOUR)
open clustering.py and set parameters in the init function (line 16, with fasttext path on line 34)
run clustering.py
inspect the result in "clusterout.txt"

-------------------------------------------------------------------------

STEP SEVEN: running the latent variable model and checking its clusters (experiment 5.3)

FROM THE ROOT FOLDER
cd wmodels
open latentmodel.py and set parameters in the init function (line 19, with fasttext path on line 41)
run latentmodel.py (WARNING: TRAINS VERY SLOWLY)
open clusteringlatent.py and set parameters in the init function (line 16, with fasttext path on line 29)
run clusteringlatent.py
inspect the printed result

-------------------------------------------------------------------------