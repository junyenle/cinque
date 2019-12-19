As you will soon notice (if you're trying to find anything in this repo),
I treat Python very much as a scripting tool rather than one for engineering anything significantly complex.

As such, the style is very "one file one purpose". In order to make it possible for anyone else to use my code,
I have to write little instruction documents like this one.

--------------------------------------------------
The data folder:
There's a (very good) argument to be made that all programs exist only to transform data,
but that's another story. This folder contains all scripts that handle data processing:

dataparse.py parses the ukWaC corpus and extracts all noun phrases modified by multiple adjectives
input: ukWaC corpus
output: out1.txt -> out25.txt, one for each ukWaC corpus file
NOTE: I removed out1->out25 before pushing, because they're huge and unnecessary

datacombine.py combines out1.txt -> out25.txt
input: out1.txt -> out25.txt
output: data.txt, a combined file of all our noun phrases
NOTE: I also removed data.txt for the same reason

datasplit.py splits data into train and test sets
input: data.txt
output: train.txt, test.txt
NOTE: I have removed train.txt as it exceeds GitHub file size limit! It is available instead on Google Drive here: 
https://drive.google.com/file/d/1qbCMLK5l-HepdCe6mqFZCVzbMtRyUweJ/view?usp=sharing

gentrainprobs.py performs the counting heuristic on the entire TRAIN SET ONLY and saves adjective bigrams and their probabilities
input: train.txt
output: probs.txt
options: you can set transitivity to True if you want to assume that AB, BC implies AC

simplemodel.py predicts the permutations of adjectives in the test set, using the counting heuristic
input: test.txt, probs.txt
output: prints accuracy, log likelihood, and a bunch of other random debug stuff

miscellaneous:
probstrans.txt is the same as probs.txt, but with transitivity True
test.py is a file for writing whatever small tests i might need

--------------------------------------------------
The nn folder:
NOTE THAT THE NN IS NOT CONFIGURED TO RUN OUR PARTICULAR PROBLEM YET

dnn.py is the main file to run the NN
for now, data input is hardcoded. look for the mnist file (which was what I tested it on)

dnn_misc.py gradients, backprop, etc. utility for dnn.py

the two .sh files just run dnn.py with different options

the non-mnist json files hold accuracies and the like

the mnist json file was my test data