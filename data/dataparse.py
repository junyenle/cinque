from nltk.corpus import brown
import nltk
from collections import Counter
import numpy as np
from conllu import parse

POS = 0
WRD = 1
LIMIT = -1
FIRSTDOC = 22
NUMFILES = 4

def CleanWord(word):
    dontuse = {"."}
    return ''.join(letter for letter in word if letter not in dontuse).lower()

class Data:
    def __init__(self): 
        
        # parse files
        for i in range(FIRSTDOC, FIRSTDOC + NUMFILES):
            words = []
            print("processing document {}".format(i))
            # open and read file
            input = open("/mnt/d/ukwac/UKWAC-" + str(i) + ".xml", "r", encoding="Latin-1")
            output = open("out" + str(i) + ".txt", "w+")
            counter = 0
            for i, line in enumerate(input):
                larr = line.split()
                if len(larr) != 3:
                    words.append(("STOP", "STOP"))
                else:
                    words.append((larr[1], larr[2]))
                if counter == LIMIT:
                    break
                else:
                    counter += 1
            # find adj nouns
            for i, word in enumerate(words):
                noun = ""
                adjs = []
                if word[POS] == "NN" or word[POS] == "NNS":
                    j = i - 1
                    while j != -1 and words[j][POS] == "JJ":
                        adjs.append(CleanWord(words[j][WRD]))
                        j -= 1
                    if len(adjs) > 1:
                        outputline = CleanWord(word[WRD]) + "\t"
                        k = len(adjs) - 1
                        while k != 0:
                            outputline += adjs[k] + " "
                            k -= 1
                        outputline += adjs[k] + "\n"
                        output.write(outputline)
            input.close()
            output.close()
                    
        
data = Data()