# put words to hold out in holdout.txt, put tests in testheld.txt


from gensim.models import FastText as ft
from gensim.test.utils import get_tmpfile
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors

HOLDOUTFLAG = False
HOLDOUT = []
holdfile = open("holdout.txt", "r")
for i, line in enumerate(holdfile):
    if i == 0:
        continue
    HOLDOUT.append(line.strip())
    print("holding out {}".format(line.strip()))

print("loading model")
cap_path = datapath("/mnt/d/cc.en.300.bin/cc.en.300.bin")
model = load_facebook_vectors(cap_path)
# model = ft.load("test.model")
print("model loaded")

if HOLDOUTFLAG:
    print("generating training vectors")
    itrain = open("train.txt", "r")
    otrain = open("trainvecs.txt", "w+")
    for j, line in enumerate(itrain):
        if j % 1000 == 0:
            print("train on line {}".format(j))
        linearr = line.split()
        if len(linearr) < 3:
            continue # bad data
        adjs = linearr[1:]
        for i in range(1, len(adjs)):
            towrite = ""
            if adjs[i] in HOLDOUT:
                continue
            if adjs[i-1] in HOLDOUT:
                continue
            first = model.wv[adjs[i-1]]
            second = model.wv[adjs[i]]
            towrite += "1"
            for item in first:
                towrite += " {}".format(item)
            for item in second:
                towrite += " {}".format(item)
            towrite += "\n"
            otrain.write(towrite)
            towrite = ""
            towrite += "0"
            for item in second:
                towrite += " {}".format(item)
            for item in first:
                towrite += " {}".format(item)
            towrite += "\n"
            otrain.write(towrite)
    itrain.close()
    otrain.close()

else:
    print("generating training vectors")
    itrain = open("train.txt", "r")
    otrain = open("/mnt/d/trainvecs.txt", "w+")
    for j, line in enumerate(itrain):
        if j % 1000 == 0:
            print("train on line {}".format(j))
        linearr = line.split()
        if len(linearr) < 3:
            continue # bad data
        adjs = linearr[1:]
        for i in range(1, len(adjs)):
            towrite = ""
            if adjs[i] in HOLDOUT:
                continue
            if adjs[i-1] in HOLDOUT:
                continue
            first = model.wv[adjs[i-1]]
            second = model.wv[adjs[i]]
            towrite += "1"
            for item in first:
                towrite += " {}".format(item)
            for item in second:
                towrite += " {}".format(item)
            towrite += "\n"
            otrain.write(towrite)
            towrite = ""
            towrite += "0"
            for item in second:
                towrite += " {}".format(item)
            for item in first:
                towrite += " {}".format(item)
            towrite += "\n"
            otrain.write(towrite)
    itrain.close()
    otrain.close()
            
    print("generating validation vectors")
    itrain = open("valid.txt", "r")
    otrain = open("validvecs.txt", "w+")
    for j, line in enumerate(itrain):
        if j % 1000 == 0:
            print("valid on line {}".format(j))
        linearr = line.split()
        if len(linearr) < 3:
            continue # bad data
        adjs = linearr[1:]
        for i in range(1, len(adjs)):
            towrite = ""
            first = model.wv[adjs[i-1]]
            second = model.wv[adjs[i]]
            towrite += "1"
            for item in first:
                towrite += " {}".format(item)
            for item in second:
                towrite += " {}".format(item)
            towrite += "\n"
            otrain.write(towrite)
            towrite = ""
            towrite += "0"
            for item in second:
                towrite += " {}".format(item)
            for item in first:
                towrite += " {}".format(item)
            towrite += "\n"
            otrain.write(towrite)
    itrain.close()
    otrain.close()

    print("generating testing vectors")
    itrain = open("test.txt", "r")
    otrain = open("testvecs.txt", "w+")
    for j, line in enumerate(itrain):
        if j % 1000 == 0:
            print("test on line {}".format(j))
        linearr = line.split()
        if len(linearr) < 3:
            continue # bad data
        adjs = linearr[1:]
        for i in range(1, len(adjs)):
            towrite = ""
            first = model.wv[adjs[i-1]]
            second = model.wv[adjs[i]]
            towrite += "1"
            for item in first:
                towrite += " {}".format(item)
            for item in second:
                towrite += " {}".format(item)
            towrite += "\n"
            otrain.write(towrite)
            towrite = ""
            towrite += "0"
            for item in second:
                towrite += " {}".format(item)
            for item in first:
                towrite += " {}".format(item)
            towrite += "\n"
            otrain.write(towrite)
    itrain.close()
    otrain.close()

