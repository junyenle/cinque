from nltk.corpus import wordnet as wn

datain = open("datanown.txt", "r", encoding = "Latin-1")
dataout = open("datawn.txt", "w+")

for line in datain:
    linearr = line.split()
    if len(linearr) < 3 or len(linearr) > 7:
        continue
    badwordfound = False
    for word in linearr[1:]:
        if len(wn.synsets(word, pos='a')) == 0:
            badwordfound = True
            break
    if badwordfound:
        continue
    else:
        dataout.write("{}".format(line))