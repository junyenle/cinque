from collections import Counter
adjcounter = Counter()
ifile = open("train.txt", "r")
ofile = open("probstrans.txt", "w+")
#seenfile = open("vocabulary.txt", "w+")
transitivity = True

seen = set()
# read file and count adj pairs
for line in ifile:
    adjectives = line.split()[1:]
    for i, adj in enumerate(adjectives):
        seen.add(adj)
        if i == 0:
            continue
        else:
            if transitivity:
                for j in range(0, i):
                    adjcounter[(adjectives[j], adj)] += 1
            else:
                for j in range(i-1, i):
                    adjcounter[(adjectives[j], adj)] += 1
# turn counts into probabilities
for item in adjcounter:
    origcount = adjcounter[item]
    if origcount == 0:
        continue
    else:
        reverse = (item[1],item[0])
        adjcounter[item] = origcount / (adjcounter[reverse] + origcount)
    if adjcounter[item] != 0:
        ofile.write("{} {} {}\n".format(item[0], item[1], adjcounter[item]))
        
#for adj in seen:
#    seenfile.write(adj + "\n")