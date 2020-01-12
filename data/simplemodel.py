from collections import Counter
from itertools import count
import numpy as np

TEST_LIMIT = 20000
NEGATIVE = -1
POSITIVE = 1
class PE:
    def __init__(self, element, direction):
        assert direction in (NEGATIVE, POSITIVE)
        self.element = element
        self.direction = direction

def get_permutations(list):
    numelem = len(list)
    permutation = [PE(element, NEGATIVE) for element in range(0, numelem)]
    toret = []
    for n in count(1):
        permlist = []
        for e in permutation:
            permlist.append(list[e.element])
        toret.append(permlist)
        mobile = None
        for i, e in enumerate(permutation):
            j = i + e.direction
            if (0 <= j < len(permutation)
                and e.element > permutation[j].element
                and (mobile is None or e.element > mobile_element)):
                mobile = i
                mobile_element = permutation[mobile].element
        if mobile is None:
            break
        for e in permutation:
            if e.element > mobile_element:
                e.direction *= -1
        i = mobile
        j = i + permutation[i].direction
        assert 0 <= j < len(permutation)
        permutation[i], permutation[j] = permutation[j], permutation[i]
    return toret
        
def score(adjs):
    """ sum of pairwise scores from probs """
    score = 0
    for i, adj in enumerate(adjs):
        if i == 0:
            continue
        else:
            score += probs[(adjs[i-1],adj)]
    return score
    
# open seen file
# seen = set()
# seenfile = open("vocabulary.txt", "r")
# for line in seenfile:
    # seen.add(line.strip())
    
# open probabilities file // TODO change to train file
probs = Counter()
probfile = open("probs.txt", "r", encoding="Latin-1")
for line in probfile:
    linearr = line.split()
    if len(linearr) > 3:
        continue
    probs[(linearr[0],linearr[1])] = float(linearr[2])

# open test file
goodcount = 0
badcount = 0
ocount = 0
loglikesum = 0
weirdcount = 0
nonecount = 0
testfile = open("test.txt", "r", encoding="Latin-1")
for linenum, line in enumerate(testfile):
    if linenum == TEST_LIMIT:
        break
    loglikescore = 0
    adjs = line.split()[1:]
    if len(adjs) < 2:
        weirdcount += 1
        continue
    if len(adjs) > 6:
        continue
    bestscore = score(adjs)
    loglikescore += bestscore
    changed = False
    neglogsum = 0
    for i, list in enumerate(get_permutations(adjs)):
        permscore = score(list)
        neglogsum += np.exp(permscore)
        if i == 0:
            continue
        if permscore >= bestscore:
            changed = True
            bestscore = permscore
    if bestscore == 0:
        ocount += 1
    loglikescore -= np.log(neglogsum)
    loglikesum += loglikescore
    if changed == False:
        goodcount += 1
    else:
        badcount += 1
print("log likelihood: {}".format(loglikesum))
print("Accuracy: {}".format(goodcount / (goodcount + badcount)))
print("BAD: {}".format(badcount))
print("GOOD: {}".format(goodcount))
print("0 count: {}".format(ocount))
print("Wrong format: {}".format(weirdcount))
print("NONERS: {}".format(nonecount))
       

# test = []
# test.append("big")
# test.append("happy")
# test.append("fat")
# for permutation in get_permutations(test):
    # print(permutation)
    # print(score(permutation))
