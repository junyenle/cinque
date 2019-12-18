from nltk.corpus import brown
import nltk
from collections import Counter
import numpy as np

words = (brown.words()[:10000])
tagged_words = nltk.pos_tag(words)
count = 0
adjpairs = Counter()
for i, tagged_word in enumerate(tagged_words):
    if i == 0:
        continue
    if tagged_word[1] == "JJ" and tagged_words[i-1][1] == "JJ":
        count += 1
        adjpairs[(tagged_words[i-1][0],tagged_word[0])] += 1
for pair in adjpairs:
    print(pair)
N = []