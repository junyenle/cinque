import torch
import pickle
from gensim.models import FastText as ft
from gensim.test.utils import get_tmpfile
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors
from utils import PE, get_permutations

def save_model(filename, model):
    file = open(filename, "wb")
    pickle.dump(model, file)
    file.close()
    
def load_model(filename):
    file = open(filename, "rb")
    model = pickle.load(file)
    file.close()
    return model

def get_permutations_of_embeddings(line, fbmodel):    
    """ input: string of words in form noun adj adj adj....
        output: list of lists, each inner list is a list of fasttext word embeddings
        basically, "give me the list of word embeddings for each possible permutation, and let the first permutation be the correct one" """
    #print(line)
    linearr = line.split()
    adjs = linearr[1:]
    permutations = get_permutations(adjs)
    correctpermutation = permutations[0]

    # fill an array with arrays of floats, each an embedding
    # index 0 contains the correct embedding
    allpermutationembeddings = []
    for permutation in permutations:
        permutationembedding = []
        for adj in permutation:
            permutationembedding.append(fbmodel.wv[adj])
        allpermutationembeddings.append(permutationembedding)

    return allpermutationembeddings
DEBUGPRINTINTERVAL = 1000 # how often to print where we are
DEBUGLIMIT = 1000000 # number of inputs to consider
TRAIN = False
MODELFILE = "wmodel"
SAVEMODEL = True
TEST = True
EPOCHS = 3
LRCONFIG = 0.01 # learning rate
LRDECAYRATE = 0.1

# load fbmodel (fasttext)
print("loading fb model")
path = datapath("/mnt/d/cc.en.300.bin/cc.en.300.bin")
fbmodel = load_facebook_vectors(path)

# initialize W matrix
torch.manual_seed(0)
w = torch.randn((300, 300), requires_grad=True)

if TRAIN:
    print("TRAINING MODE")
    for epoch in range(EPOCHS):
        # set the LR
        LR = LRCONFIG * (LRDECAYRATE ** epoch)
        
        tfile = open("/mnt/d/train.txt", "r")
        # for each train input line
        for ti, line in enumerate(tfile):
            # with torch.autograd.detect_anomaly(): # catching nans - TODO: remove when finished debugging
                if ti == DEBUGLIMIT: # end early
                    break
                # if ti < 126400:
                    # continue
                if ti % DEBUGPRINTINTERVAL == 0: # print progress
                    print("epoch = {} ti = {}".format(epoch, ti))
                    
                # check for bad input
                numwords = len(line.split())
                if numwords > 7 or numwords < 3:
                    continue
                # get permutations
                permutationembeddings = get_permutations_of_embeddings(line, fbmodel)
                
                # variables for computing negative log likelihood
                negativell = torch.tensor(0., requires_grad = True)
                scores = []
                
                # for each permutation, compute score
                for pei, permutationembedding in enumerate(permutationembeddings):
                    # correct embedding from train file
                    if pei == 0:
                        score = torch.tensor(0., requires_grad = True)
                        for adji, adjembedding in enumerate(permutationembedding):
                            if adji == 0:
                                continue
                            else:
                                # A
                                adj1e = torch.FloatTensor(permutationembedding[adji - 1])
                                # B
                                adj2e = torch.FloatTensor(adjembedding)
                                # A W B
                                score = score + torch.matmul(torch.matmul(torch.t(adj1e), w), adj2e)
                        negativell = negativell - score # the first part of negative log likelihood
                        scores.append(score) # the second part of negative log likelihood
                    # incorrect embeddings
                    else:
                        score = torch.tensor(0., requires_grad = True)
                        for adji, adjembedding in enumerate(permutationembedding):
                            if adji == 0:
                                continue
                            else:
                                # A
                                adj1e = torch.FloatTensor(permutationembedding[adji - 1])
                                # B
                                adj2e = torch.FloatTensor(adjembedding)
                                # A W B
                                score = score + torch.matmul(torch.matmul(torch.t(adj1e), w), adj2e)
                        scores.append(score) # the second part of negative log likelihood
                
                # compute negative log likelihood
                negativelldenominator = torch.logsumexp(torch.FloatTensor(scores), 0)
                negativell = negativell + negativelldenominator
                if ti % DEBUGPRINTINTERVAL == 0:
                    print("current loss is {}".format(negativell.item()))
                # backward pass and gradient updates
                negativell.backward()
                with torch.no_grad():
                    w = w - LR * w.grad
                # reset requires_grad for the next iteration
                w.requires_grad = True
        tfile.close()
    if SAVEMODEL:
        save_model(MODELFILE, w)
        print(w)
            
if TEST:
    print("TESTING MODE")
    # variables for computing accuracy
    correct = 0
    incorrect = 0
    # variable for computing sum negative log likelihood
    sumnll = 0
    consideredcases = 0
    
    # load w from file
    print("loading trained W matrix")
    w = load_model(MODELFILE)
    w.requires_grad = False
    
    tfile = open("../data/test.txt", "r")
    # for each test input line
    for ti, line in enumerate(tfile):
        if ti == DEBUGLIMIT: # end early
            break
        if ti % DEBUGPRINTINTERVAL == 0: # print progress
            print("ti = {}".format(ti))
                            
        # check for bad input
        numwords = len(line.split())
        if numwords > 7 or numwords < 3:
            continue
        
        # increase our considered counter
        consideredcases += 1
            
        # variables for picking the best score
        bestscore = -10000
        correctscore = -10000
        wrong = False
        
        # get permutations
        permutationembeddings = get_permutations_of_embeddings(line, fbmodel)
        
        # variables for computing negative log likelihood
        negativell = torch.tensor(0)
        scores = [] # negativelldenominator = torch.tensor(0)
        
        # for each permutation, compute score
        for pei, permutationembedding in enumerate(permutationembeddings):
            # correct embedding from test file
            if pei == 0:
                score = torch.tensor(0)
                for adji, adjembedding in enumerate(permutationembedding):
                    if adji == 0:
                        continue
                    else:
                        # A
                        adj1e = torch.FloatTensor(permutationembedding[adji - 1])
                        # B
                        adj2e = torch.FloatTensor(adjembedding)
                        # A W B
                        score = score + torch.matmul(torch.matmul(torch.t(adj1e), w), adj2e)
                negativell = negativell - score # the first part of negative log likelihood
                scores.append(score) # the first part of negative log likelihood
                
                # score updating
                rawscore = score.item()
                correctscore = rawscore
                bestscore = rawscore
                
            # incorrect embeddings from test file
            else:
                score = torch.tensor(0)
                for adji, adjembedding in enumerate(permutationembedding):
                    if adji == 0:
                        continue
                    else:
                        # A
                        adj1e = torch.FloatTensor(permutationembedding[adji - 1])
                        # B
                        adj2e = torch.FloatTensor(adjembedding)
                        # A W B
                        score = score + torch.matmul(torch.matmul(torch.t(adj1e), w), adj2e)
                        
                # score updating
                rawscore = score.item()
                scores.append(score)
                if not wrong: # TODO: this does not allow us to print the highest scoring permutation
                    if rawscore >= bestscore:
                        bestscore = rawscore
                        incorrect += 1
                        wrong = True
        # compute negative log likelihood
        negativelldenominator = torch.logsumexp(torch.FloatTensor(scores), 0)
        negativell = negativell + negativelldenominator
        # update sum negative log likelihood
        sumnll += negativell.item()
            
        # did we pick the best score?
        if not wrong:
            correct += 1 
    print("Accuracy: {}".format(correct / (correct + incorrect)))
    print("Average Negative Log Likelihood: {} ({}/{})".format(sumnll / consideredcases, sumnll, consideredcases))
    