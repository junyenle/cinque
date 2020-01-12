# main file for model 3: interaction matrix with gradient descent

import torch
import torch.nn.functional as fn
import pickle
from gensim.models import FastText as ft
from gensim.test.utils import get_tmpfile
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors
from utils import PE, get_permutations
from nltk.corpus import wordnet as wn

VERBOSE = False
def printd(str):
    if VERBOSE:
        print(str)

class Model:
        def __init__(self):
            """ initialize the model. set parameters here """
            # set hyperparameters
            self.learning_rate = 0.1
            self.learning_rate_decay = 0.1
            self.regularization_parameter = 0
            self.epochs = 1
            self.batch_size = 32
            # set remaining configuration variables
            self.mode = "BOTH"
            self.train_input_limit = 10000
            self.test_input_limit = 1000
            self.train_filename = "../data/train.txt"
            self.num_c = 15
            # self.test_filename = "../data/testheld.txt" 
            self.test_filename = "../data/test.txt"
            self.model_filename = "wmodellatent_{}".format(self.num_c)  
            self.debug_print_interval = 1000
            self.debug_batch_print_interval = 1
            
            # load pretrained fasttext model
            print("loading fasttext model")
            self.ftmodel = load_facebook_vectors(datapath("/mnt/g/ft/wiki.en.bin"))
            # load or create weight matrix
            if self.mode == "TEST":
                print("loading model")
                self.w, self.v = self.load_model()
                print(self.w)
                print(self.v)
            else:
                torch.manual_seed(0)
                self.w = torch.randn((self.num_c, self.num_c), requires_grad = True)
                self.w.retain_grad()
                self.v = torch.randn((300, self.num_c), requires_grad = True)
                self.v.retain_grad()
                
            # torch.autograd.set_detect_anomaly(True)
                
        def get_permutations_of_adjs(self, line):    
            """ input: string of words in form noun adj adj adj....
                output: list of lists, each inner list is a list of fasttext word embeddings
                basically, "give me the list of word embeddings for each possible permutation, and let the first permutation be the correct one" """
            #print(line)
            linearr = line.split()
            # remove the noun
            adjs = linearr[1:]
            # get permutations
            permutations = get_permutations(adjs)
            return permutations
    
        def save_model(self):
            """ save model to pickle file """
            file = open(self.model_filename, "wb")
            pickle.dump((self.w, self.v), file)
            file.close()
            
        def load_model(self):
            """ load model from pickle file """
            file = open(self.model_filename, "rb")
            (self.w, self.v) = pickle.load(file)
            file.close()
            return self.w, self.v
        
        
        def enumerate_cvecs(self, adjlist):
            # todo: do this a smarter way... im too tired
            cvecs = []
            num_loops = len(adjlist)
            for i in range(self.num_c):
                for j in range(self.num_c):
                    if num_loops == 2:
                        cvec = [i, j]
                        cvecs.append(cvec)
                    else:
                        for k in range(self.num_c):
                            if num_loops == 3:
                                cvec = [i, j, k]
                                cvecs.append(cvec)
                            else:
                                for l in range(self.num_c):
                                    if num_loops == 4:
                                        cvec = [i, j, k, l]
                                        cvecs.append(cvec)
                                    else:
                                        for m in range(self.num_c):
                                            if num_loops == 5:
                                                cvec = [i, j, k, l, m]
                                                cvecs.append(cvec)
                                            else:
                                                for b in range(self.num_c):
                                                    if num_loops == 6:
                                                        cvec = [i, j, k, l, m, b]
                                                        cvecs.append(cvec)
            return cvecs

        def compute_p_pi_given_cvec(self, adjlist, cvec):
            scorezero = torch.tensor(0.)
            scoresum = torch.tensor(0.)
            # print(adjlist)
            # print(cvec)
            # adjperms = get_permutations(adjlist)
            cperms = get_permutations(cvec)
            for i, cperm in enumerate(cperms):
                score = self.compute_pi_c_score(cperm)
                if i == 0:
                    scorezero = torch.exp(score)
                else:
                    scoresum = scoresum + torch.exp(score)
            # print("A: {}".format(score))
            toret = scorezero / scoresum
            # print("dont want 0 or nan toret:{}".format(toret))
            return toret
            
        def compute_p_c_given_a(self, adj, c):
            embedding = torch.FloatTensor(self.ftmodel.wv[adj])
            toret = torch.matmul(embedding, self.v)
            toret = fn.softmax(toret, dim=0)[c]
            # print(toret)
            return toret
            
        def compute_product_p_c_given_a(self, listofadjs, cvec):
            toret = torch.tensor(1.)
            for i, adj in enumerate(listofadjs):
                toret = toret * self.compute_p_c_given_a(adj, cvec[i])
            return toret
        
        def compute_p_pi_given_a(self, listofadjs):
            toret = torch.tensor(0.)
            for cvec in self.enumerate_cvecs(listofadjs):
                first = self.compute_p_pi_given_cvec(listofadjs, cvec)
                second = self.compute_product_p_c_given_a(listofadjs, cvec)
                # print(first)
                # print(second)
                score = first * second
                # print("B: {}".format(score))
                toret = toret + score
            return toret
            
        
        def compute_nll(self, scores):
            """ input: list of scores, where scores[0] is the true permutation's score
                output: negative log likelihood of those scores 
                UNUSED UNUSED UNUSED """
            # compute negative log likelihood
            negativell = torch.tensor(0.)
            negativell = negativell - scores[0]
            negativell = negativell + torch.logsumexp(torch.FloatTensor(scores), 0)
            return negativell
        
        def get_c_from_adj(self, adj):
            return torch.matmul(torch.FloatTensor(self.ftmodel.wv[adj]), self.v)
            
        def compute_pi_c_score(self, cvec):
            scoresum = torch.tensor(0.)
            # print(cvec)
            for i, c in enumerate(cvec):
                if i == 0:
                    continue
                else:
                    # print(cvec)
                    cleft = torch.zeros([1, self.num_c], dtype=torch.float32)
                    cright = torch.zeros([1, self.num_c], dtype=torch.float32)
                    # print(cleft)
                    # print(cright)
                    cleft[0][cvec[i-1]] = 1
                    cright[0][cvec[i]] = 1
                    # print(cleft)
                    # print(cright)
                    score = self.compute_c_pair_score(cleft, cright)
                    scoresum = scoresum + score
            return scoresum
       
        def compute_c_pair_score(self, c1, c2):
            return torch.matmul(torch.matmul(c1,self.w),torch.t(c2))
            
        def update_weights(self, sum_negativell):
            """ input: sum negative loglikelihood
                output: none
                other: computes loss (sum nll + L2), performs backward pass, updates W matrix """
            # apply L2 regularization 
            loss = sum_negativell + torch.sum(self.w**2) * self.regularization_parameter + torch.sum(self.v**2) * self.regularization_parameter
            # print("LOSS:{}".format(loss))
            # backward pass and update weights
            loss.backward()
            printd("SANITY CHECK:")
            printd("w grad sum: {}".format(torch.sum(self.w.grad)))
            printd("v grad sum: {}".format(torch.sum(self.v.grad)))
            with torch.no_grad():
                self.w = self.w - self.learning_rate * self.w.grad
                self.v = self.v - self.learning_rate * self.v.grad
            printd("w sum: {}".format(torch.sum(self.w)))
            printd("v sum: {}".format(torch.sum(self.v)))
            self.w.requires_grad = True
            self.w.retain_grad()
            self.v.requires_grad = True
            self.v.retain_grad()
            
        def execute(self):
            """ execute the model based on the set configurations """
            if self.mode == "TRAIN" or self.mode == "BOTH":
                print("TRAINING MODE")
                for epoch in range(self.epochs):
                    # set learning rate
                    self.learning_rate = self.learning_rate * (self.learning_rate_decay ** epoch)
                    # initialize variables for counting things
                    sum_negativell = torch.tensor(0.)
                    batchcount = 0
                    # stream inputs from file
                    tfile = open(self.train_filename, "r", encoding = "Latin-1")
                    
                    holdout = []
                    hfile = open("../data/holdout.txt", "r", encoding = "Latin-1")
                    for i, line in enumerate(hfile):
                        if i == 0:
                            continue
                        holdout.append(line.strip())
                        
                    badinput = 0
                    for i, line in enumerate(tfile):
                        if i == self.train_input_limit:
                            break
                        if i % self.debug_print_interval == 0:
                            print("epoch = {} i = {}".format(epoch, i))
                        # check for bad input
                        numwords = len(line.split())
                        if numwords > 3 or numwords < 3:
                            badinput += 1
                            continue
                            
                        permutations_of_adjs = self.get_permutations_of_adjs(line)
                        # score each permutation
                        likelihoodzero = torch.tensor(0.)
                        likelihoodsum = torch.tensor(0.)
                        for j, permutation in enumerate(permutations_of_adjs):
                            likelihood = self.compute_p_pi_given_a(permutation)
                            # print("LIKELIHOOD: SHOULD NOT BE NAN OR ZERO: {}".format(likelihood))
                            if j == 0:
                                likelihoodzero = torch.exp(likelihood)
                            likelihoodsum = likelihoodsum + torch.exp(likelihood)
                        # compute negative log likelihood (nll) and update sum negative log likelihood
                        # print(line)
                        nllcontri = torch.log(likelihoodzero / likelihoodsum)
                        if torch.isnan(nllcontri):
                            print("NLLCONTRI {}: SHOULD NOT BE NAN OR ZERO: {}".format(i, nllcontri))
                            print(nllcontri.size())
                            print("LIKELIHOODZERO: {}".format(likelihoodzero))
                            print("LIKELIHOODSUM: {}".format(likelihoodsum))
                            wait = input("PRESS ENTER TO CONTINUE.")
                        sum_negativell = sum_negativell - nllcontri
                        # print(sum_negativell)
                        # if at end of batch...
                        if (i + 1) % self.batch_size == 0:
                            # periodically print the batch's nll
                            batchcount += 1
                            if batchcount == self.debug_batch_print_interval:
                                print("batch loss: {}".format(sum_negativell.item() / self.batch_size))
                                batchcount = 0
                            # update weights and reset nll sum counter
                            self.update_weights(sum_negativell)
                            sum_negativell = torch.tensor(0.)
                    tfile.close()
                    print("BAD INPUT COUNT: {}".format(badinput))
                print("saving model")
                self.save_model()
                
            if self.mode == "TEST" or self.mode == "BOTH":
                print("TESTING MODE")
                # variables for computing accuracy and average negative log likelihood
                correct_count = 0
                incorrect_count = 0
                sum_negativell = 0
                inputcounter = 0
                # stream inputs from file
                tfile = open(self.test_filename, "r", encoding = "Latin-1")
                for i, line in enumerate(tfile):
                    if i == self.test_input_limit:
                        break
                    if i % self.debug_print_interval == 0:
                        print("testing i = {}".format(i))
                        if incorrect_count == 0:
                            acc = 1.0
                        else:
                            acc = correct_count / (correct_count + incorrect_count)
                        print("current acc: {}".format(acc))
                
                    # check for bad input
                    numwords = len(line.split())
                    if numwords > 3 or numwords < 3:
                        continue
                    
                    inputcounter += 1     

                    permutations_of_adjs = self.get_permutations_of_adjs(line)
                    # score each permutation
                    likelihoods = []
                    for permutation in permutations_of_adjs:
                        likelihoods.append(self.compute_p_pi_given_a(permutation))
                    # compute negative log likelihood (nll) and update sum negative log likelihood
                    likelihoods = torch.FloatTensor(likelihoods)
                    # print("LIKELIHOODS: {}".format(likelihoods))
                    sum_negativell = sum_negativell - torch.log(likelihoods[0] / torch.sum(likelihoods))
                    # print(sum_negativell)
                    # if at end of batch...
                               
                    
                    # check if the correct permutation scored (strictly) highest
                    correct = True
                    correctscore = likelihoods[0].item()
                    for score in likelihoods[1:]:
                        if score.item() >= correctscore:
                            correct = False
                            # print(line)
                            break
                    if correct:
                        correct_count += 1
                    else:
                        incorrect_count += 1
                print("Accuracy: {}".format(correct_count / (correct_count + incorrect_count)))
                print("Average Negative Log Likelihood: {} ({}/{})".format(sum_negativell / inputcounter, sum_negativell, inputcounter))
            
if __name__ == "__main__":
    model = Model()
    model.execute()

    