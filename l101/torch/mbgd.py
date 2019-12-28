# main file for model 3: interaction matrix with gradient descent

import torch
import pickle
from gensim.models import FastText as ft
from gensim.test.utils import get_tmpfile
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors
from utils import PE, get_permutations

class Model:
        def __init__(self):
            """ initialize the model. set parameters here """
            # set hyperparameters
            self.learning_rate = 0.01
            self.learning_rate_decay = 0.1
            self.regularization_parameter = 0.1
            self.epochs = 3
            self.batch_size = 32
            # set remaining configuration variables
            self.mode = "TEST"
            self.train_input_limit = 1000000
            self.test_input_limit = 100000
            self.train_filename = "/mnt/d/train.txt"
            self.test_filename = "/mnt/c/Users/Jun/cinque/l101/data/test.txt"
            self.model_filename = "wmodelbatchregfullpoint1"
            self.debug_print_interval = 1000
            self.debug_batch_print_interval = 10
            
            # load pretrained fasttext model
            print("loading fasttext model")
            self.ftmodel = load_facebook_vectors(datapath("/mnt/d/cc.en.300.bin/cc.en.300.bin"))
            # load or create weight matrix
            if self.mode == "TEST":
                print("loading model")
                self.w = self.load_model()
                print(self.w)
                self.w.requires_grad = True
            else:
                torch.manual_seed(0)
                self.w = torch.randn((300,300), requires_grad = True)
                
        def get_permutations_of_embeddings(self, line):    
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
                    permutationembedding.append(self.ftmodel.wv[adj])
                allpermutationembeddings.append(permutationembedding)
            return allpermutationembeddings
    
        def save_model(self):
            """ save model to pickle file """
            file = open(self.model_filename, "wb")
            pickle.dump(self.w, file)
            file.close()
            
        def load_model(self):
            """ load model from pickle file """
            file = open(self.model_filename, "rb")
            model = pickle.load(file)
            file.close()
            return model
        
        def compute_permutation_score(self, embeddings):
            """ input: a list of embeddings
                output: the sum of scores of consecutive embeddings """
            score = torch.tensor(0.)
            # for each pair of adjectives
            for i in range(1, len(embeddings)):
                adj1 = torch.FloatTensor(embeddings[i - 1])
                adj2 = torch.FloatTensor(embeddings[i])
                # compute A W B
                score = score + torch.matmul(torch.matmul(torch.t(adj1), self.w), adj2)
            return score
            
        def compute_nll(self, scores):
            """ input: list of scores, where scores[0] is the true permutation's score
                output: negative log likelihood of those scores """
            # compute negative log likelihood
            negativell = torch.tensor(0.)
            negativell = negativell - scores[0]
            negativell = negativell + torch.logsumexp(torch.FloatTensor(scores), 0)
            return negativell
            
        def update_weights(self, sum_negativell):
            """ input: sum negative loglikelihood
                output: none
                other: computes loss (sum nll + L2), performs backward pass, updates W matrix """
            # apply L2 regularization 
            loss = sum_negativell + torch.sum(self.w**2) * self.regularization_parameter
            # backward pass and update weights
            loss.backward()
            with torch.no_grad():
                self.w = self.w - self.learning_rate * self.w.grad
            self.w.requires_grad = True
            
        def execute(self):
            """ execute the model based on the set configurations """
            if self.mode == "TRAIN":
                print("TRAINING MODE")
                for epoch in self.epochs:
                    # set learning rate
                    self.learning_rate = self.learning_rate * (self.learning_rate_decay ** epoch)
                    # initialize variables for counting things
                    sum_negativell = torch.tensor(0.)
                    batchcounter = 0
                    # stream inputs from file
                    tfile = open(self.train_filename, "r", encoding = "Latin-1")
                    for i, line in enumerate(tfile):
                        if i == self.train_input_limit:
                            break
                        if i % self.debug_print_interval == 0:
                            print("epoch = {} i = {}".format(epoch, i))
                        # check for bad input
                        numwords = len(line.split())
                        if numwords > 7 or numwords < 3:
                            continue
                        # get permutations and translate into fasttext embeddings
                        permutations = self.get_permutations_of_embeddings(line)
                        # score each permutation
                        scores = []
                        for permutation_embedding in permutations:
                            scores.append(self.compute_permutation_score(permutation_embedding))
                        # compute negative log likelihood (nll) and update sum negative log likelihood
                        sum_negativell = sum_negativell + self.compute_nll(scores)
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
                print("saving model")
                save_model()
                print(self.w)  
                
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
                        print("i = {}".format(i))
                    # check for bad input
                    numwords = len(line.split())
                    if numwords > 7 or numwords < 3:
                        continue
                    inputcounter += 1              
                    # get permutations and translate into fasttext embeddings
                    permutations = self.get_permutations_of_embeddings(line)
                    # score each permutation
                    scores = []
                    for permutation_embedding in permutations:
                        scores.append(self.compute_permutation_score(permutation_embedding))
                    # compute negative log likelihood (nll) and update sum negative log likelihood
                    sum_negativell = sum_negativell + self.compute_nll(scores)
                    # check if the correct permutation scored (strictly) highest
                    correct = True
                    bestscore = scores[0].item()
                    for score in scores[1:]:
                        if score.item() >= bestscore:
                            correct = False
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

    