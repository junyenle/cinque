# main code file for experiment 2: clustering with SVD and interaction matrix
import torch
import pickle
from gensim.models import FastText as ft
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs
from collections import Counter

class Model:
    def __init__(self):
        self.mode = "TRAIN"
        self.train_filename = "/mnt/d/train.txt"
        self.test_filename = "/mnt/c/Users/Jun/cinque/l101/data/test.txt"
        self.wmodel_filename = "wmodelbatchregfullpoint1"
        self.num_clusters = 100
        self.kmleft = KMeans(init='k-means++', n_clusters=self.num_clusters, n_init=10)
        self.kmright = KMeans(init='k-means++', n_clusters=self.num_clusters, n_init=10)
                      
        self.debug_print_interval = 100
        self.top_count = 1000
        
        # load pretrained w
        print("loading w")
        self.w = self.load_wmodel()
        # load pretrained fasttext model
        print("loading fasttext model")
        self.ftmodel = load_facebook_vectors(datapath("/mnt/d/cc.en.300.bin/cc.en.300.bin"))
    def get_top_k_adjs(self, k):
        c = Counter()
        file = open(self.test_filename, "r", encoding = "Latin-1")
        for i, line in enumerate(file):
            linearr = line.split()
            if len(linearr) < 3 or len(linearr) > 7:
                continue
            adjs = linearr[1:]
            for adj in adjs:
                c[adj] += 1
        file.close()
        topwords = [word for word,cnt in c.most_common(k)]
        return topwords
    def save_cmodel(self):
        file = open("leftmodel", "wb")
        pickle.dump(self.kmleft, file)
        file.close()
        file = open("rightmodel", "wb")
        pickle.dump(self.kmright, file)
        file.close()
    def load_wmodel(self):
        """ load model from pickle file """
        file = open(self.wmodel_filename, "rb")
        model = pickle.load(file)
        file.close()
        return model
    def embedding(self, word):
        """ returns embedding of word as a torch tensor """
        return torch.FloatTensor(self.ftmodel.wv[word])
    def execute(self):
        # compute u, sqrt(s), v_t with svd
        print("computing svd")
        u, s, v = torch.svd(self.w, some=True, compute_uv=True)
        v_t = torch.t(v)
        s = torch.diag(s)
        sqrt_s = torch.sqrt(s)
        # get top x words
        print("getting top {} words".format(self.top_count))
        adjs = self.get_top_k_adjs(self.top_count)
        print(adjs)
        inputcount = 0
        batchcount = 0
        # create left and right embeddings for top x words
        left = []
        right = []
        for j, adj in enumerate(adjs):
            if j % self.debug_print_interval == 0:
                print("generating embeddings: {}/{}".format(j, self.top_count))
            # add each pair of adjectives to the batches
            eleft_t = torch.matmul(torch.matmul(torch.t(self.embedding(adj)), u), sqrt_s)
            eright = torch.matmul(torch.matmul(sqrt_s, v_t), self.embedding(adj))
            if torch.sum(eright).item() == 0 or torch.sum(eleft_t).item() == 0:
                continue # bad input
            left.append(eleft_t.numpy())
            right.append(eright.numpy())
                
        # train kmleft and kmright
        print("training k means")
        self.kmleft.fit(left)
        self.kmright.fit(right)

        print("saving model")
        self.save_cmodel()
        if self.mode == "APPLY":
            donothing = 1
            # compute pairwise dot products of centroids
        
if __name__ == "__main__":
    model = Model()
    model.execute()
    