# main code file for experiment 2: clustering with SVD and interaction matrix
import torch
import torch.nn.functional as fn
import pickle
from gensim.models import FastText as ft
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs
from collections import Counter
import numpy as np
from nltk.corpus import wordnet as wn

class Model:
    def __init__(self):
        self.test_filename = "../data/test.txt"
        self.num_clusters = 15
        self.wmodel_filename = "wmodellatent_{}".format(self.num_clusters)
        self.top_count = 1000
        self.w = None
        self.v = None
        
        # load pretrained w, v
        print("loading w, v")
        self.load_wmodel()
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
                # # check if adjective is actually an adjective
                # if len(wn.synsets(adj, pos='a')) > 0:
                    # c[adj] += 1
                    # # print("CORRECT")
                # # else:
                    # # print("WRONG WRONG WRONG WRONG WRONG")
        file.close()
        topwords = [word for word,cnt in c.most_common(k)]
        return topwords
    def dot_product(self, left, right):
        """ input: left, right vectors as lists of elements
            output: dot product """
        sum = 0
        for i in range(len(left)):
            sum += left[i] * right[i]
        return sum
    def load_wmodel(self):
        """ load model from pickle file """
        file = open(self.wmodel_filename, "rb")
        (w, v) = pickle.load(file)
        self.w = w
        self.v = v
        file.close()
    def embedding(self, word):
        """ returns embedding of word as a torch tensor """
        return torch.FloatTensor(self.ftmodel.wv[word])
    def execute(self):
        # get top x words
        print("getting top {} words".format(self.top_count))
        top_adjs = self.get_top_k_adjs(self.top_count)
        # print(top_adjs)
        inputcount = 0
        batchcount = 0
        # cluster top words and map them
        # words_to_clusters = {}
        clusters = []
        for i in range(self.num_clusters):
            clusters.append([])
        print("processing top {} words".format(self.top_count))
        for i, adj in enumerate(top_adjs):
            embedding = self.embedding(adj)
            scores = torch.matmul(embedding, self.v)
            bestscore = torch.argmax(scores)
            clusters[bestscore.item()].append(adj)
        for cluster in clusters:
            print(cluster)
        # find top clusters
        print("getting top clusters")
        clusterscores = Counter()
        for i in range(self.num_clusters):
            for j in range(self.num_clusters):
                clusterscores[(i,j)] = self.w[i][j].item()
        num_to_keep = 90 #self.num_clusters * self.num_clusters
        for i, clusterpair in enumerate(clusterscores.most_common(num_to_keep)):
            print("\nRANK: {} CLUSTER: {}".format(i, clusterpair))
            print(clusters[clusterpair[0][0]])
            print(clusters[clusterpair[0][1]])
                
        
        
if __name__ == "__main__":
    model = Model()
    model.execute()
    