# main code file for experiment 2: clustering with SVD and interaction matrix
from scipy.spatial.distance import cdist
import torch
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
        self.mode = "TRAIN"
        self.train_filename = "/mnt/d/train.txt"
        self.test_filename = "/mnt/c/Users/Jun/cinque/l101/data/test.txt"
        self.wmodel_filename = "wmodelbatchregfullpoint1"
        self.num_clusters = 30
        self.num_best_clusters = int((self.num_clusters ** 2) / 10) # top 10% of cluster pairs
        self.kmleft = KMeans(init='k-means++', n_clusters=self.num_clusters, n_init=10)
        self.kmright = KMeans(init='k-means++', n_clusters=self.num_clusters, n_init=10)
        self.debug_print_interval = 100
        self.top_count = 1000
        self.badadjs = ["a", "few"]
        
        # load pretrained w
        print("loading w")
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
                # check if adjective is actually an adjective
                if len(wn.synsets(adj, pos='a')) > 0:
                    c[adj] += 1
                    # print("CORRECT")
                # else:
                    # print("WRONG WRONG WRONG WRONG WRONG")
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
    def load_cmodel(self):
        file = open("leftmodel", "wb")
        self.kmleft = pickle.load(file)
        file.close()
        file = open("rightmodel", "wb")
        self.kmright = pickle.load(file)
        file.close()
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
        self.w = pickle.load(file)
        file.close()
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
        top_adjs = self.get_top_k_adjs(self.top_count)
        print(top_adjs)
        inputcount = 0
        batchcount = 0
        # create left and right embeddings for top x words
        left = []
        right = []
        adjectives = [] # keeps track of actual adjectives we have embeddings for, in the same order (so we can use indexing to map them)
        for j, adj in enumerate(top_adjs):
            if j % self.debug_print_interval == 0:
                print("generating embeddings: {}/{}".format(j, self.top_count))
            # check for bad input. badadjs is manually constructed to alleviate problems in the data set
                # for example, "a" is a determiner, not an adjective... but wacky tags it as an adjective, leading to bad data
            if adj in self.badadjs or len(adj) < 3:
                continue
            # compute adjective's left and right embeddings
            eleft_t = torch.matmul(torch.matmul(torch.t(self.embedding(adj)), u), sqrt_s)
            eright = torch.matmul(torch.matmul(sqrt_s, v_t), self.embedding(adj))
            # check for bad input
            if torch.sum(eright).item() == 0 or torch.sum(eleft_t).item() == 0:
                continue
            # add adjective to words list
            adjectives.append(adj)
            # add each adjective's left and right embeddings to the batches
            left.append(eleft_t.numpy())
            right.append(eright.numpy())
                
        # train kmleft and kmright. get clusters for each embedding
        # print("kmeans test")
        # for k in range(2, 50):
            # kml = KMeans(init='k-means++', n_clusters=k, n_init=10)
            # kml.fit(left)
            # kmr = KMeans(init='k-means++', n_clusters=k, n_init=10)
            # kmr.fit(right)
            # print("{}: {} / {}".format(k, np.average(np.min(cdist(left, kml.cluster_centers_, 'euclidean'), axis=1)), np.average(np.min(cdist(right, kmr.cluster_centers_, 'euclidean'), axis=1))))
        print("training kmeans")
        left_embeddings_to_clusters = self.kmleft.fit_predict(left).tolist()
        right_embeddings_to_clusters = self.kmright.fit_predict(right).tolist()
        # put adjs into cluster->adjs maps (note this uses the indexing established earlier to look-up the adjs by its index)
        left_clusters_to_words = {}
        right_clusters_to_words = {}
        for l, left_cluster in enumerate(left_embeddings_to_clusters):
            # add empty list if the key is new
            if left_cluster not in left_clusters_to_words:
                left_clusters_to_words[left_cluster] = []
            # add word to the cluster's list
            left_clusters_to_words[left_cluster].append(adjectives[l])
        for r, right_cluster in enumerate(right_embeddings_to_clusters):
            # add empty list if the key is new
            if right_cluster not in right_clusters_to_words:
                right_clusters_to_words[right_cluster] = []
            # add word to the cluster's list
            right_clusters_to_words[right_cluster].append(adjectives[r])
        # compute pairwise dot products of centroids
        for cluster in left_clusters_to_words:
            print("{}: {}".format(cluster, left_clusters_to_words[cluster]))
        for cluster in right_clusters_to_words:
            print("{}: {}".format(cluster, right_clusters_to_words[cluster]))
        print("clustering")
        left_centroids = self.kmleft.cluster_centers_
        right_centroids = self.kmright.cluster_centers_
        # compute pairwise dot products between centroids
        print("computing pairwise dot products")
        dot_products = Counter()
        for l, left_centroid in enumerate(left_centroids):
            for r, right_centroid in enumerate(right_centroids):
                dot_products[(l, r)] = self.dot_product(left_centroid, right_centroid)
        # pick top dot products
        print("picking {} top dot products".format(self.num_best_clusters))
        # list words in the top clusters
        ofile = open("clusterout.txt", "w+")
        for cluster_combination in dot_products.most_common(self.num_best_clusters):
            left_cluster = cluster_combination[0][0]
            right_cluster = cluster_combination[0][1]
            ofile.write("\ncluster: {}".format(cluster_combination[0]))
            ofile.write("\nscore: {}".format(cluster_combination[1]))
            ofile.write("\nleft words: {}".format(left_clusters_to_words[left_cluster]))
            ofile.write("\nright words: {}\n".format(right_clusters_to_words[right_cluster]))
        
if __name__ == "__main__":
    model = Model()
    model.execute()
    