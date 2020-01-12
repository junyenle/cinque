# main file for model 2: dnn with fasttext

import pickle
import json
import numpy as np
import sys
import dnn_misc
import os
import argparse
from collections import Counter
from itertools import count

DEBUGLIMIT = 20000
TRAIN = True
TEST = False
APPLY = True
BATCHSIZE = 100
NEGATIVE = -1
POSITIVE = 1
MODELFILE = "dnn_medium_wn"
HOLDOUT = False # for training held out, make sure you change the model to save line 239
EVALHELDONLY = False # for testing held out. files: testheld.txt

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
        
def score(adjs, nnmodel, fbmodel):
    """ sum of pairwise scores from probs """
    score = 0
    for i, adj in enumerate(adjs):
        if i == 0:
            continue
        else:
            adj1vec = fbmodel.wv[adjs[i-1]]
            adj2vec = fbmodel.wv[adjs[i]]
            xin = []
            for elem in adj1vec:
                xin.append(elem)
            for elem in adj2vec:
                xin.append(elem)
            x = []
            x.append(xin)
            # print(x)
            # print(len(x))
            # print(len(x[0]))
            # print(adj1vec)
            # print(len(adj1vec))
            # print(adj2vec)
            # print(len(adj2vec))
            a2 = predict_new(x, nnmodel)
            score += a2[0][1] - a2[0][0]
    return score
    
class DataSplit:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N, self.d = self.X.shape

    def get_example(self, idx):
        batchX = np.zeros((len(idx), self.d))
        batchY = np.zeros((len(idx), 1))
        for i in range(len(idx)):
            batchX[i] = self.X[idx[i]]
            batchY[i, :] = self.Y[idx[i]]
        return batchX, batchY

def save_model(model):
    file = open(MODELFILE, "wb")
    pickle.dump(model, file)
    file.close()
    
def load_model(model):
    file = open(MODELFILE, "rb")
    model = pickle.load(file)
    file.close()
    return model
            
def data_loader(dataset):
    # This function reads the MNIST data and separate it into train, val, and test set
    
    Xtrain = []
    Ytrain = []
    Xvalid = []
    Yvalid = []
    Xtest = []
    Ytest = []
    
    if HOLDOUT:
        trainfile = open("../data/trainvecsheld2.txt", "r")
    else:
        trainfile = open("../data/trainvecs.txt", "r")
    print("processing training file")
    for i, line in enumerate(trainfile):
        if i == DEBUGLIMIT:
            break
        if i % 1000 == 0:
            print("train on {}".format(i))
        linearr = line.split()
        Xtrain.append(linearr[1:])
        Ytrain.append(linearr[0])
    trainfile.close()
    valfile = open("../data/validvecs.txt", "r")
    print("processing validation file")
    for i, line in enumerate(valfile):
        if i == DEBUGLIMIT:
            break
        if i % 100 == 0:
            print("valid on {}".format(i))
        linearr = line.split()
        Xvalid.append(linearr[1:])
        Yvalid.append(linearr[0])
    valfile.close()
    testfile = open("../data/testvecs.txt", "r")
    print("processing testing file file")
    for i, line in enumerate(testfile):
        if i == DEBUGLIMIT:
            break
        if i % 100 == 0:
            print("test on {}".format(i))
        linearr = line.split()
        Xtest.append(linearr[1:])
        Ytest.append(linearr[0])
    testfile.close()
    
    #print(Xtrain[0])
    
    return np.array(Xtrain), np.array(Ytrain), np.array(Xvalid),\
           np.array(Yvalid), np.array(Xtest), np.array(Ytest)

def predict_label(f):
    # This is a function to determine the predicted label given scores
    label = np.argmax(f, axis=1).astype(float).reshape((f.shape[0], -1))
    #print("score of {} gives us label {}".format(f, label))
    return label

def predict_new(x, model):
    a1 = model['L1'].forward(x)
    h1 = model['nonlinear1'].forward(a1)
    a2 = model['L2'].forward(h1)
    return a2
    
def get_training_example(line):        
    linearr = line.split()
    X = linearr[1:]
    xfloats = []
    for item in X:
        xfloats.append(float(item))
    xarr = []
    xarr.append(xfloats)
    Y = linearr[0]
    yfloats = []
    yfloats.append(float(Y))
    yarr = []
    yarr.append(yfloats)
    return np.array(xarr), np.array(yarr)

def train_model(model, trainlines, _lambda, _learning_rate, _alpha, _optimizer, momentum):
    idx_order = np.random.permutation(len(trainlines))
    # print(idx_order)
    for i, index in enumerate(idx_order):
        line = trainlines[index]
        if i % 100 == 0:
            print("training {} / {} \r".format(i, BATCHSIZE))
        x, y = get_training_example(line)
        ### forward ###
        a1 = model['L1'].forward(x)
        h1 = model['nonlinear1'].forward(a1)
        a2 = model['L2'].forward(h1)
        loss = model['loss'].forward(a2, y)

        ### backward ###
        grad_a2 = model['loss'].backward(a2, y)
        grad_h1 = model['L2'].backward(h1, grad_a2)
        grad_a1 = model['nonlinear1'].backward(a1, grad_h1)
        grad_x = model['L1'].backward(x, grad_a1)

        ### gradient_update ###
        for module_name, module in model.items():
            # model is a dictionary with 'L1', 'L2', 'nonLinear1' and 'loss' as keys.
            # the values for these keys are the corresponding objects created in line 123-126 using classes 
            # defined in dnn_misc.py

            # check if the module has learnable parameters. not all modules have learnable parameters.
            # if it does, the module object will have an attribute called 'params'.
            if hasattr(module, 'params'):
                for key, _ in module.params.items():
                    # gradient computed during the backward pass + L2 regularization term
                    # _lambda is the regularization hyper parameter
                    g = module.gradient[key] + _lambda * module.params[key]

                    if _optimizer == "Minibatch_Gradient_Descent":
                        module.params[key] -= _learning_rate * g

                    elif _optimizer == "Gradient_Descent_Momentum":
                        parameter = module_name + '_' + key
                        momentum[parameter] = _alpha * momentum[parameter] + _learning_rate * g
                        module.params[key] -= momentum[parameter]
    return model
    
def main(main_params):

    ### set the random seed ###
    np.random.seed(int(main_params['random_seed']))

    ### data processing ###
    #Xtrain, Ytrain, Xval, Yval , Xtest, Ytest = data_loader(dataset = 'mnist_subset.json')
    trainfile = open("/mnt/d/trainvecs.txt", "r")
    validfile = open("/mnt/d/validvecs.txt", "r")
    testfile = open("/mnt/d/testvecs.txt", "r")
    d = 600 # length of vectors

    ### building/defining MLP ###
    """
    The network structure is input --> linear --> relu --> linear --> softmax_cross_entropy loss
    the hidden_layer size (num_L1) is 1000
    the output_layer size (num_L2) is 2
    """
    model = dict()
    num_L1 = 1000
    num_L2 = 2

    # experimental setup
    num_epoch = int(main_params['num_epoch'])
    minibatch_size = int(main_params['minibatch_size'])

    # optimization setting: _alpha for momentum, _lambda for weight decay
    _learning_rate = float(main_params['learning_rate'])
    _step = 10
    _alpha = 0.0
    _lambda = float(main_params['lambda'])
    _optimizer = main_params['optim']
    _epsilon = main_params['epsilon']

    # create objects (modules) from the module classes
    model['L1'] = dnn_misc.linear_layer(input_D = d, output_D = num_L1)
    model['nonlinear1'] = dnn_misc.relu()
    model['L2'] = dnn_misc.linear_layer(input_D = num_L1, output_D = num_L2)
    model['loss'] = dnn_misc.softmax_cross_entropy()

    momentum = None
    # create variables for momentum
    if _optimizer == "Gradient_Descent_Momentum":
        # creates a dictionary that holds the value of momentum for learnable parameters
        momentum = dnn_misc.add_momentum(model)
        _alpha = 0.9

    if TRAIN:
        train_acc_record = []
        val_acc_record = []
        train_loss_record = []
        val_loss_record = []

        ### run training and validation ###
        for t in range(num_epoch):
            print('At epoch ' + str(t + 1))
            if (t % _step == 0) and (t != 0):
                # learning_rate decay
                _learning_rate = _learning_rate * 0.1

            # NOT SHUFFLING FOR NOW todo: (shuffle)
            
            # training examples one by one
            batchedtraining = []
            for i, line in enumerate(trainfile):
                if i == DEBUGLIMIT:
                    break
                if i % BATCHSIZE == 0:
                    print("TRAINING epoch {}  line {}".format(t, i))
                  
                batchedtraining.append(line)
                if len(batchedtraining) == BATCHSIZE:
                    model = train_model(model, batchedtraining, _lambda, _learning_rate, _alpha, _optimizer, momentum)
                    batchedtraining.clear()
            # remaining examples
            model = train_model(model, batchedtraining, _lambda, _learning_rate, _alpha, _optimizer, momentum)

            ### Compute validation accuracy ###
            print("computing validation accuracy")
            val_acc = 0.0
            val_loss = 0.0
            val_count = 0
            for i, line in enumerate(validfile):
                if i == DEBUGLIMIT:
                    break
                if i % 1000 == 0:
                    print("validation processing on line {} \r".format(i))
                    
                x, y = get_training_example(line)

                ### forward ###
                a1 = model['L1'].forward(x)
                h1 = model['nonlinear1'].forward(a1)
                a2 = model['L2'].forward(h1)
                loss = model['loss'].forward(a2, y)
                val_loss += len(y) * loss
                val_acc += np.sum(predict_label(a2) == y)
                val_count += len(y)

            val_loss = val_loss / val_count
            val_acc = val_acc / val_count

            val_acc_record.append(val_acc)
            val_loss_record.append(val_loss)

            print('Validation loss at epoch ' + str(t + 1) + ' is ' + str(val_loss))
            print('Validation accuracy at epoch ' + str(t + 1) + ' is ' + str(val_acc))
        
        # save model
        print("saving model")
        save_model(model)

    if TEST:
        # load model
        print("loading model")
        model = load_model(model)
        
        ### Compute test accuracy ###    
        test_loss = 0.0
        test_acc = 0.0
        test_count = 0
        ### output file ###
        outputfile = open("/mnt/d/vecscores.txt", "w+")
        # training examples one by one
        print("computing test accuracy")
        for i, line in enumerate(testfile):
            if i == DEBUGLIMIT:
                break
            if i % 100 == 0:
                print("test processing on line {} \r".format(i))
                
            x, y = get_training_example(line)

            ### forward ###
            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            a2 = model['L2'].forward(h1)
            loss = model['loss'].forward(a2, y)
            test_loss +=  len(y) * loss
            test_acc += np.sum(predict_label(a2) == y)
            test_count += len(y)
            writestring = ""
            xarr = x[0]
            for item in xarr:
                writestring += "{} ".format(item)
            writestring += "{} {}\n".format(a2[0][0], a2[0][1])
            outputfile.write(writestring)

        test_loss = test_loss / test_count
        test_acc = test_acc / test_count
        outputfile.close()

        print('Test loss is ' + str(test_loss))
        print('Test accuracy is ' + str(test_acc))
    
    if APPLY:
        # load NN model
        print("loading NN model")
        model = load_model(model)
        # load fasttext model
        from gensim.models import FastText as ft
        from gensim.test.utils import get_tmpfile
        from gensim.test.utils import datapath
        from gensim.models.fasttext import load_facebook_vectors
        print("loading fasttext model")
        cap_path = datapath("/mnt/d/cc.en.300.bin/cc.en.300.bin")
        fbmodel = load_facebook_vectors(cap_path)
        print("fasttext model loaded")
        
        goodcount = 0
        badcount = 0
        ocount = 0
        loglikesum = 0

        wrong = Counter()
        if EVALHELDONLY:
            testfile = open("../data/testheld.txt", "r", encoding = "Latin-1")
        else:
            testfile = open("../data/test.txt", "r", encoding="Latin-1")
        for linenum, line in enumerate(testfile):
            if linenum == DEBUGLIMIT:
                break
            if linenum % 100 == 0:
                print("evaluating test input {}".format(linenum))
            loglikescore = 0
            adjs = line.split()[1:]
            if len(adjs) < 2 or len(adjs) > 6:
                continue # bad data
            bestscore = score(adjs, model, fbmodel)
            loglikescore += bestscore
            changed = False
            neglogsum = 0
            if EVALHELDONLY:
                bestlist = adjs
            for i, list in enumerate(get_permutations(adjs)):
                permscore = score(list, model, fbmodel)
                neglogsum += np.exp(permscore)
                if i == 0:
                    continue
                if permscore >= bestscore:
                    changed = True
                    bestscore = permscore
                    if EVALHELDONLY:
                        bestlist = list
            if bestscore == 0:
                ocount += 1
            loglikescore -= np.log(neglogsum)
            loglikesum += loglikescore
            if changed == False:
                goodcount += 1
            else:
                badcount += 1
                if EVALHELDONLY:
                    # if "artificial" in adjs or "esoteric" in adjs or "enormous" in adjs:
                        # badcount -= 1
                        # print("WRONG: {}".format(bestlist))
                        # print("CORRECT: {}".format(adjs))
                    # else:
                        # print("WRONG: {}".format(adjs))
                        # print("CORRECT: {}".format(bestlist))
                    for adj in adjs:
                        wrong[adj] += 1
        print("average log likelihood: {}".format(loglikesum / (goodcount + badcount)))
        print("Accuracy: {}".format(goodcount / (goodcount + badcount)))
        print("BAD: {}".format(badcount))
        print("GOOD: {}".format(goodcount))
        print("0 count: {}".format(ocount))
        if EVALHELDONLY:
            print(wrong)
        

		
    
    print('finished')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=2)
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--lambda', default=0.001)
    parser.add_argument('--num_epoch', default=1)
    parser.add_argument('--minibatch_size', default=1)
    parser.add_argument('--optim', default='Minibatch_Gradient_Descent')
    parser.add_argument('--epsilon', default=0.001)
    args = parser.parse_args()
    main_params = vars(args)
    main(main_params)