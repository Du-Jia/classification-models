import os
import torch
import numpy as np
import pandas as pd
import gensim
from stop_words import get_stop_words
import Data.WordVector.load_word_vector as lwv


class DataLoader():

    def __init__(self, dataset='trainset.txt'):
        self.dataroot = os.path.abspath('./dataset')
        self.stop_words = get_stop_words('en')
        self.dataset = os.path.join(self.dataroot, dataset)

    def set_dataset(self, dataset):
        self.dataset = os.path.join(self.dataroot, dataset)

    def get_average_length(self):
        cnt = 0
        with open(self.dataset, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                line = line.split()

                label = line[-1]
                Y.append([int(label)])

                sentence = line[:-1]
                sentence = [str.lower(word) for word in sentence]
                sentence = [word for word in sentence if word not in self.stop_words]
                cnt += len(sentence)
            average = cnt / len(lines)
        return average

    def creator(self, average_length=-1):
        """
        Parameters:
            average_length: int, if -1, don't clip or pad.
        """
        para = ['(', ')', '...', '.', '``', '--', '\'', ',', ':', '-', "''", '?', "\\*\\*\\*", ';']
        self.stop_words.extend(para)
        wv = lwv.load_wv('./WordVector')
        X = []
        Y = []
        # the sum of length for all sentences
        # cnt = 0
        with open(self.dataset, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                line = line.split()

                label = line[-1]
                Y.append([int(label)])

                sentence = line[:-1]
                sentence = [str.lower(word) for word in sentence]
                sentence = [word for word in sentence if word not in self.stop_words]
                # print(sentence, label)
                # cnt += len(sentence)
                word_vectors = []
                for word in sentence:
                    try:
                        vec = wv[word]
                    except :
                        vec = list(np.zeros(300))
                    word_vectors.append(vec)

                # padding and clipping
                if average_length != -1:
                    if len(word_vectors) >= average_length:  # 10 is average length of sentences
                        word_vectors = word_vectors[:average_length]
                    else:
                        for i in range(average_length-len(word_vectors)):
                            vec = list(np.zeros(300))
                            word_vectors.append(vec)

                X.append(np.array(word_vectors))

        X = np.array(X)
        Y = np.array(Y)
        # print(cnt/len(lines), trainset.shape, len(trainset[0]))

        return X, Y

    def loader(self, dataset, padding_length=-1):
        if padding_length == -1:
            path = os.path.abspath('./dataset')
            xpath = os.path.join(path, dataset+'X.npy')
            ypath = os.path.join(path, dataset+'Y.npy')
        else:
            path = os.path.abspath('./dataset')
            xpath = os.path.join(path, dataset + str(padding_length) + 'X.npy')
            ypath = os.path.join(path, dataset + str(padding_length) + 'Y.npy')

        return np.load(xpath, allow_pickle=True), np.load(ypath, allow_pickle=True)


if __name__ == '__main__':
    # Run thie phase to create all embedded sentences when you run the code first.
    dataloader = DataLoader()
    trainX, trainY = dataloader.creator(average_length=20)
    np.save('./dataset/trainX.npy', trainX)
    np.save('./dataset/trainY.npy', trainY)
    dataloader.set_dataset('testset.txt')
    testX, testY = dataloader.creator(average_length=20)
    np.save('./dataset/testX.npy', testX)
    np.save('./dataset/testY.npy', testY)
    dataloader.set_dataset('validset.txt')
    validX, validY = dataloader.creator(average_length=20)
    np.save('./dataset/validX.npy', validX)
    np.save('./dataset/validY.npy', validY)

    # test code for loader, print the shape of different sets.
    dataloader = DataLoader()
    X, Y = dataloader.loader('train')
    print(X.shape, Y.shape)
    dataloader = DataLoader()
    X, Y = dataloader.loader('test')
    print(X.shape, Y.shape)
    dataloader = DataLoader()
    X, Y = dataloader.loader('valid')
    print(X.shape, Y.shape)




