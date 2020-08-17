# encoding=utf-8
# python 3.7

import os
import sys

sstroot = os.path.abspath('./stanfordSentimentTreebank/')
sentences = os.path.join(sstroot, 'datasetSentences.txt')
spliter = os.path.join(sstroot, 'datasetSplit.txt')
taglist = os.path.join(sstroot, 'sentiment_labels.txt')
pharseids = os.path.join(sstroot, 'dictionary.txt')

datasetroot = os.path.abspath('./dataset/')
trainset = os.path.join(datasetroot, 'trainset.txt')
validset = os.path.join(datasetroot, 'validset.txt')
testset = os.path.join(datasetroot, 'testset.txt')

resroot = os.path.abspath('./results')


def sentence_split(sentences=sentences, spliter=spliter, train=trainset, valid=validset, test=testset):
    """
    Split sentences to 3 dataset.

    Parameters:
        sentences: string, the path of datasetSentences.txt
        spliter: string, the path of datasetSplit.txt
        train: string, the path of trainset
        test: string, the path of testset
        valid: string, the path of validset
    """
    with open(sentences, 'r') as senfile, open(spliter, 'r') as splfile:
        sens = senfile.readlines()
        spls = splfile.readlines()
        size = len(spls)
        
        labeled = {}
        labeled['1'] = []
        labeled['2'] = []
        labeled['3'] = []
        for index in range(1, size):
            spl = spls[index].strip()
            spl = spl.split(',')
            label = spl[1]

            id = int(spl[0])
            sentence = sens[id].split('\t')[1].strip()
            sentence = sentence.replace('-RRB-', ')')
            sentence = sentence.replace('-LRB-', '(')

            labeled[label].append(sentence)

        return labeled


def tag_sentences(splited_data, dictionary=pharseids, tags=taglist, train=trainset, test=testset, valid=validset):
    pharse2id = {}
    tags = [-1]

    with open(dictionary, 'r') as file, open(taglist, 'r') as file2:
        lines = file.readlines()
        for line in lines:
            line = line.split('|')
            pharse = line[0]
            id = int(line[1].strip())
            pharse2id[pharse] = id

        lines = file2.readlines()
        for index in range(1, len(lines)):
            line = lines[index]
            line.strip()
            prob = line.split('|')[1]
            prob = float(prob)
            if 0 <= prob < 0.2:
                label = 0
            elif  0.2 <= prob < 0.4:
                label = 1
            elif 0.4 <= prob < 0.6:
                label = 2
            elif 0.6 <= prob < 0.8:
                label = 3
            else:
                label = 4
            tags.append(label)

    train = tag_sentence(tags, pharse2id, splited_data['1'], trainset)
    test = tag_sentence(tags, pharse2id, splited_data['2'], testset)
    valid = tag_sentence(tags, pharse2id, splited_data['3'], validset)

    return train, test, valid


def tag_sentence(tags, pharse2id, sentences, dataset):
    with open(dataset, 'w') as file:
        for index, sentence in enumerate(sentences):
            if sentence in pharse2id.keys():
                id = pharse2id[sentence]
                label = tags[id]
                sentence = sentence + ' ' + str(label) + '\n'
                file.write(sentence)
            else:
                print(sentence)


if __name__ == "__main__":
    splited_sentence = sentence_split()
    tag_sentences(splited_sentence)
        

