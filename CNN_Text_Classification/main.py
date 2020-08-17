import argparse
import os
import torch
import sys
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.autograd as autograd
import torch.nn.functional as F

from CNN_Model import CNNNet

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=50,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')

args = parser.parse_args()

root = os.path.abspath('./Data')
root = os.path.join(root, 'dataset')

trainX_path = os.path.join(root, 'trainX.npy')
trainY_path = os.path.join(root, 'trainY.npy')

testX_path = os.path.join(root, 'testX.npy')
testY_path = os.path.join(root, 'testY.npy')

validX_path = os.path.join(root, 'validX.npy')
validY_path = os.path.join(root, 'validY.npy')

# train model

trainX = torch.tensor(np.load(trainX_path, allow_pickle=True))
trainY = torch.tensor(np.load(trainY_path, allow_pickle=True))

validX = torch.tensor(np.load(validX_path, allow_pickle=True))
validY = torch.tensor(np.load(validY_path, allow_pickle=True))


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


def train(trainX, trainY, validX, validY, model, args):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0

    model.train()

    trainX = trainX.reshape((trainX.shape[0], trainX.shape[2], trainX.shape[1]))
    trainX = trainX.type(torch.FloatTensor)
    trainY = trainY.reshape(trainY.shape[0])
    trainY = trainY.type(torch.LongTensor)

    validX = validX.reshape((validX.shape[0], validX.shape[2], validX.shape[1]))
    validX = validX.type(torch.FloatTensor)
    validY = validY.reshape(validY.shape[0])
    validY = validY.type(torch.LongTensor)

    batch_size = model.batch_size
    batch_num = len(trainX) // batch_size + 1

    for epoch in range(2, args.epochs+1):
        for batch in range(batch_num):
            feature = trainX[batch_size * batch:batch_size * (batch+1)]
            target = trainY[batch_size * batch:batch_size * (batch+1)]
            # feature.t_(), target.sub_(1)  # batch first, index align

            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})\n'.format(steps,
                                                                             loss.item(),
                                                                             accuracy.item(),
                                                                             corrects.item(),
                                                                             batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(validX, validY, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(X, Y, model, args):
    model.eval()
    corrects, avg_loss = 0, 0

    size = X.shape[0]
    batch_size = model.batch_size
    batch_num = len(X) // batch_size + 1

    for batch in range(batch_num):
        feature = X[batch_size * batch:batch_size * (batch+1)]
        target = Y[batch_size * batch:batch_size * (batch+1)]
        # feature.t_(), target.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       X.shape[0]))
    return accuracy


net = CNNNet(batch_size=32)
train(trainX, trainY, validX, validY, net, args)