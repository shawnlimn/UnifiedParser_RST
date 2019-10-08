


import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pickle
import torch
import numpy as np
import random
from torch.backends import cudnn
import argparse
from model import PointerNetworks
from solver import TrainSolver

import os


def parse_args():
    parser = argparse.ArgumentParser(description='Pointer')

    parser.add_argument('-hdim', type=int, default=64, help='hidden size')
    parser.add_argument('-rnn', type=str, default='GRU', help='rnn type')
    parser.add_argument('-rnnlayers', type=int, default=6, help='how many rnn layers')
    parser.add_argument('-fine', type=str,default='False', help='fine tuning word embedding')
    parser.add_argument('-isbi', type=str, default='True', help='is bidirctional')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-dout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('-wd', type=float, default=0.0001, help='weight decay')
    parser.add_argument('-seed', type=int, default=550, help='random seed')
    parser.add_argument('-bsize', type=int, default=80, help='batch size')
    parser.add_argument('-lrdepoch', type=int, default=10, help='lr decay each epoch')
    parser.add_argument('-isbarnor', type=str, default='True', help='batch normalization')
    parser.add_argument('-iscudnn', type=str, default='True', help='cudnn')
    parser.add_argument('-savepath', type=str, default=r'/home/lin/Segmentation/ELMo/Savings', help='save path')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    use_cuda = True

    if args.iscudnn =='False':
        iscudnn = False
    else:
        iscudnn = True

    cudnn.enabled = iscudnn
    h_Dim = args.hdim
    rnn_type = args.rnn
    rnn_layers = args.rnnlayers
    lr = args.lr
    dout = args.dout
    wd = args.wd
    myseed = args.seed
    batch_size = args.bsize
    lrdepoch = args.lrdepoch

    torch.manual_seed(myseed)
    if use_cuda:
        torch.cuda.manual_seed_all(myseed)
    np.random.seed(myseed)
    random.seed(myseed)



    if args.isbi =='False':
        IS_BI = False
    else:
        IS_BI = True


    if args.fine =='False':
        FineTurning = False
    else:
        FineTurning = True


    if args.isbarnor == 'False':
        isbarnor = False
    else:
        isbarnor = True

    print(FineTurning)
    print(type(FineTurning))
    print(type(IS_BI))


    loadpath = r'/home/lin/Segmentation/ELMo/SegData'
    tr_x = pickle.load( open(os.path.join(loadpath,"Training_InputSentences_seg.pickle"),"rb"))
    tr_y = pickle.load(open(os.path.join(loadpath,"Training_EDUBreaks_seg.pickle"), "rb"))

    dev_x = pickle.load(open(os.path.join(loadpath,"Testing_InputSentences_seg.pickle"),"rb"))
    dev_y = pickle.load(open(os.path.join(loadpath,"Testing_EDUBreaks_seg.pickle"), "rb"))


    filename = 'elmoLarge_dot_'+str(myseed) + 'seed_' + str(h_Dim) +'hidden_'+ \
               str(IS_BI)+'bi_' + rnn_type + 'rnn_' + str(FineTurning)+'Fined_'+str(rnn_layers)+\
               'rnnlayers_'+ str(lr)+'lr_'+str(dout)+'dropout_'+str(wd)+'weightdecay_'+str(batch_size)+'bsize_'+str(lrdepoch)+'lrdepoch_'+\
                str(isbarnor)+'barnor_'+str(iscudnn)+'iscudnn'

    save_path = os.path.join(args.savepath,filename)
    print(save_path)


    word_dim=1024
    hidden_dim=h_Dim
    is_bi_encoder_rnn= IS_BI
    rnn_type=rnn_type
    rnn_layers=rnn_layers
    dropout_prob=dout
    use_cuda=use_cuda
    finedtuning=FineTurning
    isbanor=isbarnor


    model = PointerNetworks(word_dim=1024,
                            hidden_dim=h_Dim,is_bi_encoder_rnn= IS_BI,rnn_type=rnn_type,rnn_layers=rnn_layers,
                 dropout_prob=dout,use_cuda=use_cuda,finedtuning=FineTurning,isbanor=isbarnor)
    if use_cuda:
        model = model.cuda()

    mysolver = TrainSolver(model,tr_x, tr_y, dev_x, dev_y,save_path,
                         batch_size=batch_size, eval_size=600, epoch=1000, lr=lr, lr_decay_epoch=lrdepoch, weight_decay=wd,
                           use_cuda=use_cuda)

    best_i, best_pre, best_rec, best_f1 = mysolver.train()

    with open(os.path.join(args.savepath,'resultTable.csv'), 'a') as f:
      f.write(filename + ',' + ','.join(map(str,[best_i, best_pre, best_rec, best_f1]))+ '\n')
