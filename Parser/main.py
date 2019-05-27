__author__ = 'Lin'



import os
import pickle
import torch
import numpy as np
import random
import argparse
from model import ParsingNet
from Training import Train
import os
from allennlp.modules.elmo import Elmo, batch_to_ids


def parse_args():
    parser = argparse.ArgumentParser(description='PointNet')
    
    parser.add_argument('--GPUforModel', type=int, default=0, help='Which GPU to run')
    parser.add_argument('--ELMo_mode', choices=['Large', 'Medium','Small'], default='Large', help='ELMo size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size of RNN')
    parser.add_argument('--rnn_layers', type=int, default=6, help='Number of RNN layers')
    parser.add_argument('--dropout_e', type=float, default=0.33, help='Dropout rate for encoder')
    parser.add_argument('--dropout_d', type=float, default=0.5, help='Dropout rate for decoder')
    parser.add_argument('--dropout_c', type=float, default=0.5, help='Dropout rate for classifier')
    parser.add_argument('--input_is_word', type=str, default='True', help='Whether the encoder input is word or EDU')
    parser.add_argument('--atten_model', choices=['Dotproduct', 'Biaffine'], default='Dotproduct', help='Attention mode')
    parser.add_argument('--classfier_input_size', type=int, default=64, help='Input size of relation classifier')
    parser.add_argument('--classfier_hidden_size', type=int, default=64, help='Hidden size of relation classifier')
    parser.add_argument('--classifier_bias', type=str, default='True', help='Whether classifier has bias')
    parser.add_argument('--seed', type=int, default=550, help='Seed number')
    parser.add_argument('--eval_size', type=int, default=600, help='Evaluation size')
    parser.add_argument('--epoch', type=int, default=300, help='Epoch number')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial lr')
    parser.add_argument('--lr_decay_epoch', type=int, default=10, help='Lr decay epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay rate')
    parser.add_argument('--highorder', type=str, default='False', help='Whether to incorperate highoreder information')
    # TO BE ADDED
    parser.add_argument('--datapath', type=str, default=r'/4TB/lin/Parsing/HIGHORDER/ParsingData/', help='Data path')
    parser.add_argument('--savepath', type=str, default=r'/4TB/lin/Parsing/HIGHORDER/test/Savings', help='Model save path')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    USE_CUDA = torch.cuda.is_available()

    device = torch.device("cuda:"+str(args.GPUforModel) if USE_CUDA else "cpu")
    
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    rnn_layers = args.rnn_layers
    dropout_e = args.dropout_e
    dropout_d = args.dropout_d
    dropout_c = args.dropout_c
    input_is_word = args.input_is_word
    atten_model = args.atten_model
    classfier_input_size = args.classfier_input_size
    classfier_hidden_size = args.classfier_hidden_size
    classifier_bias = args.classifier_bias
    ELMo_mode = args.ELMo_mode
    
    # TO BE ADDED
    data_path = args.datapath
    save_path = args.savepath
    seednumber = args.seed
    eval_size = args.eval_size
    epoch = args.epoch
    lr = args.lr
    lr_decay_epoch = args.lr_decay_epoch
    weight_decay = args.weight_decay
    
    

    # Initialize ELMo
    if ELMo_mode == 'Large':
        word_dim = 1024
        weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
        options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'

    elif ELMo_mode == 'Medium':       
        word_dim = 512
        weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5'
        options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json'

    elif ELMo_mode == 'Small':
        word_dim = 256
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    
    elmo = Elmo(options_file, weight_file, 2, dropout=0.5 ,requires_grad=False).to(device)

    # Setting random seeds 
    torch.manual_seed(seednumber)
    if USE_CUDA:
        torch.cuda.manual_seed_all(seednumber)
    np.random.seed(seednumber)
    random.seed(seednumber)

    # Process bool args       
    if args.classifier_bias == 'True':
        classifier_bias = True

    elif args.classifier_bias == 'False':
        classifier_bias = False
        

    if args.highorder == 'True':
        highorder = True

    elif args.highorder == 'False':
        highorder = False
    
    
    # Load Training data
    Tr_InputSentences = pickle.load(open(os.path.join(data_path,"Training_InputSentences.pickle"), "rb"))
    Tr_EDUBreaks = pickle.load(open(os.path.join(data_path,"Training_EDUBreaks.pickle"), "rb"))
    Tr_DecoderInput = pickle.load(open(os.path.join(data_path,"Training_DecoderInputs.pickle"), "rb"))
    Tr_RelationLabel = pickle.load(open(os.path.join(data_path,"Training_RelationLabel.pickle"), "rb"))
    Tr_ParsingBreaks = pickle.load(open(os.path.join(data_path,"Training_ParsingIndex.pickle"), "rb"))
    Tr_GoldenMetric = pickle.load(open(os.path.join(data_path,"Training_GoldenLabelforMetric.pickle"), "rb"))
    Tr_ParentsIndex = pickle.load(open(os.path.join(data_path,"Training_ParentsIndex.pickle"), "rb"))
    Tr_SiblingIndex = pickle.load(open(os.path.join(data_path,"Training_Sibling.pickle"), "rb"))
    
    # Load Testing data
    Test_InputSentences = pickle.load(open(os.path.join(data_path,"Testing_InputSentences.pickle"), "rb"))
    Test_EDUBreaks = pickle.load(open(os.path.join(data_path,"Testing_EDUBreaks.pickle"), "rb"))
    Test_DecoderInput = pickle.load(open(os.path.join(data_path,"Testing_DecoderInputs.pickle"), "rb"))
    Test_RelationLabel = pickle.load(open(os.path.join(data_path,"Testing_RelationLabel.pickle"), "rb"))
    Test_ParsingBreaks = pickle.load(open(os.path.join(data_path,"Testing_ParsingIndex.pickle"), "rb"))
    Test_GoldenMetric = pickle.load(open(os.path.join(data_path,"Testing_GoldenLabelforMetric.pickle"), "rb"))
    
#    # Check numbers
    print(len(Tr_InputSentences))
    print(len(Tr_EDUBreaks))
    print(len(Tr_DecoderInput))
    print(len(Tr_RelationLabel))
    print(len(Tr_ParsingBreaks))
    print(len(Tr_GoldenMetric))
    print(len(Tr_ParentsIndex))
    print(len(Tr_SiblingIndex))
    
    print(len(Test_InputSentences))
    print(len(Test_EDUBreaks))
    print(len(Test_DecoderInput))
    print(len(Test_RelationLabel))
    print(len(Test_ParsingBreaks))
    print(len(Test_GoldenMetric))
    
    

    
    # To check data
    sent_temp = ''
    print("Checking Data...")
    for word_temp in Tr_InputSentences[2]:
        sent_temp = sent_temp + ' ' + word_temp    
    print(sent_temp)   
    print('... ...')      
    print("That's great! No error found!")
    
    
    # To save model and data
    FileName = str(seednumber)+'_BatchSize_' + str(batch_size) + 'ELMo_' + str(ELMo_mode) + 'RnnLayer_' + str(rnn_layers) +\
                'AttenMode_' + atten_model + 'RnnHiddenSize_' + str(hidden_size) + \
                'ClassifierHidden_' + str(classfier_hidden_size)

    SavePath = os.path.join(save_path, FileName)
    print(SavePath)
    
    # Indicate model
    model = ParsingNet(elmo, batch_size, word_dim, hidden_size,
                       hidden_size, atten_model, device,
                       classfier_input_size, classfier_hidden_size, highorder, 39,
                       classifier_bias, rnn_layers, dropout_e,dropout_d,dropout_c)
    
    model = model.to(device)
    
    
    TrainingProcess = Train(model, Tr_InputSentences, Tr_EDUBreaks, Tr_DecoderInput,
                            Tr_RelationLabel, Tr_ParsingBreaks, Tr_GoldenMetric,
                            Tr_ParentsIndex, Tr_SiblingIndex,
                            Test_InputSentences, Test_EDUBreaks, Test_DecoderInput, 
                            Test_RelationLabel,Test_ParsingBreaks, Test_GoldenMetric, 
                            batch_size, eval_size, epoch, lr, lr_decay_epoch,
                            weight_decay, SavePath)
    
   
    
    best_epoch, best_F_relation, best_P_relation, best_R_relation, best_F_span, \
            best_P_span, best_R_span, best_F_nuclearity, best_P_nuclearity, \
            best_R_nuclearity = TrainingProcess.train()

    print('--------------------------------------------------------------------')
    print('Training Completed!')
    print('Processing...')
    print('The best F1 points for Relation is: %f.' % (best_F_relation))
    print('The best F1 points for Nuclearity is: %f' % (best_F_nuclearity))
    print('The best F1 points for Span is: %f' % (best_F_span))



    # Save result
    with open(os.path.join(args.savepath,'Results.csv'), 'a') as f:
      f.write(FileName + ',' + ','.join(map(str,[best_epoch, best_F_relation, \
                                best_P_relation, best_R_relation, best_F_span, \
                                best_P_span, best_R_span, best_F_nuclearity, \
                                best_P_nuclearity,  best_R_nuclearity]))+ '\n')








