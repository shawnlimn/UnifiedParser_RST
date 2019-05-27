__author__ = 'Lin'

import torch.optim as optim
import numpy as np
import torch
import random
import torch.nn as nn
import copy
import os
from Metric import getBatchMeasure, getMicroMeasure


def getBatchData_training(InputSentences, EDUBreaks, DecoderInput, RelationLabel, 
                 ParsingBreaks, GoldenMetric, ParentsIndex, Sibing, batch_size):

    # change them into np.array
    InputSentences = np.array(InputSentences)
    EDUBreaks = np.array(EDUBreaks)
    DecoderInput = np.array(DecoderInput)
    RelationLabel = np.array(RelationLabel)
    ParsingBreaks = np.array(ParsingBreaks)
    GoldenMetric = np.array(GoldenMetric)
    ParentsIndex = np.array(ParentsIndex)
    Sibing = np.array(Sibing)
    
    if len(DecoderInput) < batch_size:
        batch_size = len(DecoderInput)
        
    IndexSelected = random.sample(range(len(DecoderInput)), batch_size)
    # Get batch data
    InputSentences_batch = copy.deepcopy(InputSentences[IndexSelected])
    EDUBreaks_batch = copy.deepcopy(EDUBreaks[IndexSelected])
    DecoderInput_batch = copy.deepcopy(DecoderInput[IndexSelected])
    RelationLabel_batch = copy.deepcopy(RelationLabel[IndexSelected])
    ParsingBreaks_batch = copy.deepcopy(ParsingBreaks[IndexSelected])
    GoldenMetric_batch = copy.deepcopy(GoldenMetric[IndexSelected])
    ParentsIndex_batch = copy.deepcopy(ParentsIndex[IndexSelected])
    Sibing_batch = copy.deepcopy(Sibing[IndexSelected])
    
    # Get sorted    
    Lengths_batch = np.array([len(sent) for sent in InputSentences_batch])   
    idx = np.argsort(Lengths_batch)
    idx = idx[::-1]
    
    # Convert them back to list
    InputSentences_batch = InputSentences_batch[idx].tolist()
    EDUBreaks_batch = EDUBreaks_batch[idx].tolist()
    DecoderInput_batch = DecoderInput_batch[idx].tolist()
    RelationLabel_batch = RelationLabel_batch[idx].tolist()
    ParsingBreaks_batch = ParsingBreaks_batch[idx].tolist()
    GoldenMetric_batch = GoldenMetric_batch[idx].tolist()
    ParentsIndex_batch = ParentsIndex_batch[idx].tolist()
    Sibing_batch = Sibing_batch[idx].tolist()




    return  InputSentences_batch, EDUBreaks_batch, DecoderInput_batch, RelationLabel_batch, \
            ParsingBreaks_batch, GoldenMetric_batch, ParentsIndex_batch, Sibing_batch


def getBatchData(InputSentences, EDUBreaks, DecoderInput, RelationLabel, 
                 ParsingBreaks, GoldenMetric, batch_size):

    # change them into np.array
    InputSentences = np.array(InputSentences)
    EDUBreaks = np.array(EDUBreaks)
    DecoderInput = np.array(DecoderInput)
    RelationLabel = np.array(RelationLabel)
    ParsingBreaks = np.array(ParsingBreaks)
    GoldenMetric = np.array(GoldenMetric)
    
    
    if len(DecoderInput) < batch_size:
        batch_size = len(DecoderInput)
    IndexSelected = random.sample(range(len(DecoderInput)), batch_size)

    # Get batch data
    InputSentences_batch = copy.deepcopy(InputSentences[IndexSelected])
    EDUBreaks_batch = copy.deepcopy(EDUBreaks[IndexSelected])
    DecoderInput_batch = copy.deepcopy(DecoderInput[IndexSelected])
    RelationLabel_batch = copy.deepcopy(RelationLabel[IndexSelected])
    ParsingBreaks_batch = copy.deepcopy(ParsingBreaks[IndexSelected])
    GoldenMetric_batch = copy.deepcopy(GoldenMetric[IndexSelected])

    # Get sorted    
    Lengths_batch = np.array([len(sent) for sent in InputSentences_batch])   
    idx = np.argsort(Lengths_batch)
    idx = idx[::-1]
    
    # Convert them back to list
    InputSentences_batch = InputSentences_batch[idx].tolist()
    EDUBreaks_batch = EDUBreaks_batch[idx].tolist()
    DecoderInput_batch = DecoderInput_batch[idx].tolist()
    RelationLabel_batch = RelationLabel_batch[idx].tolist()
    ParsingBreaks_batch = ParsingBreaks_batch[idx].tolist()
    GoldenMetric_batch = GoldenMetric_batch[idx].tolist()
    




    return  InputSentences_batch, EDUBreaks_batch, DecoderInput_batch, RelationLabel_batch, ParsingBreaks_batch, GoldenMetric_batch





class Train(object):
    def __init__(self, model, Tr_Input_sentences, Tr_EDUBreaks, Tr_DecoderInput,
                 Tr_RelationLabel, Tr_ParsingBreaks, Tr_GoldenMetric,
                 Tr_ParentsIndex, Tr_SiblingIndex,
                 Test_InputSentences, Test_EDUBreaks, Test_DecoderInput, 
                 Test_RelationLabel,Test_ParsingBreaks, Test_GoldenMetric, 
                 batch_size, eval_size, epoch, lr, lr_decay_epoch, weight_decay,
                 save_path):
                 

        self.model = model
        self.Tr_Input_sentences = Tr_Input_sentences
        self.Tr_EDUBreaks = Tr_EDUBreaks
        self.Tr_DecoderInput = Tr_DecoderInput
        self.Tr_RelationLabel = Tr_RelationLabel 
        self.Tr_ParsingBreaks = Tr_ParsingBreaks
        self.Tr_GoldenMetric = Tr_GoldenMetric
        self.Tr_ParentsIndex = Tr_ParentsIndex
        self.Tr_SiblingIndex = Tr_SiblingIndex
        self.Test_InputSentences = Test_InputSentences 
        self.Test_EDUBreaks = Test_EDUBreaks
        self.Test_DecoderInput = Test_DecoderInput
        self.Test_RelationLabel = Test_RelationLabel
        self.Test_ParsingBreaks = Test_ParsingBreaks
        self.Test_GoldenMetric = Test_GoldenMetric
        self.batch_size = batch_size
        self.eval_size = eval_size
        self.epoch = epoch
        self.lr = lr
        self.lr_decay_epoch = lr_decay_epoch 
        self.weight_decay = weight_decay
        self.save_path = save_path

    
    def getTrainingEval(self):
        
        # Obtain eval_size samples of training data to evalute the model in 
        # every epoch
        
        # Conver to np.array
        Tr_Input_sentences = np.array(self.Tr_Input_sentences)
        Tr_EDUBreaks = np.array(self.Tr_EDUBreaks)
        Tr_DecoderInput = np.array(self.Tr_DecoderInput)
        Tr_RelationLabel = np.array(self.Tr_RelationLabel)
        Tr_ParsingBreaks = np.array(self.Tr_ParsingBreaks) 
        Tr_GoldenMetric = np.array(self.Tr_GoldenMetric)
        
        IndexSelected = random.sample(range(len(self.Tr_ParsingBreaks)),self.eval_size)
        
        DevTr_Input_sentences = Tr_Input_sentences[IndexSelected].tolist()
        DevTr_EDUBreaks = Tr_EDUBreaks[IndexSelected].tolist()
        DevTr_DecoderInput = Tr_DecoderInput[IndexSelected].tolist()
        DevTr_RelationLabel = Tr_RelationLabel[IndexSelected].tolist()
        DevTr_ParsingBreaks = Tr_ParsingBreaks[IndexSelected].tolist()
        DevTr_GoldenMetric = Tr_GoldenMetric[IndexSelected].tolist()
        
        return DevTr_Input_sentences, DevTr_EDUBreaks, DevTr_DecoderInput, DevTr_RelationLabel,\
                DevTr_ParsingBreaks, DevTr_GoldenMetric
        
        
    
    
    def getAccuracy(self, Input_sentences, EDUBreaks, DecoderInput, RelationLabel,\
                    ParsingBreaks, GoldenMetric):

        LoopNeeded = int(np.ceil(len(EDUBreaks) / self.batch_size))

        Loss_tree_all = []
        Loss_label_all = []
        correct_span = 0
        correct_relation = 0 
        correct_nuclearity = 0
        no_system = 0
        no_golden = 0
        
        
        for loop in range(LoopNeeded):
            
            StartPosition = loop * self.batch_size
            EndPosition =  (loop + 1) * self.batch_size
            if EndPosition > len(EDUBreaks):
                EndPosition = len(EDUBreaks)
            
            InputSentences_batch, EDUBreaks_batch, _,\
                RelationLabel_batch, ParsingBreaks_batch, GoldenMetric_batch =\
                getBatchData(Input_sentences[StartPosition:EndPosition], 
                             EDUBreaks[StartPosition:EndPosition],
                             DecoderInput[StartPosition:EndPosition], 
                             RelationLabel[StartPosition:EndPosition],
                             ParsingBreaks[StartPosition:EndPosition], 
                             GoldenMetric[StartPosition:EndPosition], self.batch_size)
            
            Loss_tree_batch, Loss_label_batch, SPAN_batch = self.model.TestingLoss(
                    InputSentences_batch, EDUBreaks_batch, RelationLabel_batch,
                    ParsingBreaks_batch, True)
            
            
            Loss_tree_all.append(Loss_tree_batch)
            Loss_label_all.append(Loss_label_batch)
            correct_span_batch, correct_relation_batch, correct_nuclearity_batch,\
                no_system_batch, no_golden_batch = getBatchMeasure(SPAN_batch, 
                                                                   GoldenMetric_batch)
            
            correct_span = correct_span + correct_span_batch
            correct_relation = correct_relation + correct_relation_batch  
            correct_nuclearity = correct_nuclearity + correct_nuclearity_batch
            no_system = no_system + no_system_batch
            no_golden = no_golden + no_golden_batch
        
        
        
        span_points, relation_points, nuclearity_points = getMicroMeasure(
                correct_span,correct_relation,correct_nuclearity,no_system,no_golden)

        return np.mean(Loss_tree_all), np.mean(Loss_label_all), span_points, relation_points, nuclearity_points
            
            
    def LearningRateAdjust(self, optimizer, epoch, lr_decay=0.5, lr_decay_epoch=50):

        if (epoch % lr_decay_epoch == 0) and (epoch != 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_decay

    
    def train(self):

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                lr=self.lr, betas=(0.9, 0.9),weight_decay=self.weight_decay)

        iteration = int(np.ceil(len(self.Tr_ParsingBreaks) / self.batch_size))

        try:
            os.mkdir(self.save_path)
        except:
            pass
            
        
        best_F_relation = 0
        best_F_span = 0
        for i in range(self.epoch):
            self.LearningRateAdjust(optimizer, i, 0.8, self.lr_decay_epoch)

            for iter in range(iteration):
                   
                print("epoch:%d, iteration:%d" % (i, iter)) 

                InputSentences_batch, EDUBreaks_batch, DecoderInput_batch, \
                RelationLabel_batch, ParsingBreaks_batch, _ ,ParentsIndex_batch,\
                                        Sibing_batch = getBatchData_training(
                                        self.Tr_Input_sentences, self.Tr_EDUBreaks, 
                                        self.Tr_DecoderInput, self.Tr_RelationLabel, 
                                        self.Tr_ParsingBreaks, self.Tr_GoldenMetric, 
                                        self.Tr_ParentsIndex, self.Tr_SiblingIndex, self.batch_size)

                self.model.zero_grad()
                Loss_tree_batch, Loss_label_batch = self.model.TrainingLoss(InputSentences_batch, 
                                                            EDUBreaks_batch, RelationLabel_batch, 
                                                            ParsingBreaks_batch, DecoderInput_batch,
                                                            ParentsIndex_batch, Sibing_batch)

                Loss = Loss_tree_batch + Loss_label_batch
                Loss.backward()

                cur_loss = float(Loss.item())
                
                print(cur_loss)

                # To avoid gradient exploration
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)

                optimizer.step()

            # Convert model to eval 
            self.model.eval() 
            
            # Obtain Training (devolopment) data
            DevTr_Input_sentences, DevTr_EDUBreaks, DevTr_DecoderInput, DevTr_RelationLabel,\
            DevTr_ParsingBreaks, DevTr_GoldenMetric = self.getTrainingEval()
        
            # Eval on training (devolopment) data
            LossTree_Trdev, LossLabel_Trdev, span_points_Trdev, relation_points_Trdev,\
            nuclearity_points_Trdev = self.getAccuracy(DevTr_Input_sentences, DevTr_EDUBreaks, 
                                                       DevTr_DecoderInput,DevTr_RelationLabel, 
                                                       DevTr_ParsingBreaks, DevTr_GoldenMetric)

        
            # Eval on Testing data
            LossTree_Test, LossLabel_Test, span_points_Test, relation_points_Test,\
            nuclearity_points_Test = self.getAccuracy(self.Test_InputSentences, self.Test_EDUBreaks, 
                                                       self.Test_DecoderInput,self.Test_RelationLabel, 
                                                       self.Test_ParsingBreaks, self.Test_GoldenMetric)
            
            # Unfold numbers
            # Test
            P_span, R_span, F_span = span_points_Test
            P_relation, R_relation, F_relation = relation_points_Test
            P_nuclearity, R_nuclearity, F_nuclearity = nuclearity_points_Test
            # Training (dev)
            _, _, F_span_Trdev = span_points_Trdev
            _, _, F_relation_Trdev = relation_points_Trdev
            _, _, F_nuclearity_Trdev = nuclearity_points_Trdev

            # Relation will take the priority consideration
            # if F_relation > best_F_relation:
            if F_span > best_F_span:
                best_epoch = i
                # relation
                best_F_relation = F_relation
                best_P_relation = P_relation
                best_R_relation = R_relation
                # span
                best_F_span = F_span
                best_P_span = P_span
                best_R_span = R_span
                # nuclearity
                best_F_nuclearity = F_nuclearity
                best_P_nuclearity = P_nuclearity
                best_R_nuclearity = R_nuclearity
            
            
            # Saving data
            save_data = [i, LossTree_Trdev, LossLabel_Trdev, F_span_Trdev, F_relation_Trdev, F_nuclearity_Trdev,
                         LossTree_Test, LossLabel_Test, F_span, F_relation, F_nuclearity]

            FileName = 'span_bs_{}_es_{}_lr_{}_lrdc_{}_wd_{}.txt'.format(self.batch_size,\
                           self.eval_size,self.lr,self.lr_decay_epoch,self.weight_decay)
            with open(os.path.join(self.save_path, FileName), 'a+') as f:
                f.write(','.join(map(str,save_data)) + '\n')

            
            # Saving model
            if best_epoch == i  and i > 20:
                torch.save(self.model, os.path.join(self.save_path, r'Epoch_%d.torchsave' % (i)))

            # Convert back to training
            self.model.train()
            
            
        return best_epoch, best_F_relation, best_P_relation, best_R_relation, best_F_span, \
            best_P_span, best_R_span, best_F_nuclearity, best_P_nuclearity, best_R_nuclearity
            





