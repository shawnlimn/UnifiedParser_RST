__author__ = 'Lin'


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from module import EncoderRNN, DecoderRNN, PointerAtten, LabelClassifier
from DataHandler import get_RelationAndNucleus


class ParsingNet(nn.Module):
    def __init__(self, elmo, batch_size, word_dim, hidden_size, decoder_input_size,
                 atten_model, device, classfier_input_size, 
                 classfier_hidden_size, highorder=False, classes_label=39, classifier_bias=True, 
                 rnn_layers=6, dropout_e=0.33,dropout_d=0.5,dropout_c=0.5):

        super(ParsingNet, self).__init__()
        '''
        
        Args:
            batch_size: batch size
            word_dim: word embedding dimension 
            hidden_size: hidden size of encoder and decoder 
            decoder_input_size: input dimension of decoder
            atten_model: pointer attention machanisam, 'Dotproduct' or 'Biaffine' 
            device: device that our model is running on 
            device2: device that ELMo is running on
            classfier_input_size: input dimension of labels classifier 
            classfier_hidden_size: classifier hidden space
            classes_label: relation(label) number, default = 39
            classifier_bias: bilinear bias in classifier, default = True
            rnn_layers: encoder and decoder layer number
            dropout: dropout rate
            
            
        '''
        self.batch_size = batch_size
        self.word_dim = word_dim
        self.hidden_size = hidden_size
        self.decoder_input_size = decoder_input_size
        self.device = device
        self.classfier_input_size = classfier_input_size
        self.classfier_hidden_size = classfier_hidden_size
        self.classes_label =  classes_label
        self.classifier_bias = classifier_bias
        self.rnn_layers = rnn_layers
        self.highorder = highorder
        self.encoder = EncoderRNN(elmo, word_dim, hidden_size, device, 
                                  rnn_layers, dropout_e)
        self.decoder = DecoderRNN(decoder_input_size, hidden_size, rnn_layers, dropout_d)
        self.pointer = PointerAtten(atten_model,hidden_size)
        self.getlabel = LabelClassifier(classfier_input_size, classfier_hidden_size, classes_label, bias=True, dropout=dropout_c)
        
        
        
        
    def forward(self):
        raise RuntimeError('Parsing Network does not have forward process.')
        
        
    def TrainingLoss(self, input_sentence, EDU_breaks, LabelIndex, ParsingIndex, DecoderInputIndex,
                     ParentsIndex, SiblingIndex):
        
        # Obtain encoder outputs and last hidden states
        EncoderOutputs, Last_Hiddenstates = self.encoder(input_sentence)
        LossFunction  = nn.NLLLoss()
        Loss_label_batch = 0
        Loss_tree_batch = 0
        Loop_label_batch = 0
        Loop_tree_batch = 0
        
        for i in range(self.batch_size):
            
            '''
            
            For a sentence containing only ONE EDU, we don't need to run the
            model. If TWO, we don't need to build a parsing tree. We only 
            predict the relation between these two EDU.
            
            Input:
                input_sentence: [batch_size, length]
                EDU_breaks: e.g. [[2,4,6,9],[2,5,8,10,13],[6,8],[6]]
                LabelIndex: e.g. [[0,3,32],[20,11,14,19],[20],[],]
                ParsingIndex: e.g. [[1,2,0],[3,2,0,1],[0],[]]
                DecoderInputIndex: e.g. [[0,1,2],[0,1,1,2],[0],[]]    
                
            Output: 
                Average loss of tree in a batch
                Average loss of relation in a batch
            
            '''
            
            cur_LabelIndex = LabelIndex[i]
            cur_LabelIndex = torch.tensor(cur_LabelIndex)
            cur_LabelIndex = cur_LabelIndex.to(self.device)
            cur_ParsingIndex = ParsingIndex[i]
            cur_DecoderInputIndex = DecoderInputIndex[i]
            cur_ParentsIndex = ParentsIndex[i]
            cur_SiblingIndex = SiblingIndex[i]
            
            if len(EDU_breaks[i]) == 1:
                
                continue
                
            elif len(EDU_breaks[i]) == 2:
                
                # Take the last hidden state of an EDU as the representation of 
                # this EDU. The dimension: [2,hidden_size]
                cur_EncoderOutputs = EncoderOutputs[i][EDU_breaks[i]]
                
                # Use the last hidden state of a span to predict the relation 
                # beween these two span.
                input_left = cur_EncoderOutputs[0].unsqueeze(0)
                input_right = cur_EncoderOutputs[1].unsqueeze(0)
                _ , log_relation_weights = self.getlabel(input_left, input_right)
                
                Loss_label_batch = Loss_label_batch + LossFunction(log_relation_weights, cur_LabelIndex)
                Loop_label_batch = Loop_label_batch + 1
            
            else:    

                # Take the last hidden state of an EDU as the representation of this EDU
                # The dimension: [NO_EDU,hidden_size]
                cur_EncoderOutputs = EncoderOutputs[i][EDU_breaks[i]]

                # Obtain last hidden state of encoder
                tempp = torch.transpose(Last_Hiddenstates, 0, 1)[i].unsqueeze(0)
                cur_Last_Hiddenstates = torch.transpose(tempp, 0, 1)
                cur_Last_Hiddenstates = cur_Last_Hiddenstates.contiguous()
                
                if self.highorder:

                    # To incorperate parents information
                    cur_DecoderInputs_P = cur_EncoderOutputs[cur_ParentsIndex]
                    cur_DecoderInputs_P[0] = 0

                    # To incorperate sibling information
                    cur_DecoderInputs_S = cur_EncoderOutputs[cur_SiblingIndex]
                    for n in range(len(cur_SiblingIndex)):
                        if cur_SiblingIndex[n] == 99:
                            cur_DecoderInputs_S[n] = 0

                    # Original input
                    cur_DecoderInputs = cur_EncoderOutputs[cur_DecoderInputIndex]

                    # One-layer self attention
                    inputs_all = torch.cat((cur_DecoderInputs.unsqueeze(0).transpose(0,1),cur_DecoderInputs_S.unsqueeze(0).transpose(0,1),\
                                            cur_DecoderInputs_P.unsqueeze(0).transpose(0,1)),1)
                    new_inputs_all = torch.matmul(F.softmax(torch.matmul(inputs_all,inputs_all.transpose(1,2)),1),inputs_all)
                    cur_DecoderInputs =  new_inputs_all[:,0,:] + new_inputs_all[:,1,:] + new_inputs_all[:,2,:]

                    
                else: 
                    cur_DecoderInputs = cur_EncoderOutputs[cur_DecoderInputIndex]
                
                # Obtain decoder outputs
                # cur_DecoderOutputs: [No_decoderinput,hidden_state]
                cur_DecoderOutputs, _ = self.decoder(cur_DecoderInputs.unsqueeze(0), cur_Last_Hiddenstates)
                cur_DecoderOutputs = cur_DecoderOutputs.squeeze(0)
                
                
                EDU_index = [x for x in range(len(cur_EncoderOutputs))]               
                stacks = ['__StackRoot__',EDU_index]
    
                for j in range(len(cur_DecoderOutputs)):
                                                           
                    if stacks[-1] is not '__StackRoot__':
                        stack_head = stacks[-1]
                        
                        
                        if len(stack_head) < 3:

                            # We remove this from stacks after compute the
                            # relation between these two EDUS
                        
                            # Compute Classifier Loss
                            input_left = cur_EncoderOutputs[cur_ParsingIndex[j]].unsqueeze(0)
                            input_right = cur_EncoderOutputs[stack_head[-1]].unsqueeze(0)
                            _ , log_relation_weights = self.getlabel(input_left,input_right)      
                            
                            Loss_label_batch = Loss_label_batch + LossFunction(log_relation_weights, cur_LabelIndex[j].unsqueeze(0))
                            
                            del stacks[-1]
                            Loop_label_batch = Loop_label_batch + 1 
                            
                            
                        else: # Length of stack_head >= 3
                            
                            # Compute Tree Loss
                            # We don't attend to the last EDU of a span to be parsed
                            _ , log_atten_weights = self.pointer(cur_EncoderOutputs[stack_head[:-1]],
                                                                            cur_DecoderOutputs[j])            
                            cur_ground_index = torch.tensor([int(cur_ParsingIndex[j]) - int(stack_head[0])])
                            cur_ground_index = cur_ground_index.to(self.device)
                            Loss_tree_batch = Loss_tree_batch + LossFunction(log_atten_weights, cur_ground_index)
                                                  
                            
                            # Compute Classifier Loss
                            input_left = cur_EncoderOutputs[cur_ParsingIndex[j]].unsqueeze(0)
                            input_right = cur_EncoderOutputs[stack_head[-1]].unsqueeze(0)
                            _ , log_relation_weights = self.getlabel(input_left, input_right)
                            
                            Loss_label_batch = Loss_label_batch + LossFunction(log_relation_weights, cur_LabelIndex[j].unsqueeze(0))
                            
                            # Stacks stuff
                            stack_down = stack_head[(cur_ParsingIndex[j] - stack_head[0] + 1):]
                            stack_top = stack_head[:(cur_ParsingIndex[j] - stack_head[0] + 1)]                            
                            del stacks[-1]
                            Loop_label_batch = Loop_label_batch + 1
                            Loop_tree_batch = Loop_tree_batch + 1

                            # Remove ONE-EDU part, TWO-EDU span will be removed after classifier in next step
                            if len(stack_down) > 1:
                                stacks.append(stack_down)
                            if len(stack_top) > 1:
                                stacks.append(stack_top)
                            
        Loss_label_batch = Loss_label_batch / Loop_label_batch
        Loss_tree_batch = Loss_tree_batch / Loop_tree_batch
        
        return Loss_tree_batch, Loss_label_batch
        
        
        
    
    
    
    
    def TestingLoss(self, input_sentence, EDU_breaks, LabelIndex, ParsingIndex, GenerateTree = True):
        '''
            
            Input:
                input_sentence: [batch_size, length]
                EDU_breaks: e.g. [[2,4,6,9],[2,5,8,10,13],[6,8],[6]]
                LabelIndex: e.g. [[0,3,32],[20,11,14,19],[20],[],]
                ParsingIndex: e.g. [[1,2,0],[3,2,0,1],[0],[]]
                   
                
            Output: 
                Average loss of tree in a batch
                Average loss of relation in a batch
            
        '''
        # Obtain encoder outputs and last hidden states
        EncoderOutputs, Last_Hiddenstates = self.encoder(input_sentence)
        
        LossFunction  = nn.NLLLoss()
        Loss_label_batch = 0
        Loss_tree_batch = 0
        Loop_label_batch = 0
        Loop_tree_batch = 0
        cur_label = []
        Label_batch = []
        cur_tree = []
        Tree_batch = []
        
        if GenerateTree:
            SPAN_batch = []   
        
        for i in range(len(EDU_breaks)):
            
            
            
            cur_LabelIndex = LabelIndex[i]
            cur_LabelIndex = torch.tensor(cur_LabelIndex)
            cur_LabelIndex = cur_LabelIndex.to(self.device)
            cur_ParsingIndex = ParsingIndex[i]
            
            if len(EDU_breaks[i]) == 1:
                
                # For a sentence containing only ONE EDU, it has no 
                # corresponding relation label and parsing tree break.
                Tree_batch.append([])
                Label_batch.append([])
                
                if GenerateTree:
                    SPAN_batch.append(['NONE'])
                
            elif len(EDU_breaks[i]) == 2:
                
                # Take the last hidden state of an EDU as the repretation of 
                # this EDU. The dimension: [2,hidden_size]
                cur_EncoderOutputs = EncoderOutputs[i][EDU_breaks[i]]
                
                #  Directly run the classifier to obain predicted label
                input_left = cur_EncoderOutputs[0].unsqueeze(0)
                input_right = cur_EncoderOutputs[1].unsqueeze(0)
                relation_weights , log_relation_weights = self.getlabel(input_left, input_right)               
                _ , topindex = relation_weights.topk(1)
                LabelPredict = int(topindex[0][0])                 
                Tree_batch.append([0])
                Label_batch.append([LabelPredict])
                

                Loss_label_batch = Loss_label_batch + LossFunction(log_relation_weights, cur_LabelIndex)
                Loop_label_batch = Loop_label_batch + 1
                
                if GenerateTree:
                    
                    # Generate a span structure
                    # e.g. (1:Nucleus=span:8,9:Satellite=Attribution:12)
                    Nuclearity_left, Nuclearity_right, Relation_left, Relation_right = \
                                            get_RelationAndNucleus(LabelPredict)
                                            
                    Span = '(1:'+str(Nuclearity_left)+'='+str(Relation_left)+\
                    ':1,2:'+str(Nuclearity_right)+'='+str(Relation_right)+':2)' 
                    SPAN_batch.append([Span])
                
                
            else:    
                
                # Take the last hidden state of an EDU as the repretation of this EDU
                # The dimension: [NO_EDU,hidden_size]
                cur_EncoderOutputs = EncoderOutputs[i][EDU_breaks[i]]

                EDU_index = [x for x in range(len(cur_EncoderOutputs))]               
                stacks = ['__StackRoot__', EDU_index]
                
                # cur_decoder_input: [1,1,hidden_size]  
                # Alternative way is to take the last one as the input. You need to prepare data accordingly for training  
                cur_decoder_input = cur_EncoderOutputs[0].unsqueeze(0).unsqueeze(0)
                
                # Obtain last hidden state
                temptest = torch.transpose(Last_Hiddenstates, 0, 1)[i].unsqueeze(0)
                cur_Last_Hiddenstates = torch.transpose(temptest, 0, 1)
                cur_Last_Hiddenstates = cur_Last_Hiddenstates.contiguous()
                
                cur_decoder_hidden = cur_Last_Hiddenstates             
                LoopIndex = 0
                
                if GenerateTree:
                    Span = ''
                if self.highorder:
                    cur_sibling = {} 
                   
                                        
                while stacks[-1] is not '__StackRoot__':
                    stack_head = stacks[-1]

                    if len(stack_head) < 3: 
                    
                        # Predict relation label
                        input_left = cur_EncoderOutputs[stack_head[0]].unsqueeze(0)
                        input_right = cur_EncoderOutputs[stack_head[-1]].unsqueeze(0)
                        relation_weights , log_relation_weights = self.getlabel(input_left, input_right)
                        _ , topindex = relation_weights.topk(1)
                        LabelPredict = int(topindex[0][0])
                        cur_label.append(LabelPredict)
                        
                        # For 2 EDU case, we directly point the first EDU 
                        # as the current parsing tree break 
                        cur_tree.append(stack_head[0])
                        
                        # To keep decoder hidden states consistent
                        _ , cur_decoder_hidden = self.decoder(cur_decoder_input, cur_decoder_hidden)                             
                        
                        # Align ground true label
                        if LoopIndex > (len(cur_ParsingIndex) - 1):
                            cur_Label_true = cur_LabelIndex[-1]
                        else:
                            cur_Label_true = cur_LabelIndex[LoopIndex]
                            
                       
                        Loss_label_batch = Loss_label_batch + LossFunction(log_relation_weights, cur_Label_true.unsqueeze(0))
                        Loop_label_batch = Loop_label_batch + 1 
                        LoopIndex = LoopIndex + 1
                        del stacks[-1]
                        
                        
                        if GenerateTree:
                            # To generate a tree structure
                            Nuclearity_left, Nuclearity_right, Relation_left, Relation_right = \
                                                    get_RelationAndNucleus(LabelPredict)           

                            cur_span = '('+str(stack_head[0]+1)+':'+str(Nuclearity_left)+'='+str(Relation_left)+\
                            ':'+str(stack_head[0]+1)+','+str(stack_head[-1]+1)+':'+str(Nuclearity_right)+'='+\
                            str(Relation_right)+':'+str(stack_head[-1]+1)+ ')' 
                            
                            Span = Span +' ' + cur_span
                        
                        
                        
                        
                        
                    else: # Length of stack_head >= 3
                    
                        # Alternative way is to take the last one as the input. You need to prepare data accordingly for training  
                        cur_decoder_input = cur_EncoderOutputs[stack_head[0]].unsqueeze(0).unsqueeze(0)

                        if self.highorder:
                            if LoopIndex != 0:
                                # Incoperate Parents information
                                cur_decoder_input_P = cur_EncoderOutputs[stack_head[-1]]
                                # To incorperate Sibling information
                                if str(stack_head) in cur_sibling.keys():
                                    cur_decoder_input_S = cur_EncoderOutputs[cur_sibling[str(stack_head)]]


                                    inputs_all = torch.cat((cur_decoder_input.squeeze(0),cur_decoder_input_S.unsqueeze(0),\
                                                    cur_decoder_input_P.unsqueeze(0)),0)
                                    new_inputs_all = torch.matmul(F.softmax(torch.matmul(inputs_all,inputs_all.transpose(0,1)),0),inputs_all)
                                    cur_decoder_input =  new_inputs_all[0,:] + new_inputs_all[1,:] + new_inputs_all[2,:]
                                    cur_decoder_input = cur_decoder_input.unsqueeze(0).unsqueeze(0)
                                    

                                    # cur_decoder_input = cur_decoder_input + cur_decoder_input_P + cur_decoder_input_S
                                else:
                                    inputs_all = torch.cat((cur_decoder_input.squeeze(0),cur_decoder_input_P.unsqueeze(0)),0)
                                    new_inputs_all = torch.matmul(F.softmax(torch.matmul(inputs_all,inputs_all.transpose(0,1)),0),inputs_all)
                                    cur_decoder_input =  new_inputs_all[0,:] + new_inputs_all[1,:]
                                    cur_decoder_input = cur_decoder_input.unsqueeze(0).unsqueeze(0)


                      
                        # Predict the parsing tree break
                        cur_decoder_output, cur_decoder_hidden = self.decoder(cur_decoder_input, cur_decoder_hidden)
                        atten_weights , log_atten_weights = self.pointer(cur_EncoderOutputs[stack_head[:-1]],
                                                                        cur_decoder_output.squeeze(0).squeeze(0))
                        _ , topindex_tree = atten_weights.topk(1)
                        TreePredict = int(topindex_tree[0][0]) + stack_head[0]
                        cur_tree.append(TreePredict)
                        
                        # Predict the Label 
                        input_left = cur_EncoderOutputs[TreePredict].unsqueeze(0)
                        input_right = cur_EncoderOutputs[stack_head[-1]].unsqueeze(0)
                        relation_weights , log_relation_weights = self.getlabel(input_left, input_right)
                        _ , topindex_label = relation_weights.topk(1)
                        LabelPredict = int(topindex_label[0][0])
                        cur_label.append(LabelPredict)
                        
                        # Align ground true label and tree
                        if LoopIndex > (len(cur_ParsingIndex) - 1):
                            cur_Label_true = cur_LabelIndex[-1]
                            cur_Tree_true = cur_ParsingIndex[-1]
                        else:
                            cur_Label_true = cur_LabelIndex[LoopIndex]
                            cur_Tree_true = cur_ParsingIndex[LoopIndex]

                        temp_ground = max(0,(int(cur_Tree_true) - int(stack_head[0])))
                        if temp_ground >= (len(stack_head) - 1):
                            temp_ground = stack_head[-2] - stack_head[0]
                        # Compute Tree Loss
                        cur_ground_index = torch.tensor([temp_ground])
                        cur_ground_index = cur_ground_index.to(self.device)
                        Loss_tree_batch = Loss_tree_batch + LossFunction(log_atten_weights, cur_ground_index)
                        
                        # Compute Classifier Loss     
                        Loss_label_batch = Loss_label_batch + LossFunction(log_relation_weights, cur_Label_true.unsqueeze(0))
                        
                        # Stacks stuff
                        stack_down = stack_head[(TreePredict - stack_head[0] + 1):]
                        stack_top = stack_head[:(TreePredict - stack_head[0] + 1)]                            
                        del stacks[-1]
                        Loop_label_batch = Loop_label_batch + 1
                        Loop_tree_batch = Loop_tree_batch + 1
                        LoopIndex = LoopIndex + 1
                        
                        # Sibling inforamtion
                        if self.highorder:
                            if len(stack_down) > 2:
                                cur_sibling.update({str(stack_down):stack_top[-1]})
                        
                                     
                        # Remove ONE-EDU part
                        if len(stack_down) > 1:
                            stacks.append(stack_down)
                        if len(stack_top) > 1:
                            stacks.append(stack_top)
                          
                            
                        if GenerateTree:    
                            # Generate a span structure
                            # e.g. (1:Nucleus=span:8,9:Satellite=Attribution:12)
                            Nuclearity_left, Nuclearity_right, Relation_left, Relation_right = \
                                                    get_RelationAndNucleus(LabelPredict)
                                                    
                            cur_span = '('+str(stack_head[0]+1)+':'+str(Nuclearity_left)+'='+str(Relation_left)+\
                            ':'+str(TreePredict+1)+','+str(TreePredict+2)+':'+str(Nuclearity_right)+'='+\
                            str(Relation_right)+':'+str(stack_head[-1]+1)+ ')' 
                            Span = Span +' ' + cur_span 
                
                            
                Tree_batch.append(cur_tree)
                Label_batch.append(cur_label)    
                if GenerateTree:
                    SPAN_batch.append([Span.strip()])
                    
        Loss_label_batch = Loss_label_batch / Loop_label_batch
        Loss_tree_batch = Loss_tree_batch / Loop_tree_batch
        
        Loss_label_batch = Loss_label_batch.detach().cpu().numpy()
        Loss_tree_batch = Loss_tree_batch.detach().cpu().numpy()
        
        
        return Loss_tree_batch, Loss_label_batch, (SPAN_batch if GenerateTree else None)
        






       
    














