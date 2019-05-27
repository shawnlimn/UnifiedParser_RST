__author__ = 'Lin'


import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids



class EncoderRNN(nn.Module):
    def __init__(self, elmo, word_dim, hidden_size, device, rnn_layers=6, dropout=0.2):

        super(EncoderRNN, self).__init__()
        '''
        Input:
            [batch,length]
        Output: 
            encoder_output: [batch,length,hidden_size]    
            encoder_hidden: [rnn_layers,batch,hidden_size]
        '''
        
        self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size
        self.device = device
        self.word_dim = word_dim
        self.nnDropout = nn.Dropout(dropout)
        self.elmo = elmo.to(device)
        # Initialize GRU; 
        self.gru = nn.GRU(word_dim, hidden_size, rnn_layers, batch_first=True,
                          dropout=(0 if rnn_layers == 1 else dropout), bidirectional=True)
        self.batchnorm_input = nn.BatchNorm1d(word_dim, affine=False, track_running_stats=False)
    


    def forward(self, input_sentence):
        
        # [batch, max_length,word_dimension]
        embeddings = self.GetEDURepresentation(input_sentence)

        # barch normalization
        embeddings = embeddings.permute(0,2,1) 
        embeddings = self.batchnorm_input(embeddings)
        embeddings = embeddings.permute(0,2,1) 

        # apply dropout
        embeddings = self.nnDropout(embeddings)
        
        input_lengths = [len(line) for line in input_sentence]    
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, input_lengths, batch_first = True)        
        #initialize hidden states
        batch_size = embeddings.size(0)
        hidden_initial = self.initHidden(batch_size)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden_initial)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # apply dropout
        outputs = outputs.contiguous()
        outputs = self.nnDropout(outputs)
        # Sum bidirectional GRU outputs (or concatenate)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]        
        # Obtain last hidden state of encoder
        hidden = hidden.contiguous()
        hidden = hidden[:self.rnn_layers,:,:] + hidden[self.rnn_layers:,:,:]
        return outputs, hidden
    
    

    def GetEDURepresentation(self, input_sentence):
          
        character_ids = batch_to_ids(input_sentence)
        character_ids = character_ids.to(self.device)
        elmo_embeddings = self.elmo(character_ids)
        input_embeddings = elmo_embeddings['elmo_representations'][0]
        input_embeddings = input_embeddings.to(self.device)

        return input_embeddings
        

    
    def initHidden(self, batch_size):

            h_0 = torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_size)
            h_0 = h_0.to(self.device)

            return h_0
    
  

   
class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_layers=6, dropout=0.2):
        
        super(DecoderRNN, self).__init__()
        
        '''
        
        
        Input:
            input: [1,length,input_size]
            initial_hidden_state: [rnn_layer,1,hidden_size]

        Output:
            output: [1,length,input_size]
            hidden_states: [rnn_layer,1,hidden_size]
            
        '''
        
        # Define GRU layer
        self.gru = nn.GRU(input_size, hidden_size, rnn_layers, batch_first=True, dropout=(0 if rnn_layers == 1 else dropout))
        


    def forward(self, input_hidden_states, last_hidden):
        
        # Forward through unidirectional GRU        
        
        outputs, hidden = self.gru(input_hidden_states, last_hidden)  

        # Return output and final hidden state
        return outputs, hidden
   
   
    

class PointerAtten(nn.Module):
    def __init__(self, atten_model,hidden_size):
        super(PointerAtten, self).__init__()       
        
        '''       
        Input:
            Encoder_outputs: [length,encoder_hidden_size]
            Current_decoder_output: [decoder_hidden_size] 
            Attention_model: 'Biaffine' or 'Dotproduct' 
            
        Output:
            attention_weights: [1,length]
            log_attention_weights: [1,length]
            
        '''

        self.atten_model = atten_model
        self.weight1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.weight2 = nn.Linear(hidden_size, 1, bias=False)
        

    def forward(self, encoder_outputs, cur_decoder_output):
        
        if self.atten_model == 'Biaffine':
            
            EW1_temp = self.weight1(encoder_outputs)
            EW1 = torch.matmul(EW1_temp,cur_decoder_output).unsqueeze(1)
            EW2 = self.weight2(encoder_outputs)
            bi_affine = EW1 + EW2
            bi_affine = bi_affine.permute(1,0)   

            # Obtain attention weights and logits (to compute loss)
            atten_weights = F.softmax(bi_affine,0)
            log_atten_weights = F.log_softmax(bi_affine,0)

             
        elif self.atten_model == 'Dotproduct':
            
            dot_prod = torch.matmul(encoder_outputs,cur_decoder_output).unsqueeze(0)            
            # Obtain attention weights and logits (to compute loss)
            atten_weights = F.softmax(dot_prod,1)
            log_atten_weights = F.log_softmax(dot_prod,1)

            
        # Return attention weights and log attention weights
        return atten_weights, log_atten_weights
  
    
    
class LabelClassifier(nn.Module):
    def __init__(self, input_size, classifier_hidden_size, classes_label=39, 
                 bias=True, dropout=0.5):

        super(LabelClassifier, self).__init__() 
        '''
        
        Args:
            input_size: input size
            classifier_hidden_size: project input to classifier space
            classes_label: corresponding to 39 relations we have. 
                           (e.g. Contrast_NN)
            bias: If set to False, the layer will not learn an additive bias.
                Default: True               

        Input:
            input_left: [1,input_size]
            input_right: [1,input_size]
        Output:
            relation_weights: [1,classes_label]
            log_relation_weights: [1,classes_label]
            
        '''
        self.classifier_hidden_size = classifier_hidden_size   
        self.labelspace_left = nn.Linear(input_size, classifier_hidden_size, bias=False)
        self.labelspace_right = nn.Linear(input_size, classifier_hidden_size, bias=False)
        self.weight_left = nn.Linear(classifier_hidden_size, classes_label, bias=False)
        self.weight_right = nn.Linear(classifier_hidden_size, classes_label, bias=False)
        self.nnDropout = nn.Dropout(dropout)
        
        if bias:
            self.weight_bilateral = nn.Bilinear(classifier_hidden_size, classifier_hidden_size, classes_label)
        else:
            self.weight_bilateral = nn.Bilinear(classifier_hidden_size, classifier_hidden_size, classes_label, bias=False)
        
        
    def forward(self,input_left,input_right):
        
        left_size = input_left.size()
        right_size = input_right.size()
        
        labelspace_left = F.elu(self.labelspace_left(input_left))
        labelspace_right = F.elu(self.labelspace_right(input_right))
        
        # Apply dropout
        union = torch.cat((labelspace_left,labelspace_right),1)
        union = self.nnDropout(union)
        labelspace_left = union[:,:self.classifier_hidden_size]
        labelspace_right = union[:,self.classifier_hidden_size:]
        output = (self.weight_bilateral(labelspace_left, labelspace_right) + 
                  self.weight_left(labelspace_left) + self.weight_right(labelspace_right))

        # Obtain relation weights and log relation weights (for loss) 
        relation_weights = F.softmax(output,1)
        log_relation_weights = F.log_softmax(output,1)

        return relation_weights, log_relation_weights

    
    
    
    
    
