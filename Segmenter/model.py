

import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import numpy as np



#Using ELMo contextual word embeddings
from allennlp.modules.elmo import Elmo, batch_to_ids


weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
elmo = Elmo(options_file, weight_file, 2, dropout=0.5,requires_grad=False)
elmo.cuda()







class PointerNetworks(nn.Module):
    def __init__(self,word_dim, hidden_dim,is_bi_encoder_rnn,rnn_type,rnn_layers,
                 dropout_prob,use_cuda,finedtuning,isbanor):
        super(PointerNetworks,self).__init__()

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.is_bi_encoder_rnn = is_bi_encoder_rnn
        self.num_rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.finedtuning = finedtuning

        self.nnDropout = nn.Dropout(dropout_prob)

        self.isbanor = isbanor


        if rnn_type in ['LSTM', 'GRU']:



            self.decoder_rnn = getattr(nn, rnn_type)(input_size= 2 * hidden_dim if is_bi_encoder_rnn else hidden_dim,
                                                     hidden_size=2 * hidden_dim if is_bi_encoder_rnn else hidden_dim,
                                                     num_layers=rnn_layers,
                                                     dropout=dropout_prob,
                                                     batch_first=True)

            self.encoder_rnn = getattr(nn, rnn_type)(input_size=word_dim,
                                       hidden_size=hidden_dim,
                                       num_layers=rnn_layers,
                                       bidirectional=is_bi_encoder_rnn,
                                       dropout=dropout_prob,
                                       batch_first=True)



        else:
            print('rnn_type should be LSTM,GRU')



        self.use_cuda = use_cuda


        if self.is_bi_encoder_rnn:
            self.num_encoder_bi = 2
        else:
            self.num_encoder_bi = 1



    def initHidden(self,hsize,batchsize):


        if self.rnn_type == 'LSTM':

            h_0 = Variable(torch.zeros(self.num_encoder_bi*self.num_rnn_layers, batchsize, hsize))
            c_0 = Variable(torch.zeros(self.num_encoder_bi*self.num_rnn_layers, batchsize, hsize))

            if self.use_cuda:
                h_0 = h_0.cuda()
                c_0 = c_0.cuda()

            return (h_0, c_0)
        else:

            h_0 = Variable(torch.zeros(self.num_encoder_bi*self.num_rnn_layers, batchsize, hsize))

            if self.use_cuda:
                h_0 = h_0.cuda()


            return h_0







    def _run_rnn_packed(self, cell, x, x_lens, h=None):
        x_packed = R.pack_padded_sequence(x, x_lens,
                                          batch_first=True)

        if h is not None:
            output, h = cell(x_packed, h)
        else:
            output, h = cell(x_packed)

        output, _ = R.pad_packed_sequence(output, batch_first=True)

        return output, h





    def pointerEncoder(self,Xin_ELMo,lens):
        
                
        self.bn_inputdata = nn.BatchNorm1d(self.word_dim, affine=False, track_running_stats=False)

        batch_size = len(Xin_ELMo) 
        
        #to convert input to ELMo embeddings
        character_ids = batch_to_ids(Xin_ELMo)
        if self.use_cuda:
            character_ids = character_ids.cuda()
        embeddings = elmo(character_ids)
        X_ELMo = embeddings['elmo_representations'][0] #two layers output  [batch,length,d_elmo]     
        if self.use_cuda:
            X_ELMo = X_ELMo.cuda()
           

        
        X = X_ELMo
        if self.isbanor:
            X= X.permute(0,2,1) # N C L
            X = self.bn_inputdata(X)
            X = X.permute(0, 2, 1) # N L C

        X = self.nnDropout(X)



        encoder_lstm_co_h_o = self.initHidden(self.hidden_dim, batch_size)
        output_encoder, hidden_states_encoder = self._run_rnn_packed(self.encoder_rnn, X, lens, encoder_lstm_co_h_o)  # batch_first=True
        output_encoder = output_encoder.contiguous()
        output_encoder = self.nnDropout(output_encoder)

        


        return output_encoder, hidden_states_encoder


    def pointerLayer(self,encoder_states,cur_decoder_state):
        """

        :param encoder_states:  [Length, hidden_size]
        :param cur_decoder_state:  [hidden_size,1]
        """
        
        
        
        #we use simple dot product attention to computer pointer
        attention_pointer = torch.matmul(encoder_states,cur_decoder_state).unsqueeze(1)
        attention_pointer = attention_pointer.permute(1,0)
        
        #TODO: for log loss
        att_weights = F.softmax(attention_pointer)
        logits = F.log_softmax(attention_pointer)




        return logits,att_weights







    def training_decoder(self,hn,hend,Xindex,Yindex,lens):
        """
        Here, we use encoder hidden states as the input of decoder, instead of 
        corresponding word embedding.

        """


        loss_function  = nn.NLLLoss()
        batch_loss =0
        LoopN = 0
        batch_size = len(lens)
        for i in range(len(lens)): #Loop batch size

            curX_index = Xindex[i]
            curY_index = Yindex[i]
            curL = lens[i]
            curencoder_hn = hn[i,0:curL,:]  #[length, encoder_hidden_size]
            
            # x_index_var = Variable(torch.from_numpy(curX_index.astype(np.int64)))
            x_index_var = torch.tensor(curX_index)
            if self.use_cuda:
                x_index_var = x_index_var.cuda()
                
            curEncoder_Hidden_states = curencoder_hn[x_index_var]    #[no_segmentation, encoder_hidden_size]
            curEncoder_Hidden_states = curEncoder_Hidden_states.unsqueeze(0)    


            if self.rnn_type =='LSTM':# need h_end,c_end


                h_end = hend[0].permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)
                c_end = hend[1].permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)

                curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                curc0 = c_end[i].unsqueeze(0).permute(1, 0, 2)


                h_pass = (curh0,curc0)
            else:


                h_end = hend.permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)  
                curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                h_pass = curh0



            decoder_out, _ = self.decoder_rnn(curEncoder_Hidden_states,h_pass)
            decoder_out = decoder_out.squeeze(0)   #[no_segmentation,decoder_hidden_size]

            
            for j in range(len(decoder_out)):  #Loop every decoder hidden states
                cur_dj = decoder_out[j]
                cur_groundy = curY_index[j]

                cur_start_index = curX_index[j]
                predict_range = list(range(cur_start_index,curL))

                # TODO: make it point backward, only consider predict_range in current time step
                # align groundtruth
                cur_groundy_var = Variable(torch.LongTensor([int(cur_groundy) - int(cur_start_index)]))
                if self.use_cuda:
                    cur_groundy_var = cur_groundy_var.cuda()

                curencoder_hn_back = curencoder_hn[predict_range,:]



                
                cur_logists, cur_weights = self.pointerLayer(curencoder_hn_back,cur_dj)

                batch_loss = batch_loss + loss_function(cur_logists,cur_groundy_var)
                LoopN = LoopN +1

        batch_loss = batch_loss/LoopN

        return batch_loss


    def neg_log_likelihood(self,BatchX,IndexX, IndexY,lens):

        '''
        :param Xin_glove: [batch,length]
        :param Xin_ELMo: [batch,length(no padding)]
        :param lens: stack of lenth of sentences in a batch
        '''


        encoder_hn, encoder_h_end = self.pointerEncoder(BatchX,lens)

        loss = self.training_decoder(encoder_hn, encoder_h_end, IndexX, IndexY,lens)

        return loss




    def test_decoder(self,hn,hend,Yindex,lens):
        
        
#        Similar to training a decoder, we use encoder hidden states as the input of decoder, instead of 
#        corresponding word embedding.

        
       
        
        loss_function = nn.NLLLoss()
        batch_loss = 0
        LoopN = 0

        batch_boundary =[]
        batch_boundary_start =[]
        batch_align_matrix =[]

        batch_size = len(lens)

        for i in range(len(lens)):  # Loop batch size



            curL = lens[i]
            curY_index = Yindex[i]
            curencoder_hn = hn[i,0:curL,:]  #length * encoder_hidden_size  
            cur_end_boundary =curY_index[-1]

            cur_boundary = []
            cur_b_start = []
            cur_align_matrix = []



            if self.rnn_type =='LSTM':# need h_end,c_end


                h_end = hend[0].permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)
                c_end = hend[1].permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)

                curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                curc0 = c_end[i].unsqueeze(0).permute(1, 0, 2)

                h_pass = (curh0,curc0)
            else: # only need h_end


                h_end = hend.permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)
                curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                h_pass = curh0


            Not_break = True

            loop_hc = h_pass
            loop_in = curencoder_hn[0,:].unsqueeze(0).unsqueeze(0)  #[ 1, 1, encoder_hidden_size] (first start)

            loopstart =0
            loop_j =0
            
            while (Not_break): #if not end

                loop_o, loop_hc = self.decoder_rnn(loop_in,loop_hc)


                #TODO: make it point backward

                predict_range = list(range(loopstart,curL))
                curencoder_hn_back = curencoder_hn[predict_range,:]
                cur_logists, cur_weights = self.pointerLayer(curencoder_hn_back, loop_o.squeeze(0).squeeze(0))

                cur_align_vector = np.zeros(curL)
                cur_align_vector[predict_range]=cur_weights.data.cpu().numpy()[0]
                cur_align_matrix.append(cur_align_vector)

                #TODO:align groundtruth
                if loop_j > len(curY_index)-1:
                    cur_groundy = curY_index[-1]
                else:
                    cur_groundy = curY_index[loop_j]


                cur_groundy_var = Variable(torch.LongTensor([max(0,int(cur_groundy) - loopstart)]))
                if self.use_cuda:
                    cur_groundy_var = cur_groundy_var.cuda()

                batch_loss = batch_loss + loss_function(cur_logists, cur_groundy_var)


                #TODO: get predicted boundary
                topv, topi = cur_logists.data.topk(1)

                pred_index = topi[0][0]


                #TODO: align pred_index to original seq
                ori_pred_index =pred_index + loopstart


                if cur_end_boundary <= ori_pred_index:
                    cur_boundary.append(cur_end_boundary)
                    cur_b_start.append(loopstart)
                    Not_break = False
                    loop_j = loop_j + 1
                    LoopN = LoopN + 1
                    break
                else:
                    cur_boundary.append(ori_pred_index)
                    
                    loop_in = curencoder_hn[ori_pred_index+1,:].unsqueeze(0).unsqueeze(0)
                    cur_b_start.append(loopstart)

                    loopstart = ori_pred_index+1  # start =  pred_end + 1

                    loop_j = loop_j + 1
                    LoopN = LoopN + 1


            #For each instance in batch
            batch_boundary.append(cur_boundary)
            batch_boundary_start.append(cur_b_start)
            batch_align_matrix.append(cur_align_matrix)

        batch_loss = batch_loss / LoopN

        batch_boundary=np.array(batch_boundary)
        batch_boundary_start = np.array(batch_boundary_start)
        batch_align_matrix = np.array(batch_align_matrix)

        return batch_loss,batch_boundary,batch_boundary_start,batch_align_matrix



    def predict(self,Xin_batch, IndexY,lens):
 
        batch_boundary = []
        batch_boundary_start = []

        for i in range(len(Xin_batch)):
            X_in = [Xin_batch[i]]
            len_in = [lens[i]]
            cur_IndexY = [IndexY[i]]

            cur_encoder_hn, cur_encoder_h_end = self.pointerEncoder(X_in, len_in)
            
            

            # cur_loss, cur_boundary, cur_boundary_start, _ = self.test_decoder(encoder_hn,encoder_h_end,IndexY,lens)
            cur_loss, cur_boundary, cur_boundary_start, _ = self.test_decoder(cur_encoder_hn,cur_encoder_h_end,cur_IndexY,len_in)


            batch_loss = batch_loss + cur_loss
            batch_boundary.append(cur_boundary)
            batch_boundary_start.append(cur_boundary_start)


        return  batch_loss,batch_boundary,batch_boundary_start, None





















