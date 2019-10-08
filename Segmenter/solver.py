
import pickle
import torch.optim as optim
import numpy as np
import torch
from torch.autograd import Variable

import random
from torch.nn.utils import clip_grad_norm
import copy
import os





def sample_a_sorted_batch_from_numpy(inputX,outputY,batch_size,use_cuda):


    inputX = np.array(inputX)
    outputY = np.array(outputY)

    if batch_size != None:
        select_index = random.sample(range(len(outputY)), batch_size)
    else:
        select_index = np.array(range(len(outputY)))

    batch_x = copy.deepcopy(inputX[select_index])
    batch_y = copy.deepcopy(outputY[select_index])

    all_lens = np.array([len(x) for x in batch_x])
    
    idx = np.argsort(all_lens)
    idx = idx[::-1]  # decreasing

    batch_x = batch_x[idx]

    batch_y = batch_y[idx]

    # decoder input
    batch_x_index = []

    for i in range(len(batch_y)):
        cur_y = batch_y[i]
        temp = cur_y
        
        temp = [x+1 for x in cur_y]
        temp.insert(0,0)
        temp.pop()
        batch_x_index.append(temp)





    all_lens = all_lens[idx]
    


    return  batch_x, batch_x_index, batch_y, all_lens


def get_batch_test(inputX,outputY,batch_size,use_cuda):



    inputX = np.array(inputX)
    outputY = np.array(outputY)


    if batch_size != None:
        select_index = random.sample(range(len(outputY)), batch_size)
    else:
        select_index = np.array(range(len(outputY)))

    batch_x = copy.deepcopy(inputX[select_index])
    batch_y = copy.deepcopy(outputY[select_index])

    all_lens = np.array([len(x) for x in batch_x])
    

    return  batch_x, batch_y, all_lens





class TrainSolver(object):
    def __init__(self, model,train_x,train_y,dev_x,dev_y,save_path,batch_size,eval_size,epoch, lr,lr_decay_epoch,weight_decay,use_cuda):

        self.lr = lr
        self.model = model
        self.epoch = epoch
        self.train_x = train_x
        self.train_y = train_y
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.lr_decay_epoch = lr_decay_epoch
        self.eval_size  = eval_size
        self.dev_x, self.dev_y = dev_x, dev_y
        self.model = model
        self.save_path = save_path
        self.weight_decay =weight_decay




    def sample_dev(self):

        select_index = random.sample(range(len(self.train_y)),self.eval_size)

        train_x = np.array(self.train_x)
        train_y = np.array(self.train_y)
        test_tr_x = train_x[select_index]
        test_tr_y = train_y[select_index]

        return test_tr_x,test_tr_y







    def get_batch_micro_metric(self,pre_b, ground_b):



        All_C = []
        All_R = []
        All_G = []
        for i in range(len(ground_b)):
            index_of_1 = np.array(ground_b[i])
            index_pre = pre_b[i]

            index_pre = np.array(index_pre)

            
            END_B = index_of_1[-1]
            index_pre = index_pre[index_pre != END_B]
            index_of_1 = index_of_1[index_of_1 != END_B]

            no_correct = len(np.intersect1d(list(index_of_1), list(index_pre)))
            All_C.append(no_correct)
            All_R.append(len(index_pre))
            All_G.append(len(index_of_1))


        return All_C,All_R,All_G





    def get_batch_metric(self,pre_b, ground_b):

        b_pr =[]
        b_re =[]
        b_f1 =[]
        for i,cur_seq_y in enumerate(ground_b):
            index_of_1 = np.where(cur_seq_y==1)[0]
            index_pre = pre_b[i]

            no_correct = len(np.intersect1d(index_of_1,index_pre))

            cur_pre = no_correct / len(index_pre)
            cur_rec = no_correct / len(index_of_1)
            cur_f1 = 2*cur_pre*cur_rec/ (cur_pre+cur_rec)

            b_pr.append(cur_pre)
            b_re.append(cur_rec)
            b_f1.append(cur_f1)

        return b_pr,b_re,b_f1



    def check_accuracy(self,dataX,dataY):


        need_loop = int(np.ceil(len(dataY) / self.batch_size))

        all_ave_loss =[]
        all_boundary =[]
        all_boundary_start = []
        all_align_matrix = []
        all_index_decoder_y =[]
        all_x_save = []

        all_C =[]
        all_R =[]
        all_G =[]
        for lp in range(need_loop):
            startN = lp*self.batch_size
            endN =  (lp+1)*self.batch_size
            if endN > len(dataY):
                endN = len(dataY)

            batch_x, batch_y, all_lens = get_batch_test(
                dataX[startN:endN], dataY[startN:endN], None, self.use_cuda)


            batch_ave_loss, batch_boundary, batch_boundary_start, _ = self.model.predict(batch_x, batch_y,
                                                                                                  all_lens)
            all_ave_loss.extend([batch_ave_loss.data[0]])
            all_boundary.extend(batch_boundary)
            all_boundary_start.extend(batch_boundary_start)



            ba_C,ba_R,ba_G = self.get_batch_micro_metric(batch_boundary, batch_y)

            all_C.extend(ba_C)
            all_R.extend(ba_R)
            all_G.extend(ba_G)


        ba_pre = np.sum(all_C)/ np.sum(all_R)
        ba_rec = np.sum(all_C)/ np.sum(all_G)
        ba_f1 = 2*ba_pre*ba_rec/ (ba_pre+ba_rec)


        return np.mean(all_ave_loss),ba_pre,ba_rec,ba_f1, (all_x_save,all_index_decoder_y,all_boundary, all_boundary_start)#, all_align_matrix)


    def adjust_learning_rate(self,optimizer,epoch,lr_decay=0.5, lr_decay_epoch=50):

        if (epoch % lr_decay_epoch == 0) and (epoch != 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay



    def train(self):

        self.test_train_x, self.test_train_y = self.sample_dev()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay)



        num_each_epoch = int(np.round(len(self.train_y) / self.batch_size))

        os.mkdir(self.save_path)

        best_i =0
        best_f1 =0

        for epoch in range(self.epoch):

            self.adjust_learning_rate(optimizer, epoch, 0.8, self.lr_decay_epoch)

            track_epoch_loss = []
            for iter in range(num_each_epoch):
                print("epoch:%d,iteration:%d" % (epoch, iter))
               


                batch_x, batch_x_index, batch_y, all_lens = sample_a_sorted_batch_from_numpy(
                    self.train_x, self.train_y, self.batch_size, self.use_cuda)
                

                self.model.zero_grad()

                neg_loss = self.model.neg_log_likelihood(batch_x, batch_x_index, batch_y, all_lens)



                neg_loss_v = float(neg_loss.data[0])

                print(neg_loss_v)
                track_epoch_loss.append(neg_loss_v)

                neg_loss.backward()

                clip_grad_norm(self.model.parameters(), 5)
                optimizer.step()


            #TODO: after each epoch,check accuracy


            self.model.eval()

            tr_batch_ave_loss, tr_pre, tr_rec, tr_f1 ,visdata1 = self.check_accuracy(self.test_train_x,self.test_train_y)

            dev_batch_ave_loss, dev_pre, dev_rec, dev_f1, visdata = self.check_accuracy(self.dev_x,self.dev_y)
            _, _, all_boundary, _ = visdata
            
            print()

            if best_f1 < dev_f1:
                best_f1 = dev_f1
                best_rec = dev_rec
                best_pre = dev_pre
                best_i = epoch



            save_data = [epoch,tr_batch_ave_loss,tr_pre,tr_rec,tr_f1,
                         dev_batch_ave_loss,dev_pre,dev_rec,dev_f1]
            # save_data = [epoch, dev_batch_ave_loss,dev_pre,dev_rec,dev_f1]


            save_file_name = 'bs_{}_es_{}_lr_{}_lrdc_{}_wd_{}_epoch_loss_acc_pk_wd.txt'.format(self.batch_size,self.eval_size,self.lr,self.lr_decay_epoch,self.weight_decay)
            with open(os.path.join(self.save_path,save_file_name), 'a+') as f:
                f.write(','.join(map(str,save_data))+'\n')


            if epoch == best_i and epoch >= 50:
                torch.save(self.model, os.path.join(self.save_path,r'model_epoch_%d.torchsave'%(epoch)))
                
                with open(os.path.join(self.save_path, r'Best_Segmentation_%d.pickle' % (epoch)), 'wb') as f:
                        pickle.dump(all_boundary , f)


            self.model.train()
            
            

        return best_i,best_pre,best_rec,best_f1



