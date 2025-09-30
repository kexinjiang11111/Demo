# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

from basicModel import Lang
from Model import dualModel

class bridgeModel(nn.Module):
    def __init__(self, FLAGS, vocab=None, embed=None):
        super(bridgeModel, self).__init__()

        self.device = FLAGS.device
        self.max_length_sen = FLAGS.max_length_sen
        self.n_class = FLAGS.n_class
        self.learning_rate = FLAGS.learning_rate
        self.batch_size = FLAGS.batch_size
        self.lambda1 = FLAGS.lambda1
        self.bidirectional = True if FLAGS.bidirectional else False
        print('****Bidirectional in bridgeModel is {}'.format(self.bidirectional))
        self.margin = FLAGS.margin
        self.supervised = True if FLAGS.supcon else False
        self.t_sne = FLAGS.t_sne

        #Lang处理词汇表，用于文本预处理和词索引转换
        self.lang = Lang(vocab)

        self.model = dualModel(FLAGS, len(vocab), embed)
        self.model.to(self.device)

        #优化器配置
        self.optimizer = getattr(optim, FLAGS.optim_type)([
            #参数组1: 基础参数 学习率: 继承全局基础学习率 权重衰减: FLAGS.weight_decay (启用)
            {'params': self.model.base_params, 'weight_decay': FLAGS.weight_decay},
            #参数组2: 句子嵌入层 学习率: FLAGS.lr_word_vector (覆盖全局) 权重衰减: 0 (禁用)
            {'params': self.model.sen.embed.parameters(), 'lr': FLAGS.lr_word_vector, 'weight_decay':0},
            #参数组3: 字面嵌入层 → 冻结 学习率: 0 (固定参数) 权重衰减: 0
            {'params': self.model.literal.embed.parameters(), 'lr': 0, 'weight_decay':0},
            #参数组4: 深度嵌入层 → 冻结 学习率: 0 (固定参数) 权重衰减: 0
            {'params': self.model.deep.embed.parameters(), 'lr': 0, 'weight_decay': 0},
            #参数组5: ADGCN嵌入层 → 冻结 学习率: 0 (固定参数) 权重衰减: 0
            {'params': self.model.adgcn.embed.parameters(), 'lr': 0, 'weight_decay': 0}],
            lr=self.learning_rate)

    #数据预处理
    def gen_batch_data(self, batched_data):
        dict_data = {}
        #处理主句子，将句子转化为词索引张量并记录长度
        dict_data['sens'] = self.lang.VariablesFromSentences(batched_data['sentences'], True, self.device)
        dict_data['len_sen'] = batched_data['length_sen']
        #处理情感表达句子
        dict_data['senti_sens'] = self.lang.VariablesFromSentences(batched_data['sentis'], True, self.device)
        dict_data['senti_len_sen'] = batched_data['length_senti']
        #处理非情感表达句子
        dict_data['nonsenti_sens'] = self.lang.VariablesFromSentences(batched_data['nonsentis'], True, self.device)
        dict_data['nonsenti_len_sen'] = batched_data['length_nonsenti']
        # 处理ADGCN相关数据
        dict_data['dependency_graph'] = torch.FloatTensor(batched_data['dependency_graphs']).to(self.device)
        dict_data['sentic_graph'] = torch.FloatTensor(batched_data['sentic_graphs']).to(self.device)
        #处理标签数据
        literals = Variable(torch.LongTensor(batched_data['literals']))
        dict_data['literals'] = literals.to(self.device) if self.device else literals
        deeps = Variable(torch.LongTensor(batched_data['deeps']))
        dict_data['deeps'] = deeps.to(self.device) if self.device else deeps
        sarcasms = Variable(torch.LongTensor(batched_data['sarcasms']))
        dict_data['sarcasms'] = sarcasms.to(self.device) if self.device else sarcasms
        return dict_data

    #预测
    def predict(self, batched_data):
        self.model.eval() #设置为评估模式
        b_data = self.gen_batch_data(batched_data) # padded idx
        #禁用梯度计算
        with torch.no_grad():
            if self.t_sne:
                prob, _, _, _, literal_rep, deep_rep,  sen_rep = self.model(b_data)
                # prob, _, _, _, literal_rep, deep_rep, adgcn_rep, sen_rep = self.model(b_data)
            else:
                prob, _, _, _ = self.model(b_data)
        #
        label_idx = [tmp.item() for tmp in torch.argmax(prob, dim=-1)]
        if self.t_sne:
            # return label_idx, literal_rep, deep_rep, adgcn_rep, sen_rep
            return label_idx, literal_rep, deep_rep, sen_rep
        else:
            return label_idx

    def stepTrain(self, batched_data, inference=False):
        self.model.eval() if inference else self.model.train()

        if inference == False:
            self.optimizer.zero_grad()

        b_data = self.gen_batch_data(batched_data)

        prob, prob_senti, prob_nonsenti = self.model(b_data)
        loss_ce = F.nll_loss(torch.log(prob), b_data['sarcasms'])
        loss_senti = F.nll_loss(torch.log(prob_senti), b_data['literals'])
        loss_nonsenti = F.nll_loss(torch.log(prob_nonsenti), b_data['deeps'])
        loss_all = loss_ce + loss_senti + loss_nonsenti 


        if inference == False:
            loss_all.backward()
            self.optimizer.step()
        return np.array([loss_all.data.cpu().numpy(), loss_ce.data.cpu().numpy(), loss_senti.data.cpu().numpy(),
                         loss_nonsenti.data.cpu().numpy()]).reshape(
            4), prob.data.cpu().numpy(), prob_senti.data.cpu().numpy(), prob_nonsenti.data.cpu().numpy()

    def save_model(self, dir, idx):
        os.mkdir(dir) if not os.path.isdir(dir) else None
        torch.save(self.state_dict(), '%s/model%s.pth' % (dir, idx))
        print('****save state dict****')
        print(self.state_dict().keys())

    def load_model(self, dir, idx=-1, device='cpu'):
        if idx < 0:
            params = torch.load('%s' % dir)
            self.load_state_dict(params)
        else:
            print('****load state dict****')
            print(self.state_dict().keys())
            self.load_state_dict(torch.load(f'{dir}/model{idx}.pth', map_location=device))