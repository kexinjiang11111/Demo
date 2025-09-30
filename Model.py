# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from DynamicRNN import DynamicRNN
from GraphModel import ADGCN  # 新增导入

class dualModel(nn.Module):
    def __init__(self, opt, n_vocab, embed_list):
        super(dualModel, self).__init__()
        self.n_layers = opt.n_layers
        self.dim_input = opt.dim_input
        self.dim_hidden = opt.dim_hidden
        self.bidirecitional = True if opt.bidirectional else False  # int->bool
        self.rnn_type = opt.rnn_type
        self.device = opt.device
        self.multi_dim = opt.multi_dim
        self.max_length_sen = opt.max_length_sen
        self.dim_profile = 64
        dim_sen = opt.dim_hidden * (2 if self.bidirecitional else 1)
        self.t_sne = opt.t_sne # always together with predict mode

        # 创建三个BiRNN子模块，分别处理不同类型的输入
        self.add_module('literal', BiRNN_dualModel(opt, n_vocab, embed_list))  # 字面意义分析
        self.add_module('deep', BiRNN_dualModel(opt, n_vocab, embed_list))  # 深层语义分析
        self.add_module('sen', BiRNN_dualModel(opt, n_vocab, embed_list))  # 整体句子分析
        self.add_module('adgcn', ADGCN(opt, n_vocab, embed_list))  # 情感图分析

        # 降维层：将拼接后的特征投影回原始维度
        self.add_module('reduce_literal', nn.Linear(dim_sen*2, dim_sen))
        self.add_module('reduce_deep', nn.Linear(dim_sen*2, dim_sen))
        self.add_module('reduce_adgcn', nn.Linear(((dim_sen * 3)//2) , dim_sen))
        self.add_module('relu', nn.ReLU())  # 激活函数

        # 分类器：分别用于字面情感、深层语义和最终综合预测
        self.add_module('dense_literal', nn.Linear(dim_sen, opt.n_class))
        self.add_module('dense_deep', nn.Linear(dim_sen, opt.n_class))
        self.add_module('dense_adgcn', nn.Linear(dim_sen//2, opt.n_class))
        self.dense = nn.Linear(dim_sen*3, opt.n_class)

        params_senti = list(map(id, self.literal.embed.parameters()))
        params_nonsenti = list(map(id, self.deep.embed.parameters()))
        params_adgcn = list(map(id, self.adgcn.embed.parameters()))
        params_sen = list(map(id, self.sen.embed.parameters()))
        self.base_params = list(
            filter(lambda p: id(p) not in (params_sen + params_senti + params_nonsenti + params_adgcn),
                   self.parameters()))
        print('****parameter all set****')

    def forward(self, dict_inst):
        # 获取四个通道的特征表示
        literal_rep = self.literal(dict_inst['senti_sens'], dict_inst['senti_len_sen'])
        deep_rep = self.deep(dict_inst['nonsenti_sens'], dict_inst['nonsenti_len_sen'])
        sen_rep = self.sen(dict_inst['sens'], dict_inst['len_sen'])
        adgcn_rep = self.adgcn(dict_inst['sens'], dict_inst['len_sen'], dict_inst['dependency_graph'], dict_inst['sentic_graph'])  # 新增ADGCN输入
 
        # print("adgcn_rep 维度：", adgcn_rep.shape) 
        # print("sen_rep 维度：", sen_rep.shape) 
        # 特征融合与降维
        literal_rep_proj = self.relu(self.reduce_literal(torch.cat([literal_rep, sen_rep], dim=-1)))
        deep_rep_proj = self.relu(self.reduce_deep(torch.cat([deep_rep, sen_rep], dim=-1)))
        # adgcn_rep_proj = self.relu(self.reduce_adgcn(torch.cat([adgcn_rep, sen_rep], dim=-1)))  # 新增ADGCN处理

        # 中间分类器
        prob_senti = torch.softmax(self.dense_literal(literal_rep), dim=-1)
        prob_nonsenti = torch.softmax(self.dense_deep(deep_rep), dim=-1)
        # prob_adgcn = torch.softmax(self.dense_adgcn(adgcn_rep), dim=-1)  # 新增ADGCN分类结果

        # 最终分类器 - 拼接三个通道的特征
        # dense_input1 = torch.cat([literal_rep_proj, deep_rep_proj], dim=-1)  
        dense_input = torch.cat([literal_rep_proj, deep_rep_proj, adgcn_rep], dim=-1)  
        prob = torch.softmax(self.dense(dense_input), dim=-1)
        
        
        if self.t_sne:
            return prob, prob_senti, prob_nonsenti, literal_rep_proj, deep_rep_proj, dense_input
        else:
            return prob, prob_senti, prob_nonsenti
        
        
        # 返回结果
        # if self.t_sne:
        #     return prob, prob_senti, prob_nonsenti, prob_adgcn, literal_rep_proj, deep_rep_proj, adgcn_rep_proj, dense_input
        # else:
        #     return prob, prob_senti, prob_nonsenti, prob_adgcn

class BiRNN_dualModel(nn.Module):
    def __init__(self, opt, n_vocab, embed_list):
        super(BiRNN_dualModel, self).__init__()
        self.n_layers = opt.n_layers
        self.dim_input = opt.dim_input
        self.dim_hidden = opt.dim_hidden
        self.bidirecitional = True if opt.bidirectional else False  # int->bool
        self.rnn_type = opt.rnn_type
        self.device = opt.device

        self.add_module('embed', nn.Embedding(n_vocab, self.dim_input))
        self.add_module('embed_dropout', nn.Dropout(opt.embed_dropout_rate))
        self.add_module('rnn_sen', DynamicRNN(opt.dim_input, opt.dim_hidden, opt.n_layers,
                                           dropout=(opt.cell_dropout_rate if opt.n_layers > 1 else 0),
                                           bidirectional=self.bidirecitional, rnn_type=opt.rnn_type, device=opt.device))
        self.init_weights(embed_list)

    def init_weights(self, embed_list):
        self.embed.weight.data.copy_(torch.from_numpy(embed_list))

    def forward(self, embed_list, length_list):
        embedded = self.embed_dropout(self.embed(embed_list))

        output_pad, hidden_encoder = self.rnn_sen(embedded, lengths=length_list, flag_ranked=False)

        max_len = output_pad.size(1)
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        mask = Variable((idxes < torch.LongTensor(length_list).unsqueeze(1)).float()).to(self.device)
        output_pad = output_pad * mask.unsqueeze(-1)
        length_tensor = torch.from_numpy(length_list).float().to(self.device)
        r_s = torch.sum(output_pad, dim=1).transpose(0, 1) / length_tensor
        r_s = r_s.transpose(0, 1)
      


        return r_s
