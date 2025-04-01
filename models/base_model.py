import torch
import dgl
from torch.nn import Module
import torch.nn as nn
import numpy as np
from utils import compute_norm, node_norm_to_edge_norm, filter_none
import torch.nn.functional as F
from pprint import pprint
from utils import *


class BaseModel(Module):
    def __init__(self, args, data, device):
        super(BaseModel, self).__init__()
        self.args = args
        self.data = data
        self.device = device
        self.num_ent, self.train_data, self.valid_data, self.test_data, self.num_rels = self.data.num_nodes, self.data.train, self.data.valid, self.data.test, self.data.num_rels
        self.edge_index, self.edge_type = self.construct_adj()

        # self.ent_embeds = nn.Parameter(torch.Tensor(self.num_ent, self.args.embed_size)).to(self.device)
        # self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels*2, self.args.embed_size)).to(self.device)

        # nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        self.ent_embeds = get_param((self.num_ent, self.args.embed_size)).to(self.device)
        self.rel_embeds = get_param((self.num_rels*2, self.args.embed_size)).to(self.device)
        self.loss = nn.BCELoss()
        # self.loss = nn.CrossEntropyLoss()
        self.ent_encoder = None
        
        # self.g = self.build_graph().to(self.device)
        # self.g = self.g.local_var()
        # self.g.edata["id"] = self.edge_type.to(self.device)
        # # self.g.ndata['norm'] = compute_norm(self.g).view(-1, 1).to(self.device)
        # norm = compute_norm(self.g).view(-1, 1).to(self.device)
        # self.g.edata["norm"] = node_norm_to_edge_norm(self.g, norm)
        
        
        self.bias = nn.Parameter(torch.zeros(self.num_rels*2)).to(self.device)

        # ConvE
        # self.conve_hid_drop = 0.3
        # self.feat_drop = 0.3
        # self.input_drop = 0.2
        # self.k_w = 10
        # self.k_h = 20
        # self.num_filt = 200
        # self.ker_sz = 7

        self.conve_hid_drop = 0.4
        self.feat_drop = 0.1
        self.input_drop = 0.2
        self.k_w = 10
        self.k_h = 20
        self.num_filt = 250
        self.ker_sz = 7
        
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(self.num_filt)  # do bn on output of conv
        self.bn2 = nn.BatchNorm1d(self.args.embed_size)
        
        self.input_drop = nn.Dropout(
            self.input_drop)  # stacked input dropout
        self.feature_drop = nn.Dropout(
            self.feat_drop)  # feature map dropout
        self.hidden_drop = nn.Dropout(
            self.conve_hid_drop)  # hidden layer dropout
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=self.num_filt,
                                      kernel_size=(self.ker_sz, self.ker_sz), stride=1, padding=0, bias=True)
        
        # flat_sz_h = int(3 * self.k_h) - self.ker_sz + 1  # height after conv #头 尾 pair-wise
        # flat_sz_w = self.k_w - self.ker_sz + 1  # width after conv
        flat_sz_h = int(3 * self.k_w) - self.ker_sz + 1  # height after conv #头 尾 pair-wise
        flat_sz_w = self.k_h - self.ker_sz + 1  # width after conv
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        # fully connected projection
        self.fc = nn.Linear(self.flat_sz, self.args.embed_size)
    
    
    def construct_adj(self):
        """
        Constructor of the runner class

        Parameters
        ----------

        Returns
        -------
        Constructs the adjacency matrix for GCN

        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.train_data:
            edge_index.append((sub, obj))
            edge_type.append(rel)
        # Adding inverse edges
        for sub, rel, obj in self.train_data:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.num_rels)
    
        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type
    
    def concat(self, ent_embed, sub_embed, Heuristic):
        """
        :param ent_embed: [batch_size, embed_dim]
        :param sub_embed: [batch_size, embed_dim]
        :return: stack_input: [B, C, H, W]
        """
        ent_embed = ent_embed.view(-1, 1, self.args.embed_size)
        sub_embed = sub_embed.view(-1, 1, self.args.embed_size)
        Heuristic = Heuristic.view(-1, 1, self.args.embed_size)
        # [batch_size, 2, embed_dim]
        stack_input = torch.cat([ent_embed, sub_embed, Heuristic], 1)

        assert self.args.embed_size == self.k_h * self.k_w
        # reshape to 2D [batch, 1, 2*k_h, k_w]
        # stack_input = stack_input.reshape(-1, 1, 3 * self.k_w, self.k_h)
        stack_input = torch.transpose(stack_input, 2, 1).reshape(-1, 1, 3 * self.k_w, self.k_h)
        return stack_input
    
    def forward(self, triplets):
        # 输入一个batch数据，返回loss 
        sub, obj = triplets[:, 0].to(self.device), triplets[:, 2].to(self.device)
        # sub_gnn_node, obj_gnn_node, optpair_info_embed, self.rel_embeds = self.ent_encoder.forward(self.g, self.rel_embeds, sub, obj)  # [batch_size, num_ent]
        sub_gnn_node, obj_gnn_node, optpair_info_embed, rel_embeds = self.ent_encoder.forward(self.ent_embeds, self.edge_index, self.edge_type, self.rel_embeds, sub, obj)  # [batch_size, num_ent]
        score = self.decoder(sub_gnn_node, obj_gnn_node, optpair_info_embed, rel_embeds, score_function='conve')
        
        return score

    def evaluate(self, triplets, labels, split='valid'):
        rel =  triplets[:, 1].to(self.device)
        labels = labels.to(self.device)
        pred = self.forward(triplets)
        pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred) #处理nan值
        # pred[pred!=pred] = 0 # 处理nan值
        loss = self.calc_loss(pred, labels)
        b_range = torch.arange(pred.shape[0], device=self.device)
        # [batch_size, 1], get the predictive score of obj
        target_pred = pred[b_range, rel]
        # label=>-1000000, not label=>pred, filter out other objects with same sub&rel pair
        pred = torch.where(
            labels.bool(), -torch.ones_like(pred) * 10000000, pred)
        # copy predictive score of obj to new pred
        pred[b_range, rel] = target_pred
        ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
            b_range, rel]  # get the rank of each (sub, rel, obj)
        ranks = ranks.float()
        
        return ranks, loss
    
    # def build_graph(self):
    #     g = dgl.DGLGraph()
    #     g.add_nodes(self.num_ent)
    #     g.add_edges(self.train_data[:, 0], self.train_data[:, 2])
    #     g.add_edges(self.train_data[:, 2], self.train_data[:, 0])
    #     return g
    
    def calc_loss(self, pred, label):
        return self.loss(pred, label)
      

    def decoder(self, sub_gnn_node, obj_gnn_node, optpair_info_embed, rel_embeds, score_function='conve'):
        if score_function == "conve":
          # [batch_size, 1, 2*k_h, k_w]
          stack_input = self.concat(sub_gnn_node, obj_gnn_node, optpair_info_embed)
          x = self.bn0(stack_input)
          x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
          x = self.bn1(x)
          x = F.relu(x)
          x = self.feature_drop(x)
          x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
          x = self.fc(x)  # [batch_size, embed_dim]
          x = self.hidden_drop(x)
          x = self.bn2(x)
          # print(x)
          x = F.relu(x)
          # if self.add_heuristic:
          #     # x = self.mlp2(torch.cat([x, Heuristic], dim=1))
          #     gate = self.mlp3(torch.cat([x, Heuristic], dim=1))
          #     x = gate * x + (1-gate) * Heuristic
          # print(rel_embeds)
          # print(x)
          x = torch.mm(x, rel_embeds.transpose(1, 0))  # [batch_size, ent_num]
          x += self.bias.expand_as(x)
          # score = torch.sigmoid(x)
          # print(x.grad_fn)
          score = torch.softmax(x, 1)
          # print(score)
          return score


    
