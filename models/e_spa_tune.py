from torch.nn import Module, ModuleList
import torch.nn as nn
import torch
from models.operations_new import G_OPS, LAYER_FUSION_OPS, LAYER_CONNECT_OPS, NODE_INFO_OPS, PAIR_INFO_OPS, NODE_FUSION_OPS, PAIR_FUSION_OPS
import torch.nn.functional as F
import numpy as np

class NaOp(Module):
    def __init__(self, primitive, input_dim, output_dim, rel_num, base_num, dropout, head_num):
        super(NaOp, self).__init__()
        self._op = G_OPS[primitive](input_dim, output_dim, rel_num, base_num, dropout, head_num)

    def forward(self, x, edge_index, edge_type, rel_embeds):
        return self._op(x, edge_index, edge_type, rel_embeds)



class LcOp(Module):
    def __init__(self, primitive, hidden_dim):
        super(LcOp, self).__init__()
        self._op = LAYER_CONNECT_OPS[primitive](hidden_dim)

    def forward(self, fts):
        return self._op(fts)


class LfOp(Module):
    def __init__(self, primitive, hidden_dim, layer_num):
        super(LfOp, self).__init__()
        self._op = LAYER_FUSION_OPS[primitive](hidden_dim, layer_num)

    def forward(self, fts):
        return self._op(fts)

class NiOp(Module):
    def __init__(self, primitive, hidden_dim, ent_num):
        super(NiOp, self).__init__()
        self._op = NODE_INFO_OPS[primitive](hidden_dim, ent_num)

    def forward(self, node_info_embed):
        return self._op(node_info_embed)

class PiOp(Module):
    def __init__(self, primitive, hidden_dim, batch_pair_num):
        super(PiOp, self).__init__()
        self._op = PAIR_INFO_OPS[primitive](hidden_dim, batch_pair_num)

    def forward(self, pair_info_embed):
        return self._op(pair_info_embed)

class NfOp(Module):
    def __init__(self, primitive, hidden_dim):
        super(NfOp, self).__init__()
        self._op = NODE_FUSION_OPS[primitive](hidden_dim)

    def forward(self, sub_emb, obj_emb, subNodeEmbed, objNodeEmbed):
        return self._op(sub_emb, obj_emb, subNodeEmbed, objNodeEmbed)
    
class ESPATune(Module):
    def __init__(self, genotype, args, input_dim, hidden_dim, output_dim, rel_num, s2o, node_wise_feature, batch_pair_num):
        super(ESPATune, self).__init__()
        self.genotype = genotype
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rel_num = rel_num
        self.layer_num = self.args.gnn_layer_num
        self.s2o = s2o
        self.node_wise_feature = node_wise_feature
        self.ent_num = self.node_wise_feature.shape[0]
        self.batch_pair_num = batch_pair_num
        self.device = self.node_wise_feature.device
        self.gcn_drop = args.gcn_drop
        # 层之间的dropout
        self.hidden_drop = torch.nn.Dropout(self.gcn_drop).to(self.device)

        # pair_info_weights
        self.pair_info_dim = self.output_dim
        self.dropout = 0.1
        self.ln = False
        self.tailact = False
        self.lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()
        
        self.cn = nn.Sequential(
            nn.Linear(1, self.pair_info_dim),
            nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(self.pair_info_dim, self.pair_info_dim),
            self.lnfn(self.pair_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(self.pair_info_dim, self.pair_info_dim) if not self.tailact else nn.Identity()).to(self.device)
        self.jaccaard = nn.Sequential(
            nn.Linear(1, self.pair_info_dim),
            nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(self.pair_info_dim, self.pair_info_dim),
            self.lnfn(self.pair_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(self.pair_info_dim, self.pair_info_dim) if not self.tailact else nn.Identity()).to(self.device)
        self.ra = nn.Sequential(
            nn.Linear(1, self.pair_info_dim),
            nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(self.pair_info_dim, self.pair_info_dim),
            self.lnfn(self.pair_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(self.pair_info_dim, self.pair_info_dim) if not self.tailact else nn.Identity()).to(self.device)
        self.adamic = nn.Sequential(
            nn.Linear(1, self.pair_info_dim),
            nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(self.pair_info_dim, self.pair_info_dim),
            self.lnfn(self.pair_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(self.pair_info_dim, self.pair_info_dim) if not self.tailact else nn.Identity()).to(self.device)
        self.pair_mlp = nn.Linear(self.output_dim*4, self.output_dim).to(self.device)
        
        # node_info_value
        self.Degree = self.node_wise_feature[:,0].view(-1,1).to(self.device)
        self.Cen = self.node_wise_feature[:,1].view(-1,1).to(self.device)
        self.Pgrank = self.node_wise_feature[:,2].view(-1,1).to(self.device)
        self.Netw = self.node_wise_feature[:,3].view(-1,1).to(self.device)
        self.Katz = self.node_wise_feature[:,4].view(-1,1).to(self.device)

        # node_info_weights
        self.node_info_dim = self.output_dim
        self.dropout = 0.1
        self.ln = False
        self.tailact = False
        self.lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()
        self.degree = nn.Sequential(
              nn.Linear(1, self.node_info_dim),
              # nn.Linear(self.p.num_rel*2, self.node_info_dim),
              nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
              nn.Linear(self.node_info_dim, self.node_info_dim),
              self.lnfn(self.node_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
              nn.ReLU(inplace=True), nn.Linear(self.node_info_dim, self.node_info_dim) if not self.tailact else nn.Identity()).to(self.device)
        self.cen = nn.Sequential(
            nn.Linear(1, self.node_info_dim),
            nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(self.node_info_dim, self.node_info_dim),
            self.lnfn(self.node_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(self.node_info_dim, self.node_info_dim) if not self.tailact else nn.Identity()).to(self.device)
        self.pgrank = nn.Sequential(
            nn.Linear(1, self.node_info_dim),
            nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(self.node_info_dim, self.node_info_dim),
            self.lnfn(self.node_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(self.node_info_dim, self.node_info_dim) if not self.tailact else nn.Identity()).to(self.device)
        self.netw = nn.Sequential(
            nn.Linear(1, self.node_info_dim),
            nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(self.node_info_dim, self.node_info_dim),
            self.lnfn(self.node_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(self.node_info_dim, self.node_info_dim) if not self.tailact else nn.Identity()).to(self.device)
        self.katz = nn.Sequential(
            nn.Linear(1, self.node_info_dim),
            nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(self.node_info_dim, self.node_info_dim),
            self.lnfn(self.node_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(self.node_info_dim, self.node_info_dim) if not self.tailact else nn.Identity()).to(self.device)
        self.node_mlp = nn.Linear(self.output_dim*5, self.output_dim).to(self.device)


        self.na_layers = ModuleList()
        self.lc_layers = ModuleList()
        ops = genotype.split('||')
        for idx in range(self.layer_num):
            if idx == 0:
                self.na_layers.append(
                    NaOp(ops[2 * idx], self.input_dim, self.hidden_dim, self.rel_num, self.args.base_num,
                         self.args.dropout,
                         self.args.head_num))
                self.lc_layers.append(LcOp(ops[2 * idx + 1], self.hidden_dim))
            else:
                if idx == self.layer_num - 1:
                    self.na_layers.append(
                        NaOp(ops[2 * idx], self.hidden_dim, self.output_dim, self.rel_num,
                             self.args.base_num,
                             self.args.dropout, 1))
                    self.lf_layer = LfOp(ops[2 * idx + 1], self.hidden_dim, self.layer_num)
                else:
                    self.na_layers.append(
                        NaOp(ops[2 * idx], self.hidden_dim,
                             self.hidden_dim, self.rel_num, self.args.base_num,
                             self.args.dropout, self.args.head_num))
                    self.lc_layers.append(LcOp(ops[2 * idx + 1], self.hidden_dim))
        self.ni_module = NiOp(ops[2 * self.layer_num], self.output_dim*5, self.ent_num)
        self.nf_layer = NfOp(ops[2 * self.layer_num + 1], self.output_dim)
        self.pi_module = PiOp(ops[2 * self.layer_num + 2], self.output_dim*4, self.batch_pair_num)
    
    
    def concat(self, ent_embed, sub_embed, Heuristic):
        """
        :param ent_embed: [batch_size, embed_dim]
        :param sub_embed: [batch_size, embed_dim]
        :return: stack_input: [B, C, H, W]
        """
        ent_embed = ent_embed.view(-1, 1, self.output_dim)
        sub_embed = sub_embed.view(-1, 1, self.output_dim)
        Heuristic = Heuristic.view(-1, 1, self.output_dim)
        # [batch_size, 2, embed_dim]
        stack_input = torch.cat([ent_embed, sub_embed, Heuristic], 1)

        assert self.output_dim == self.k_h * self.k_w
        # reshape to 2D [batch, 1, 2*k_h, k_w]
        # stack_input = stack_input.reshape(-1, 1, 3 * self.k_h, self.k_w)
        stack_input = torch.transpose(stack_input, 2, 1).reshape(-1, 1, 3 * self.k_h, self.k_w)
        return stack_input
    
    def forward(self, ent_embeds, edge_index, edge_type, rel_embeds, sub, obj, mode=None):
        lf_list = []
        for i, layer in enumerate(self.na_layers):
            lc_list = []
            lc_list.append(ent_embeds)
            # graph, rel_embeds = layer(graph, rel_embeds)
            ent_embeds, rel_embeds = layer(ent_embeds, edge_index, edge_type, rel_embeds)
            ent_embeds = self.hidden_drop(ent_embeds)
            lf_list.append(ent_embeds)
            lc_list.append(ent_embeds)
            if i != self.layer_num - 1:
                ent_embeds = self.lc_layers[i](lc_list)
        
        ent_embeds = self.lf_layer(lf_list)
        sub_embed = torch.index_select(ent_embeds, 0, sub)
        obj_embed = torch.index_select(ent_embeds, 0, obj)

        # final_gnn_embed = self.lf_layer(lf_list)
        # sub_embed = torch.index_select(final_gnn_embed, 0, sub)
        # obj_embed = torch.index_select(final_gnn_embed, 0, obj)
        
        node_info_embed = torch.cat([self.degree(self.Degree), self.cen(self.Cen), self.pgrank(self.Pgrank), self.netw(self.Netw),self.katz(self.Katz)], dim=1).to(self.device)
        optnode_info_embed = torch.relu(self.node_mlp(self.ni_module(node_info_embed))).to(self.device)
        sub_node_info, obj_node_info = optnode_info_embed[sub],optnode_info_embed[obj]

        sub_gnn_node, obj_gnn_node = self.nf_layer(sub_embed, obj_embed, sub_node_info, obj_node_info)

        # 启发式信息计算(低阶 pair-wise)
        s_neighbor = list(map(lambda x: self.s2o.get(int(x)) if self.s2o.get(int(x)) else [], sub))
        o_neighbor = list(map(lambda x: self.s2o.get(int(x)) if self.s2o.get(int(x)) else [], obj))
        CN = [set(s_neighbor[i]) & set(o_neighbor[i]) for i in range(len(sub))]

        CNnum = torch.tensor([len(i) for i in CN], dtype=torch.float32).view(-1,1).to(self.device)
        Jaccaard = torch.tensor([CNnum[i] / len(set(s_neighbor[i]) | set(o_neighbor[i])) for i in range(len(sub))], dtype=torch.float32).view(-1,1).to(self.device)
        RA = torch.tensor([sum([1 / len(self.s2o.get(x)) for x in CN[i]])  for i in range(len(sub))], dtype=torch.float32).view(-1,1).to(self.device)
        Adamic = torch.tensor([sum( [1 / np.log(len(self.s2o.get(x))) if len(self.s2o.get(x)) > 1  else 0 for x in CN[i]] )  for i in range(len(sub))], dtype=torch.float32).view(-1,1).to(self.device)
        pair_info_embed = torch.cat([self.cn(CNnum), self.jaccaard(Jaccaard),self.ra(RA), self.adamic(Adamic)],dim=1).to(self.device)
        optpair_info_embed = torch.relu(self.pair_mlp(self.pi_module(pair_info_embed))).to(self.device)

        # 现获得首尾实体融合实体的嵌入以及pair-wise的嵌入，如何评分？（score函数？）
        return sub_gnn_node, obj_gnn_node, optpair_info_embed, rel_embeds
        # return sub_gnn_node, obj_gnn_node, optpair_info_embed, rel_embeds
    
