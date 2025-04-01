from torch.nn import Module, ModuleList
import torch
from numpy.random import choice
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.operations_new import G_OPS, LAYER_FUSION_OPS, LAYER_CONNECT_OPS, NODE_INFO_OPS, PAIR_INFO_OPS, NODE_FUSION_OPS
from models.genotypes_new import NA_PRIMITIVES, LC_PRIMITIVES, LF_PRIMITIVES, NODE_INFO_PRIMITIVES, PAIR_INFO_PRIMITIVES, NF_PRIMITIVES


class GOpBlock(Module):
    def __init__(self, input_dim, output_dim, rel_num, base_num, dropout, head_num):
        super(GOpBlock, self).__init__()
        self._ops = ModuleList()
        for primitive in NA_PRIMITIVES:
            self._op = G_OPS[primitive](input_dim, output_dim, rel_num, base_num, dropout, head_num)
            self._ops.append(self._op)

    def forward(self, x, edge_index, edge_type, rel_embeds, primitive):
        return self._ops[NA_PRIMITIVES.index(primitive)](x, edge_index, edge_type, rel_embeds)

# class TOpBlock(Module):
#     def __init__(self, input_dim, hidden_dim, seq_head_num):
#         super(TOpBlock, self).__init__()
#         self._ops = ModuleList()
#         for primitive in SEQ_PRIMITIVES:
#             self._op = T_OPS[primitive](input_dim, hidden_dim, seq_head_num)
#             self._ops.append(self._op)

#     def forward(self, current_embed, previous_embed, previous_embed_transformer, local_attn_mask, primitive):
#         return self._ops[SEQ_PRIMITIVES.index(primitive)](current_embed, previous_embed, previous_embed_transformer,
#                                                           local_attn_mask)

class LcOpBlock(Module):
    def __init__(self, hidden_dim):
        super(LcOpBlock, self).__init__()
        self._ops = ModuleList()
        for primitive in LC_PRIMITIVES:
            self._op = LAYER_CONNECT_OPS[primitive](hidden_dim)
            self._ops.append(self._op)

    def forward(self, fts, primitive):
        return self._ops[LC_PRIMITIVES.index(primitive)](fts)

class LfOpBlock(Module):
    def __init__(self, hidden_dim, layer_num):
        super(LfOpBlock, self).__init__()
        self._ops = ModuleList()
        for primitive in LF_PRIMITIVES:
            self._op = LAYER_FUSION_OPS[primitive](hidden_dim, layer_num)
            self._ops.append(self._op)

    def forward(self, fts, primitive):
        return self._ops[LF_PRIMITIVES.index(primitive)](fts)

class NodeInfoOpBlock(Module):
    def __init__(self, hidden_dim, ent_num):
        super(NodeInfoOpBlock, self).__init__()
        self._ops = ModuleList()
        for primitive in NODE_INFO_PRIMITIVES:
            self._op = NODE_INFO_OPS[primitive](hidden_dim, ent_num)
            self._ops.append(self._op)

    def forward(self, node_info_embed, primitive):
        return self._ops[NODE_INFO_PRIMITIVES.index(primitive)](node_info_embed)
    
class PairInfoOpBlock(Module):
    def __init__(self, hidden_dim, batch_pair_num):
        super(PairInfoOpBlock, self).__init__()
        self._ops = ModuleList()
        for primitive in PAIR_INFO_PRIMITIVES:
            self._op = PAIR_INFO_OPS[primitive](hidden_dim, batch_pair_num)
            self._ops.append(self._op)

    def forward(self, pair_info_embed, primitive):
        return self._ops[PAIR_INFO_PRIMITIVES.index(primitive)](pair_info_embed)

class NfOpBlock(Module):
    def __init__(self, hidden_dim):
        super(NfOpBlock, self).__init__()
        self._ops = ModuleList()
        for primitive in NF_PRIMITIVES:
            self._op = NODE_FUSION_OPS[primitive](hidden_dim)
            self._ops.append(self._op)

    def forward(self, sub_emb, obj_emb, subNodeEmbed, objNodeEmbed, primitive):
        return self._ops[NF_PRIMITIVES.index(primitive)](sub_emb, obj_emb, subNodeEmbed, objNodeEmbed)
    
class ESPASPOSSearch(Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, rel_num, s2o, node_wise_feature, batch_pair_num):
        super(ESPASPOSSearch, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rel_num = rel_num
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
            nn.ReLU(inplace=True), nn.Linear(self.pair_info_dim, self.pair_info_dim) if not self.tailact else nn.Identity())
        self.jaccaard = nn.Sequential(
            nn.Linear(1, self.pair_info_dim),
            nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(self.pair_info_dim, self.pair_info_dim),
            self.lnfn(self.pair_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(self.pair_info_dim, self.pair_info_dim) if not self.tailact else nn.Identity())
        self.ra = nn.Sequential(
            nn.Linear(1, self.pair_info_dim),
            nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(self.pair_info_dim, self.pair_info_dim),
            self.lnfn(self.pair_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(self.pair_info_dim, self.pair_info_dim) if not self.tailact else nn.Identity())
        self.adamic = nn.Sequential(
            nn.Linear(1, self.pair_info_dim),
            nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(self.pair_info_dim, self.pair_info_dim),
            self.lnfn(self.pair_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(self.pair_info_dim, self.pair_info_dim) if not self.tailact else nn.Identity())
        self.pair_mlp = nn.Linear(self.output_dim*4, self.output_dim)
        
        # node_info_value
        self.Degree = self.node_wise_feature[:,0].view(-1,1)
        self.Cen = self.node_wise_feature[:,1].view(-1,1)
        self.Pgrank = self.node_wise_feature[:,2].view(-1,1)
        self.Netw = self.node_wise_feature[:,3].view(-1,1)
        self.Katz = self.node_wise_feature[:,4].view(-1,1)

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
              nn.ReLU(inplace=True), nn.Linear(self.node_info_dim, self.node_info_dim) if not self.tailact else nn.Identity())
        self.cen = nn.Sequential(
            nn.Linear(1, self.node_info_dim),
            nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(self.node_info_dim, self.node_info_dim),
            self.lnfn(self.node_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(self.node_info_dim, self.node_info_dim) if not self.tailact else nn.Identity())
        self.pgrank = nn.Sequential(
            nn.Linear(1, self.node_info_dim),
            nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(self.node_info_dim, self.node_info_dim),
            self.lnfn(self.node_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(self.node_info_dim, self.node_info_dim) if not self.tailact else nn.Identity())
        self.netw = nn.Sequential(
            nn.Linear(1, self.node_info_dim),
            nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(self.node_info_dim, self.node_info_dim),
            self.lnfn(self.node_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(self.node_info_dim, self.node_info_dim) if not self.tailact else nn.Identity())
        self.katz = nn.Sequential(
            nn.Linear(1, self.node_info_dim),
            nn.Dropout(self.dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(self.node_info_dim, self.node_info_dim),
            self.lnfn(self.node_info_dim, self.ln), nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(self.node_info_dim, self.node_info_dim) if not self.tailact else nn.Identity())
        self.node_mlp = nn.Linear(self.output_dim*5, self.output_dim)


        self.layer_num = self.args.gnn_layer_num
        self.g_layers = ModuleList()
        self.lc_layers = ModuleList()
        self.ops = None

        for idx in range(self.layer_num):
            if idx == 0:
                self.g_layers.append(
                    GOpBlock(self.input_dim, self.hidden_dim, self.rel_num, self.args.base_num,
                              self.args.dropout,
                              self.args.head_num))
                self.lc_layers.append(LcOpBlock(self.hidden_dim))
            else:
                if idx == self.layer_num - 1:
                    self.g_layers.append(
                        GOpBlock(self.hidden_dim, self.output_dim, self.rel_num,
                                  self.args.base_num,
                                  self.args.dropout, 1))
                    self.lf_layer = LfOpBlock(self.hidden_dim, self.layer_num)
                else:
                    self.g_layers.append(
                        GOpBlock(self.hidden_dim,
                                  self.hidden_dim, self.rel_num, self.args.base_num,
                                  self.args.dropout, self.args.head_num))
                    self.lc_layers.append(LcOpBlock(self.hidden_dim))

        self.ni_block = NodeInfoOpBlock(self.output_dim*5, self.ent_num)
        self.pi_block = PairInfoOpBlock(self.output_dim*4, self.batch_pair_num)
        self.nf_block = NfOpBlock(self.output_dim)

    def generate_single_path(self, op_subsupernet=''):
        single_path_list = []
        for i in range(self.layer_num):
            single_path_list.append(choice(NA_PRIMITIVES))
            if i != self.layer_num - 1:
                single_path_list.append(choice(LC_PRIMITIVES))
            else:
                if op_subsupernet != '':
                    single_path_list.append(op_subsupernet)
                else:
                    single_path_list.append(choice(LF_PRIMITIVES))
        single_path_list.append(choice(NODE_INFO_PRIMITIVES))
        single_path_list.append(choice(NF_PRIMITIVES))
        single_path_list.append(choice(PAIR_INFO_PRIMITIVES))
        return single_path_list
    
    
    def forward(self, ent_embeds, edge_index, edge_type, rel_embeds, sub, obj, mode=None):
        lf_list = []
        for i, layer in enumerate(self.g_layers):
            lc_list = []
            lc_list.append(ent_embeds)
            ent_embeds, rel_embeds = layer(ent_embeds, edge_index, edge_type, rel_embeds, self.ops[2 * i])
            ent_embeds = self.hidden_drop(ent_embeds)
            lf_list.append(ent_embeds)
            lc_list.append(ent_embeds)
            if i != self.layer_num - 1:
                ent_embeds = self.lc_layers[i](lc_list, self.ops[2 * i + 1])
        final_gnn_embed = self.lf_layer(lf_list, self.ops[self.layer_num * 2-1])

        sub_embed = torch.index_select(final_gnn_embed, 0, sub)
        obj_embed = torch.index_select(final_gnn_embed, 0, obj)
        # sub_embed, obj_embed = final_gnn_embed[sub], final_gnn_embed[obj]
        node_info_embed = torch.cat([self.degree(self.Degree), self.cen(self.Cen), self.pgrank(self.Pgrank), self.netw(self.Netw),self.katz(self.Katz)], dim=1)
        optnode_info_embed = torch.relu(self.node_mlp(self.ni_block(node_info_embed, self.ops[self.layer_num * 2])))
        sub_node_info, obj_node_info = optnode_info_embed[sub],optnode_info_embed[obj]

        sub_gnn_node, obj_gnn_node = self.nf_block(sub_embed, obj_embed, sub_node_info, obj_node_info,self.ops[self.layer_num * 2 + 1])

        # 启发式信息计算(低阶 pair-wise)
        s_neighbor = list(map(lambda x: self.s2o.get(int(x)) if self.s2o.get(int(x)) else [], sub))
        o_neighbor = list(map(lambda x: self.s2o.get(int(x)) if self.s2o.get(int(x)) else [], obj))
        CN = [set(s_neighbor[i]) & set(o_neighbor[i]) for i in range(len(sub))]

        CNnum = torch.tensor([len(i) for i in CN], dtype=torch.float32).view(-1,1).to(self.device)
        Jaccaard = torch.tensor([CNnum[i] / len(set(s_neighbor[i]) | set(o_neighbor[i])) for i in range(len(sub))], dtype=torch.float32).view(-1,1).to(self.device)
        RA = torch.tensor([sum([1 / len(self.s2o.get(x)) for x in CN[i]])  for i in range(len(sub))], dtype=torch.float32).view(-1,1).to(self.device)
        Adamic = torch.tensor([sum( [1 / np.log(len(self.s2o.get(x))) if len(self.s2o.get(x)) > 1  else 0 for x in CN[i]] )  for i in range(len(sub))], dtype=torch.float32).view(-1,1).to(self.device)
        pair_info_embed = torch.cat([self.cn(CNnum), self.jaccaard(Jaccaard),self.ra(RA), self.adamic(Adamic)],dim=1)
        optpair_info_embed = torch.relu(self.pair_mlp(self.pi_block(pair_info_embed, self.ops[self.layer_num * 2 + 2])))

        # 现获得首尾实体融合实体的嵌入以及pair-wise的嵌入，如何评分？（score函数？）
        
        return  sub_gnn_node, obj_gnn_node, optpair_info_embed, rel_embeds

        
