from torch.nn import Module, ModuleList, Parameter, Dropout, BatchNorm1d, Linear
from torch.nn.functional import relu, elu, softmax
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from models.operations_new import G_OPS, LAYER_FUSION_OPS, LAYER_CONNECT_OPS, NODE_INFO_OPS, PAIR_INFO_OPS, NODE_FUSION_OPS
from models.genotypes_new import NA_PRIMITIVES, LC_PRIMITIVES, LF_PRIMITIVES,NODE_INFO_PRIMITIVES, PAIR_INFO_PRIMITIVES, NF_PRIMITIVES
from numpy.random import choice
from pprint import pprint


def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")


class GMixedOp(Module):
    def __init__(self, input_dim, output_dim, rel_num, base_num, dropout, head_num):
        super(GMixedOp, self).__init__()
        self._ops = ModuleList()

        for primitive in NA_PRIMITIVES:
            op = G_OPS[primitive](input_dim, output_dim, rel_num, base_num, dropout, head_num)
            self._ops.append(op)

    def forward(self, x, edge_index, edge_type, rel_embeds, weights):
        mixed_ent = []
        mixed_rel = []
        for w, op in zip(weights, self._ops):
            x, rel = op(x, edge_index, edge_type, rel_embeds)
            mixed_ent.append(w * x)
            mixed_rel.append(w * rel)
        x = sum(mixed_ent)
        rel = sum(mixed_rel)
        return x, rel


class LcMixedOp(Module):
    def __init__(self, hidden_dim):
        super(LcMixedOp, self).__init__()
        self._ops = ModuleList()
        for primitive in LC_PRIMITIVES:
            op = LAYER_CONNECT_OPS[primitive](hidden_dim)
            self._ops.append(op)

    def forward(self, fts, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * op(fts))
        return sum(mixed_res)


class LaMixedOp(Module):
    def __init__(self, hidden_dim, layer_num):
        super(LaMixedOp, self).__init__()
        self._ops = ModuleList()
        for primitive in LF_PRIMITIVES:
            op = LAYER_FUSION_OPS[primitive](hidden_dim, layer_num)
            self._ops.append(op)

    def forward(self, fts, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * (op(fts)))
        return sum(mixed_res)

class NodeMixedOp(Module):
    def __init__(self, hidden_dim, ent_num):
        super(NodeMixedOp, self).__init__()
        self._ops = ModuleList()
        for primitive in NODE_INFO_PRIMITIVES:
            op = NODE_INFO_OPS[primitive](hidden_dim, ent_num)
            self._ops.append(op)

    def forward(self, node_info_embed, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * (op(node_info_embed)))
        return sum(mixed_res)

class PairMixedOp(Module):
    def __init__(self, hidden_dim, batch_pair_num):
        super(PairMixedOp, self).__init__()
        self._ops = ModuleList()
        for primitive in PAIR_INFO_PRIMITIVES:
            op = PAIR_INFO_OPS[primitive](hidden_dim, batch_pair_num)
            self._ops.append(op)

    def forward(self, pair_info_embed, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * (op(pair_info_embed)))
        return sum(mixed_res)

class NfMixedOp(Module):
    def __init__(self, hidden_dim):
        super(NfMixedOp, self).__init__()
        self._ops = ModuleList()
        for primitive in NF_PRIMITIVES:
            self._op = NODE_FUSION_OPS[primitive](hidden_dim)
            self._ops.append(self._op)

    def forward(self, sub_emb, obj_emb, subNodeEmbed, objNodeEmbed, weights):
        mixed_sub = []
        mixed_obj = []
        for w, op in zip(weights, self._ops):
            sub, obj = op(sub_emb, obj_emb, subNodeEmbed, objNodeEmbed)
            mixed_sub.append(w * sub)
            mixed_obj.append(w * obj)
        return sum(mixed_sub), sum(mixed_obj)

class ESPADartsSearch(Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim, rel_num, s2o, node_wise_feature, batch_pair_num):
        super(ESPADartsSearch, self).__init__()
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
        self.la_layers = ModuleList()
        self.ops = None
        for idx in range(self.layer_num):
            if idx == 0:
                self.g_layers.append(
                    GMixedOp(self.input_dim, self.hidden_dim, self.rel_num, self.args.base_num,
                             self.args.dropout,
                             self.args.head_num))
                self.lc_layers.append(LcMixedOp(self.hidden_dim))
            else:
                if idx == self.layer_num - 1:
                    self.g_layers.append(
                        GMixedOp(self.hidden_dim, self.output_dim, self.rel_num,
                                 self.args.base_num,
                                 self.args.dropout, 1))
                    self.la_layer = LaMixedOp(self.hidden_dim, self.layer_num)
                else:
                    self.g_layers.append(
                        GMixedOp(self.hidden_dim,
                                 self.hidden_dim, self.rel_num, self.args.base_num,
                                 self.args.dropout, self.args.head_num))
                    self.lc_layers.append(LcMixedOp(self.hidden_dim))
        self.ni_block = NodeMixedOp(self.output_dim*5, self.ent_num)
        self.pi_block = PairMixedOp(self.output_dim*4, self.batch_pair_num)
        self.nf_block = NfMixedOp(self.output_dim)

        self._initialize_alphas()

    def _initialize_alphas(self):
        na_ops_num = len(NA_PRIMITIVES)
        lc_ops_num = len(LC_PRIMITIVES)
        la_ops_num = len(LF_PRIMITIVES)
        node_ops_num = len(NODE_INFO_PRIMITIVES)
        pair_ops_num = len(PAIR_INFO_PRIMITIVES)
        nf_ops_num = len(NF_PRIMITIVES)
        if "darts" in self.args.search_mode:
            self.na_alphas = Variable(1e-3 * torch.randn(self.layer_num, na_ops_num).cuda(), requires_grad=True)
            self.lc_alphas = Variable(1e-3 * torch.randn(self.layer_num - 1, lc_ops_num).cuda(), requires_grad=True)
            self.la_alphas = Variable(1e-3 * torch.randn(1, la_ops_num).cuda(), requires_grad=True)
            self.node_alphas = Variable(1e-3 * torch.randn(1, node_ops_num).cuda(), requires_grad=True)
            self.pair_alphas = Variable(1e-3 * torch.randn(1, pair_ops_num).cuda(), requires_grad=True)
            self.nf_alphas = Variable(1e-3 * torch.randn(1, nf_ops_num).cuda(), requires_grad=True)
        self._arch_parameters = [
                self.na_alphas,
                self.lc_alphas,
                self.la_alphas,
                self.node_alphas,
                self.pair_alphas,
                self.nf_alphas
        ]

    def get_one_hot_alpha(self, alpha):
        one_hot_alpha = torch.zeros_like(alpha, device=alpha.device)
        idx = torch.argmax(alpha, dim=-1)

        for i in range(one_hot_alpha.size(0)):
            one_hot_alpha[i, idx[i]] = 1.0

        return one_hot_alpha

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        gene = []
        na_max, na_indices = torch.max(softmax(self.na_alphas, dim=-1).data.cpu(), dim=-1)
        lc_max, lc_indices = torch.max(softmax(self.lc_alphas, dim=-1).data.cpu(), dim=-1)
        la_max, la_indices = torch.max(softmax(self.la_alphas, dim=-1).data.cpu(), dim=-1)
        node_max, node_indices = torch.max(softmax(self.node_alphas, dim=-1).data.cpu(), dim=-1)
        pair_max, pair_indices = torch.max(softmax(self.pair_alphas, dim=-1).data.cpu(), dim=-1)
        nf_max, nf_indices = torch.max(softmax(self.nf_alphas, dim=-1).data.cpu(), dim=-1)
        pprint(na_max)
        pprint(lc_max)
        pprint(la_max)
        print(node_max)
        print(pair_max)
        print(nf_max)
        for idx in range(self.layer_num):
            gene.append(NA_PRIMITIVES[na_indices[idx]])
            if idx != self.layer_num - 1:
                gene.append(LC_PRIMITIVES[lc_indices[idx]])
        gene.append(LF_PRIMITIVES[la_indices[-1]])
        gene.append(NODE_INFO_PRIMITIVES[node_indices[-1]])
        gene.append(PAIR_INFO_PRIMITIVES[pair_indices[-1]])
        gene.append(NF_PRIMITIVES[nf_indices[-1]])
        return "||".join(gene)


    def forward(self, ent_embeds, edge_index, edge_type, rel_embeds, sub, obj, mode=None):
        if "darts" in self.args.search_mode:
            na_weights = softmax(self.na_alphas, dim=-1)
            lc_weights = softmax(self.lc_alphas, dim=-1)
            la_weights = softmax(self.la_alphas, dim=-1)
            node_weights = softmax(self.node_alphas, dim=-1)
            pair_weights = softmax(self.pair_alphas, dim=-1)
            nf_weights = softmax(self.nf_alphas, dim=-1)
        if mode == 'evaluate_single_path':
            na_weights = self.get_one_hot_alpha(na_weights)
            lc_weights = self.get_one_hot_alpha(lc_weights)
            la_weights = self.get_one_hot_alpha(la_weights)
            node_weights = self.get_one_hot_alpha(node_weights)
            pair_weights = self.get_one_hot_alpha(pair_weights)
            nf_weights = self.get_one_hot_alpha(nf_weights)
            
        la_list = []
        for i, layer in enumerate(self.g_layers):
            lc_list = []
            lc_list.append(ent_embeds)
            ent_embeds, rel_embeds = layer(ent_embeds, edge_index, edge_type, rel_embeds, na_weights[i])
            lc_list.append(ent_embeds)
            la_list.append(ent_embeds)
            if i != self.layer_num - 1:
                ent_embeds = self.lc_layers[i](lc_list, lc_weights[i])
        final_gnn_embed = self.la_layer(la_list, la_weights[0])

        sub_embed = torch.index_select(final_gnn_embed, 0, sub)
        obj_embed = torch.index_select(final_gnn_embed, 0, obj)
      
        node_info_embed = torch.cat([self.degree(self.Degree), self.cen(self.Cen), self.pgrank(self.Pgrank), self.netw(self.Netw),self.katz(self.Katz)], dim=1)
        optnode_info_embed = torch.relu(self.node_mlp(self.ni_block(node_info_embed, node_weights[0])))
        sub_node_info, obj_node_info = optnode_info_embed[sub],optnode_info_embed[obj]

        sub_gnn_node, obj_gnn_node = self.nf_block(sub_embed, obj_embed, sub_node_info, obj_node_info,nf_weights[0])

        # 启发式信息计算(低阶 pair-wise)
        s_neighbor = list(map(lambda x: self.s2o.get(int(x)) if self.s2o.get(int(x)) else [], sub))
        o_neighbor = list(map(lambda x: self.s2o.get(int(x)) if self.s2o.get(int(x)) else [], obj))
        CN = [set(s_neighbor[i]) & set(o_neighbor[i]) for i in range(len(sub))]

        CNnum = torch.tensor([len(i) for i in CN], dtype=torch.float32).view(-1,1).to(self.device)
        Jaccaard = torch.tensor([CNnum[i] / len(set(s_neighbor[i]) | set(o_neighbor[i])) for i in range(len(sub))], dtype=torch.float32).view(-1,1).to(self.device)
        RA = torch.tensor([sum([1 / len(self.s2o.get(x)) for x in CN[i]])  for i in range(len(sub))], dtype=torch.float32).view(-1,1).to(self.device)
        Adamic = torch.tensor([sum( [1 / np.log(len(self.s2o.get(x))) if len(self.s2o.get(x)) > 1  else 0 for x in CN[i]] )  for i in range(len(sub))], dtype=torch.float32).view(-1,1).to(self.device)
        pair_info_embed = torch.cat([self.cn(CNnum), self.jaccaard(Jaccaard),self.ra(RA), self.adamic(Adamic)],dim=1)
        optpair_info_embed = torch.relu(self.pair_mlp(self.pi_block(pair_info_embed, pair_weights[0])))

        # 现获得首尾实体融合实体的嵌入以及pair-wise的嵌入，如何评分？（score函数？）
        
        return  sub_gnn_node, obj_gnn_node, optpair_info_embed, rel_embeds
