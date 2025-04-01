from torch.nn import Module, Linear, GRU
import torch
from models.rgcn import RGCNLayer
from models.rgat import RGATLayer, MultiHeadRGATLayer
from models.compgcn_conv import CompGCNLayer
from models.jknet import JKNet
from models.self_attention import SelfAttention
import numpy as np

G_OPS = {
    "rgcn": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim, output_dim,
                                                                                             rel_num,
                                                                                             base_num, dropout,
                                                                                             head_num,
                                                                                             'rgcn'),
    "rgat_vanilla": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim,
                                                                                                     output_dim,
                                                                                                     rel_num,
                                                                                                     base_num, dropout,
                                                                                                     head_num,
                                                                                                     'rgat_vanilla'),
    "rgat_sym": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim, output_dim,
                                                                                                 rel_num,
                                                                                                 base_num, dropout,
                                                                                                 head_num,
                                                                                                 'rgat_sym'),
    "rgat_cos": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim, output_dim,
                                                                                                 rel_num,
                                                                                                 base_num, dropout,
                                                                                                 head_num,
                                                                                                 'rgat_cos'),
    "rgat_linear": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim,
                                                                                                    output_dim,
                                                                                                    rel_num,
                                                                                                    base_num, dropout,
                                                                                                    head_num,
                                                                                                    'rgat_linear'),
    "rgat_gene-linear": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim,
                                                                                                         output_dim,
                                                                                                         rel_num,
                                                                                                         base_num,
                                                                                                         dropout,
                                                                                                         head_num,
                                                                                                         'rgat_gene'
                                                                                                         '-linear'),
    "compgcn_add": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim,
                                                                                                    output_dim,
                                                                                                    rel_num,
                                                                                                    base_num, dropout,
                                                                                                    head_num,
                                                                                                    'compgcn_add'),
    "compgcn_sub": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim,
                                                                                                    output_dim,
                                                                                                    rel_num,
                                                                                                    base_num, dropout,
                                                                                                    head_num,
                                                                                                    'compgcn_sub'),
    "compgcn_mult": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim,
                                                                                                     output_dim,
                                                                                                     rel_num,
                                                                                                     base_num, dropout,
                                                                                                     head_num,
                                                                                                     'compgcn_mult'),
    "compgcn_corr": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim,
                                                                                                     output_dim,
                                                                                                     rel_num,
                                                                                                     base_num, dropout,
                                                                                                     head_num,
                                                                                                     'compgcn_corr'),
    "compgcn_rotate": lambda input_dim, output_dim, rel_num, base_num, dropout, head_num: NaAggregator(input_dim,
                                                                                                     output_dim,
                                                                                                     rel_num,
                                                                                                     base_num, dropout,
                                                                                                     head_num,
                                                                                                     'compgcn_rotate'),
}

LAYER_CONNECT_OPS = {
    "lc_skip": lambda hidden_dim: LayerConnector("lc_skip", hidden_dim),
    "lc_sum": lambda hidden_dim: LayerConnector("lc_sum", hidden_dim),
    "lc_concat": lambda hidden_dim: LayerConnector("lc_concat", hidden_dim)
}

LAYER_FUSION_OPS = {
    "lf_skip": lambda hidden_dim, layer_num: LayerFusion("lf_skip", hidden_dim, layer_num),
    "lf_sum": lambda hidden_dim, layer_num: LayerFusion("lf_sum", hidden_dim, layer_num),
    "lf_concat": lambda hidden_dim, layer_num: LayerFusion("lf_concat", hidden_dim, layer_num),
    "lf_max": lambda hidden_dim, layer_num: LayerFusion("lf_max", hidden_dim, layer_num),
    "lf_mean": lambda hidden_dim, layer_num: LayerFusion("lf_mean", hidden_dim, layer_num)
}

# T_OPS = {
#     'gru': lambda input_dim, hidden_dim, seq_head_num: SeqEncoder('gru', input_dim, hidden_dim, seq_head_num),
#     'identity': lambda input_dim, hidden_dim, seq_head_num: SeqEncoder('identity', input_dim, hidden_dim, seq_head_num),
#     'sa': lambda input_dim, hidden_dim, seq_head_num: SeqEncoder('sa', input_dim, hidden_dim, seq_head_num)
# }

# SEQ_OPS = {
#     'gru': lambda input_dim, seq_head_num: SeqEncoder('gru', input_dim, seq_head_num),
#     'identity': lambda input_dim, seq_head_num: SeqEncoder('identity', input_dim, seq_head_num)
# }

# node-wise heuristic degree eigenvector_centrality pagerank betweenness_centrality katz
NODE_INFO_OPS = {
    'na_na_na_na_na':lambda input_dim, ent_num: NodeInfoOps('na_na_na_na_na', input_dim, ent_num),
    'na_na_na_na_katz':lambda input_dim, ent_num: NodeInfoOps('na_na_na_na_katz', input_dim, ent_num),
    'na_na_na_betw_na':lambda input_dim, ent_num: NodeInfoOps('na_na_na_betw_na', input_dim, ent_num),
    'na_na_na_betw_katz':lambda input_dim, ent_num: NodeInfoOps('na_na_na_betw_katz', input_dim, ent_num),
    'na_na_pgrank_na_na':lambda input_dim, ent_num: NodeInfoOps('na_na_pgrank_na_na', input_dim, ent_num),
    'na_na_pgrank_na_katz':lambda input_dim, ent_num: NodeInfoOps('na_na_pgrank_na_katz', input_dim, ent_num),
    'na_na_pgrank_betw_na':lambda input_dim, ent_num: NodeInfoOps('na_na_pgrank_betw_na', input_dim, ent_num),
    'na_na_pgrank_betw_katz':lambda input_dim, ent_num: NodeInfoOps('na_na_pgrank_betw_katz', input_dim, ent_num),
    'na_eigen_na_na_na':lambda input_dim, ent_num: NodeInfoOps('na_eigen_na_na_na', input_dim, ent_num),
    'na_eigen_na_na_katz':lambda input_dim, ent_num: NodeInfoOps('na_eigen_na_na_katz', input_dim, ent_num),
    'na_eigen_na_betw_na':lambda input_dim, ent_num: NodeInfoOps('na_eigen_na_betw_na', input_dim, ent_num),
    'na_eigen_na_betw_katz':lambda input_dim, ent_num: NodeInfoOps('na_eigen_na_betw_katz', input_dim, ent_num),
    'na_eigen_pgrank_na_na':lambda input_dim, ent_num: NodeInfoOps('na_eigen_pgrank_na_na', input_dim, ent_num),
    'na_eigen_pgrank_na_katz':lambda input_dim, ent_num: NodeInfoOps('na_eigen_pgrank_na_katz', input_dim, ent_num),
    'na_eigen_pgrank_betw_na':lambda input_dim, ent_num: NodeInfoOps('na_eigen_pgrank_betw_na', input_dim, ent_num),
    'na_eigen_pgrank_betw_katz':lambda input_dim, ent_num: NodeInfoOps('na_eigen_pgrank_betw_katz', input_dim, ent_num),
    'degree_na_na_na_na':lambda input_dim, ent_num: NodeInfoOps('degree_na_na_na_na', input_dim, ent_num),
    'degree_na_na_na_katz':lambda input_dim, ent_num: NodeInfoOps('degree_na_na_na_katz', input_dim, ent_num),
    'degree_na_na_betw_na':lambda input_dim, ent_num: NodeInfoOps('degree_na_na_betw_na', input_dim, ent_num),
    'degree_na_na_betw_katz':lambda input_dim, ent_num: NodeInfoOps('degree_na_na_betw_katz', input_dim, ent_num),
    'degree_na_pgrank_na_na':lambda input_dim, ent_num: NodeInfoOps('degree_na_pgrank_na_na', input_dim, ent_num),
    'degree_na_pgrank_na_katz':lambda input_dim, ent_num: NodeInfoOps('degree_na_pgrank_na_katz', input_dim, ent_num),
    'degree_na_pgrank_betw_na':lambda input_dim, ent_num: NodeInfoOps('degree_na_pgrank_betw_na', input_dim, ent_num),
    'degree_na_pgrank_betw_katz':lambda input_dim, ent_num: NodeInfoOps('degree_na_pgrank_betw_katz', input_dim, ent_num),
    'degree_eigen_na_na_na':lambda input_dim, ent_num: NodeInfoOps('degree_eigen_na_na_na', input_dim, ent_num),
    'degree_eigen_na_na_katz':lambda input_dim, ent_num: NodeInfoOps('degree_eigen_na_na_katz', input_dim, ent_num),
    'degree_eigen_na_betw_na':lambda input_dim, ent_num: NodeInfoOps('degree_eigen_na_betw_na', input_dim, ent_num),
    'degree_eigen_na_betw_katz':lambda input_dim, ent_num: NodeInfoOps('degree_eigen_na_betw_katz', input_dim, ent_num),
    'degree_eigen_pgrank_na_na':lambda input_dim, ent_num: NodeInfoOps('degree_eigen_pgrank_na_na', input_dim, ent_num),
    'degree_eigen_pgrank_na_katz':lambda input_dim, ent_num: NodeInfoOps('degree_eigen_pgrank_na_katz', input_dim, ent_num),
    'degree_eigen_pgrank_betw_na':lambda input_dim, ent_num: NodeInfoOps('degree_eigen_pgrank_betw_na', input_dim, ent_num),
    'degree_eigen_pgrank_betw_katz':lambda input_dim, ent_num: NodeInfoOps('degree_eigen_pgrank_betw_katz', input_dim, ent_num)
}

# pair-wise heuristic cn jaccard ra(resource allocation) aa(AdamicAdar)
PAIR_INFO_OPS = {
    'na_na_na_na':lambda input_dim, batch_pair_num: PairInfoOps('na_na_na_na', input_dim, batch_pair_num),
    'na_na_na_aa':lambda input_dim, batch_pair_num: PairInfoOps('na_na_na_aa', input_dim, batch_pair_num),
    'na_na_ra_na':lambda input_dim, batch_pair_num: PairInfoOps('na_na_ra_na', input_dim, batch_pair_num),
    'na_na_ra_aa':lambda input_dim, batch_pair_num: PairInfoOps('na_na_ra_aa', input_dim, batch_pair_num),
    'na_jaccard_na_na':lambda input_dim, batch_pair_num: PairInfoOps('na_jaccard_na_na', input_dim, batch_pair_num),
    'na_jaccard_na_aa':lambda input_dim, batch_pair_num: PairInfoOps('na_jaccard_na_aa', input_dim, batch_pair_num),
    'na_jaccard_ra_na':lambda input_dim, batch_pair_num: PairInfoOps('na_jaccard_ra_na', input_dim, batch_pair_num),
    'na_jaccard_ra_aa':lambda input_dim, batch_pair_num: PairInfoOps('na_jaccard_ra_aa', input_dim, batch_pair_num),
    'cn_na_na_na':lambda input_dim, batch_pair_num: PairInfoOps('cn_na_na_na', input_dim, batch_pair_num),
    'cn_na_na_aa':lambda input_dim, batch_pair_num: PairInfoOps('cn_na_na_aa', input_dim, batch_pair_num),
    'cn_na_ra_na':lambda input_dim, batch_pair_num: PairInfoOps('cn_na_ra_na', input_dim, batch_pair_num),
    'cn_na_ra_aa':lambda input_dim, batch_pair_num: PairInfoOps('cn_na_ra_aa', input_dim, batch_pair_num),
    'cn_jaccard_na_na':lambda input_dim, batch_pair_num: PairInfoOps('cn_jaccard_na_na', input_dim, batch_pair_num),
    'cn_jaccard_na_aa':lambda input_dim, batch_pair_num: PairInfoOps('cn_jaccard_na_aa', input_dim, batch_pair_num),
    'cn_jaccard_ra_na':lambda input_dim, batch_pair_num: PairInfoOps('cn_jaccard_ra_na', input_dim, batch_pair_num),
    'cn_jaccard_ra_aa':lambda input_dim, batch_pair_num: PairInfoOps('cn_jaccard_ra_aa', input_dim, batch_pair_num)
}

# node-wise fusion operator
NODE_FUSION_OPS = {
    "node_sum": lambda hidden_dim: NodeFusion("node_sum", hidden_dim),
}

# pair-wise fusion operator
PAIR_FUSION_OPS = {
    
}


class NaAggregator(Module):
    def __init__(self, input_dim, output_dim, rel_num, base_num, dropout, head_num, aggregator):
        super(NaAggregator, self).__init__()
        if aggregator == "rgcn":
            self._op = RGCNLayer(input_dim, output_dim, rel_num, base_num, dropout=dropout)
        elif aggregator == "rgat_vanilla":
            self._op = MultiHeadRGATLayer(input_dim, output_dim // head_num, rel_num, base_num, dropout=dropout,
                                          head_num=head_num,
                                          att_type="vanilla")
        elif aggregator == "rgat_sym":
            self._op = MultiHeadRGATLayer(input_dim, output_dim // head_num, rel_num, base_num, dropout=dropout,
                                          head_num=head_num,
                                          att_type="sym")
        elif aggregator == "rgat_cos":
            self._op = MultiHeadRGATLayer(input_dim, output_dim // head_num, rel_num, base_num, dropout=dropout,
                                          head_num=head_num,
                                          att_type="cos")
        elif aggregator == "rgat_linear":
            self._op = MultiHeadRGATLayer(input_dim, output_dim // head_num, rel_num, base_num, dropout=dropout,
                                          head_num=head_num,
                                          att_type="linear")
        elif aggregator == "rgat_gene-linear":
            self._op = MultiHeadRGATLayer(input_dim, output_dim // head_num, rel_num, base_num, dropout=dropout,
                                          head_num=head_num,
                                          att_type="gene-linear")
        elif aggregator == "compgcn_add":
            self._op = CompGCNLayer(input_dim, output_dim, rel_num, base_num, dropout=dropout, comp_op='add')
        elif aggregator == "compgcn_sub":
            self._op = CompGCNLayer(input_dim, output_dim, rel_num, base_num, dropout=dropout, comp_op='sub')
        elif aggregator == "compgcn_mult":
            self._op = CompGCNLayer(input_dim, output_dim, rel_num, base_num, dropout=dropout, comp_op='mult')
        elif aggregator == "compgcn_corr":
            self._op = CompGCNLayer(input_dim, output_dim, rel_num, base_num, dropout=dropout, comp_op="corr")
        elif aggregator == "compgcn_rotate":
            self._op = CompGCNLayer(input_dim, output_dim, rel_num, base_num, dropout=dropout, comp_op="rotate")
            # delattr(self._op, "rel")

    def forward(self, x, edge_index, edge_type, rel_embeds):
        return self._op(x, edge_index, edge_type, rel_embeds)



class LaAggregator(Module):
    def __init__(self, mode, hidden_dim, layer_num):
        super(LaAggregator, self).__init__()
        self.mode = mode
        if self.mode in ['lstm', 'max', 'cat']:
            self.jump = JKNet(mode, channels=hidden_dim, layer_num=layer_num)
        if self.mode == 'cat':
            self.lin = Linear(hidden_dim * layer_num, hidden_dim)
        else:
            self.lin = Linear(hidden_dim, hidden_dim)

    def forward(self, fts):
        if self.mode in ['lstm', 'max', 'cat']:
            return self.lin((self.jump(fts)))
        elif self.mode == 'sum':
            return torch.stack(fts, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            return torch.stack(fts, dim=-1).mean(dim=-1)
        # return self.lin(elu(self.jump(fts)))


# class SeqEncoder(Module):
#     def __init__(self, mode, input_dim, hidden_dim, seq_head_num, layer_num=1):
#         super(SeqEncoder, self).__init__()
#         self.mode = mode
#         if mode == 'gru':
#             self.seq = GRU(input_dim, hidden_dim)
#         elif mode == 'sa':
#             self.seq = SelfAttention(input_dim, seq_head_num)

#     def forward(self, current_embed, previous_embed, previous_embed_transformer, local_attn_mask):
#         if self.mode == 'gru':
#             _, hidden = self.seq(current_embed, previous_embed)
#             return hidden[-1]
#         elif self.mode == 'sa':
#             return self.seq(current_embed.squeeze(0), previous_embed_transformer, local_attn_mask)
#         else:
#             return current_embed.squeeze(0)


class Identity(Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, embed):
        return embed


class Zero(Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, embed):
        embed = embed.mul(0.)
        return embed


class LayerConnector(Module):
    def __init__(self, mode, hidden_dim):
        super(LayerConnector, self).__init__()
        self.mode = mode
        if self.mode == "lc_concat":
            self.linear = Linear(2 * hidden_dim, hidden_dim)

    def forward(self, fts):
        if self.mode == "lc_skip":
            return fts[0]
        elif self.mode == "lc_sum":
            return torch.stack(fts, dim=-1).sum(dim=-1)
        elif self.mode == "lc_concat":
            return self.linear(torch.cat(fts, dim=-1))
        else:
            raise NotImplementedError


class LayerFusion(Module):
    def __init__(self, mode, hidden_dim, layer_num):
        super(LayerFusion, self).__init__()
        self.mode = mode
        if self.mode == "lf_concat":
            self.linear = Linear(layer_num * hidden_dim, hidden_dim)

    def forward(self, fts):
        if self.mode == "lf_skip":
            return fts[-1]
        elif self.mode == "lf_sum":
            return torch.stack(fts, dim=-1).sum(dim=-1)
        elif self.mode == "lf_max":
            return torch.stack(fts, dim=-1).max(dim=-1)[0]
        elif self.mode == "lf_mean":
            return torch.stack(fts, dim=-1).mean(dim=-1)
        elif self.mode == "lf_concat":
            return self.linear(torch.cat(fts, dim=-1))
        else:
            raise NotImplementedError


class NodeInfoOps(Module):
    def __init__(self, mode, input_dim, ent_num):
        super(NodeInfoOps, self).__init__()
        self.mode = mode
        self.ops_list = [0 if i == 'na' else 1 for i in self.mode.split('_')]
        # 创建mask矩阵，其中需要掩盖的部分为0，其余部分为1
        self.mask_matrix = torch.ones((ent_num, input_dim), dtype=torch.float32).to("cuda")
        node_dim = int(input_dim / 5)  # 5种node-wise启发式信息
        for index, ops in enumerate(self.ops_list):
          if ops==0:
            self.mask_matrix[:, index*node_dim:(index+1)*node_dim] = 0

    def forward(self, node_info_embed):
        # 将特征矩阵与mask矩阵相乘，实现掩盖操作
        return node_info_embed*self.mask_matrix
    
class PairInfoOps(Module):
    def __init__(self, mode, input_dim, batch_pair_num):
        super(PairInfoOps, self).__init__()
        self.mode = mode
        self.ops_list = [0 if i == 'na' else 1 for i in self.mode.split('_')]
        # 创建mask矩阵，其中需要掩盖的部分为0，其余部分为1
        self.mask_matrix = torch.ones((batch_pair_num, input_dim), dtype=torch.float32).to("cuda")
        pair_dim = int(input_dim / 4)  # 4种pair-wise启发式信息
        for index, ops in enumerate(self.ops_list):
          if ops==0:
            self.mask_matrix[:, index*pair_dim:(index+1)*pair_dim] = 0

    def forward(self, pair_info_embed):
        # 将特征矩阵与mask矩阵相乘，实现掩盖操作
        return pair_info_embed*self.mask_matrix
    
class NodeFusion(Module):
    def __init__(self, mode, hidden_dim):
        super(NodeFusion, self).__init__()
        self.mode = mode
        if self.mode == "node_sum":
          self.learnable_vec_sub = torch.nn.Parameter(torch.rand(size=(hidden_dim,2)))
          self.learnable_vec_obj = torch.nn.Parameter(torch.rand(size=(hidden_dim,2)))

    def forward(self, sub_emb, obj_emb, subNodeEmbed, objNodeEmbed):
        if self.mode == "node_sum":
            alpha_obj = torch.softmax(self.learnable_vec_obj, dim = 1)
            alpha_sub = torch.softmax(self.learnable_vec_sub, dim = 1)
            sub_emb = alpha_sub[:,0] * sub_emb + alpha_sub[:,1] * subNodeEmbed
            obj_emb = alpha_obj[:,0] * obj_emb + alpha_obj[:,1] * objNodeEmbed
            return sub_emb, obj_emb
        # elif self.mode == "lf_sum":
        #     return torch.stack(fts, dim=-1).sum(dim=-1)
        # elif self.mode == "lf_max":
        #     return torch.stack(fts, dim=-1).max(dim=-1)[0]
        # elif self.mode == "lf_mean":
        #     return torch.stack(fts, dim=-1).mean(dim=-1)
        # elif self.mode == "lf_concat":
        #     return self.linear(torch.cat(fts, dim=-1))
        else:
            raise NotImplementedError
        
# class PairFusion(Module):
#     def __init__(self, mode, hidden_dim, layer_num):
#         super(PairFusion, self).__init__()
#         self.mode = mode
#         if self.mode == "lf_concat":
#             self.linear = Linear(layer_num * hidden_dim, hidden_dim)

#     def forward(self, fts):
#         if self.mode == "lf_skip":
#             return fts[-1]
#         elif self.mode == "lf_sum":
#             return torch.stack(fts, dim=-1).sum(dim=-1)
#         elif self.mode == "lf_max":
#             return torch.stack(fts, dim=-1).max(dim=-1)[0]
#         elif self.mode == "lf_mean":
#             return torch.stack(fts, dim=-1).mean(dim=-1)
#         elif self.mode == "lf_concat":
#             return self.linear(torch.cat(fts, dim=-1))
#         else:
#             raise NotImplementedError