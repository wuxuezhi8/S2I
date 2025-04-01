from torch.nn import Module, ModuleList, Parameter, Dropout, BatchNorm1d
from torch.nn.init import xavier_uniform_, xavier_normal_, zeros_, calculate_gain
from torch.nn.functional import relu
import dgl.function as fn
import torch
from torch import nn
from utils import ccorr, rotate
from pprint import pprint


class CompGCNLayer(Module):
    def __init__(self, input_size, output_size, rel_num, base_num,  dropout=0.0, comp_op="corr",
                 bias=True):
        super(CompGCNLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.base_num = base_num
        self.rel_num = rel_num
        self.activation = torch.tanh
        self.dropout = dropout
        self.comp_op = comp_op
        # self.rel = None
        # CompGCN loop weight
        self.w_loop = self.get_param([input_size, output_size])
        # CompGCN weights
        self.w_in = self.get_param([input_size, output_size])
        self.w_out = self.get_param([input_size, output_size])
        self.w_rel = self.get_param([input_size, output_size])  # transform embedding of relations to next layer
        self.loop_rel = self.get_param([1, input_size])  # self-loop embedding
        if base_num > 0:
            self.rel_wt = self.get_param([rel_num * 2, base_num])
        else:
            self.rel_wt = None

        # CompGCN dropout
        if dropout:
            self.dropout = Dropout(dropout)
        # CompGCN bias
        if bias:
            self.w_bias = Parameter(torch.zeros(self.output_size))
        else:
            self.register_parameter("w_bias", None)
        self.bn = BatchNorm1d(self.output_size)


    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param
    
    def get_message(self, edges):
        edge_type = edges.data['id']  # [E, 1]
        edge_num = edge_type.shape[0]
        edge_data = self.rel_transform(
            edges.src['ft'], self.rel[edge_type])  # [E, in_channel]
        # NOTE: first half edges are all in-directions, last half edges are out-directions.
        msg = torch.cat([torch.matmul(edge_data[:edge_num // 2, :], self.w_in),
                         torch.matmul(edge_data[edge_num // 2:, :], self.w_out)])
        msg = msg * edges.data['norm'].reshape(-1, 1)  # [E, D] * [E, 1]
        return {'msg': msg}

    # def apply_func(self, nodes):
    #     return {'ft': nodes.data['ft'] * nodes.data['norm']}

    def reduce_func(self, nodes):
        return {'ft': self.dropout(nodes.data['ft'])}

    def forward(self, graph, rel_embeds):
        if self.rel_wt is None:
            self.rel = rel_embeds
        else:
            # [num_rel*2, num_base] @ [num_base, in_c]
            self.rel = torch.mm(self.rel_wt, rel_embeds)

        
        graph = graph.local_var()
        node_repr = graph.ndata['ft']

        graph.update_all(self.get_message, fn.sum(
            msg='msg', out='ft'), self.reduce_func)
        
        node_repr = (graph.ndata.pop('ft') +
                 torch.mm(self.rel_transform(node_repr, self.loop_rel), self.w_loop)) / 3

        if self.w_bias is not None:
            node_repr = node_repr + self.w_bias

        node_repr = self.bn(node_repr)
        if self.activation:
            node_repr = self.activation(node_repr)
        # if self.dropout:
        #     x = self.dropout(x)
        graph.ndata['ft'] = node_repr
        # return graph
        return graph, torch.matmul(self.rel, self.w_rel)
    


    def rel_transform(self, ent_embed, rel_embed, op='corr'):
        if op == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif op == 'rotate':
            trans_embed = rotate(ent_embed, rel_embed)
        elif op == 'add':
            trans_embed = ent_embed + rel_embed
        elif op == 'sub':
            trans_embed = ent_embed - rel_embed
        elif op == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError

        return trans_embed


class CompGCN(Module):
    def __init__(self, args, input_size, hidden_size, output_size, rel_num):
        super(CompGCN, self).__init__()
        self.args = args
        self.layers = ModuleList()
        self.layer_num = self.args.gnn_layer_num
        self.activation = torch.tanh
        self.rel_num = rel_num
        for idx in range(self.layer_num):
            if idx == 0:
                self.layers.append(CompGCNLayer(
                    input_size, hidden_size, self.args.base_num, self.rel_num, self.activation, self.args.dropout,
                    self.args.comp_op
                ))
            else:
                if idx == self.layer_num - 1:
                    self.layers.append(CompGCNLayer(
                        hidden_size, output_size, self.args.base_num, self.rel_num, self.activation, self.args.dropout,
                        self.args.comp_op
                    ))
                else:
                    self.layers.append(CompGCNLayer(
                        hidden_size, hidden_size, self.args.base_num, self.rel_num, self.activation, self.args.dropout,
                        self.args.comp_op
                    ))

    def forward(self, graph):
        for i, layer in enumerate(self.layers):
            graph = layer(graph)
        return graph

    def forward_isolated(self, ent_embed):
        for i, layer in enumerate(self.layers):
            ent_embed = layer.forward_isolated(ent_embed)
        return ent_embed
