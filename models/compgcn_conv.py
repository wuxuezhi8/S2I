from utils import *
from models.message_passing import MessagePassing
from torch_scatter.scatter import scatter_add

class CompGCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, base_num, dropout=0.0, comp_op="corr", bias=True):
        super(self.__class__, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.dropout = dropout
        self.comp_op = comp_op
        self.act = torch.tanh
        self.device = None

        self.w_loop = self.get_param((in_channels, out_channels))
        self.w_in = self.get_param((in_channels, out_channels))
        self.w_out = self.get_param((in_channels, out_channels))
        self.w_rel = self.get_param((in_channels, out_channels))
        self.loop_rel = self.get_param((1, in_channels))

        self.drop = torch.nn.Dropout(self.dropout)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        
        if bias:
            self.register_parameter(
                'bias', Parameter(torch.zeros(out_channels)))

    def get_param(self, shape):
      param = Parameter(torch.Tensor(*shape))
      xavier_normal_(param.data)
      # xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
      return param

    def forward(self, x, edge_index, edge_type, rel_embed):
        if self.device is None:
            self.device = edge_index.device
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        self.in_index, self.out_index = edge_index[:,
                                                   :num_edges], edge_index[:, num_edges:]
        self.in_type,  self.out_type = edge_type[:
                                                 num_edges], 	 edge_type[num_edges:]

        self.loop_index = torch.stack(
            [torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(
            0)-1, dtype=torch.long).to(self.device)

        self.in_norm = self.compute_norm(self.in_index,  num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)

        in_res = self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,
                                rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in')
        loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type,
                                  rel_embed=rel_embed, edge_norm=None, 		mode='loop')
        out_res = self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,
                                 rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
        out = self.drop(in_res)*(1/3) + self.drop(out_res) * \
            (1/3) + loop_res*(1/3)

        out = out + self.bias
        out = self.bn(out)

        # Ignoring the self loop inserted
        # return self.act(out), rel_embed[:-1]
        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]

    def rel_transform(self, ent_embed, rel_embed):
        if self.comp_op == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.comp_op == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.comp_op == 'mult':
            trans_embed = ent_embed * rel_embed
        elif self.comp_op == 'add':
            trans_embed = ent_embed + rel_embed
        elif self.comp_op == 'rotate':
            trans_embed = rotate(ent_embed, rel_embed)
        else:
            raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight = getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        # Summing number of weights of the edges
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)
        deg_inv = deg.pow(-0.5)							# D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

        return norm

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
