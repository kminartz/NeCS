import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch_scatter as ts

class MessagePassingLayer(nn.Module):
    """
    gnn layer that can handle varying topologies as opposed to only fully connected graphs
    """
    def __init__(self, output_nf, hidden_nf, act_fn=nn.ReLU(), bias=True, attention=False,
                 residual=True, agg_fn=None, types=(1,2)):
        super().__init__()

        self.attention = attention
        self.residual = residual
        if agg_fn is None:
            agg_fn = 'sum'
        self.agg_fn = agg_fn
        self.types = types
        self.hidden_nf = hidden_nf
        self.output_nf = output_nf

        self.edge_mlps = nn.ModuleDict()

        # for each node type combination, we have a separate set of weights for the message function:
        for i in types:
            for j in types:
                self.edge_mlps[f'{i}_{j}'] = nn.Sequential(
                    nn.LazyLinear(hidden_nf, bias=bias),
                    act_fn,
                    nn.Linear(hidden_nf, hidden_nf, bias=bias),
                    act_fn)

        if self.attention:
            self.att_mlps = nn.ModuleDict()
            for i in types:
                for j in types:
                    self.att_mlps[f'{i}_{j}'] = nn.Sequential(
                        nn.LazyLinear(hidden_nf, bias=bias),
                        act_fn,
                        nn.Linear(hidden_nf, 1, bias=bias),
                        nn.Sigmoid())

        self.node_mlps = nn.ModuleDict()

        # for each node type, a separate set of weights for the node function:
        for i in types:
            self.node_mlps[f'{i}'] = nn.Sequential(
                nn.LazyLinear(hidden_nf, bias=bias),
                act_fn,
                nn.Linear(hidden_nf, output_nf, bias=bias))


    def forward(self, tup):
        assert len(tup) == 4, 'expected 4-tuple input of nodes, edge idx, edge feat and node type here!'
        # note: edge_attr can be None
        x, edge_index, edge_attr, node_type = tup

        # perform some sanity checks
        assert min(self.types) <= torch.min(node_type) <= torch.max(node_type) <= max(self.types), 'expected node type to be in range of num_types!'
        assert len(x.shape) == 3, 'expected x to have three axes as per the global convention of graph-domain problems!'
        assert edge_index.shape[0] == 2, 'edge index should have shape 2 at first dim!'
        assert node_type.shape[0] == x.shape[0], 'expected node type to have same batch size as x!'

        x = x[:, 0]  # remove the unsqueeze 1 dim that was necessary for the logistics to play nicely in the enc-proc-dec and trainer code
        row, col = edge_index
        edge_feat = self.edge_model(x[row], x[col], edge_attr, node_type[row], node_type[col])
        h = self.node_model(x, edge_index, edge_feat, node_type)
        h = h.unsqueeze(1)  # add back the unnecessary dim which helps in handling the logistics in the global code
        return h, edge_index, edge_feat, node_type


    def edge_model(self, source, target, edge_attr, type1, type2):
        # message function \phi
        edge_in = torch.cat([source, target], dim=1)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1)

        out = torch.zeros(size=(*edge_in.shape[:-1], self.hidden_nf)).to(edge_in.device)
        for t1 in self.types:
            for t2 in self.types:
                edges_for_type_mask = torch.logical_and(type1 == t1, type2 == t2)
                mlp = self.edge_mlps[f'{t1}_{t2}']
                out[edges_for_type_mask] = mlp(edge_in[edges_for_type_mask])

        if self.attention:
            att = torch.zeros(size=(*edge_in.shape[:-1], 1)).to(edge_in.device)
            for t1 in self.types:
                for t2 in self.types:
                    edges_for_type_mask = torch.logical_and(type1 == t1, type2 == t2)
                    mlp = self.att_mlps[f'{t1}_{t2}']
                    att[edges_for_type_mask] = mlp(torch.abs(source[edges_for_type_mask] - target[edges_for_type_mask]))
            out = out * att

        return out

    def node_model(self, h, edge_index, edge_attr, type):
        # node update function \psi
        row, col = edge_index
        agg = self.unsorted_segment_agg(edge_attr, row, num_segments=h.size(0))
        node_in = torch.cat([h, agg], dim=1)

        out = torch.zeros(size=(*node_in.shape[:-1], self.output_nf)).to(node_in.device)

        for t in self.types:
            nodes_for_type_mask = type == t
            mlp = self.node_mlps[f'{t}']
            out[nodes_for_type_mask] = mlp(node_in[nodes_for_type_mask])

        if self.residual:
            out = out + h[:, :out.shape[1]]

        return out


    def unsorted_segment_agg(self, data, segment_ids, num_segments):
        """Custom PyTorch op to do nodewise aggregation of data"""
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)  # Init empty result tensor.
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        ts.scatter(src=data, index=segment_ids, dim=0, out=result, reduce=self.agg_fn)
        return result


class MessagePassingGNN(nn.Module):

    def __init__(self, num_layers, emb_dim, activation, **kwargs):
        super().__init__()
        self.forward_network = nn.Sequential(*[MessagePassingLayer(
            output_nf=emb_dim, hidden_nf=emb_dim, act_fn=activation, bias=True, **kwargs
        ) for _ in range(num_layers)])

    def forward(self, tup):
        assert isinstance(tup, tuple), 'expected tuple input of nodes, edge idx, edge feat, node_type here!'
        out = self.forward_network(tup)
        return out


