import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, GraphConv


class GAT(nn.Module):
    def __init__(self, g, num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, negative_slope, residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.gat_layers.append(GATConv(in_dim, num_hidden, heads[0], feat_drop, attn_drop, negative_slope, False, None))
        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(num_hidden * heads[l-1], num_hidden, heads[l], feat_drop, attn_drop, negative_slope, residual, None))
        self.gat_layers.append(GATConv(num_hidden * heads[-2], num_classes, heads[-1], feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs, middle=False):
        h = inputs
        middle_feats = []
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g.to(torch.device("cuda:0")), h).flatten(1)
            middle_feats.append(h)
            h = self.activation(h)
        logits = self.gat_layers[-1](self.g.to(torch.device("cuda:0")), h).mean(1)
        if middle:
            return logits, middle_feats
        return logits


class GCN(nn.Module):
    def __init__(self, g, num_layers, in_dim, num_hidden, num_classes):
        super(GCN, self).__init__()
        self.g = g
        self.gat_layers = []
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GraphConv(in_dim, num_hidden))
        for _ in range(1, num_layers):
            self.gat_layers.append(GraphConv(num_hidden, num_hidden))

        self.gat_layers.append(GraphConv(num_hidden, num_classes))

    def forward(self, inputs, middle=False):
        h = inputs
        middle_feats = []
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g.to(torch.device("cuda:0")), h)
            middle_feats.append(h)
            h = F.relu(h)
        logits = self.gat_layers[-1](self.g.to(torch.device("cuda:0")), h)
        if middle:
            return logits, middle_feats
        return logits