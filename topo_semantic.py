import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.softmax import edge_softmax


class _GAT(nn.Module):
    def __init__(self, in_dim, out_dim, retatt=False):
        super(_GAT, self).__init__()
        self.GATConv1 = GATConv(in_feats=in_dim, out_feats=out_dim, num_heads=1)
        self.nonlinear = nn.LeakyReLU(negative_slope=0.2)
        self.retatt = retatt

    def forward(self, graph, feats):
        feats = self.nonlinear(feats)
        if self.retatt:
            rst, att = self.GATConv1(graph, feats, self.retatt)
            return rst.flatten(1), att
        else:
            rst = self.GATConv1(graph, feats, self.retatt)
            return rst.flatten(1)


class distance(nn.Module):
    def __init__(self):
        super(distance, self).__init__()

    def forward(self, graph, feats):
        graph = graph.local_var().to(torch.device("cuda:0"))
        feats = feats.view(-1, 1, feats.shape[1])
        graph.ndata.update({'ftl': feats, 'ftr': feats})

        graph.apply_edges(fn.u_sub_v('ftl', 'ftr', 'diff'))
        e = graph.edata.pop('diff')
        e = torch.exp((-1.0 / 100) * torch.sum(torch.abs(e), dim=-1))

        e = edge_softmax(graph, e)
        return e


def get_loc_model(feat_info, upsampling=False):
    return distance()


def get_upsamp_model(feat_info):
    upsamp_model1 = _GAT(feat_info['s_feat'][0], feat_info['t1_feat'][0])
    upsamp_model2 = _GAT(feat_info['s_feat'][1], feat_info['t2_feat'][1])
    upsamp_model3 = _GAT(feat_info['s_feat'][2], feat_info['t3_feat'][2])
    return upsamp_model1, upsamp_model2, upsamp_model3