import torch
import torch.nn as nn
import dgl.function as fn


def kd_loss(logits, logits_t, alpha=1.0, T=10.0):
    ce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    kl_loss_fn = nn.KLDivLoss()
    labels_t = torch.where(logits_t > 0.0, torch.ones(logits_t.shape).to(logits_t.device),
                           torch.zeros(logits_t.shape).to(logits_t.device))
    ce_loss = ce_loss_fn(logits, labels_t)
    d_s = torch.log(torch.cat((torch.sigmoid(logits / T), 1 - torch.sigmoid(logits / T)), dim=1))
    d_t = torch.cat((torch.sigmoid(logits_t / T), 1 - torch.sigmoid(logits_t / T)), dim=1)
    kl_loss = kl_loss_fn(d_s, d_t) * T * T
    return ce_loss * alpha + (1 - alpha) * kl_loss


def KLDiv(graph, edgex, edgey):
    with graph.local_scope():
        nnode = graph.number_of_nodes()
        graph.ndata.update({'kldiv': torch.ones(nnode, 1).to(edgex.device)})
        diff = edgey * (torch.log(edgey) - torch.log(edgex))
        graph.edata.update({'diff': diff})
        graph.update_all(fn.u_mul_e('kldiv', 'diff', 'm'), fn.sum('m', 'kldiv'))
        return torch.mean(torch.flatten(graph.ndata['kldiv']))


def graphKL_loss(models, middle_feats_s, subgraph, feats, class_loss, epoch, args):
    if epoch < args.s_epochs:
        t_model = models['t1_model']['model']
        with torch.no_grad():
            t_model.g = subgraph
            for layer in t_model.gat_layers:
                layer.g = subgraph
            _, middle_feats_t = t_model(feats.float(), middle=True)
            middle_feats_t = middle_feats_t[0]
    elif args.s_epochs <= epoch < 2 * args.s_epochs:
        t_model = models['t2_model']['model']
        with torch.no_grad():
            t_model.g = subgraph
            for layer in t_model.gat_layers:
                layer.g = subgraph
            _, middle_feats_t = t_model(feats.float(), middle=True)
            middle_feats_t = middle_feats_t[1]
    else:
        t_model = models['t3_model']['model']
        with torch.no_grad():
            t_model.g = subgraph
            for layer in t_model.gat_layers:
                layer.g = subgraph
            _, middle_feats_t = t_model(feats.float(), middle=True)
            middle_feats_t = middle_feats_t[2]

    dist_t = models['local_model']['model'](subgraph, middle_feats_t)
    dist_s = models['local_model']['model'](subgraph, middle_feats_s)
    return KLDiv(subgraph.to(torch.device("cuda:0")), dist_s, dist_t)


def optimizing(models, loss, model_list):
    for model in model_list:
        models[model]['optimizer'].zero_grad()
    loss.backward()
    for model in model_list:
        models[model]['optimizer'].step()