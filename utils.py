import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import f1_score

import dgl
from dgl.data.ppi import LegacyPPIDataset as PPIDataset
from gnns import GAT
from topo_semantic import get_loc_model, get_upsamp_model


def parameters(model):
    num_params = 0
    for params in model.parameters():
        cur = 1
        for size in params.data.shape:
            cur *= size
        num_params += cur
    return num_params


def evaluate(feats, model, subgraph, labels, loss_fcn):
    model.eval()
    with torch.no_grad():
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(feats.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), predict, average='micro')
    model.train()

    return score, loss_data.item()


def test_model(test_dataloader, model, device, loss_fcn):
    test_score_list = []
    model.eval()
    with torch.no_grad():
        for batch, test_data in enumerate(test_dataloader):
            subgraph, feats, labels = test_data
            feats = feats.to(device)
            labels = labels.to(device)
            test_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn)[0])
        mean_score = np.array(test_score_list).mean()
        print('\033[95m' + f"F1-Score on testset:        {mean_score:.4f}" + '\033[0m')
    model.train()
    return mean_score


def generate_label(t_model, subgraph, feats, middle=False):
    t_model.eval()
    with torch.no_grad():
        t_model.g = subgraph
        for layer in t_model.gat_layers:
            layer.g = subgraph
        if not middle:
            logits_t = t_model(feats.float())
            return logits_t.detach()
        else:
            logits_t, middle_feats = t_model(feats.float(), middle)
            return logits_t.detach(), middle_feats


def evaluate_model(valid_dataloader, device, s_model, loss_fcn):
    score_list = []
    val_loss_list = []
    s_model.eval()
    with torch.no_grad():
        for batch, valid_data in enumerate(valid_dataloader):
            subgraph, feats, labels = valid_data
            feats = feats.to(device)
            labels = labels.to(device)
            score, val_loss = evaluate(feats.float(), s_model, subgraph, labels.float(), loss_fcn)
            score_list.append(score)
            val_loss_list.append(val_loss)
    mean_score = np.array(score_list).mean()
    print(f"F1-Score on valset  :        {mean_score:.4f} ")
    s_model.train()
    return mean_score


def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels


def get_teacher(args, data_info):
    heads1 = ([args.t_num_heads] * args.t1_num_layers) + [args.t_num_out_heads]
    heads2 = ([args.t_num_heads] * args.t2_num_layers) + [args.t_num_out_heads]
    heads3 = ([args.t_num_heads] * args.t3_num_layers) + [args.t_num_out_heads]
    model1 = GAT(data_info['g'], args.t1_num_layers, data_info['num_feats'], args.t1_num_hidden, data_info['n_classes'],
                 heads1, F.elu, args.in_drop, args.attn_drop, args.alpha, args.residual)
    model2 = GAT(data_info['g'], args.t2_num_layers, data_info['num_feats'], args.t2_num_hidden, data_info['n_classes'],
                 heads2, F.elu, args.in_drop, args.attn_drop, args.alpha, args.residual)
    model3 = GAT(data_info['g'], args.t3_num_layers, data_info['num_feats'], args.t3_num_hidden, data_info['n_classes'],
                 heads3, F.elu, args.in_drop, args.attn_drop, args.alpha, args.residual)
    return model1, model2, model3


def get_student(args, data_info):
    heads = ([args.s_num_heads] * args.s_num_layers) + [args.s_num_out_heads]
    model = GAT(data_info['g'], args.s_num_layers, data_info['num_feats'], args.s_num_hidden, data_info['n_classes'],
                heads, F.elu, args.in_drop, args.attn_drop, args.alpha, args.residual)
    return model


def mlp(dim, logits, device):
    output = logits
    # linear = nn.Linear(dim, dim).to(device)
    # relu = nn.ReLU()
    # return linear(relu(linear(output)))
    return output


def get_feat_info(args):
    feat_info = {}
    feat_info['s_feat'] = [args.s_num_heads * args.s_num_hidden] * args.s_num_layers
    feat_info['t1_feat'] = [args.t_num_heads * args.t1_num_hidden] * args.t1_num_layers
    feat_info['t2_feat'] = [args.t_num_heads * args.t2_num_hidden] * args.t2_num_layers
    feat_info['t3_feat'] = [args.t_num_heads * args.t3_num_hidden] * args.t3_num_layers
    return feat_info


def get_data_loader(args):
    train_dataset = PPIDataset(mode='train')
    valid_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4, shuffle=True)
    fixed_train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=2)

    n_classes = train_dataset.labels.shape[1]
    num_feats = train_dataset.features.shape[1]
    g = train_dataset.graph
    data_info = {}
    data_info['n_classes'] = n_classes
    data_info['num_feats'] = num_feats
    data_info['g'] = g
    return (train_dataloader, valid_dataloader, test_dataloader, fixed_train_dataloader), data_info


def save_checkpoint(model, path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    torch.save(model.state_dict(), path)
    print(f"save model to {path}")


def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Load model from {path}")


def collect_model(args, data_info):
    device = torch.device("cuda:0")

    feat_info = get_feat_info(args)

    t1_model, t2_model, t3_model = get_teacher(args, data_info)
    t1_model.to(device)
    t2_model.to(device)
    t3_model.to(device)

    s_model = get_student(args, data_info)
    s_model.to(device)

    local_model = get_loc_model(feat_info)
    local_model.to(device)
    local_model_s = get_loc_model(feat_info, upsampling=True)
    local_model_s.to(device)

    upsampling_model1, upsampling_model2, upsampling_model3 = get_upsamp_model(feat_info)
    upsampling_model1.to(device)
    upsampling_model2.to(device)
    upsampling_model3.to(device)

    s_model_optimizer = torch.optim.Adam(s_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t1_model_optimizer = torch.optim.Adam(t1_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t2_model_optimizer = torch.optim.Adam(t2_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t3_model_optimizer = torch.optim.Adam(t3_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    local_model_optimizer = None
    local_model_s_optimizer = None
    upsampling_model1_optimizer = torch.optim.Adam(upsampling_model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    upsampling_model2_optimizer = torch.optim.Adam(upsampling_model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    upsampling_model3_optimizer = torch.optim.Adam(upsampling_model3.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_dict = {}
    model_dict['s_model'] = {'model': s_model, 'optimizer': s_model_optimizer}
    model_dict['local_model'] = {'model': local_model, 'optimizer': local_model_optimizer}
    model_dict['local_model_s'] = {'model': local_model_s, 'optimizer': local_model_s_optimizer}
    model_dict['t1_model'] = {'model': t1_model, 'optimizer': t1_model_optimizer}
    model_dict['t2_model'] = {'model': t2_model, 'optimizer': t2_model_optimizer}
    model_dict['t3_model'] = {'model': t3_model, 'optimizer': t3_model_optimizer}
    model_dict['upsampling_model1'] = {'model': upsampling_model1, 'optimizer': upsampling_model1_optimizer}
    model_dict['upsampling_model2'] = {'model': upsampling_model2, 'optimizer': upsampling_model2_optimizer}
    model_dict['upsampling_model3'] = {'model': upsampling_model3, 'optimizer': upsampling_model3_optimizer}
    return model_dict