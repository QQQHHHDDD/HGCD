import dgl
from sklearn.metrics import roc_auc_score
import torch
import torch.optim as optim
import numpy as np
from dataloader import DataLoader
from model import HGCD
import sys
from utils import *
import warnings
warnings.filterwarnings("ignore")

def get_graph_efeat(adjM, edge2type, device, flag=0, pre=0):
    g = dgl.DGLGraph(adjM + (adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        if flag != 1:
            e_feat.append(edge2type[(u, v)])
        else:
            e_feat.append(edge2type[(u+pre, v+pre)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    return g, e_feat

def train(args):
    data_loader = DataLoader(args, dataset = args.dataset, mode ='train')
    test_loader = DataLoader(args, dataset=args.dataset, mode='test')
    net = HGCD(args.num_class, args.num_stu, args.num_exer, args.num_skill, args.t, args=args, device=device)
    net.to(device)
    optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.reg_r)
    print('training model...')

    rmsem_g = 1.0
    maem_g = 1.0
    accuracy_i = 0
    rmse_i = 1.0
    auc_i = 0
    for epoch in range(args.num_train_epochs):
        running_loss = 0.0
        count = 1
        for i in range(data_loader.num_class):
            adjM2 = data_loader.adjMs[i]
            edge2type = data_loader.edge2types[i]
            kn_emb_g = data_loader.kn_embs_g[i].to(device)
            stu_list_g = data_loader.stu_lists_g[i].to(device)
            class_id_g = data_loader.class_ids_g[i].to(device)
            exer_list_g = data_loader.exer_lists_g[i].to(device)
            labels_g = data_loader.labels_g[i].to(device)
            stu_list_i = data_loader.stu_lists_i[i].to(device)
            exer_list_i = data_loader.exer_lists_i[i].to(device)
            labels_i = data_loader.labels_i[i].to(device)
            kn_emb_i = data_loader.kn_embs_i[i].to(device)
            adjM1 = adjM2[:1+stu_list_g.shape[0], :1+stu_list_g.shape[0]]
            adjM3 = adjM2[1+stu_list_g.shape[0]:, 1+stu_list_g.shape[0]:]
            g1, e_feat1 = get_graph_efeat(adjM1, edge2type, device)
            g2, e_feat2 = get_graph_efeat(adjM2, edge2type, device)
            g3, e_feat3 = get_graph_efeat(adjM3, edge2type, device, flag=1, pre=1 + stu_list_g.shape[0])

            
            optimizer.zero_grad()
            loss = net.forward(g1, e_feat1, g2, e_feat2, g3, e_feat3, class_id_g, stu_list_g, kn_emb_g, exer_list_g, None, stu_list_i, exer_list_i, kn_emb_i, labels_g, labels_i)
            # print(loss)
            loss.backward()
            optimizer.step()
            net.apply_clipper()
            running_loss += loss.item()
            count = count + 1
            if count%100 == 0 :
                print('[%d,%5d] loss: %f'%(epoch+1, count, running_loss))
                running_loss = 0.0
            
        if epoch % 5 == 0:
            rmse_g, mae_g, rmse_i, auc_i, accuracy_i = test(test_loader, net, args)
            if rmse_g < rmsem_g:
                rmsem_g = rmse_g
                maem_g = mae_g
                accuracym_i = accuracy_i
                rmsem_i = rmse_i
                aucm_i = auc_i
                torch.save(net,f'./saved_ours/{args.dataset}/model.pth')
    return rmsem_g, maem_g, accuracym_i, rmsem_i, aucm_i

def test(test_loader, net, args):
    pred = []
    real = []
    pred_i = []
    real_i = []
    correct_count = 0
    exer_count = 0
    for i in range(test_loader.num_class):
        # print(i)
        adjM = test_loader.adjMs[i]
        edge2type = test_loader.edge2types[i]
        kn_emb_g = test_loader.kn_embs_g[i].to(device)
        stu_list_g = test_loader.stu_lists_g[i].to(device)
        class_id_g = test_loader.class_ids_g[i].to(device)
        exer_list_g = test_loader.exer_lists_g[i].to(device)
        labels_g = test_loader.labels_g[i].to(device)
        stu_list_i = test_loader.stu_lists_i[i].to(device)
        exer_list_i = test_loader.exer_lists_i[i].to(device)
        labels_i = test_loader.labels_i[i].to(device)
        kn_emb_i = test_loader.kn_embs_i[i].to(device)
        exer_test_g = test_loader.exer_tests_g[i].to(device)
        
        g = dgl.DGLGraph(adjM + (adjM.T))
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g = g.to(device)
        e_feat = []
        for u, v in zip(*g.edges()):
            u = u.cpu().item()
            v = v.cpu().item()
            e_feat.append(edge2type[(u,v)])
        e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)        
        
        output1_g, output1_i = net.forward(None, None, g, e_feat, None, None, class_id_g, stu_list_g, kn_emb_g, exer_list_g, exer_test_g, stu_list_i, exer_list_i, kn_emb_i)
        
        for i in range(len(labels_i)):
            if (labels_i[i] == 1 and output1_i[i] > 0.5) or (labels_i[i] == 0 and output1_i[i] < 0.5):
                correct_count += 1
        exer_count += len(labels_i)        
        pred.extend(output1_g.tolist())
        real.extend(labels_g.tolist())
        pred_i.extend(output1_i.tolist())
        real_i.extend(labels_i.tolist())

    pred = np.array(pred)
    real = np.array(real)
    pred_i = np.array(pred_i)
    real_i = np.array(real_i)
    accuracy_i = correct_count / exer_count
    rmse_i = np.sqrt(np.mean((real_i - pred_i) ** 2))
    auc_i = roc_auc_score(real_i, pred_i)
    rmse_value_g = rmse(pred, real)
    mae_value_g = mae(pred, real)
    return rmse_value_g, mae_value_g, rmse_i, auc_i, accuracy_i

if __name__ == '__main__':
    sys.stdout = Logger("output.log", sys.stdout)
    args = set_args()
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)
    set_seed(args.seed)
    rmsem_g, maem_g, accuracy_i, rmse_i, auc_i = train(args)