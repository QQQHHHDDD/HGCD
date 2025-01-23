import torch
import torch.nn as nn
import torch.nn.functional as F
from euclidean import Euclidean
import geoopt
from structure_modules.models import *
import torch.nn.functional as F
from GNN import myGAT, hyperGAT
from structure_modules.hyp_layers import *

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def con_loss(z1: torch.Tensor, z2: torch.Tensor, len1: int, tau: float, flag: bool = False):
    f = lambda x: torch.exp(x / tau)
    sim_matrix = sim(z1, z2) 
    if flag:
        pos_sim = f(sim_matrix[:, :len1].diag())
    else:
        pos_sim = f(sim_matrix[:, len1:].diag())
    all_sim = f(sim_matrix)
    denominator = all_sim.sum(dim=1)
    loss = -torch.log(pos_sim / denominator)
    return loss.mean()

def infonce_loss(x1, x2, x3, len1, temperature=0.07):
    loss_1 = con_loss(x1, x2, len1, temperature, flag=True)
    loss_2 = con_loss(x3, x2, len1, temperature, flag=False)
    return (loss_1 + loss_2) * 0.5

class HGCD(nn.Module):
    def __init__(self, class_n, stu_n, exer_n, skill_n, t, args, device='cuda'):
        self.class_n = class_n
        self.device = device
        self.args = args
        self.stu_n = stu_n
        self.exer_n = exer_n
        self.skill_n = skill_n
        self.prednet_input_len = skill_n
        self.prednet_len1,self.prednet_len2,self.prednet_len3 = 256, 128, 1
        self.pi = t 
        super(HGCD, self).__init__()

        self.class_emb = nn.Embedding(self.class_n, self.skill_n)
        self.stu_emb = nn.Embedding(self.stu_n, self.skill_n)
        self.exer_diff = nn.Embedding(self.exer_n, self.skill_n)
        self.exer_dis = nn.Embedding(self.exer_n, 1)
        self.know_emb = nn.Embedding(self.skill_n, self.skill_n)
        self.manifold_in = Euclidean()
        self.manifold_out1 = geoopt.PoincareBall(c=args.c1)
        self.manifold_out2 = geoopt.PoincareBall(c=args.c2)
        self.heads = [args.num_heads] * args.num_layers + [1]
        self.in_dim = args.num_skill
        self.loss_function_g = nn.MSELoss()
        self.loss_function_i = nn.NLLLoss()
        self.conv1 = hyperGAT(self.manifold_in, self.manifold_out1, args.edge_feats, 10*2+1, self.in_dim, args.hidden_dim, args.num_skill, args.num_layers, self.heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)
        self.conv2 = myGAT(args.edge_feats, 10*2+1, self.in_dim, args.hidden_dim, args.num_skill, args.num_layers, self.heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)
        self.conv3 = hyperGAT(self.manifold_in, self.manifold_out2, args.edge_feats, 10*2+1, self.in_dim, args.hidden_dim, args.num_skill, args.num_layers, self.heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)
        
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        for name,param in self.named_parameters():
            if 'weight' in name and 'conv1' not in name and 'conv2' not in name:
                nn.init.xavier_normal_(param)
    
    def forward(self, g1, e_feat1, g2, e_feat2, g3, e_feat3, class_id_g, stu_list_g, kn_emb_g, exer_list_g, exer_test_g=None, stu_list_i=None, exer_list_i=None, kn_emb_i=None, labels_g=None, labels_i=None):
        stu_embeddings_g = self.stu_emb(stu_list_g)
        # print(exer_list)
        exer_embeddings_g = self.exer_diff(exer_list_g)
        class_emb_g = self.class_emb(class_id_g)
        know_emb = self.know_emb(torch.arange(self.skill_n).to(self.device))
        stu_embeddings_i = self.stu_emb(stu_list_i)
        exer_embeddings_i = self.exer_diff(exer_list_i)
        
        x1 = torch.concat([class_emb_g.unsqueeze(0), stu_embeddings_g], dim=0)
        x2 = torch.concat([class_emb_g.unsqueeze(0), stu_embeddings_g, exer_embeddings_g, know_emb], dim=0)
        x3 = torch.concat([exer_embeddings_g, know_emb], dim=0)
        
        if exer_test_g == None:
            x1 = self.conv1(g1, x1, e_feat1)
            x2 = self.conv2(g2, x2, e_feat2)
            x3 = self.conv3(g3, x3, e_feat3)
            exer_k_g = torch.sigmoid(exer_embeddings_g)
            exer_dis_g = torch.sigmoid(self.exer_dis(exer_list_g))*10
            exer_dis_i = torch.sigmoid(self.exer_dis(exer_list_i))*10
        else:
            x2 = self.conv2(g2, x2, e_feat2)
            exer_k_g = torch.sigmoid(self.exer_diff(exer_test_g))
            exer_dis_g = torch.sigmoid(self.exer_dis(exer_test_g))*10
            exer_dis_i = torch.sigmoid(self.exer_dis(exer_list_i))*10

        class_emb_g = x2[0]
        class_k = torch.sigmoid(class_emb_g)
        # group
        input_x_g = exer_dis_g * (class_k - exer_k_g) * kn_emb_g
        input_x_g = self.prednet_full1(input_x_g)
        input_x_g = self.drop_1(input_x_g)
        input_x_g = self.prednet_full2(input_x_g)
        input_x_g = self.drop_2(input_x_g)

        # individual
        input_x_i = exer_dis_i * (stu_embeddings_i - exer_embeddings_i) * kn_emb_i
        input_x_i = self.prednet_full1(input_x_i)
        input_x_i = self.drop_1(input_x_i)
        input_x_i = self.prednet_full2(input_x_i)
        input_x_i = self.drop_2(input_x_i)

        output0_g = torch.sigmoid(self.prednet_full3(input_x_g))
        output0_i = torch.sigmoid(self.prednet_full3(input_x_i))
        
        output1_g = output0_g.reshape([-1])
        output1_i = output0_i
        output0_i = torch.ones(output1_i.size()).to(self.device) - output1_i
        

        if exer_test_g == None:
            output_i = torch.cat((output0_i, output1_i), 1)
            output_i.clamp_(1e-7)
            con_loss = infonce_loss(x1, x2, x3, len1=1+stu_list_g.shape[0], temperature=0.5)
            loss = self.loss_function_g(output1_g, labels_g) + \
                self.args.beta * con_loss
                # self.args.alpha * self.loss_function_i(torch.log(output_i), labels_i)
            # loss =  self.loss_function_i(torch.log(output_i), labels_i) + \
            #     self.args.beta * con_loss    
            return loss
        else:
            return output1_g, output1_i
    
    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)
    
    def proj_tan0(self, u, manifold):
        if manifold.name == 'Lorentz':
            narrowed = u.narrow(-1, 0, 1)
            vals = torch.zeros_like(u)
            vals[:, 0:1] = narrowed
            return u - vals
        else:
            return u
