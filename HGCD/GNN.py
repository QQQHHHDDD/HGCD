import torch
import torch.nn as nn
from conv import myGATConv, hyperGATConv


class myGAT(nn.Module):
    def __init__(self,
                 edge_dim,
                 num_etypes,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha):
        super(myGAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc = nn.Linear(in_dim, num_hidden, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(edge_dim, num_etypes,
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

    def forward(self, g, x, e_feat):
        x = self.fc(x)
        res_attn = None
        for l in range(self.num_layers):
            x, res_attn = self.gat_layers[l](g, x, e_feat, res_attn=res_attn)
            x = x.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](g, x, e_feat, res_attn=None)
        logits = logits.mean(1)
        return logits
    

class hyperGAT(nn.Module):
    def __init__(self,
                 manifold_in,
                 manifold_out,
                 edge_dim,
                 num_etypes,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha):
        super(hyperGAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # print(activation)
        self.fc = nn.Linear(in_dim, num_hidden, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(hyperGATConv(manifold_in, manifold_out, edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(hyperGATConv(manifold_out, manifold_out, edge_dim, num_etypes,
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
        # output projection
        self.gat_layers.append(hyperGATConv(manifold_out, manifold_in, edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))

    def forward(self, g, x, e_feat):
        # print(g)
        x = self.fc(x)
        res_attn = None
        for l in range(self.num_layers):
            x, res_attn = self.gat_layers[l](g, x, e_feat, res_attn=res_attn)
            x = x.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](g, x, e_feat, res_attn=None)
        logits = logits.mean(1)
        return logits