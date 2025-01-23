import torch
import numpy as np
import argparse
import random
import sys


def rmse(a, b):
    return np.sqrt(np.mean((a-b)**2))

def mae(a, b):
    return np.mean(np.abs(a-b))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Logger(object):
    def __init__(self, filename='output.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
        # self.log = open(filename, 'a+')

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SLP_math', type=str, help='')
    parser.add_argument('--num_stu', default=3922, type=int, help='num_stu')
    parser.add_argument('--seed', default=2024, type=int, help='seed')
    parser.add_argument('--num_exer', default=120, type=int, help='num_exer')
    parser.add_argument('--num_class', default=145, type=int, help='number of groups')
    parser.add_argument('--num_train_epochs', default=100, type=int, help='')
    parser.add_argument('--num_skill', default=21, type=int, help='')
    parser.add_argument('--lr', default=0.001, type=float, help='')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--t', default=0.77, type=float, help='')
    parser.add_argument('--reg_r', default=1e-4, type=float, help='')
    parser.add_argument('--kl_r', default=1e-6, type=float, help='')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    parser.add_argument('--patience', type=int, default=30, help='Patience.')
    parser.add_argument('--slope', type=float, default=0.05)
    parser.add_argument('--edge_feats', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--c1', type=float, default=1.0, help='Hyperbolic radius for the first hyperbolic space.')
    parser.add_argument('--c2', type=float, default=1.0, help='Hyperbolic radius for the second hyperbolic space.')
    parser.add_argument('--k', type=float, default=0.8, help='Threshold for the knowledge graph.')
    parser.add_argument('--alpha', type=float, default=0.1, help='balance the two loss')
    parser.add_argument('--beta', type=float, default=0.01, help='balance the two loss')
    parser.add_argument('--kn_edge', action='store_true')
    
    return parser.parse_args()