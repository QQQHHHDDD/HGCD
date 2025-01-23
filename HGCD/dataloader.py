import sys
import numpy as np
import torch
import json
from scipy.sparse import csr_matrix
torch.set_printoptions(profile="full")

class DataLoader(object):
    def __init__(self, args, dataset='', mode = 'train'):
        self.ptr = 0
        self.data1 = []
        self.dataset = dataset
        self.mode = mode
        self.k = args.k
        self.args = args
        if mode == 'train':
            data_file = f'./datasets/{dataset}/train3.json'
        else:
            data_file = f'./datasets/{dataset}/test3.json'
        with open(data_file, encoding='utf8') as i_f:
            self.data1 = json.load(i_f)
        self.num_class = len(self.data1)
        self.adjMs = []
        self.edge2types = []
        self.kn_embs_g = []
        self.stu_lists_g = []
        self.class_ids_g = []
        self.exer_lists_g = []
        self.exer_tests_g = []
        self.labels_g = []
        self.stu_lists_i = []
        self.exer_lists_i = []
        self.labels_i = []
        self.kn_embs_i = []
        self.get_data()

    def get_data(self):
        if self.mode == 'train':
            self.reset()
            while not self.is_end():
                adjM, edge2type, kn_emb_g, stu_list_g, class_id_g, exer_list_g, labels_g, stu_list_i, exer_list_i, labels_i, kn_emb_i = self.next_batch()
                self.adjMs.append(adjM)
                self.edge2types.append(edge2type)
                self.kn_embs_g.append(kn_emb_g)
                self.stu_lists_g.append(stu_list_g)
                self.class_ids_g.append(class_id_g)
                self.exer_lists_g.append(exer_list_g)
                self.labels_g.append(labels_g)
                self.stu_lists_i.append(stu_list_i)
                self.exer_lists_i.append(exer_list_i)
                self.labels_i.append(labels_i)
                self.kn_embs_i.append(kn_emb_i)
        elif self.mode == 'test':
            self.reset()
            while not self.is_end():
                adjM, edge2type, kn_emb_g, stu_list_g, class_id_g, exer_list_g, exer_test_g, labels_g, stu_list_i, exer_list_i, labels_i, kn_emb_i = self.next_batch()
                self.adjMs.append(adjM)
                self.edge2types.append(edge2type)
                self.kn_embs_g.append(kn_emb_g)
                self.stu_lists_g.append(stu_list_g)
                self.class_ids_g.append(class_id_g)
                self.exer_lists_g.append(exer_list_g)
                self.exer_tests_g.append(exer_test_g)
                self.labels_g.append(labels_g)
                self.stu_lists_i.append(stu_list_i)
                self.exer_lists_i.append(exer_list_i)
                self.labels_i.append(labels_i)
                self.kn_embs_i.append(kn_emb_i)

    def next_batch(self):
        if self.mode == 'train':
            if self.is_end():
                return None, None, None, None, None, None, None
        else:
            if self.is_end():
                return None, None, None, None, None, None, None, None
        log = self.data1[self.ptr]
        self.ptr = self.ptr + 1
        class_id_g = log['student_class_id'] 
        stu_list_g = log['stu_list']         
        exer_list_g = log['exer_list']       
        labels_g = log['score_list'] 
        stu_exer_true = log['s_e_t']  
        stu_exer_false = log['s_e_f'] 
        kn_emb = log['skill_list']    
        knowledge_emb = log['kn_embs']
        edge_t, edge_f = torch.LongTensor(stu_exer_true), torch.LongTensor(stu_exer_false)
        kn_emb, stu_list_g = torch.tensor(kn_emb), torch.tensor(stu_list_g)
        knowledge_emb = torch.tensor(knowledge_emb)
        class_id_g, exer_list_g, labels_g = torch.tensor(class_id_g), torch.tensor(exer_list_g), torch.tensor(labels_g)
        
        if self.mode == 'test':
            exer_test_g = log['exer_test']
            exer_test_g = torch.tensor(exer_test_g)
            stu_list_i = log['students_test']
            exer_list_i = log['questions_test']
            labels_i = log['labels_test']
            kn_emb_i = log['kn_emb_test_i']
        else:
            stu_list_i = log['students_train']
            exer_list_i = log['questions_train']
            labels_i = log['labels_train']
            kn_emb_i = log['kn_emb_train_i']
        stu_list_i, exer_list_i, labels_i, kn_emb_i = torch.tensor(stu_list_i), \
            torch.tensor(exer_list_i), torch.tensor(labels_i), torch.tensor(kn_emb_i)

        edge2type = {}

        if len(edge_t) != 0:
            edge_t[0, :] += 1
            edge_t[1, :] += 1 + stu_list_g.shape[0]
            for i in range(edge_t.shape[1]):
                edge2type[(edge_t[0, i].item(), edge_t[1, i].item())] = 0
                edge2type[(edge_t[1, i].item(), edge_t[0, i].item())] = 3
        if len(edge_f) != 0:
            edge_f[0, :] += 1
            edge_f[1, :] += 1 + stu_list_g.shape[0]
            for i in range(edge_f.shape[1]):
                edge2type[(edge_f[0, i].item(), edge_f[1, i].item())] = 1
                edge2type[(edge_f[1, i].item(), edge_f[0, i].item())] = 4
        edge_all = torch.cat((edge_t, edge_f), 1)

        for i in range(stu_list_g.shape[0]):
            new_column = torch.tensor([[0], [i + 1]])
            edge_all = torch.cat((edge_all, new_column), 1)
            edge2type[(0, i + 1)] = 2
            edge2type[(i + 1, 0)] = 5
        for i in range(exer_list_g.shape[0] + stu_list_g.shape[0] + kn_emb.shape[1] + 1):
            edge2type[(i, i)] = 6
        
        exer_indices, kn_indices = torch.nonzero(kn_emb, as_tuple=True)
        edges_kn_exer = torch.stack((exer_indices + 1 + stu_list_g.shape[0], kn_indices + 1 + stu_list_g.shape[0] + exer_list_g.shape[0]), dim=0)
        for i in range(edges_kn_exer.shape[1]):
            edge2type[(edges_kn_exer[0, i].item(), edges_kn_exer[1, i].item())] = 7
            edge2type[(edges_kn_exer[1, i].item(), edges_kn_exer[0, i].item())] = 8      
        
        edge_all = torch.cat((edge_all, edges_kn_exer), 1)

        knowledge_emb = knowledge_emb / torch.norm(knowledge_emb, dim=1, keepdim=True)

        similarity = torch.mm(knowledge_emb, knowledge_emb.t())
        
        kn_sim_indices = torch.nonzero(similarity > self.k , as_tuple=True)
        edges_kn_kn = torch.stack((kn_sim_indices[0] + 1 + stu_list_g.shape[0] + exer_list_g.shape[0], kn_sim_indices[1] + 1 + stu_list_g.shape[0] + exer_list_g.shape[0]), dim=0)
        for i in range(edges_kn_kn.shape[1]):
            edge2type[(edges_kn_kn[0, i].item(), edges_kn_kn[1, i].item())] = 9
            edge2type[(edges_kn_kn[1, i].item(), edges_kn_kn[0, i].item())] = 9
        edge_all = torch.cat((edge_all, edges_kn_kn), 1)

        row = edge_all[0].numpy() 
        col = edge_all[1].numpy() 

        data = np.ones(row.shape[0])

        num_nodes = stu_list_g.shape[0] + exer_list_g.shape[0] + kn_emb.shape[1] + 1

        adjM = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

        if self.mode == 'train':
            return adjM, edge2type, kn_emb, stu_list_g, class_id_g, exer_list_g, labels_g, stu_list_i, exer_list_i, labels_i, kn_emb_i
        else:
            return adjM, edge2type, kn_emb, stu_list_g, class_id_g, exer_list_g, exer_test_g, labels_g, stu_list_i, exer_list_i, labels_i, kn_emb_i


    def is_end(self):
        if self.ptr + 1 > len(self.data1):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0