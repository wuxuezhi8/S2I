from collections import defaultdict as ddict
from torch.utils.data import Dataset
import numpy as np
import torch

def process(dataset, num_rel):
    """
    pre-process dataset
    :param dataset: a dictionary containing 'train', 'valid' and 'test' data.
    :param num_rel: relation number
    :return:
    """
    # sr2o = ddict(set)
    # for subj, rel, obj in dataset['train']:
    #     sr2o[(subj, rel)].add(obj)
    #     sr2o[(obj, rel + num_rel)].add(subj)
    # sr2o_train = {k: list(v) for k, v in sr2o.items()}
    # for split in ['valid', 'test']:
    #     for subj, rel, obj in dataset[split]:
    #         sr2o[(subj, rel)].add(obj)
    #         sr2o[(obj, rel + num_rel)].add(subj)
    # sr2o_all = {k: list(v) for k, v in sr2o.items()}
    # triplets = ddict(list)

    # for (subj, rel), obj in sr2o_train.items():
    #     triplets['train'].append({'triple': (subj, rel, -1), 'label': sr2o_train[(subj, rel)]})
    # for split in ['valid', 'test']:
    #     for subj, rel, obj in dataset[split]:
    #         triplets[f"{split}_tail"].append({'triple': (subj, rel, obj), 'label': sr2o_all[(subj, rel)]})
    #         triplets[f"{split}_head"].append(
    #             {'triple': (obj, rel + num_rel, subj), 'label': sr2o_all[(obj, rel + num_rel)]})
    # triplets = dict(triplets)

    # 在每个三元组后面添加其相反三元组 
    # data = ddict(list)
    # for split in ['train', 'test', 'valid']:
    #   for subj, rel, obj in dataset[split]:
    #       data[split].append((subj, rel, obj))
    #       data[split].append((obj, rel + num_rel, subj))   #添加反向
    # data = dict(data)

    # 在所有三元组后面依次添加相反三元组
    data = ddict(list)
    for split in ['train', 'test', 'valid']:
      for subj, rel, obj in dataset[split]:
          data[split].append((subj, rel, obj))
          # data[split].append((obj, rel + num_rel, subj))   #添加反向
    data = dict(data)
    data_all = ddict(list)
    for split in ['train', 'test', 'valid']:
      for subj, rel, obj in data[split]:
          data_all[split].append((subj, rel, obj))
    for split in ['train', 'test', 'valid']:
      for subj, rel, obj in data[split]:
          data_all[split].append((obj, rel + num_rel, subj))
    data = dict(data_all)

    # 头尾对应的关系
    so2r = ddict(set)
    for sub, rel, obj in data['train']:
        so2r[(sub,obj)].add(rel)
    so2rResult = {k: list(v) for k, v in so2r.items()}
    for split in ['test', 'valid']:
      for sub, rel, obj in data[split]: 
          so2r[(sub, obj)].add(rel)
    so2r_all = {k: list(v) for k, v in so2r.items()}

    triples = ddict(list)
    # triples格式：{'train':[{'triple':(3, 4, -1), 'label':[2], 'sub_samp': 1},...]}
    # for (sub, obj), rel in so2rResult.items():
    #     triples['train'].append({'triple': (sub, -1, obj), 'label': so2rResult[(sub, obj)]})
    # for split in ['test', 'valid']:
    #     for sub, rel, obj in data[split]:
    #         triples[split].append(
    #             {'triple': (sub, rel, obj), 'label': so2r_all[(sub, obj)]})            
    for sub, rel, obj in data['train']:
        triples['train'].append({'triple': (sub, -1, obj), 'label': so2rResult[(sub, obj)]})
    for split in ['test', 'valid']:
        for sub, rel, obj in data[split]:
            triples[split].append(
                {'triple': (sub, rel, obj), 'label': so2r_all[(sub, obj)]})
            
    triples = dict(triples)

    s2o = ddict(set)
    for sub, rel, obj in data['train']:
        s2o[sub].add(obj)
        s2o[obj].add(sub)
    s2o = {k: list(v) for k, v in s2o.items()}
    
    return triples, s2o


class TrainDataset(Dataset):
    def __init__(self, triplets, num_rel, params):
        super(TrainDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.label_smooth = params.lbl_smooth
        self.num_rel = num_rel

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label = torch.tensor(ele['triple'], dtype=torch.long),np.int32(ele['label']) 
        label = self.get_label(label)
        if self.label_smooth != 0.0:
            label = (1.0 - self.label_smooth) * label + self.label_smooth * (1.0 / (self.num_rel*2))
            # label = (1.0 - self.label_smooth) * label + (1.0 / (self.num_rel*2))
        return triple, label

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        return triple, trp_label
    
    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_rel*2], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, triplets, num_rel, params):
        super(TestDataset, self).__init__()
        self.triplets = triplets
        self.num_rel = num_rel

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple,label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        label = self.get_label(label)
        return triple, label

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        return triple, trp_label
        
    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_rel*2], dtype=np.float32)
        y[label] = 1.0
        return torch.tensor(y, dtype=torch.float32)