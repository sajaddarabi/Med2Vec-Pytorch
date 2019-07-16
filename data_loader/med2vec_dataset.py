#################################################################
# Code written by Sajad Darabi (sajad.darabi@cs.ucla.edu)
# For bug report, please contact author using the email address
#################################################################

import torch
import torch.utils.data as data
import os
import pickle
import numpy as np

class Med2VecDataset(data.Dataset):

    def __init__(self, root, num_codes, train=True, med=False,  diag=False,proc=False, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.proc = proc
        self.med = med
        self.diag = diag
        self.num_codes = num_codes
        if download:
            raise ValueError('cannot download')

        self.train_data = pickle.load(open(root, 'rb'))
        self.test_data = []

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def convert_to_med2vec_data(self, data):
        data = []
        for k, vv in self.train_data.items():
            for v in vv:
                di = {k:v[k] for k in ['diagnoses', 'procedures', 'medications', 'demographics', 'cptproc']}
                data.append(di)
            data.append(-1)
        return data

    def __getitem__(self, index):
        x, ivec, jvec, d = self.preprocess1(self.train_data[index])
        return x, ivec, jvec, d

    def preprocess(self, seq):
        """ create one hot vector of idx in seq, with length self.num_codes

            Args:
                seq: list of ideces where code should be 1

            Returns:
                x: one hot vector
                ivec: vector for learning code representation
                jvec: vector for learning code representation
        """
        x = torch.zeros((self.num_codes, ), dtype=torch.long)

        ivec = []
        jvec = []
        d = []
        if seq == [-1]:
            return x, torch.LongTensor(ivec), torch.LongTensor(jvec), d

        x[seq] = 1
        for i in seq:
            for j in seq:
                if i == j:
                    continue
                ivec.append(i)
                jvec.append(j)
        return x, torch.LongTensor(ivec), torch.LongTensor(jvec), d

    def preprocess(self, seq):
        """ create one hot vector of idx in seq, with length self.num_codes

            Args:
                seq: list of ideces where code should be 1

            Returns:
                x: one hot vector
                ivec: vector for learning code representation
                jvec: vector for learning code representation
        """
        x = torch.zeros((self.num_codes, ), dtype=torch.long)

        ivec = []
        jvec = []
        d = torch.zeros((self.demographics_len, ))
        if seq == -1:
            return x, torch.LongTensor(ivec), torch.LongTensor(jvec), d

        codes = seq['diagnoses'] + seq['procedures'] + seq['medications'] + seq['cptproc']
        x[codes] = 1
        d = torch.Tensor(seq['demographics'])
        for i in codes:
            for j in codes:
                if i == j:
                    continue
                ivec.append(i)
                jvec.append(j)
        return x, torch.LongTensor(ivec), torch.LongTensor(jvec), d

def collate_fn(data):
    """ Creates mini-batch from x, ivec, jvec tensors

    We should build custom collate_fn, as the ivec, and jvec have varying lengths. These should be appended
    in row form

    Args:
        data: list of tuples contianing (x, ivec, jvec)

    Returns:
        x: one hot encoded vectors stacked vertically
        ivec: long vector
        jvec: long vector
    """

    x, ivec, jvec, d = zip(*data)
    x = torch.stack(x, dim=0)
    mask = torch.sum(x, dim=1) > 0
    mask = mask[:, None]
    ivec = torch.cat(ivec, dim=0)
    jvec = torch.cat(jvec, dim=0)
    d = torch.stack(d, dim=0)

    return x, ivec, jvec, mask, d

def get_loader(root, num_codes, train=True, transform=None, target_transform=None, download=False, batch_size=1000):
    """ returns torch.utils.data.DataLoader for Med2Vec dataset """
    med2vec = Med2VecDataset(root, num_codes, train, transform, target_transform, download)
    data_loader = torch.utils.data.DataLoader(dataset=med2vec, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader
