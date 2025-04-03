import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
import numpy as np
import scipy.io as sio
import libs.countsub as cs





class Grammardataset(InMemoryDataset):
    def __init__(self, root,grammar,transform = None,pre_transform = None):
        super(Grammardataset,self).__init__(root,transform,pre_transform)

        self.process2(grammar)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only = False)
    
    @property
    def raw_file_names(self):
        return ['savetest.dat']    
    
    @property
    def processed_file_names(self):
        return 'data.pt'



    def process2(self,grammar):
        data_list = []
        val_word = {}
        seq = torch.load(self.raw_paths[0])
        max_l = 0
        max_val = -100
        for dic in seq:
            if dic['len'] >max_l:
                max_l = dic['len']
            if dic['value']> max_val:
                max_val = dic['value']
        for dic in seq:
            w = str(dic['word'])
            if w not in val_word:
                val_word[w] = (dic['value'].item())
                value =  dic['value']
                prob = dic['prob'].unsqueeze(0)
                var = dic['var']
                mask = torch.zeros((max_l,1))
                if dic['len']<max_l:
                    pad = torch.zeros((max_l - dic['len'],1),dtype=torch.int32) + grammar.dict_wtv[grammar.padding]
                    state = torch.cat([dic['state'],pad]).T
                    mask[dic['len']:,0] = torch.ones(max_l - dic['len'])*float('-inf')
                else:
                    state = dic['state'].T
                if not torch.isnan(value)  and not torch.isinf(value):
                    data_list.append(Data(value = value,var = var,state = state,prob = prob,mask = mask.T))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GraphCountnodeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCountnodeDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only = False)

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):


    
        A = [torch.tensor([[0,1,1,1,1,1,1],
                          [1,0,1,0,0,0,0],
                          [1,1,0,1,0,0,0],
                          [1,0,1,0,1,0,0],
                          [1,0,0,1,0,1,0],
                          [1,0,0,0,1,0,1],
                          [1,0,0,0,0,1,0]])*1.,
            torch.tensor([[0,1,1,1,1,1],
                          [1,0,1,0,0,0],
                          [1,1,0,1,0,1],
                          [1,0,1,0,1,0],
                          [1,0,0,1,0,1],
                          [1,0,1,0,1,0]])*1.,
            torch.tensor([[0,1,1,1,1],
                          [1,0,1,0,0],
                          [1,1,0,1,0],
                          [1,0,1,0,1],
                          [1,0,0,1,0]])*1.]
        

        nmax = 0
        for i in range(len(A)):
            if A[i].shape[0]> nmax:
                nmax = A[i].shape[0]
        data_list = []
        for i in range(len(A)):
            adj = torch.zeros((1,nmax,nmax))
            I = torch.zeros((1,nmax,nmax))
            J = torch.zeros((1,nmax,nmax))
            ad = A[i]
            n = ad.shape[0]
            adj[:,:ad.shape[0],:ad.shape[1]] = ad
            I[:,:ad.shape[0],:ad.shape[1]] = torch.eye(ad.shape[0])
            J[:,:ad.shape[0],:ad.shape[1]] = torch.ones(ad.shape)
            expy = torch.zeros((1,10,nmax,nmax))
            J = J-I
            for j in range(5):
                expy[:,j,:ad.shape[0],:ad.shape[1]] = cs.path(ad,j+2)
                expy[:,4+j,:ad.shape[0],:ad.shape[1]] = cs.cycle(ad,j+3)
           

            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)          
            data_list.append(Data(edge_index=edge_index, A=adj,I=I,J=J, y=expy,n_node  = n))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        

