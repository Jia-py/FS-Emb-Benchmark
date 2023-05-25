import torch
import torch.nn as nn
import numpy as np
from ..module import ctr

'''
For embedding dimension search, we define the search space as [32,16,8,4]
'''

class AutoDim(nn.Module):
    def __init__(self, fields, config, train_data, device,reduction='mean',
                 share_dense_embedding=True, dense_emb_bias=False, dense_emb_norm=True,
                 with_dense_kernel=False) -> None:
        super().__init__()
        self.embed_dim = 32
        self.frating = train_data.frating
        self.field2types = {f: train_data.field2type[f] for f in fields if f != train_data.frating}
        self.fields = list(self.field2types.keys())
        self.reduction = reduction
        self.share_dense_embedding = share_dense_embedding
        self.dense_emb_bias = dense_emb_bias
        self.dense_emb_norm = dense_emb_norm
        self.with_dense_kernel = with_dense_kernel
        self.num_features = len(self.field2types)
        self.gate = nn.ParameterDict({field : nn.Parameter(torch.Tensor(np.ones([1,4])/4).to(device)) for field in self.field2types})
        self.tau = 1.0

        self.epochs = config['train']['epochs']
        self.mode = 'train'
        self.device = device
        self.embedding = ctr.Embeddings(fields, self.embed_dim, train_data)
        self.in_dim = [4,8,16,32]
        self.linear = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.randn([len(self.fields),input_dim,self.embed_dim]))) for input_dim in self.in_dim])
        self.BN = nn.ModuleList([nn.BatchNorm1d(self.embed_dim,affine=False) for input_dim in self.in_dim])


    def forward(self, batch):
        if self.mode == 'train':
            emb = self.embedding(batch) # b,f,e
            b,f,e = emb.shape
            emb = [self.BN[i](torch.einsum('bfi,fie->bfe',emb[:,:,:self.in_dim[i]],self.linear[i]).transpose(1,2)).transpose(1,2) for i in range(len(self.in_dim))]
            emb = torch.cat([e.unsqueeze(2) for e in emb],2)
            if self.tau > 0.01:
                self.tau -= 0.00005
            gate_ = torch.ones([1, f, 4]).to(self.device)
            for i in range(f):
                field=self.fields[i]
                gate_[:,i,:] = torch.nn.functional.gumbel_softmax(self.gate[field], tau=self.tau, hard=False, dim=-1)
            x_=torch.mul(emb,gate_.unsqueeze(-1)).mean(2)
            return x_
        else:
            emb = []
            for f, d in batch.items():
                if f in self.embedding:
                    emb.append(self.linear[f](self.embedding[f]({f:d})).squeeze(1))
            #x_=self.BN(torch.cat([e.unsqueeze(-1) for e in emb],-1)).transpose(1,2)
            x_=torch.cat([e.unsqueeze(1) for e in emb],1)
            return x_
        
    def retrain_prepare_after_ini(self, train_data,decision):
        self.mode = 'retrain'
        print(decision)
        setattr(self,'embedding',nn.ModuleDict({field: ctr.Embeddings([field],self.in_dim[decision[field]],train_data) for field in decision}))
        setattr(self,'linear', nn.ModuleDict({field: nn.Linear(self.in_dim[decision[field]],self.embed_dim) for field in decision}))
        setattr(self,'BN',nn.BatchNorm1d(self.embed_dim,affine=False))
        del self.gate 
        return None

    def retrain_prepare_before_ini(self, train_data):
        self.mode = 'retrain'
        decision = {field: self.gate[field].argmax() for field in self.gate}
        return decision
    

    def set_optimizer(self, model):
        optimizer_DRS = torch.optim.Adam([params for name,params in model.named_parameters() if 'embedding.gate' not in name], lr = model.config['train']['learning_rate'])
        optimizer_Controller = torch.optim.Adam([params for name,params in model.named_parameters() if 'embedding.gate' in name], lr = model.config['train']['learning_rate'])
        model.Controller_optimizer = optimizer_Controller
        return [{'optimizer': optimizer_DRS}]