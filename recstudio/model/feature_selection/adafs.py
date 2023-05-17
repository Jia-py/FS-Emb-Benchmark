import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):

    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0):
        super().__init__()
        if dims is None:
            dims = []
        layers = list()
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class AdaFS(nn.Module):

    def __init__(self, field2token2idx, config, field2type, device) -> None:
        super().__init__()
        self.field2token2idx = field2token2idx
        self.feature_num = len(field2type)-1
        self.batchnorm = nn.BatchNorm1d(self.feature_num)
        self.mlp = MLP(self.feature_num * config['model']['embed_dim'], False, [self.feature_num])

        self.epochs = config['train']['epochs']
        self.field2type = field2type

        self.device = device
        self.pretrain_epoch_num = config['fs']['pretrain_epoch_num']

    def forward(self, x, current_epoch, fields, batch_data):
        fields = list(fields)
        b,f,e = x.shape
        token_fields = [field for field in fields if self.field2type[field]=='token']
        if current_epoch <= self.pretrain_epoch_num:
            return x
        else:
            x = self.batchnorm(x)
            weight = self.mlp(x.reshape(b, -1)) # b, f
            x = torch.mul(x, weight.unsqueeze(-1))
            return x

    def set_optimizer(self, model):
        optimizer_DRS = torch.optim.Adam([params for name,params in model.named_parameters() if 'feature_selection_layer' not in name], lr = model.config['train']['learning_rate'])
        optimizer_Controller = torch.optim.Adam([params for name,params in model.named_parameters() if 'feature_selection_layer' in name], lr = model.config['train']['learning_rate'])
        model.Controller_optimizer = optimizer_Controller
        return [{'optimizer': optimizer_DRS}]