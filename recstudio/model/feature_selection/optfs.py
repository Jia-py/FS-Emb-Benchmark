import torch
import torch.nn as nn

class optFS(nn.Module):

    def __init__(self, field2token2idx, config, field2type, device) -> None:
        super().__init__()
        self.field2token2idx = field2token2idx
        self.gate = {field : torch.Tensor(len(field2token2idx[field]), 1).to(device) for field in field2token2idx}
        for field in self.gate:
            torch.nn.init.xavier_uniform_(self.gate[field].data)

        self.raw_gate = {field : self.gate[field].clone().detach().to(device) for field in self.gate}
        self.raw_gc = torch.concat([self.raw_gate[field] for field in self.gate], dim=0) # feature_val_num,1
        
        self.g = {field : torch.ones_like(self.gate[field]).to(device) for field in self.gate} # final g vector
        self.gate = {field : torch.nn.Parameter(self.gate[field]).to(device) for field in self.gate} # gc vector in paper
        self.gate = torch.nn.ParameterDict(self.gate)

        self.epochs = config['train']['epochs']
        self.field2type = field2type

        self.mode = 'train'
        self.device = device

    def forward(self, x, current_epoch, fields, batch_data):
        fields = list(fields)
        b,f,e = x.shape
        token_fields = [field for field in fields if self.field2type[field]=='token']
        gc = torch.concat([self.gate[field] for field in token_fields], dim=0) # feature_val_num,1
        t = 200 * (current_epoch / self.epochs)
        if self.mode == 'train':
            self.g_tmp = torch.sigmoid(gc * t) / torch.sigmoid(self.raw_gc)
        elif self.mode == 'retrain':
            self.g_tmp = torch.concat([self.g[field] for field in token_fields], dim=0)
        
        # 把g_tmp分段赋值给self.g
        for field in token_fields:
            self.g[field] = self.g_tmp[:len(self.gate[field])]
            self.g_tmp = self.g_tmp[len(self.gate[field]):]

        x_ = torch.zeros_like(x)

        for i in range(b):
            for j in range(f):
                field = fields[j]
                if self.field2type[field]!='token':
                    break
                x_[i,j,:] = x[i,j,:] * self.g[field][batch_data[field][i]]

        return x_
    
    def retrain_prepare_before_ini(self, k):
        self.mode = 'retrain'
        # fix self.g
        for field in self.g:
            self.g[field].requires_grad = False

        return None
    
    def retrain_prepare_after_ini(self):
        pass
