import torch
import torch.nn as nn
import numpy as np

class AutoField(nn.Module):

    def __init__(self, field2token2idx, config, field2type, device) -> None:
        super().__init__()
        self.field2token2idx = field2token2idx
        self.gate = {field : torch.Tensor(np.ones([1,2])*0.5).to(device) for field in field2token2idx}
        self.gate = {field : torch.nn.Parameter(self.gate[field]) for field in self.gate}
        self.gate = torch.nn.ParameterDict(self.gate)
        self.tau = 1.0

        self.epochs = config['train']['epochs']
        self.field2type = field2type

        self.mode = 'train'
        self.device = device

    def forward(self, x, current_epoch, fields, batch_data):
        fields = list(fields)
        b,f,e = x.shape
        # token_fields = [field for field in fields if self.field2type[field]=='token']
        if self.mode == 'retrain':
            return x
        elif self.mode == 'train':
            if self.tau > 0.01:
                self.tau -= 0.00005
        # gate = torch.concat([self.gate[field] for field in token_fields], dim=0) # token_num, 2
        # gate = torch.nn.functional.gumbel_softmax(gate, tau=self.tau, hard=False, dim=-1)[:,-1] # token_num, 1
        gate_ = torch.ones([1, f, 1]).to(self.device)
        for i in range(f):
            field = fields[i]
            if self.field2type[field]!='token' and self.field2type[field]!='token_seq':
                continue
            gate_[:,i,:] = torch.nn.functional.gumbel_softmax(self.gate[field], tau=self.tau, hard=False, dim=-1)[:,-1].reshape(1,1,1)
        x_ = torch.mul(x, gate_)
        return x_
    
    def retrain_prepare_before_ini(self, k, user_id, item_id):
        self.mode = 'retrain'
        fields = [field for field in self.gate]
        gate = torch.concat([self.gate[field] for field in self.gate], dim=0)[:,-1] # token_num, 2
        indices = torch.argsort(gate, descending=True)
        # use_fields = [user_id, item_id]
        use_fields = []
        tmp_num = 0
        for i in indices:
            if fields[i] in [user_id, item_id]:
                continue
            elif tmp_num < k:
                use_fields.append(fields[i])
                tmp_num += 1
            else:
                break
        print('use_fields: ', use_fields)
        return use_fields

    def retrain_prepare_after_ini(self):
        self.mode = 'retrain'
        return None

    def set_optimizer(self, model):
        optimizer_DRS = torch.optim.Adam([params for name,params in model.named_parameters() if 'feature_selection_layer' not in name], lr = model.config['train']['learning_rate'])
        optimizer_Controller = torch.optim.Adam([params for name,params in model.named_parameters() if 'feature_selection_layer' in name], lr = model.config['train']['learning_rate'])
        model.Controller_optimizer = optimizer_Controller
        return [{'optimizer': optimizer_DRS}]