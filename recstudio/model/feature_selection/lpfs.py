import torch
import torch.nn as nn
import numpy as np

class LPFS(nn.Module):

    def __init__(self, field2token2idx, config, field2type, device) -> None:
        super().__init__()
        self.field2token2idx = field2token2idx
        self.feature_num = len(field2type)-1
        self.x = nn.Parameter(torch.ones(self.feature_num, 1).to(device))

        self.epochs = config['train']['epochs']
        self.field2type = field2type
        self.epsilon_update_frequency = config['fs']['epsilon_update_frequency']

        self.device = device
        self.epsilon = 0.1

    def forward(self, x, current_epoch, fields, batch_data):
        fields = list(fields)
        b,f,e = x.shape
        token_fields = [field for field in fields if self.field2type[field]=='token']
        if current_epoch> 0 and current_epoch % self.epsilon_update_frequency == 0:
            self.epsilon = self.epsilon * 0.9978
        g = self.lpfs_pp(self.x, self.epsilon).reshape(1,self.feature_num, 1)
        x_ = torch.zeros_like(x)
        x_ = x * g
        return x_

    def set_optimizer(self, model):
        optimizer_DRS = torch.optim.Adagrad([params for name,params in model.named_parameters() if 'feature_selection_layer' not in name], lr = model.config['train']['learning_rate'])
        optimizer_Controller = torch.optim.SGD([params for name,params in model.named_parameters() if 'feature_selection_layer' in name], lr = model.config['train']['learning_rate'], momentum=0.9)
        return [{'optimizer': optimizer_DRS}, {'optimizer': optimizer_Controller}]

    def lpfs_pp(self, x, epsilon=0.01, alpha=10, tau=2, init_val=1.0):
        g1 = x*x/(x*x+epsilon)
        g2 = alpha * epsilon ** (1.0/tau)*torch.atan(x)
        g = torch.where(x>0, g2+g1, g2-g1)/init_val
        return g