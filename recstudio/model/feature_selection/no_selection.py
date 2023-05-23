import torch
import torch.nn as nn
import numpy as np

class No_Selection(nn.Module):

    def __init__(self, field2token2idx, config, field2type, device) -> None:
        super().__init__()

    def forward(self, x, current_epoch, fields, batch_data):
        return x