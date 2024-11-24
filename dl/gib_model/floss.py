import torch
from torch import nn

class FLoss(nn.Module):
    def __init__(self, beta = 0.5, log_like = False):
        super(FLoss, self).__init__()
        
        self.beta = beta
        self.log_like = log_like
        
    def forward(self, pred, target):
        eps = 1e-10
        N = pred.size(0)
        TP = (pred * target).view(N, -1).sum()
        H = self.beta * target.view(N, -1).sum() + pred.view(N, -1).sum()
        fmeasure = (1 + self.beta) * TP / (H + eps)
        
        if self.log_like:
            floss = -torch.log(fmeasure)
        else:
            floss = (1 - fmeasure)
        
        return floss