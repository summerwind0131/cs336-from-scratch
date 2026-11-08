import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    def __init__(self,d_model,eps:float=1e-5,device=None,dtype=None):
        super().__init__()

        self.d_model=d_model
        self.eps=eps

        self.device=device
        self.dtype=dtype

        value=torch.empty(d_model,device=device,dtype=dtype)
        self.g=nn.Parameter(value)
        nn.init.constant_(self.g,1)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        in_dtype=x.dtype
        x=x.to(torch.float32)

        rms=torch.sqrt(x.pow(2).sum(dim=-1,keepdim=True)/self.d_model+self.eps)
        result=x/rms *self.g

        return result.to(in_dtype)