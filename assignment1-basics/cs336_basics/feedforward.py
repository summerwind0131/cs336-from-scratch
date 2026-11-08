import torch
import torch.nn as nn

from cs336_basics.linear import Linear
class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ffn,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        self.d_ffn=d_ffn

        self.fc1=Linear(d_model,d_ffn,device=device,dtype=dtype)
        self.fc2=Linear(d_ffn,d_model,device=device,dtype=dtype)
        self.fc3=Linear(d_model,d_ffn,device=device,dtype=dtype)
        self.sigmoid=torch.sigmoid

    def forward(self,x):
        out=self.fc1(x)
        out=out * self.sigmoid(out)
        out=out * self.fc3(x)

        return self.fc2(out)

    