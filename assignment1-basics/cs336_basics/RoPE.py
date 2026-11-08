import torch
import torch.nn as nn
from einops import einsum
from einops import rearrange


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta,d_k,max_seq_len,device=None):
        super().__init__()
        self.theta=theta

        dims=torch.arange(0,d_k,2,dtype=torch.float32,device=device)
        position=torch.arange(0,max_seq_len,dtype=torch.float32,device=device)

        frequency=self.theta ** (-dims/d_k)

        angles=einsum(position,frequency,"position,pair-> position pair")

        cos_angles=torch.cos(angles)
        sin_angles=torch.sin(angles)

        self.register_buffer("cos_angles", cos_angles,persistent=False)
        self.register_buffer("sin_angles",sin_angles,persistent=False)
        
        

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:

        cos_angles=self.cos_angles[token_positions]
        sin_angles=self.sin_angles[token_positions]
        pairs=rearrange(x,"... sqe (pair coord)-> ... sqe pair coord",coord=2)

        a=pairs[...,0]
        b=pairs[...,1]

        
        a_new=a*cos_angles-b*sin_angles
        b_new=a*sin_angles+b*cos_angles

        result=torch.stack([a_new,b_new],dim=-1)
        result=rearrange(result,"... sqe pair coord -> ... sqe (pair coord)")
        return result


        

        
        
        