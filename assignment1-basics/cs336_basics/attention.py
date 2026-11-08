import torch
import torch.nn as nn

from einops import einsum
from einops import rearrange

import math

from cs336_basics.softmax import softmax
from cs336_basics.linear import Linear
from cs336_basics.RoPE import RotaryPositionalEmbedding

def scaled_dot_product_attention(query:torch.Tensor,key:torch.Tensor,
                                 value:torch.Tensor,mask=None):
    dim=query.shape[-1]

    scores=einsum(query,key,"batch ... q_seq d,batch ... k_seq d-> batch ... q_seq k_seq")
    scores=scores/math.sqrt(dim)

    if mask is not None:
        scores = scores.masked_fill(
        ~mask,
        float("-inf"))

    scores=softmax(scores,d=-1)
    attention=einsum(scores,value,"batch ... q_seq seq, batch ... seq dv-> batch ... q_seq dv")

    return attention



class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads,
                 rope:bool,max_seq_len=None,theta=None,
                 device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads

        self.query=Linear(d_model,d_model,device,dtype)
        self.key=Linear(d_model,d_model,device,dtype)
        self.value=Linear(d_model,d_model,device,dtype)

        self.O=Linear(d_model,d_model,device,dtype)

        if rope:
            self.rope=RotaryPositionalEmbedding(theta,d_k=d_model/num_heads,max_seq_len=max_seq_len,
                                            device=device)
            self.Rope=True
        else:
            self.Rope=False


    
        
        

    def forward(self,x:torch.Tensor,token_positions=None):
        query=self.query(x)
        key=self.key(x)
        value=self.value(x)

        num=self.num_heads
        seq_len=x.shape[-2]

        query=rearrange(query,"batch ... (num d_head)->batch  num ... d_head",num=self.num_heads,)
        key=rearrange(key,"batch ... (num d_head)->batch  num ... d_head",num=self.num_heads,)
        value=rearrange(value,"batch ... (num d_head)->batch  num ... d_head",num=self.num_heads,)

        if self.Rope:
            if token_positions is not None:
                token_positions=rearrange(token_positions,"... seq_len -> ... 1 seq_len")
            else:
                token_positions=torch.arange(0,seq_len)
                token_positions=rearrange(token_positions,"... seq_len -> ... 1 seq_len")
            query=self.rope(query,token_positions)
            key=self.rope(key,token_positions)

        casual_mask=torch.ones(
                (seq_len,seq_len),
                dtype=torch.bool,
                device=x.device,
            ).tril()


        attention=scaled_dot_product_attention(query,key,value,mask=casual_mask)
        attention=rearrange(attention,"batch num ... d_head->batch ... (num d_head)")

        result=self.O(attention)

        return result
    
        


        



