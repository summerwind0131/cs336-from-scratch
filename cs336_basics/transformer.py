import torch 
import torch.nn as nn


from cs336_basics.attention import MultiHeadSelfAttention,scaled_dot_product_attention
from cs336_basics.RMSnorm import RMSNorm
from cs336_basics.feedforward import SwiGLU
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,
                 rope:bool,max_seq_len=None,theta=None,
                 device=None,dtype=None):
        super().__init__()
        self.RMSNorm1=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.RMSNorm2=RMSNorm(d_model=d_model,device=device,dtype=dtype)

        self.MultiHeadAttention=MultiHeadSelfAttention(d_model=d_model,
                                                       num_heads=num_heads,
                                                       rope=rope,max_seq_len=max_seq_len,theta=theta,
                                                       device=device,dtype=dtype)
        self.SwiGLU=SwiGLU(d_model=d_model,d_ffn=d_ff,
                           device=device,dtype=dtype)


    def forward(self,x,token_positions=None):

        # MultiHeadAttention
        shortcut=x
        output=shortcut+self.MultiHeadAttention(self.RMSNorm1(x),token_positions)

        # feed forward
        shortcut=output
        output=shortcut+self.SwiGLU(self.RMSNorm2(output))

        return output


class TransformerLM(nn.Module):
    def __init__(self,vocab_size,context_length,num_layers, 
                 d_model,num_heads,d_ff,
                  rope:bool,max_seq_len=None,theta=None,eps=0.00001,
                  device=None,dtype=None):
        super().__init__()

        self.embedding=Embedding(vocab_size,d_model,
                                 device=device,dtype=dtype)
        self.layers=nn.ModuleList([
            TransformerBlock(
                d_model=d_model,num_heads=num_heads,d_ff=d_ff,
                rope=True,max_seq_len=context_length,theta=theta,
                device=device,dtype=dtype
            )for _ in range(num_layers)
        ])

        self.RMSNorm=RMSNorm(d_model,eps=eps,device=device,dtype=dtype)
        self.LinearHead=Linear(d_model,vocab_size,device=device,dtype=dtype)

    def forward(self,x,token_positions=None):
        output=self.embedding(x)
        for layer in self.layers:
            output=layer(output,token_positions)
        output=self.RMSNorm(output)
        output=self.LinearHead(output)

        return output
    