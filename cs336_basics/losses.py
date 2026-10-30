import torch
import math

def cross_entropy_loss(logits:torch.Tensor,targets):
    """
    logits:batch seq_len,vocab_size
    targets:batch seq_len
    """
    
    max,_=torch.max(logits,dim=-1,keepdim=True)
    logits=logits-max

    target_logits=logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    logits_sum=torch.log(torch.exp(logits).sum(dim=-1))

    loss=logits_sum-target_logits

    return loss.mean()

