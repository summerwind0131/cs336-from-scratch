import torch


def softmax(x:torch.Tensor,d:int)->torch.Tensor:
    max_value=torch.max(x,dim=d,keepdim=True).values

    out=torch.exp(x-max_value)
    sum_value=out.sum(dim=d,keepdim=True)

    out=out/sum_value

    return out


    