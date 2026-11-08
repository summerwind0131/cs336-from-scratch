import torch
import numpy as np
import torch.nn as nn
def data_loading(data_set,batch_size,context_length,device=None):
    starts=np.random.randint(
        0,len(data_set)-context_length,batch_size
    )

    inputs=np.stack([
        data_set[i:i+context_length]
        for i in starts]
    )
    targets=np.stack([
        data_set[i+1:i+context_length+1]
        for i in starts]
    )

    return (
        torch.tensor(inputs,dtype=torch.long,device=device),
        torch.tensor(targets,dtype=torch.long,device=device),
    )

def save_checkpoint(model:nn.Module,
                    optimizer:torch.optim.Optimizer,
                    iteration,out):
    torch.save(
        {
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "iteration":iteration,
        },
        out,)


def load_checkpoint(src,model:nn.Module,
                    optimizer:torch.optim.Optimizer):
    checkpoint=torch.load(
        src,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]


