from collections.abc import Callable,Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3
                 ,betas=(0.9,0.999),
                 eps=1e-8,weight_decay=0.01):
        if lr<0:
            raise ValueError(f"Invaliad learning rate:{lr}")
        defaults={"lr":lr,"betas":betas,"eps":eps,"weight_decay":weight_decay}
        super().__init__(params,defaults)

    @torch.no_grad()
    def step(self,closure:Optional[Callable]=None):
        loss=None if closure is None else closure()
        for group in self.param_groups:
            lr=group["lr"]
            beta1,beta2=group["betas"]
            eps=group["eps"]
            weight_decay=group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state=self.state[p]
                grad=p.grad.data

                if not state:
                    state["t"]=0
                    state["m"]=torch.zeros_like(p)
                    state["v"]=torch.zeros_like(p)

                state["t"]+=1
                t=state["t"]
                m=state["m"]
                v=state["v"]

                alpha_t=lr*math.sqrt(1-beta2**t)/(1-beta1**t)
                m.mul_(beta1).add_(grad,alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad,grad,value=1-beta2)

                p.mul_(1-lr*weight_decay)
                p.addcdiv_(m,
                           v.sqrt().add_(eps),
                           value=-alpha_t)
        return loss

    
def lr_cosine_schedule(t,alpha_max,alpha_min,Tw,Tc):
    if t<Tw:
        return t*alpha_max/Tw
    elif Tw<=t<=Tc:
        return alpha_min+0.5*(1+math.cos((t-Tw)*math.pi/(Tc-Tw)))*(alpha_max-alpha_min)
    elif t>Tc:
        return alpha_min

    
                