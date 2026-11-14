import torch 

def weighted_sum(X,weight):
    return (weight*X).sum(axis=-1)

