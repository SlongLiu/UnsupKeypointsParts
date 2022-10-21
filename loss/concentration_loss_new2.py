import torch

def concentration_loss_new2(pred):
    # get pred mean
    B,C,H,W = pred.shape
    epsilon = 1e-3
    pred = pred + 1e-3
    pred = pred / pred.sum((-1, -2), keepdim=True)

    

