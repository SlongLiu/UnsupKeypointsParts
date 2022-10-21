# ref: https://gist.github.com/nasimrahaman/a5fb23f096d7b0c3880e1622938d0901

import torch
import torch.nn as nn


# def log_sum_exp(x):
#     # See implementation detail in
#     # http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
#     # b is a shift factor. see link.
#     # x.size() = [N, C]:
#     b, _ = torch.max(x, 1)
#     y = b + torch.log(torch.exp(x - b.expand_as(x)).sum(1))
#     # y.size() = [N, 1]. Squeeze to [N] and return
#     return y.squeeze(1)

def log_sum_exp(x):
    b, _ = torch.max(x, 1)
    # b.size() = [N, ], unsqueeze() required
    y = b + torch.log(torch.exp(x - b.unsqueeze(dim=1).expand_as(x)).sum(1))
    # y.size() = [N, ], no need to squeeze()
    return y


def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .cuda(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(one_hot_mask)


def cross_entropy_with_weights(pred, target, weights=None):
    assert pred.dim() == 2
    assert not target.requires_grad
    assert target.dim() == 1

    logits = torch.softmax(pred, 1)
    loss = log_sum_exp(logits) - class_select(logits, target)
    if weights is not None:
        # loss.size() = [N]. Assert weights has the same shape
        assert list(loss.size()) == list(weights.size()), f"{list(loss.size())} != {list(weights.size())}"
        # Weight the loss
        loss = loss * weights
    return loss