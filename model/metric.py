#################################################################
# Code written by Sajad Darabi (sajad.darabi@cs.ucla.edu)
# For bug report, please contact author using the email address
#################################################################

import torch

def recall_k(output, target, mask, k=10, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
        tm = mi[:-i - 1]
        im = mi[i + 1:]

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        #ii = ii.long()
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)
        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / torch.sum(target, dim=1))
        if r != r:
            r = 0
    return r
