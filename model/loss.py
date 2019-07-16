import torch
import torch.nn as nn
import torch.nn.functional as F

def med2vec_loss(inputs, mask, probits, bce_loss, emb_w, ivec, jvec, window=1, eps=1.0e-8):
    """ returns the med2vec loss
    """
    def visit_loss(x, mask, probits, window=1):
        loss = 0
        for i in range(0, window):
            if loss != loss:
                import pdb; pdb.set_trace()
            if (i == 0):
                maski = mask[i + 1:] * mask[:-i - 1]
            else:
                maski = mask[i + 1:] * mask[i:-i] * mask[:-i - 1]
            backward_preds = probits[i+1:] * maski
            forward_preds = probits[:-i-1] * maski
            loss += bce_loss(forward_preds, x[i+1:].float()) + bce_loss(backward_preds, x[:-i-1].float())
        return loss

    def code_loss(emb_w, ivec, jvec, eps=1.e-6):
        norm = torch.sum(torch.exp(torch.mm(emb_w.t(), emb_w)), dim=1)

        cost = -torch.log((torch.exp(torch.sum(emb_w[:, ivec].t() * emb_w[:, jvec].t(), dim=1)) / norm[ivec]) + eps)
        cost = torch.mean(cost)
        return cost

    vl = visit_loss(inputs, mask, probits, window=window)
    cl = code_loss(emb_w, ivec, jvec, eps=1.e-6)
    return {'visit_loss': vl, 'code_loss': cl}
