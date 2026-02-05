import torch
import torch.nn.functional as F
import numpy as np

from utils.utils import get_clean_loss_tensor_mask, get_clean_mask

def small_loss_selection(args, outs, y, epoch, loss_all, inds, update_inds):
    out1, out2, out3 = outs[0], outs[1], outs[2]

    loss1 = -torch.sum(F.log_softmax(out1, dim=1) * y, dim=1)
    loss2 = -torch.sum(F.log_softmax(out2, dim=1) * y, dim=1)
    loss3 = -torch.sum(F.log_softmax(out3, dim=1) * y, dim=1)

    losses_total = loss1 + loss2 + loss3
    loss_all[inds] = losses_total.detach().cpu().numpy()

    if epoch >= args.start_mask_epoch:
        remember_rate = 1 - args.label_noise_rate
        if epoch >= args.start_refurb:
            final_mask = get_clean_mask(losses_total, inds, update_inds, remember_rate=remember_rate)
        else:
            final_mask = get_clean_loss_tensor_mask(losses_total, remember_rate=remember_rate)
    else:
        final_mask = torch.ones_like(losses_total)

    final_loss1 = torch.sum(final_mask * loss1) / max(torch.count_nonzero(final_mask).item(), 1)
    final_loss2 = torch.sum(final_mask * loss2) / max(torch.count_nonzero(final_mask).item(), 1)
    final_loss3 = torch.sum(final_mask * loss3) / max(torch.count_nonzero(final_mask).item(), 1)

    return final_loss1, final_loss2, final_loss3, loss_all

