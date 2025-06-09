import torch
import torch.nn.functional as F
import numpy as np

from utils.utils import get_clean_loss_tensor_mask, get_clean_mask


class Discrimn_Loss(torch.nn.Module):
    def __init__(self, gam1=1.0, eps=0.01):
        super(Discrimn_Loss, self).__init__()
        self.gam1 = gam1
        self.eps = eps

    def compute_discrimn_loss_empirical(self, x):
        """Empirical Discriminative Loss."""
        m, d = x.shape
        I = torch.eye(m).cuda()
        scalar = d / (m * self.eps)
        res = torch.logdet(I + self.gam1 * scalar * x.matmul(x.T)) / 2

        if torch.isinf(x).any() or torch.isnan(x).any() or torch.isinf(res).any() or torch.isnan(res).any():
            res = torch.tensor(float(0), device=x.device)

        return res

    def forward(self, X):
        discrimn_loss_empi = self.compute_discrimn_loss_empirical(X)

        return discrimn_loss_empi

def compute_delay_loss(args, out, y, k_losses, inds, epoch):
    loss = F.cross_entropy(out, y, reduction='none').cuda()

    k_losses[inds, epoch % args.delay_loss_k] = loss.detach().cpu().numpy()
    if epoch >= args.start_delay_loss:
        t = torch.tensor(data=np.sum(k_losses, axis=1) / k_losses.shape[1], dtype=torch.float32, requires_grad=True, device=loss.device)
        loss = 0.01 * loss + t[inds]

    return loss, k_losses

def delay_loss(args, outs, y, epoch, loss_all, inds, update_inds):
    """Compute the sum of loss for each k in k_list."""
    out1, out2, out3 = outs[0], outs[1], outs[2]

    delay_loss1 = -torch.sum(F.log_softmax(out1, dim=1) * y, dim=1)
    delay_loss2 = -torch.sum(F.log_softmax(out2, dim=1) * y, dim=1)
    delay_loss3 = -torch.sum(F.log_softmax(out3, dim=1) * y, dim=1)
    # delay_loss1 = F.cross_entropy(out1, y, reduction='none')
    # delay_loss2 = F.cross_entropy(out2, y, reduction='none')
    # delay_loss3 = F.cross_entropy(out3, y, reduction='none')

    losses_total = delay_loss1 + delay_loss2 + delay_loss3
    loss_all[inds] = losses_total.detach().cpu().numpy()

    if epoch >= args.start_mask_epoch:
        remember_rate = 1 - args.label_noise_rate
        if epoch >= args.start_refurb:
            final_mask = get_clean_mask(losses_total, inds, update_inds, remember_rate=remember_rate)
        else:
            final_mask = get_clean_loss_tensor_mask(losses_total, remember_rate=remember_rate)
    else:
        final_mask = torch.ones_like(losses_total)

    final_loss1 = torch.sum(final_mask * delay_loss1) / max(torch.count_nonzero(final_mask).item(), 1)
    final_loss2 = torch.sum(final_mask * delay_loss2) / max(torch.count_nonzero(final_mask).item(), 1)
    final_loss3 = torch.sum(final_mask * delay_loss3) / max(torch.count_nonzero(final_mask).item(), 1)

    return final_loss1, final_loss2, final_loss3, loss_all