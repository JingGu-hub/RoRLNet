import numpy as np
import torch
import torch.nn.functional as F

def generate_pseudo_labels(args, epoch, targets, refurb_matrixs, indexes, unselected_inds, alpha=1.0):
    with torch.no_grad():
        refurb_matrix1, refurb_matrix2, refurb_matrix3 = refurb_matrixs[0], refurb_matrixs[1], refurb_matrixs[2]
        logits1 = torch.mean(torch.from_numpy(refurb_matrix1[indexes]), dim=1)
        logits2 = torch.mean(torch.from_numpy(refurb_matrix2[indexes]), dim=1)
        logits3 = torch.mean(torch.from_numpy(refurb_matrix3[indexes]), dim=1)
        logits = torch.stack([logits1, logits2, logits3], dim=1).mean(dim=1).cuda()

        if epoch >= args.start_refurb:
            ground_mask = np.where(np.isin(indexes, list(unselected_inds)), 0, 1)
            ground_mask = torch.from_numpy(ground_mask).unsqueeze(1).cuda()
        else:
            ground_mask = torch.ones(logits.shape[0], 1).cuda()
        pesudo_mask = 1 - ground_mask

        # Transform label to one-hot
        labels_x = torch.zeros(targets.shape[0], args.num_classes).cuda().scatter_(1, targets.view(-1, 1), 1)

        probs = torch.softmax(logits, dim=1)
        # pesudo_probs = probs ** (1 / args.T)
        pesudo_probs = (args.lam * labels_x + (1 - args.lam) * probs) ** (1 / args.T)
        px = labels_x * ground_mask + pesudo_probs * pesudo_mask

        targets_x = px / px.sum(dim=1, keepdim=True)  # normalize

    return targets_x
