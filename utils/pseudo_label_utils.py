import numpy as np
import torch
from scipy.stats import mode


def generate_pseudo_labels(args, epoch, targets, refurb_matrixs, indexes, unselected_inds):
    with torch.no_grad():
        refurb_matrix1, refurb_matrix2, refurb_matrix3 = refurb_matrixs[0], refurb_matrixs[1], refurb_matrixs[2]
        matrix1, matrix2, matrix3 = refurb_matrix1[indexes], refurb_matrix2[indexes], refurb_matrix3[indexes]

        logits1 = torch.mean(torch.from_numpy(matrix1), dim=1)
        logits2 = torch.mean(torch.from_numpy(matrix2), dim=1)
        logits3 = torch.mean(torch.from_numpy(matrix3), dim=1)
        logits = torch.stack([logits1, logits2, logits3], dim=1).mean(dim=1).cuda()

        update_ids = []
        if epoch >= args.start_refurb:
            ground_mask = np.where(np.isin(indexes, list(unselected_inds)), 0, 1)
            ground_mask = torch.from_numpy(ground_mask).unsqueeze(1).cuda()

            pred_matrix1 = mode(matrix1.argmax(axis=-1), axis=1).mode
            pred_matrix2 = mode(matrix2.argmax(axis=-1), axis=1).mode
            pred_matrix3 = mode(matrix3.argmax(axis=-1), axis=1).mode
            update_ids = np.where((pred_matrix1 == pred_matrix2) & (pred_matrix2 == pred_matrix3))[0].tolist()
        else:
            ground_mask = torch.ones(logits.shape[0], 1).cuda()
        pesudo_mask = 1 - ground_mask

        # Transform label to one-hot
        labels_x = torch.zeros(targets.shape[0], args.num_classes).cuda().scatter_(1, targets.view(-1, 1), 1)

        probs = torch.softmax(logits, dim=1)
        pesudo_probs = probs ** (1 / args.T)
        px = labels_x * ground_mask + pesudo_probs * pesudo_mask

        targets_x = px / px.sum(dim=1, keepdim=True)  # normalize

    return targets_x, update_ids

