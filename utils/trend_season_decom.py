import torch
import torch.nn as nn
import numpy as np
import random

import tsaug


class ts_decom(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=5, block_size=2, alpha=0.01, beta=0.5):
        super(ts_decom, self).__init__()

        self.kernel_size = kernel_size
        self.block_size = block_size
        self.alpha = alpha
        self.beta = beta

        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def moving_avg(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

    def forward(self, args, x, y):
        moving_mean = self.moving_avg(x)
        long_term_dep = x - moving_mean
        short_term_x = (1 - self.alpha) * x + self.alpha * moving_mean
        long_term_x = (1 - self.beta) * x + self.beta * long_term_dep

        res_x = torch.cat([x, short_term_x, long_term_x], dim=0)
        res_y = torch.cat([y, y, y], dim=0)

        return res_x.cpu().numpy(), res_y.cpu().numpy()
