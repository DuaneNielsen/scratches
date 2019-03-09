import tensorboardX
import torch
import numpy as np

tb = tensorboardX.SummaryWriter(r'c:\data\runs\scratch18')

for epoch in range(4):

    hist = None
    for step in range(10):
        x = torch.randn(10,4)
        zero = x[:,0]
        if hist is None:
            hist = zero.cpu().numpy()
        else:
            hist = np.append(hist, zero.cpu().numpy(), axis=0)

    tb.add_histogram('0', hist, epoch)