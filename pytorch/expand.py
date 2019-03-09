import torch

M = torch.Tensor(2,128,26,26)
y = torch.Tensor(2,64)

y = y.unsqueeze(-1).unsqueeze(-1)
y = y.expand(-1,-1,26,26)

M_plus = torch.cat((M,y), dim=1)