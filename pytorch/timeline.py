import torch

""" A timeline driver for an RNN
"""

timesteps = 3
batch_size = 10
z_size = 2

timesteps = torch.linspace(0, timesteps-1, timesteps).unsqueeze(1).repeat(1, z_size)

batch = []
for _ in batch_size:
    batch_size.append(timesteps)

# if not packed sequence then...
timesteps = torch.linspace(0, timesteps-1, timesteps).unsqueeze(1).repeat(batch_size, 1, z_size)

print(timesteps.shape)