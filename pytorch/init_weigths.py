import torch.nn as nn

def print_weights_and_bias(m):
    print(m)
    if type(m) in [nn.Linear, nn.Conv2d]:
        m.weight.data.fill_(1.0)
        print(m.weight.shape)
        print(m.bias.shape)

net = nn.Sequential(nn.Linear(2, 2, bias=True), nn.Conv2d(1, 1, 2))

net.apply(print_weights_and_bias)
