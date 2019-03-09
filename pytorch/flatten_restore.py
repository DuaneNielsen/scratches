import torch
import torch.nn as nn

net = nn.Sequential(nn.Conv2d(1,1,2,1), nn.Linear(1,3))

flat_weights = []


def capture(net):

    w = []

    def _capture(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            w.append(m.weight.data)
            w.append(m.bias.data)
    net.apply(_capture)

    t = list( map(lambda x: x.view(-1), w))
    return torch.cat(t)



def restore(net, t):

    start = 0

    def _restore(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nonlocal start

            length = m.weight.data.numel()
            chunk = t[range(start, start+length)]
            m.weight.data = chunk.view(m.weight.data.shape)
            start += length

            length = m.bias.data.numel()
            chunk = t[range(start, start+length)]
            m.bias.data = chunk.view(m.bias.data.shape)
            start += length

    net.apply(_restore)


t = capture(net)
t = torch.linspace(0,t.numel()-1, t.numel())
restore(net, t)
v = capture(net)
print(v)
