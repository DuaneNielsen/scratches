import torch
from torch.distributions.normal import Normal

class Bandit:
    def __init__(self, k):
        self.mu = torch.empty(k, dtype=torch.float).uniform_(1, 3)
        self.sigma = torch.empty(k, dtype=torch.float).uniform_(0, 1)
        self.distrib = Normal(self.mu, self.sigma)

    def action(self, k):
        r = self.distrib.sample()
        return r[k]

if __name__ == '__main__':
    arms = 3
    bandit = Bandit(arms)
    action = 2
    reward = bandit.action(action)
    print('reward for action %i was %f' % (action, reward.item()))
