import torch
from torch.distributions.normal import Normal
from torch.distributions.multinomial import Multinomial


if __name__ == '__main__':
    """ k-armed bandit, with rewards of mu/sigma for each arm
    """
    arms = 3
    trials = 1
    pulls = 1

    mu = torch.empty(arms, dtype=torch.float).uniform_(1, 3)
    sigma = torch.empty(arms, dtype=torch.float).uniform_(0, 1)
    distrib = Normal(mu, sigma)

    total_actions = torch.zeros(arms)
    total_rewards = torch.zeros(arms)

    for t in range(10):
        outcomes = distrib.sample(torch.Size((trials,)))

        uniform_probs = torch.softmax(torch.ones(arms), dim=0)
        uniform_random_action_distribution = Multinomial(pulls, probs=uniform_probs)
        actions = uniform_random_action_distribution.sample(torch.Size((trials,)))

        reward = actions * outcomes

        total_rewards = total_rewards + reward
        total_actions = total_actions + actions
        greed_score = total_rewards / total_actions
        greed_score[torch.isnan(greed_score)] = 0
        greed_probs = torch.softmax(greed_score, dim=1)
        pass
    #print('reward for action %i was %f' % (action, reward.item()))
