import torch

answers = torch.Tensor([0.0,1.0,0.0,1.0])

print(answers)

non_zero = answers[answers != 0.0]

print(non_zero)

non_zero = answers[answers.nonzero()]

