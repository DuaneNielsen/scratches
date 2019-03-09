import torch

y = torch.Tensor([[0, 1, 0], [1, 0, 0]])
target = torch.Tensor([1, 2]).long()


def confusion(y, target):
    _, max_y = y.max(1)
    confusion = torch.zeros(3,3)
    for item in zip(max_y, target):
        confusion[item[0], item[1]] += 1

    correct = confusion * torch.eye(confusion.shape[0])
    incorrect = confusion - correct
    correct = correct.sum(0)
    incorrect = incorrect.sum(0)
    precision = correct / correct + incorrect
    return confusion, precision

print(confusion)
#print(correct)
#print(incorrect)
#print(precision)