import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as TVT

dataset = torchvision.datasets.ImageFolder(
    root=r'c:/data/cartpole/images/raw',
    transform=TVT.Compose([TVT.ToTensor()])
)

loader = data_utils.DataLoader(dataset=dataset, batch_size=10, shuffle=False, drop_last=True)

for images, target in loader:
    print('hello')