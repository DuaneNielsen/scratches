import torch.utils.data
from pathlib import Path

class ActionEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        torch.utils.data.Dataset.__init__(self)
        self.path = Path(directory)
        self.count = 0
        for _ in self.path.glob('*.np'):
            self.count += 1

    def __getitem__(self, index):
        filepath = self.path / str(index)
        filepath = filepath.with_suffix('*.np')
        data = None

        return data

    def __len__(self):
        return self.count
