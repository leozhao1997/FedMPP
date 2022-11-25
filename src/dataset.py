import torch
from torch.utils.data import Dataset

class FMNISTDataset(Dataset):
    def __init__(self, x, y, client_id):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.client_id = client_id

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index] 


"""
TODO:

Implement your own Dataset here
"""


