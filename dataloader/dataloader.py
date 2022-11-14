import os
import torch
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from logger import variable_logger as logger

log = logger()

class dataset(Dataset):
    def __init__(self, path: str, device="cpu"):
        self.path = path
        self.filenames = os.listdir(path)
        self.images = []
        self.device = device
        # self.transforms = ToTensor()
    def __getitem__(self, index):
        return (read_image(os.path.join(self.path, self.filenames[index]), 
                           torchvision.io.ImageReadMode.GRAY) / 255.0).to(self.device)
    def __len__(self):
        return len(self.filenames)
