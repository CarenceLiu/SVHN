'''
Wenrui Liu
2024-4-14

dataset to load SVHN
'''
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class SVHNFullDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_files = [os.path.join(self.root_dir, file) for file in os.listdir(self.root_dir) if file.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image