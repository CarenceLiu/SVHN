'''
Wenrui Liu
2024-4-15

dataset to load SVHN
'''
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

class SVHNFullDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filename = [file for file in os.listdir(self.root_dir) if file.endswith('.png')]
        self.images = [Image.open(os.path.join(self.root_dir, file)).convert('RGB') for file in tqdm(self.image_filename, desc='read image')]
        self.images = [self.transform(image) if self.transform else image for image in tqdm(self.images, desc="image transform")]
        label_map = {row["Filename"]:row["Label"] for idx, row in pd.read_csv(os.path.join(self.root_dir, "image_label_map.csv")).iterrows()}
        self.labels = [str(label_map[file]) for file in self.image_filename]
        self.label_lens = [len(label) for label in self.labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.label_lens[idx]

    def getOriginImage(self, idx):
        return Image.open(os.path.join(self.root_dir, self.image_filename[idx]))

def extract_labels(file_path):
    with h5py.File(file_path, 'r') as file:
        names = file['digitStruct/name']
        bboxes = file['digitStruct/bbox']

        data = []
        
        for i in tqdm(range(len(names)), desc="Extracting labels"):
            name_ref = names[i][0]
            img_name = ''.join(chr(c[0]) for c in file[name_ref])

            bbox = bboxes[i].item()
            label = file[bbox]['label']
            if isinstance(label, h5py.Dataset) and label.shape[0] > 1:
                label_vals = [str(int(file[label[j].item()][()][0][0])) if file[label[j].item()][()][0][0] != 10 else '0'
                              for j in range(label.shape[0])]
            else:
                label_val = label[()] if isinstance(label, h5py.Dataset) else label.item()
                label_vals = [str(int(label_val[0][0])) if label_val[0][0] != 10 else '0']
            label_str = ''.join(label_vals)
            
            data.append([img_name, label_str])
    
    return pd.DataFrame(data, columns=['Filename', 'Label'])


if __name__ == "__main__":
    train_path = '../data/train/digitStruct.mat'
    test_path = '../data/test/digitStruct.mat'
    train_output_path = '../data/train/image_label_map.csv'
    test_output_path = '../data/test/image_label_map.csv'
    df_labels = extract_labels(train_path)
    df_labels.to_csv(train_output_path, index=False)
    df_labels = extract_labels(test_path)
    df_labels.to_csv(test_output_path, index=False)
