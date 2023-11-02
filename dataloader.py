import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import json
from torchvision import transforms, utils
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



class CelebAMaskDataset(Dataset):
    def __init__(self, label_file, root_dir, transform=None):

        with open(label_file) as f: 
            self.data = f.read() 
        self.data = json.loads(self.data)

        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        labels = list(self.data.values()) 

        names = list(self.data.keys())

        img_name = os.path.join(self.root_dir,
                                names[idx])
        image = Image.open(img_name)

        # sample = {'image': image, 'label':labels[idx]}
        
        if self.transform:
            image = self.transform(image)

        return image, labels[idx]
    

# if __name__ == "__main__":
#     dataset = CelebAMaskDataset(label_file='/p/scratch/holistic-vid-westai/ganji1/identity_CelebAHQ.txt',
#                                     root_dir='/p/scratch/holistic-vid-westai/ganji1/data256x256/')
    
#     for i, sample in enumerate(dataset):
#         print(type(sample))
        
#         print(i, sample['image'].shape, sample['label'])
    