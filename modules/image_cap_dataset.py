import os
import random

from datasets import load_dataset

from torch.utils.data import Dataset
from PIL import Image
from modules.training_utils import center_crop_arr
from torchvision.transforms import transforms


class ImageCaptionDataset(Dataset):
    def __init__(self, hf_dataset_name, token, transform=None, target_transform=None, res=256):
        self.hf_dataset = load_dataset(hf_dataset_name, use_auth_token=token)["test"]
        self.token = token

        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, res)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.target_transform = target_transform

    def __len__(self):
        return len(self.hf_dataset)

    def get_item(self, idx):
        if random.random() < 0.5:
            image, prompt = self.hf_dataset[idx]["image_0"], self.hf_dataset[idx]["caption_0"]
        else:
            image, prompt = self.hf_dataset[idx]["image_1"], self.hf_dataset[idx]["caption_1"]

        image = image.rotate(90).convert("RGB")

        prompt = prompt.lower()

        if self.transform:
            image = self.transform(image)
        return image, prompt

    def __getitem__(self, idx):
        try:
            image, prompt = self.get_item(idx)
        except Exception as e:
            print(e)
            image, prompt = self.get_item(0)
        return image, prompt
