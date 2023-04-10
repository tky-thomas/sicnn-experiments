import os

import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import crop
from torchvision.transforms import ToPILImage, ToTensor


class AAR_Lightbox_Dataset(Dataset):
    def __init__(self, img_labels_path="LightBox_annotation.csv", img_dir="", transform=None):
        self.image_labels = pd.read_csv(img_labels_path)
        self.img_dir = img_dir
        self.transform = transform

        self.root = None

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path = self.image_labels.iloc[idx, 0]
        img_path = os.path.relpath(img_path, "/content/drive/My Drive/AAR/dataset/LightBox/")
        img_path = os.path.join(self.img_dir, img_path).replace("\\", "/")
        img = read_image(img_path, ImageReadMode.UNCHANGED)

        # Crops out the lightbox
        img_x_min = self.image_labels.iloc[idx, 1]
        img_y_min = self.image_labels.iloc[idx, 2]
        img_x_max = self.image_labels.iloc[idx, 3]
        img_y_max = self.image_labels.iloc[idx, 4]

        img_h = img_y_max - img_y_min
        img_w = img_x_max - img_x_min

        img = crop(img, img_y_min, img_x_min, img_h, img_w)

        # Applies transformation
        if self.transform:
            img = ToPILImage()(img)
            img = self.transform(img)

        # Creates a pytorch tensor label
        label = self.image_labels.iloc[idx, 5]
        if label == "GOOD":
            label = 1
        else:
            label = 0
        label = torch.tensor(label)

        return img, label

    def extra_repr(self):
        return "None"