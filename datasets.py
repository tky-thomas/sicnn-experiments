import os

import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import crop
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder


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


class ImageNet_Classification(ImageFolder):
    def find_classes(self, directory: str):
        numeric_classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not numeric_classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        # Reads the Labels.json file
        class_labels_f = open(os.path.abspath(os.path.join(directory, os.path.pardir, "Labels.json")))
        class_names = json.load(class_labels_f)

        # Replaces the class names with the actual named labels

        classes = list()
        for numeric_class in numeric_classes:
            classes.append(class_names[numeric_class].split(",")[0])

        class_labels_f.close()

        class_to_idx = {cls_name: i for i, cls_name in enumerate(numeric_classes)}
        return numeric_classes, class_to_idx


# Tests the datasets
if __name__ == '__main__':
    imagenet_classification_dataset = ImageNet_Classification("data/imagenet-classification/train")
    imagenet_classification_dataset.find_classes("data/imagenet-classification/train")
    print(imagenet_classification_dataset[4000], len(imagenet_classification_dataset))