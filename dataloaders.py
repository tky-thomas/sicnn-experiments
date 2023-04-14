from matplotlib import pyplot as plt

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datasets import AAR_Lightbox_Dataset, ImageNet_Classification

mean = {
    'stl10': (0.4467, 0.4398, 0.4066),
    'scale_mnist': (0.0607,),
}

std = {
    'stl10': (0.2603, 0.2566, 0.2713),
    'scale_mnist': (0.2161,),
}

AAR_DATA_SPLIT = [0.8, 0.1, 0.1]
IMAGENET_CLASSIFY_DATA_SPLIT = [0.8, 0.1, 0.1]


def make_aar_loaders(batch_size=64, extra_scaling=1):
    transform_modules_train = []
    if not extra_scaling == 1:
        if extra_scaling > 1:
            extra_scaling = 1 / extra_scaling
        scale = (extra_scaling, 1 / extra_scaling)
        print('-- extra scaling ({:.3f} - {:.3f}) is used'.format(*scale))
        scaling = transforms.RandomAffine(0, scale=scale, resample=3)
        transform_modules_train.append(scaling)

    transform_modules = [
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['scale_mnist'], std['scale_mnist']),
        transforms.Resize((224, 224))
    ]

    transform_modules_train = transform_modules_train + transform_modules

    transform_modules = transforms.Compose(transform_modules)
    transform_modules_train = transforms.Compose(transform_modules_train)

    dataset = AAR_Lightbox_Dataset(img_labels_path="data/aar-lightbox/LightBox_annotation.csv",
                                   img_dir="data/aar-lightbox/",
                                   transform=transform_modules)
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=AAR_DATA_SPLIT)
    train_dataset.transform = transform_modules_train

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    return train_loader, val_loader, test_loader


def make_imagenet_loaders(batch_size=64, extra_scaling=1):
    transform_modules_train = []
    if not extra_scaling == 1:
        if extra_scaling > 1:
            extra_scaling = 1 / extra_scaling
        scale = (extra_scaling, 1 / extra_scaling)
        print('-- extra scaling ({:.3f} - {:.3f}) is used'.format(*scale))
        scaling = transforms.RandomAffine(0, scale=scale, resample=3)
        transform_modules_train.append(scaling)

    transform_modules = [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean['scale_mnist'], std['scale_mnist']),
        transforms.Resize((224, 224))
    ]

    transform_modules_train = transform_modules_train + transform_modules

    transform_modules = transforms.Compose(transform_modules)
    transform_modules_train = transforms.Compose(transform_modules_train)

    dataset = ImageNet_Classification("data/imagenet-classification/train", transform=transform_modules)
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=IMAGENET_CLASSIFY_DATA_SPLIT)
    train_dataset.transform = transform_modules_train

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    return train_loader, val_loader, test_loader


# Tests the loaders
if __name__ == '__main__':
    train_loader, val_loader, test_loader = make_aar_loaders()
    print(len(train_loader.dataset),
          len(val_loader.dataset),
          len(test_loader.dataset))
    plt.imshow(train_loader.dataset[0][0].permute(1, 2, 0))
    plt.show()

    train_loader, val_loader, test_loader = make_imagenet_loaders()
    print(len(train_loader.dataset),
          len(val_loader.dataset),
          len(test_loader.dataset))
    plt.imshow(train_loader.dataset[0][0].permute(1, 2, 0))
    plt.show()


