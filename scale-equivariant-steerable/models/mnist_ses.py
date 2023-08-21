'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .impl.ses_conv import SESMaxProjection
from .impl.ses_conv import SESConv_Z2_H, SESConv_H_H


class MNIST_SES_Scalar(nn.Module):

    def __init__(self, pool_size=4, kernel_size=11, scales=[1.0], basis_type='A', **kwargs):
        super().__init__()
        C1, C2, C3 = 32, 63, 95
        self.main = nn.Sequential(
            SESConv_Z2_H(1, C1, kernel_size, 7, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type, **kwargs),
            SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C1),

            SESConv_Z2_H(C1, C2, kernel_size, 7, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type, **kwargs),
            SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C2),

            SESConv_Z2_H(C2, C3, kernel_size, 7, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type, **kwargs),
            SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(pool_size, padding=2),
            nn.BatchNorm2d(C3),
        )

        self.linear = nn.Sequential(
            nn.Linear(C3, 256, bias=False),  # TODO: Find a way to make this line a dependent variable
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(kwargs.get('dropout', 0.7)),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.main(x)
        print(x.size())
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MNIST_SES_V(nn.Module):

    def __init__(self, pool_size=4, kernel_size=11, scales=[1.0], basis_type='A', dropout=0.7, num_classes=10, **kwargs):
        super().__init__()
        C1, C2, C3 = 32, 63, 95
        self.main = nn.Sequential(
            SESConv_Z2_H(1, C1, kernel_size, 7, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C1),

            SESConv_H_H(C1, C2, 1, kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C2),

            SESConv_H_H(C2, C3, 1, kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(pool_size, padding=2),
            nn.BatchNorm2d(C3),
        )

        self.linear = nn.Sequential(
            nn.Linear(C3, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MNIST_SES_V_SEG(nn.Module):

    def __init__(self, pool_size=4, kernel_size=11, scales=[1.0], basis_type='A', dropout=0.7,
                 **kwargs):
        super().__init__()
        C1, C2, C3 = 16, 32, 48
        self.main = nn.Sequential(
            SESConv_Z2_H(1, C1, kernel_size, 7, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C1),

            SESConv_H_H(C1, C2, 1, kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C2),

            SESConv_H_H(C2, C3, 1, kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            SESMaxProjection(),
            nn.ReLU(True),
            # nn.MaxPool2d(pool_size, padding=2),
            # nn.BatchNorm2d(C3),
        )

        self.dim_reducer = nn.Sequential(
            nn.Conv2d(C3, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv2d(16, 4, 3, stride=1, padding=1),
            nn.Conv2d(4, 1, 3, stride=1, padding=1),
            # nn.Linear(C3, 256, bias=False),
            # nn.BatchNorm1d(256),
            # nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.main(x)
        # x = x.view(x.size(0), -1)
        x = self.dim_reducer(x)
        return x


class MNIST_SES_V_SEG_COMPLETE(nn.Module):

    def __init__(self, pool_size=4, kernel_size=11, scales=[1.0], basis_type='A', dropout=0.7,
                 **kwargs):
        super().__init__()
        C1, C2, C3 = 12, 24, 48
        self.main = nn.Sequential(
            SESConv_Z2_H(3, C1, kernel_size, 7, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C1),

            SESConv_H_H(C1, C2, 1, kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            # nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C2),

            SESConv_H_H(C2, C3, 1, kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            # nn.MaxPool2d(pool_size, padding=2),
            # nn.BatchNorm2d(C3),
        )

        C4, C5, C6 = 48, 24, 12
        self.dim_reducer = nn.Sequential(
            SESConv_H_H(C3, C4, 1, kernel_size, 7, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C4),

            SESConv_H_H(C4, C5, 1, kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            # nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C5),

            SESConv_H_H(C5, C6, 1, kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            SESMaxProjection(),
            # nn.MaxPool2d(pool_size, padding=2),
            # nn.BatchNorm2d(C3),

            nn.Conv2d(C6, 1, 3, stride=1, padding=1),
            # nn.BatchNorm2d(4),
            # nn.ReLU(True),
            # nn.Dropout(dropout),
            # nn.Conv2d(4, 1, 3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.main(x)
        # x = x.view(x.size(0), -1)
        x = self.dim_reducer(x)
        return x

def mnist_ses_vector_56p(**kwargs):
    num_scales = 4
    factor = 2.0
    min_scale = 1.5
    mult = 1.4
    size = 13
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q**i for i in range(num_scales)]
    scales = [round(s, 2) for s in scales]
    model = MNIST_SES_V(pool_size=8, kernel_size=size, scales=scales,
                        basis_type='B', mult=mult, max_order=4, dropout=dropout)
    return nn.Sequential(nn.Upsample(scale_factor=2), model)


def mnist_ses_vector_56(**kwargs):
    num_scales = 4
    factor = 2.0
    min_scale = 2.0
    mult = 1.5
    size = 15
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q**i for i in range(num_scales)]
    scales = [round(s, 2) for s in scales]
    model = MNIST_SES_V(pool_size=8, kernel_size=size, scales=scales,
                        basis_type='B', mult=mult, max_order=4, dropout=dropout)
    return nn.Sequential(nn.Upsample(scale_factor=2), model)


def mnist_ses_scalar_56p(**kwargs):
    num_scales = 3
    factor = 3.0
    min_scale = 1.0
    mult = 1.4
    size = 17
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q**i for i in range(num_scales)]
    scales = [round(s, 2) for s in scales]
    model = MNIST_SES_Scalar(pool_size=8, kernel_size=size, scales=scales,
                             basis_type='B', mult=mult, max_order=4, dropout=dropout)
    return nn.Sequential(nn.Upsample(scale_factor=2), model)


def mnist_ses_scalar_56(**kwargs):
    num_scales = 3
    factor = 2.0
    min_scale = 2.0
    mult = 1.4
    size = 15
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q**i for i in range(num_scales)]
    scales = [round(s, 2) for s in scales]
    model = MNIST_SES_Scalar(pool_size=8, kernel_size=size, scales=scales,
                             basis_type='B', mult=mult, max_order=4, dropout=dropout)
    return nn.Sequential(nn.Upsample(scale_factor=2), model)


def mnist_ses_vector_28p(**kwargs):
    num_scales = 3
    factor = 3.0
    min_scale = 1.5
    mult = 1.4
    size = 15
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q**i for i in range(num_scales)]
    scales = [round(s, 2) for s in scales]
    model = MNIST_SES_V(pool_size=4, kernel_size=size, scales=scales,
                        basis_type='B', mult=mult, max_order=4, dropout=dropout)
    return model


def mnist_ses_vector_28(**kwargs):
    num_scales = 3
    factor = 3.0
    min_scale = 1.5
    mult = 1.5
    size = 13
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q**i for i in range(num_scales)]
    scales = [round(s, 2) for s in scales]
    model = MNIST_SES_V(pool_size=4, kernel_size=size, scales=scales,
                        basis_type='B', mult=mult, max_order=4, dropout=dropout)
    return model


def mnist_ses_scalar_28p(**kwargs):
    num_scales = 4
    factor = 3.0
    min_scale = 1.5
    mult = 1.4
    size = 13
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q**i for i in range(num_scales)]
    scales = [round(s, 2) for s in scales]
    model = MNIST_SES_Scalar(pool_size=4, kernel_size=size, scales=scales,
                             basis_type='B', mult=mult, max_order=4, dropout=dropout)
    return model


def mnist_ses_scalar_28(**kwargs):
    num_scales = 4
    factor = 3.0
    min_scale = 1.7
    mult = 1.5
    size = 15
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q**i for i in range(num_scales)]
    scales = [round(s, 2) for s in scales]
    model = MNIST_SES_Scalar(pool_size=4, kernel_size=size, scales=scales,
                             basis_type='B', mult=mult, max_order=4, dropout=dropout)
    return model


def ses_classification_224(num_classes=10, **kwargs):
    num_scales = 4
    factor = 1.2
    min_scale = 2.6
    mult = 1.2
    size = 19
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q ** i for i in range(num_scales)]
    scales = [round(s, 2) for s in scales]
    model = MNIST_SES_V(pool_size=32, kernel_size=size, scales=scales,
                             basis_type='B', mult=mult, max_order=4, dropout=dropout, num_classes=num_classes)
    return model


def ses_classification_224_segment(num_classes=10, **kwargs):
    num_scales = 4
    factor = 2
    min_scale = 2.6
    mult = 1.2
    size = 19
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q ** i for i in range(num_scales)]
    scales = [round(s, 2) for s in scales]
    model = MNIST_SES_V_SEG(pool_size=32, kernel_size=size, scales=scales,
                             basis_type='B', mult=mult, max_order=4, dropout=dropout)
    return nn.Sequential(nn.Upsample(scale_factor=4), model)


def ses_classification_224_segment_complete(**kwargs):
    num_scales = 4
    factor = 1.
    min_scale = 1.4
    mult = 1.4
    size = 15
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q ** i for i in range(num_scales)]
    scales = [round(s, 2) for s in scales]
    model = MNIST_SES_V_SEG_COMPLETE(pool_size=32, kernel_size=size, scales=scales,
                             basis_type='B', mult=mult, max_order=4, dropout=dropout)
    return nn.Sequential(nn.Upsample(scale_factor=2), model)


# def ses_classification_224_segment_complete(**kwargs):
#     num_scales = 4
#     factor = 1.2
#     min_scale = 2.6
#     mult = 1.2
#     size = 19
#     dropout = 0.7
#     q = factor ** (1 / (num_scales - 1))
#     scales = [min_scale * q ** i for i in range(num_scales)]
#     scales = [round(s, 2) for s in scales]
#     model = MNIST_SES_V_SEG_COMPLETE(pool_size=32, kernel_size=size, scales=scales,
#                              basis_type='B', mult=mult, max_order=4, dropout=dropout)
#     return nn.Sequential(nn.Upsample(scale_factor=2), model)


