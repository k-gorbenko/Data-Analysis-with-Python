import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def encoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels, 
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''

    # Реализуйте блок вида conv -> relu -> max_pooling. 
    # Параметры слоя conv заданы параметрами функции encoder_block. 
    # MaxPooling должен быть с ядром размера 2.
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2)  # MaxPooling с ядром размера 2
    )

    return block

def decoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels, 
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''

    # Реализуйте блок вида conv -> relu -> upsample. 
    # Параметры слоя conv заданы параметрами функции encoder_block. 
    # Upsample должен быть со scale_factor=2. Тип upsampling (mode) можно выбрать любым.
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.ReLU(inplace=False),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Увеличение с масштабом 2
    )

    return block

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        параметры: 
            - in_channels: количество каналов входного изображения
            - out_channels: количество каналов выхода нейросети
        '''
        super().__init__()

        self.enc1_block = encoder_block(in_channels, 32, 7, 3)
        self.enc2_block = encoder_block(32, 64, 3, 1)
        self.enc3_block = encoder_block(64, 128, 3, 1)

        # Определяем декодирующие блоки
        self.dec1_block = decoder_block(128, 64, 3, 1)  # Симметричен enc3_block
        self.dec2_block = decoder_block(64 + 64, 32, 3, 1)  # enc2_block + skip connection
        self.dec3_block = decoder_block(32 + 32, out_channels, 3, 1)  # enc1_block + skip connection

    def forward(self, x):
        # downsampling part
        enc1 = self.enc1_block(x)
        enc2 = self.enc2_block(enc1)
        enc3 = self.enc3_block(enc2)

        # upsampling part
        dec1 = self.dec1_block(enc3)
        dec2 = self.dec2_block(torch.cat([dec1, enc2], 1))  # skip connection
        dec3 = self.dec3_block(torch.cat([dec2, enc1], 1))  # skip connection

        return dec3


def create_model(in_channels, out_channels):
    # your code here
    # return model instance (None is just a placeholder)

    return UNet(in_channels, out_channels)
