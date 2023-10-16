"""
Copyright (C) 2023 Yukara Ikemiya
"""

from torch import nn

from .common_layers import Resnet2D


class Encoder(nn.Module):
    def __init__(self, in_ch, width, depth, num_down, stride, **kwargs):
        super().__init__()

        blocks = []
        for ii in range(num_down):
            # Down-sampling
            down = nn.Conv2d(in_ch if ii == 0 else width, width, stride * 2, stride, stride // 2)
            # ResNet
            resnet = Resnet2D(width, depth, **kwargs)

            blocks.extend([down, resnet])

        # output layer
        conv_ch_out = nn.Conv2d(width, width, 3, 1, 1)
        blocks.append(conv_ch_out)

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    # Each argument name corresponds to that of EncoderBlock
    def __init__(self, in_ch, width, depth, num_down, stride, **kwargs):
        super().__init__()
        kernel_size = stride * 2 if stride != 1 else 3
        padding = stride // 2 if stride != 1 else 1

        blocks = [nn.Conv2d(width, width, 3, 1, 1)]
        for ii in range(num_down):
            # ResNet
            resnet = Resnet2D(width, depth, **kwargs)
            # Up-sampling
            up = nn.ConvTranspose2d(width, in_ch if ii == (num_down - 1) else width,
                                    kernel_size, stride, padding)

            blocks.extend([resnet, up])

        # output layer for 0-1 image date
        out = nn.Sigmoid()

        blocks.append(out)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
