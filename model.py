import torch
import torch.nn as nn
import numpy as np


class Flatten(nn.Module):
    """https://github.com/sksq96/pytorch-vae/blob/master/vae.py"""
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    """https://github.com/sksq96/pytorch-vae/blob/master/vae.py"""
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class ResidualBlock(nn.Module):
    """Residual Block with some normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
    def forward(self, x):
        return x + self.main(x)


class Encoder(nn.Module):
    """Encoder for translating an image to a latent value, W = (W - F + 2P) /S + 1"""
    def __init__(self, start_channel=64, target_channel=25, nlayers=6, last_layer='max', image_layer ='tanh'):
        super(Encoder, self).__init__()
        encodings = list()
        decodings = list()

        self.channel = start_channel
        self.target_channel = target_channel
        self.last_layer = last_layer
        self.image_layer = image_layer
        self.nlayers = nlayers

        encodings.append(nn.Conv2d(3, self.channel, kernel_size=4, stride=2, padding=1, bias=False))
        encodings.append(nn.InstanceNorm2d(self.channel, affine=True, track_running_stats=True))
        encodings.append(nn.ReLU(inplace=True))

        for i in range(self.nlayers - 2):
            encodings.append(nn.Conv2d(self.channel, self.channel * 2, kernel_size=4, stride=2, padding=1, bias=False))
            encodings.append(nn.InstanceNorm2d(self.channel * 2, affine=True, track_running_stats=True))
            encodings.append(nn.ReLU(inplace=True))
            self.channel *= 2

        encodings.append(nn.Conv2d(self.channel, self.target_channel, kernel_size=4, stride=2, padding=1, bias=False))
        encodings.append(nn.InstanceNorm2d(self.target_channel, affine=True, track_running_stats=True))
        encodings.append(nn.ReLU(inplace=True))

        if self.last_layer == 'max':
            encodings.append(nn.MaxPool2d(kernel_size=4))
        elif self.last_layer == 'avg':
            encodings.append(nn.AvgPool2d(kernel_size=4))
        elif self.last_layer == 'conv':
            encodings.append(nn.Conv2d(self.target_channel, self.target_channel, kernel_size=4, stride=1, bias=False))
            encodings.append(nn.InstanceNorm2d(self.target_channel, affine=True, track_running_stats=True))
            encodings.append(nn.ReLU(inplace=True))
        else:
            raise Exception('error on encoder last mode')

        self.encoder = nn.Sequential(*encodings)

        decodings.append(nn.ConvTranspose2d(self.target_channel, self.target_channel * 2, kernel_size=4, bias=False))
        decodings.append(nn.InstanceNorm2d(self.target_channel * 2, affine=True, track_running_stats=True))
        decodings.append(nn.ReLU(inplace=True))

        self.mchannel = self.target_channel * 2

        for i in range(self.nlayers - 1):
            decodings.append(nn.ConvTranspose2d(self.mchannel, self.mchannel // 2, kernel_size=4, stride=2, padding=1, bias=False))
            decodings.append(nn.InstanceNorm2d(self.mchannel // 2, affine=True, track_running_stats=True))
            decodings.append(nn.ReLU(inplace=True))
            self.mchannel = self.mchannel // 2

        decodings.append(nn.ConvTranspose2d(self.mchannel, 3, kernel_size=4, stride=2, padding=1, bias=False))
        if self.image_layer == 'tanh':
            decodings.append(nn.Tanh())
        elif self.image_layer == 'sigmoid':
            decodings.append(nn.Sigmoid)

        self.decoder = nn.Sequential(*decodings)

    def forward(self, x):
        z = self.encoder(x)
        image = self.decoder(z)
        return image


class VarEncoder(nn.Module):
    """Encoder for translating an image to a latent value"""

    def __init__(self, start_channel=32, target_channel=25, nlayers=6, image_layer ='tanh'):
        super(VarEncoder, self).__init__()
        self.target_channel = target_channel
        self.image_layer = image_layer
        self.nlayers = nlayers
        self.channel = start_channel
        encodings = list()
        decodings = list()

        self.fc1 = nn.Linear(64 * 64 * 2 * self.channel, self.target_channel)
        self.fc2 = nn.Linear(64 * 64 * 2 * self.channel, self.target_channel)

        encodings.append(nn.Conv2d(3, self.channel, kernel_size=4, stride=2, padding=1, bias=False))
        encodings.append(nn.InstanceNorm2d(self.channel, affine=True, track_running_stats=True))
        encodings.append(nn.ReLU(inplace=True))

        encodings.append(nn.Conv2d(self.channel, self.channel * 2, kernel_size=4, stride=2, padding=1, bias=False))
        encodings.append(nn.InstanceNorm2d(self.channel * 2, affine=True, track_running_stats=True))
        encodings.append(nn.ReLU(inplace=True))
        encodings.append(Flatten())
        #self.channel *= 2
        # add one-hot vector coressponding
        self.encoder = nn.Sequential(*encodings)

        decodings.append(nn.ConvTranspose2d(self.target_channel, self.target_channel * 2, kernel_size=4, bias=False))
        decodings.append(nn.InstanceNorm2d(self.target_channel * 2, affine=True, track_running_stats=True))
        decodings.append(nn.ReLU(inplace=True))

        self.mchannel = self.target_channel * 2

        for i in range(self.nlayers - 1):
            decodings.append(
                nn.ConvTranspose2d(self.mchannel, self.mchannel // 2, kernel_size=4, stride=2, padding=1, bias=False))
            decodings.append(nn.InstanceNorm2d(self.mchannel // 2, affine=True, track_running_stats=True))
            decodings.append(nn.ReLU(inplace=True))
            self.mchannel = self.mchannel // 2

        decodings.append(nn.ConvTranspose2d(self.mchannel, 3, kernel_size=4, stride=2, padding=1, bias=False))
        if self.image_layer == 'tanh':
            decodings.append(nn.Tanh())
        elif self.image_layer == 'sigmoid':
            decodings.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decodings)

    def reparameterize(self, mu, logvar):
        """https://github.com/sksq96/pytorch-vae/blob/master/vae.py"""
        std = logvar.mul(0.5).exp_()
        esp = torch.randn_like(std) # _like follows tensor_type
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        """https://github.com/sksq96/pytorch-vae/blob/master/vae.py"""
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = z.unsqueeze(len(z.size())).unsqueeze(len(z.size()))
        image = self.decoder(z)

        return image, mu, logvar