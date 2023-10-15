"""
Copyright (C) 2023 Yukara Ikemiya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .encdec import Encoder, Decoder
from .stochastic_quantizer import SQuantizer


class SQVAE(nn.Module):
    def __init__(self, kwargs_encdec: dict, kwargs_quantizer: dict):
        super().__init__()
        assert (kwargs_encdec['width'] == kwargs_quantizer['dim_dict'])

        self.encoder = Encoder(**kwargs_encdec)
        self.decoder = Decoder(**kwargs_encdec)
        self.quantizer = SQuantizer(**kwargs_quantizer)
        self.quantizer.set_data_dimension(2)

    def forward(self, x):
        # encode
        z = self.encoder(x)
        # quantize
        z_q, loss_latent, metrics_latent = self.quantizer(z)
        # decode
        x_rec = self.decoder(z_q)
        # reconstruction loss
        loss_rec, rmse = self.compute_rec_loss(x_rec, x)

        loss = loss_latent + loss_rec

        metrics = {}
        metrics['rmse'] = rmse
        metrics['loss'] = loss.detach()
        metrics['loss_rec'] = loss_rec.detach()
        metrics.update(metrics_latent)

        return loss, x_rec, metrics

    def compute_rec_loss(self, x_rec, x_gt):
        # Reconstruction loss
        bs, *shape = x_rec.shape
        dim_x = np.prod(shape)

        # square-error
        se = F.mse_loss(x_rec, x_gt, reduction="sum") / bs

        # "Preventing Posterior Collapse Induced by Oversmoothing in Gaussian VAE"
        # https://arxiv.org/abs/2102.08663
        loss_rec = dim_x * torch.log(se) / 2

        rmse = (se.detach() / dim_x).sqrt()

        return loss_rec, rmse

    def set_temperature(self, t):
        self.quantizer.set_temperature(t)
