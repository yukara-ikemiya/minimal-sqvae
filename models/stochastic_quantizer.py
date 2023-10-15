"""
Copyright (C) 2023 Yukara Ikemiya
"""

import torch
from torch import nn
import torch.nn.functional as F


class SQuantizer(nn.Module):
    # Type-I of the Gaussian SQ-VAE in the paper
    def __init__(self, size_dict: int, dim_dict: int, var_q_init: float):
        super().__init__()
        self.size_dict = size_dict
        self.dim_dict = dim_dict

        # Codebook
        self.codebook = nn.Parameter(torch.randn(self.size_dict, self.dim_dict))
        self.var_q = nn.Parameter(torch.zeros(1))
        self.temperature = 1.0

        # self.init
        self.register_buffer('init', torch.zeros(1, dtype=torch.bool))

        # Dimension of input data
        # 1 -> 1d (e.g. audio), 2 -> 2d (e.g. image)
        self.dim_data = 2

        # Initial variance
        self.var_q_init = var_q_init
        self.register_buffer('var_init', torch.tensor([var_q_init]))

    def set_temperature(self, value: float):
        self.temperature = value

    def set_data_dimension(self, dim: int):
        self.dim_data = dim

    def forward(self, z):
        # Pre-reshape
        z = self._pre_reshape(z)

        # Initilize codebook
        if self.training and not self.init[0]:
            self._init_codebook(z)

        # limit variance range using sigmoid
        var_q = torch.sigmoid(self.var_q) * 2. * self.var_init

        # Quantize
        z_quantize, loss, metrics = self.quantize(z, var_q)

        # Post-reshape
        z_quantize = self._post_reshape(z_quantize)

        metrics['variance_q'] = float(var_q.mean())

        return z_quantize, loss, metrics

    def quantize(self, z: torch.Tensor, var_q: torch.Tensor):
        # Posterior distance
        weight_q = 0.5 / torch.clamp(var_q, min=1e-10)
        logit = -self._calc_distance_bw_enc_codes(z, weight_q)
        probs = torch.softmax(logit, dim=-1)
        log_probs = torch.log_softmax(logit, dim=-1)

        # Quantization
        if self.training:
            encodings = F.gumbel_softmax(logit, tau=self.temperature)  # [L, size_dict]
            z_quantized = torch.mm(encodings, self.codebook)
        else:
            idxs_enc = torch.argmax(logit, dim=1)  # [L]
            z_quantized = F.embedding(idxs_enc, self.codebook)

        # Latent loss

        # KLD regularization
        loss_kld_reg = torch.sum(probs * log_probs) / self.bs

        # commitment loss
        loss_commit = self._calc_distance_bw_enc_dec(z, z_quantized, weight_q) / self.bs

        loss_latent = loss_kld_reg + loss_commit

        metrics = {}  # logging
        metrics['loss_commit'] = loss_commit.detach()
        metrics['loss_kld_reg'] = loss_kld_reg.detach()
        metrics['loss_latent'] = loss_latent.detach()

        return z_quantized, loss_latent, metrics

    def _calc_distance_bw_enc_codes(self, z, weight_q):
        distances = weight_q * self._se_codebook(z)
        return distances

    def _se_codebook(self, z):
        # z         : [L, dim_z]
        # codebook  : [size_dict, dim_z]
        # distances : [L, size_dict]

        distances = torch.sum(z**2, dim=1, keepdim=True)\
            + torch.sum(self.codebook**2, dim=1) - 2 * torch.mm(z, self.codebook.t())
        return distances

    def _calc_distance_bw_enc_dec(self, z1, z2, weight_q):
        return torch.sum((z1 - z2)**2 * weight_q)

    def _init_codebook(self, z):
        def _tile(z_, scale_rand=0.2):
            L, dim = z_.shape
            if L < self.size_dict:
                n_repeats = (self.size_dict - 1) // L + 1
                z_ = z_.repeat(n_repeats, 1)
            z_ = z_ + torch.randn_like(z_, requires_grad=False) * scale_rand * var_z
            return z_

        var_z = torch.var(z, dim=0).mean()
        y = _tile(z)
        _k_rand = y[torch.randperm(y.shape[0])][:self.size_dict]

        # if dist.is_available():
        #     dist.broadcast(_k_rand, 0)

        self.codebook.data[:, :] = _k_rand.clone()

        var_init = torch.var(y, dim=0).mean().clone().detach() * self.var_q_init
        self.var_init[:] = var_init
        self.init[0] = True

        print(f'Variance was initialized to {var_init}')

    def _pre_reshape(self, z):
        # (bs, dim_z, *in_shape) -> (bs * prod(in_shape), dim_z)

        if self.dim_data == 1:
            self.bs, self.dim_z, self.num_d1 = z.shape
            self.num_d2 = 1
        elif self.dim_data == 2:
            self.bs, self.dim_z, self.num_d1, self.num_d2 = z.shape
        else:
            raise Exception("Undefined dimension size.")

        dim_z = z.shape[1]

        if self.dim_data == 1:
            z = z.permute(0, 2, 1).contiguous()
            z = z.view(-1, dim_z)
        elif self.dim_data == 2:
            z = z.permute(0, 2, 3, 1).contiguous()
            z = z.view(-1, dim_z)
        else:
            raise Exception("Undefined dimension size.")

        return z

    def _post_reshape(self, z):
        # (bs * prod(in_shape), dim_z) -> (bs, dim_z, *in_shape)

        if self.dim_data == 1:
            z = z.view(self.bs, self.num_d1, -1).permute(0, -1, 1).contiguous()
        elif self.dim_data == 2:
            z = z.view(self.bs, self.num_d1, self.num_d2, -1).permute(0, -1, 1, 2).contiguous()
        else:
            raise Exception("Undefined dimension size.")

        return z
