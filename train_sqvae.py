"""
Copyright (C) 2023 Yukara Ikemiya
"""

import argparse
import os
import sys
import yaml
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.sqvae import SQVAE


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, required=True, help="path to MNIST dataset folder")
    parser.add_argument("--config", type=str, default="./config/params.yaml", help="path to a configuration file")
    parser.add_argument("--logdir", type=str, default="./logs", help="directory storing log files")
    parser.add_argument("--device", type=int, default=0, help="gpu device to use")
    parser.add_argument("--num_sample", type=int, default=4, help="number of samples to generate during test")
    parser.add_argument("--jobname", type=str, default="none", help="job/directory name used for tensorboard outputs")
    parser.add_argument("--disable_tqdm", action='store_true', help='disable tqdm print')

    return parser.parse_args()


def main(args):
    with open(args.config, 'r') as yml:
        cfg = yaml.safe_load(yml)

    device = f'cuda:{args.device}' if args.device is not None else 'cpu'

    # dataloading
    resize = cfg['data_resize']
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(resize, antialias=True)])
    train_dataset = datasets.MNIST(root=args.datadir, transform=transform, train=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], num_workers=4,
                              pin_memory=True, persistent_workers=True, shuffle=True)
    test_dataset = datasets.MNIST(root=args.datadir, transform=transform, train=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], num_workers=4,
                             pin_memory=True, persistent_workers=True, shuffle=False)

    # model
    vae = SQVAE(cfg['encdec'], cfg['quantizer'])
    vae = vae.to(device)

    # optimizer
    betas = (cfg['beta_1'], cfg['beta_2'])
    optimizer = optim.Adam(vae.parameters(), lr=cfg['lr'], betas=betas)

    ckpt_dir = f'{args.logdir}/{args.jobname}/'
    tb_dir = f'{args.logdir}/tensorboard/{args.jobname}/'
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    writer = SummaryWriter(tb_dir)

    num_epoch = cfg['num_epoch']
    num_train_data = len(train_loader)
    num_sample = args.num_sample
    temp_decay = cfg['temperature_decay']

    msg = ["\t{0}: {1}".format(key, val) for key, val in cfg.items()]
    print("hyperparameters: \n" + "\n".join(msg))

    # main training loop
    for n in range(num_epoch):

        print(f"epoch: {n*1}/{num_epoch}")

        vae.train()
        m_gather = {}
        for idx, (x, _) in enumerate(tqdm.tqdm(train_loader, disable=args.disable_tqdm)):
            x = x.to(device)

            # temperature annealing
            step = idx + n * num_train_data
            temp = np.max([np.exp(- temp_decay * step), 1e-5])
            vae.set_temperature(temp)

            # update
            optimizer.zero_grad()
            loss, _, metrics = vae(x)
            loss.backward()
            optimizer.step()

            # gather metrics
            writer.add_scalar('temperature', temp, step)
            for k, v in metrics.items():
                writer.add_scalar(f'train/{k}', v, step)
                m_gather[k] = m_gather.get(k, 0.) + metrics[k]

        for k, v in m_gather.items():
            m_gather[k] /= len(train_loader)

        print(f'train (average) : {m_gather}')

        # eval
        vae.eval()
        m_gather = {}
        with torch.no_grad():
            # test
            for idx, (x, _) in enumerate(tqdm.tqdm(test_loader, disable=args.disable_tqdm)):
                x = x.to(device)
                loss, x_rec, metrics = vae(x)

                # gather metrics
                for k, v in metrics.items():
                    m_gather[k] = m_gather.get(k, 0.) + metrics[k]

                if idx == 0:
                    sample_gt, sample_rec = x[:num_sample], x_rec[:num_sample]

            for k, v in m_gather.items():
                m_gather[k] /= len(test_loader)
                writer.add_scalar(f'test/{k}', m_gather[k], (n + 1) * num_train_data)

            print(f'test (average) : {m_gather}')

            # save sample images
            samples = torch.cat([sample_gt, sample_rec], dim=0)
            writer.add_images("reconstruction images", samples, n + 1)

    # save final checkpoint
    torch.save(vae.state_dict(), f'{ckpt_dir}/sqvae.pt')

    writer.close()


if __name__ == '__main__':
    args = get_args()
    main(args)
