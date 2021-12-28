from tqdm import trange

import torch.optim as optim
import torch.nn as nn
import torch
import wandb
import numpy as np

from utils.mnistlayout import create_grid_image, get_dataloader, layout_to_image
from utils.generator import Generator
from utils.discriminator import RelationBasedDiscriminator
from utils.tool import get_wandb


class Trainer:
    def __init__(
            self,
            latent_dim=100,
            class_num=1,
            geoparam_num=2,
            batch_size=128,
            learning_rate=0.00002,
            n_epoch=100):
        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
        else:
            self.device = torch.device("cpu")
        self.real_label = 1
        self.fake_label = 0

        self.class_num = class_num
        self.geoparam_num = geoparam_num
        self.latent_dim = latent_dim
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        lr = learning_rate

        self.dataloader = get_dataloader(batch_size=self.batch_size)
        self.element_num = self.dataloader.dataset.element_num
        self.fixed_z = self.generate_z(40)
        self.G = Generator(
            class_num=self.class_num,
            geoparam_num=self.geoparam_num)
        self.D = RelationBasedDiscriminator(
            element_num=self.element_num,
            class_num=self.class_num,
            geoparam_num=self.geoparam_num
        )
        self.G.to(self.device)
        self.D.to(self.device)

        self.criterion = nn.BCELoss()
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lr)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lr)
        self.run = get_wandb({"group": "train_mnist"})

    def exec_train(self):
        self.G.train()
        self.D.train()

        for epoch in trange(self.n_epoch, desc="epoch"):
            D_losses, G_losses = [], []
            for x in self.dataloader:
                x = x.to(self.device)
                D_losses.append(self.D_train(x))
                G_losses.append(self.G_train())
            self.run.log({"D_losses": np.mean(D_losses),
                          "G_losses": np.mean(G_losses),
                          "generated": wandb.Image(self.generate_example()),
                          "epoch": epoch})

    def generate_z(self, size):
        batch = []
        norm_mean = torch.zeros(self.geoparam_num, dtype=torch.float32)
        norm_std = torch.ones(self.geoparam_num, dtype=torch.float32)
        for _ in range(size):
            z = []
            for _ in range(self.element_num):
                z.append(torch.cat(
                    [torch.rand(self.class_num),
                     torch.normal(mean=norm_mean, std=norm_std)]))
            batch.append(torch.stack(z, dim=0))
        return torch.stack(batch, dim=0).to(self.device)

    def generate_example(self):
        layouts = self.G(self.fixed_z)
        grid_image = create_grid_image([layout_to_image(l) for l in layouts])
        return grid_image

    def D_train(self, x: torch.tensor):
        self.D.zero_grad()

        # 本物のデータが入力の場合の Discriminator の損失関数を計算する。
        y_pred = self.D(x)
        y_real = torch.full_like(y_pred, self.real_label)
        loss_real = self.criterion(y_pred, y_real)

        # 偽物のデータが入力の場合の Discriminator の損失関数を計算する。
        z = self.generate_z(size=self.batch_size)
        y_pred = self.D(self.G(z))
        y_fake = torch.full_like(y_pred, self.fake_label)
        loss_fake = self.criterion(y_pred, y_fake)

        loss = loss_real + loss_fake
        loss.backward()
        self.D_optimizer.step()

        return float(loss)

    def G_train(self):
        self.G.zero_grad()

        # 損失関数を計算する。
        z = self.generate_z(size=self.batch_size)
        y_pred = self.D(self.G(z))
        y = torch.full_like(y_pred, self.real_label)
        loss = self.criterion(y_pred, y)

        loss.backward()
        self.G_optimizer.step()

        return float(loss)
