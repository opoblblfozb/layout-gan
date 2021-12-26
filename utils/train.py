import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from utils.model import get_generator, get_discriminator, Generator, Discriminator
from utils.data import get_dataloader


class Trainer:
    def __init__(self, latent_dim=100, img_heigh=28, img_width=28, fixed_z_num=100, batch_size=128, learning_rate=0.0002, n_epoch=100):
        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
        else:
            self.device = torch.device("cpu")
        self.real_label = 1
        self.fake_label = 0
        
        self.latent_dim = latent_dim
        self.data_dim = img_heigh * img_width
        self.fixed_z = torch.randn(fixed_z_num, self.latent_dim, device=self.device)
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        lr = learning_rate
        
        self.G: Generator = get_generator(self.latent_dim, self.data_dim, self.device)
        self.D: Discriminator = get_discriminator(self.data_dim, self.device)
        self.dataloader = get_dataloader(batch_size=self.batch_size)

        self.criterion = nn.BCELoss()
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lr)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lr)

        self.history = []

        
    def exec_train(self):
        self.G.train()
        self.D.train()

        for epoch in trange(self.n_epoch, desc="epoch"):
            D_losses, G_losses = [], []
            for x, _ in self.dataloader:
                x = x.to(self.device)
                D_losses.append(self.D_train(x))
                G_losses.append(self.G_train(x))

            # 途中経過を確認するために画像を生成する。
            img = self.generate_img(self.G, self.fixed_z)
            # 途中経過を記録する。
            info = {
                "epoch": epoch + 1,
                "D_loss": np.mean(D_losses),
                "G_loss": np.mean(G_losses),
                "img": img,
            }
            self.history.append(info)
        
        self.write_result(self.history)


    def D_train(self, x: torch.tensor):
        self.D.zero_grad()
        # (N, H, W) -> (N, H * W) に形状を変換する。
        x = x.flatten(start_dim=1)

        # 本物のデータが入力の場合の Discriminator の損失関数を計算する。
        y_pred = self.D(x)
        y_real = torch.full_like(y_pred, self.real_label)
        loss_real = self.criterion(y_pred, y_real)

        # 偽物のデータが入力の場合の Discriminator の損失関数を計算する。
        z = torch.randn(x.size(0), self.latent_dim, device=self.device)
        y_pred = self.D(self.G(z))
        y_fake = torch.full_like(y_pred, self.fake_label)
        loss_fake = self.criterion(y_pred, y_fake)

        loss = loss_real + loss_fake
        loss.backward()
        self.D_optimizer.step()

        return float(loss)

    def G_train(self, x:torch.tensor):
        self.G.zero_grad()

        # 損失関数を計算する。
        z = torch.randn(x.size(0), self.latent_dim, device=self.device)
        y_pred = self.D(self.G(z))
        y = torch.full_like(y_pred, self.real_label)
        loss = self.criterion(y_pred, y)

        loss.backward()
        self.G_optimizer.step()

        return float(loss)

    def generate_img(self, G, fixed_z):
        with torch.no_grad():
            # 画像を生成する。
            x = G(fixed_z)

        # (N, C * H * W) -> (N, C, H, W) に形状を変換する。
        x = x.view(-1, 1, 28, 28).cpu()
        # 画像を格子状に並べる。
        img = torchvision.utils.make_grid(x, nrow=10, normalize=True, pad_value=1)
        # テンソルを PIL Image に変換する。
        img = transforms.functional.to_pil_image(img)

        return img

    def write_result(self, history):
        history: pd.DataFrame = pd.DataFrame(history)

        _, ax = plt.subplots()
        # 損失の推移を描画する。
        ax.set_title("Loss")
        ax.plot(history["epoch"], history["D_loss"], label="Discriminator")
        ax.plot(history["epoch"], history["G_loss"], label="Generator")
        ax.set_xlabel("Epoch")
        ax.legend()
        plt.savefig("./result/loss_history.png")


        imgs = history["img"]
        imgs[0].save(
        "./result/history.gif", save_all=True, append_images=imgs[1:], duration=500, loop=0
        )
