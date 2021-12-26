import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.main = nn.Sequential(
            # fc1
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # fc2
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # fc3
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # fc4
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.main = nn.Sequential(
            # fc1
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # fc2
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # fc3
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # fc4
            nn.Linear(128, 1),
            nn.Sigmoid(),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.main(x)

def get_generator(latent_dim, data_dim, device):
    return Generator(latent_dim, data_dim).to(device)

def get_discriminator(data_dim, device):
    return Discriminator(data_dim).to(device)