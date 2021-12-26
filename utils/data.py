import torch
from torchvision import datasets, transforms


def get_dataloader(batch_size: int):
    # Transform を作成する。
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    # Dataset を作成する。
    download_dir = "./data"  # ダウンロード先は適宜変更してください
    dataset = datasets.MNIST(download_dir, train=True, transform=transform, download=True)

    # DataLoader を作成する。
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader