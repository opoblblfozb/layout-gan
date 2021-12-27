import random
from pathlib import Path
from typing import List
from PIL import Image
from PIL.Image import Image as ImageType
import numpy as np
from rich import print

import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, DataLoader


def create_grid_image(images: List[ImageType]) -> ImageType:
    """[summary]

    Args:
        images (List[ImageType]): input images

    Returns:
        ImageType: grid layout of input images
    """
    image_tensor = []
    for image in images:
        image_tensor.append(transforms.PILToTensor()(image))
    image_tensor = torch.stack(image_tensor, dim=0)
    grid_image = utils.make_grid(image_tensor)
    return transforms.ToPILImage()(grid_image)


def layout_to_image(layout: torch.tensor) -> ImageType:
    """[summary]

    Args:
        tensor (torch.tensor): [element_num, 3]
        3 = [prob, y(tate), x(yoko)]
    """
    layout = layout.to("cpu").detach().numpy().copy()
    img_arr = np.zeros([28, 28], dtype=np.int16)
    for point in layout:
        dense, y, x = point
        y = round(y * 27)
        x = round(x * 27)
        img_arr[y, x] = round(dense * 255)
    return Image.fromarray(np.uint8(img_arr), mode="L")


class MnistLayout(Dataset):
    """
        ----> x方向
        |
        |      array
        y方向
    """

    def __init__(
            self,
            MnistDataset: Dataset,
            threshold: int = 1,
            element_num: int = 128):
        self.mnist = MnistDataset
        self.threshold = threshold
        self.element_num = element_num

    def __getitem__(self, index):
        arr = np.array(self.mnist.__getitem__(index)[0])
        ys, xs = np.nonzero(arr >= self.threshold)
        points = [[arr[y, x] / 255,
                   y / 27,
                   x / 27] for y, x in zip(ys, xs)]
        sampled = random.choices(points, k=self.element_num)
        return torch.tensor(sampled, dtype=torch.float32)

    def __len__(self):
        return len(self.mnist)


def get_dataloader(batch_size: int):
    # Dataset を作成する。
    download_dir = "./data"  # ダウンロード先は適宜変更してください
    if not Path(download_dir).exists():
        Path(download_dir).mkdir(parents=True)
    dataset = datasets.MNIST(
        download_dir,
        train=True,
        download=True)
    layoutset = MnistLayout(dataset)
    dataloader = DataLoader(layoutset, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == "__main__":
    get_dataloader(64)
