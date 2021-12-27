import unittest

import numpy as np
import torch
from utils.mnistlayout import create_grid_image, layout_to_image, get_dataloader


class TestMnistLayout(unittest.TestCase):
    def test_layout_to_image(self):
        points = []
        xs = np.linspace(0, 1, num=50)
        ys = np.linspace(0, 1, num=50)
        # top
        for x in xs:
            points.append([1, 0, x])
        # bottom
        for x in xs:
            points.append([1, 1, x])
        # center
        for y in ys:
            points.append([1, y, 0.5])
        image = layout_to_image(torch.tensor(points, dtype=torch.float32))
        image.save("./test/test_layout_to_image.jpeg")

    def test_mnistlayout(self):
        dataloder = get_dataloader(batch_size=16)
        layouts = next(iter(dataloder))
        images = []
        for layout in layouts:
            images.append(layout_to_image(layout))
        grid_image = create_grid_image(images)
        grid_image.save("./test/test_mnistlayout.jpeg")


if __name__ == "__main__":
    unittest.main()
