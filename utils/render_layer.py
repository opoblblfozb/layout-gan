import torch
import torch.nn as nn


class PointRender(nn.Module):
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width
        self.size = img_width * img_height
        self.xiyi = torch.tensor(
            [self.img_height, self.img_width], dtype=torch.float32)
        self.relu = nn.ReLU()

    def __forward__(self, x):
        """
            in : [batchsize, element_num, geoparam_num(2)]
            # geoparam_num 0~1
            out: [batchsize, element_num, img_height, img_width]
        """
        s = x.shape
        base_img = torch.arange(
            s[0] * self.size).view(-1, self.img_height, self.img_width)
        x.view(s[0], s[1], 1, 1)
        x = x * self.xiyi
        diff = torch.tensor(1, dtype=torch.float32) - \
            torch.absolute(base_img - x)
        out = self.relu(diff)
        pass
