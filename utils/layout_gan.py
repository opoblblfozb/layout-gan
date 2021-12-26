import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter


def get_relation_module():
    pass

# attention機構について、使えそう
# https://github.com/The-AI-Summer/self-attention-cv/blob/main/examples/mhsa.py


class Relation(nn.Module):
    def __init__(self, feature_num):
        super().__init__()
        self.unary = nn.Linear(feature_num, feature_num)
        self.alpha = nn.Linear(feature_num, feature_num, bias=False)
        self.beta = nn.Linear(feature_num, feature_num, bias=False)
        self.r = nn.Linear(feature_num, feature_num, bias=False)

    def calucuate_on(self, alpha_k, beta_k, unary_k, x_k):
        """
        Args:
            alpha_k ([type]): [element_num,  feature_num]]
            beta_k ([type]): [element_num,  feature_num]
            unary_k ([type]): [element_num,  feature_num]

        return:
            [element_num, feature_num]
        """
        res = []  # element_numを可変にできないか？
        element_num, feature_num = alpha_k.shape
        for i in range(element_num):
            attention = torch.zeros(feature_num)
            for j in range(element_num):
                if i == j:
                    continue
                else:
                    h = torch.dot(alpha_k[i], beta_k[j])
                    attention += h * unary_k[j]
            attention /= element_num
            attention = self.r(attention)
            res.append(attention + x_k[i])
        return torch.stack(res)

    def forward(self, x):
        """
        Args:
            x (torch.tensor): [batch_size, element_num,  feature_num]
        """
        res = []
        alpha_out = self.alpha(x)
        beta_out = self.beta(x)
        unary_out = self.unary(x)
        batch_size = alpha_out.shape[0]
        for k in range(batch_size):
            batch_result = self.calucuate_on(
                alpha_out[k], beta_out[k], unary_out[k], x[k])
            res.append(batch_result)
        return torch.stack(res)


class Generator(nn.Module):
    def __init__(self, class_num: int, geoparam_num: int):
        super().__init__()
        self.class_num = class_num
        self.geoparam_num = geoparam_num
        self.feature_num = class_num + geoparam_num

        self.encoder = nn.Sequential(
            nn.Linear(self.feature_num, self.feature_num * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.feature_num * 2, self.feature_num * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.feature_num * 4, self.feature_num * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.relation = Relation(self.feature_num * 8)

    def forward(self, x):
        """
        Args:
            x (torch.tensor): [batch_size, element_num,  feature_num]
        """
        out = self.encoder(x)  # [batch_size, element_num, embedding_num]
        out = self.relation(out)  # [batch_size, element_num, embeddng_num]

        return out


if __name__ == "__main__":
    batch_size = 128
    element_num = 10
    feature_num = 4
    dummy_input = torch.randn(batch_size, element_num, feature_num)
    model = Generator(class_num=2, geoparam_num=2)
    torch.onnx.export(
        model,
        dummy_input,
        "./test/generator.onnx",
        verbose=True,
        input_names=["input"],
        output_names=["output"])
    writer = SummaryWriter(log_dir="./test/generator_model")
    writer.add_graph(model, input_to_model=dummy_input, verbose=True)
    writer.close()
