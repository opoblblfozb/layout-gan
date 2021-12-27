import torch.nn as nn
import torch
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from torch.utils.tensorboard import SummaryWriter
from self_attention_cv import MultiHeadSelfAttention


class RelationBasedDiscriminator(nn.Module):
    def __init__(self, element_num: int, class_num: int, geoparam_num: int):
        super().__init__()
        self.element_num = element_num
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
        # non-local neuralnetworkのところ、一旦multiheadselfattentionで代用
        # 処理がちょっと若干違うので、要修正
        self.relation1 = MultiHeadSelfAttention(self.feature_num * 8)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.relation2 = MultiHeadSelfAttention(self.feature_num * 8)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.relation3 = MultiHeadSelfAttention(self.feature_num * 8)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.relation4 = MultiHeadSelfAttention(self.feature_num * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.max_pooling = nn.MaxPool1d(self.feature_num * 8)

        self.pred = nn.Sequential(
            nn.Linear(self.element_num, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x (torch.tensor): [batch_size, element_num,  feature_num]
        """
        # [batch_size, element_num, embedding_num]
        out = self.encoder(x)

        # [batch_size, element_num, embeddng_num]
        out = self.relu1(self.relation1(out))
        out = self.relu2(self.relation2(out))
        out = self.relu3(self.relation3(out))
        out = self.relu4(self.relation4(out))
        out = self.max_pooling(out)
        out = torch.squeeze(out)
        out = self.pred(out)

        return out


if __name__ == "__main__":
    batch_size = 128
    element_num = 10
    feature_num = 4
    dummy_input = torch.randn(batch_size, element_num, feature_num)
    model = RelationBasedDiscriminator(
        element_num=element_num, class_num=2, geoparam_num=2)
    writer = SummaryWriter(log_dir="./test/relationbasedD")
    writer.add_graph(model, input_to_model=dummy_input, verbose=True)
    writer.close()
