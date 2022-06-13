import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1_VQA = nn.Linear(1, 20)
        self.layer1_R1 = nn.Linear(1, 5)
        self.layer1_R2 = nn.Embedding(3, 5)
        self.layer1_M = nn.Linear(1, 10)
        self.layer1_I = nn.Linear(1, 10)

        self.fc1 = nn.Linear(50, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x1, x2, x3, x4, x5):
        x_R2 = self.layer1_R2(x3).view(-1, 5)
        h = torch.cat((self.layer1_VQA(x1), self.layer1_R1(x2), x_R2,
                       self.layer1_M(x4), self.layer1_I(x5)), 1)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        fc2 = F.relu(self.fc2(h))
        h = F.dropout(fc2, p=0.5, training=self.training)
        h = self.fc3(h)
        return h

    @staticmethod
    def get_model(*args):
        """Obtaining an instance of this model."""
        return Model()