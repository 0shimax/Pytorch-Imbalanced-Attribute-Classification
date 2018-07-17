import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, in_ch, n_class):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, n_class, kernel_size=1)
        self.conv2 = nn.Conv2d(in_ch, n_class, kernel_size=1)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        exped = torch.exp(z)
        normalized_z = exped / exped.sum(2).sum(2).unsqueeze(2).unsqueeze(3)

        confidence_weight = F.sigmoid(self.conv2(x))
        return normalized_z * confidence_weight


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.ab1 = AttentionBlock(10, 10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.ab2 = AttentionBlock(20, 10)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.fc1_a1 = nn.Linear(1440, 128)
        self.fc2_a1 = nn.Linear(128, 10)

        self.fc1_a2 = nn.Linear(160, 32)
        self.fc2_a2 = nn.Linear(32, 10)


    def forward(self, x):
        n_batch = x.shape[0]

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        a1 = self.ab1(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        a2 = self.ab2(x)

        x = x.view(n_batch, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        yp = self.fc2(x)

        x_a1 = F.relu(self.fc1_a1(a1.view(n_batch, -1)))
        y_a1 = self.fc2_a1(x_a1)

        x_a2 = F.relu(self.fc1_a2(a2.view(n_batch, -1)))
        y_a2 = self.fc2_a2(x_a2)

        # y = yp + y_a1 + y_a2
        # return F.log_softmax(y, dim=1)
        return yp, y_a1, y_a2
