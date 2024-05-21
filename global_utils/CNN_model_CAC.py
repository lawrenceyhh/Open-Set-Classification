import torch
import torch.nn as nn


class COB_CNN_CAC(nn.Module):
    def __init__(self, num_classes=8):
        super(COB_CNN_CAC, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.dropout1 = nn.Dropout2d(p=0.45)
        self.dropout2 = nn.Dropout(p=0.45)

        self.fc1 = nn.Linear(11776, 128)
        self.fc2 = nn.Linear(128, self.num_classes)
        self.anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes).double(), requires_grad=False)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)

        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)

        out_linear = self.fc2(x)
        out_distance = self.distance_classifier(out_linear)
        # x = nn.functional.softmax(x, dim=1)
        return out_linear, out_distance

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def set_anchors(self, means):
        self.anchors = nn.Parameter(means.double(), requires_grad=False)

    def distance_classifier(self, x):
        """
        Calculates Euclidean distance from x to each class anchor
        Returns n x m array of distance from input of batch_size n to anchors of size m
        """

        n = x.size(0)
        m = self.num_classes
        d = self.num_classes

        x = x.unsqueeze(1).expand(n, m, d).double()
        anchors = self.anchors.unsqueeze(0).expand(n, m, d)
        dists = torch.norm(x - anchors, 2, 2)

        return dists
