import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        self.relu5 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(in_features=(256 * 6 * 6), out_features=4096)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu7 = nn.ReLU()
        # dropout 0.5?
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.maxpool2(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.maxpool3(x)
        # change shape for linear feed
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.relu6(self.fc1(x))
        x = self.relu7(self.fc2(x))
        return self.fc3(x)


alexnet = AlexNet()
a = torch.randn((1, 3, 227, 227))
print(alexnet)
print(a.size())
out = alexnet(a)
print(out.size())
