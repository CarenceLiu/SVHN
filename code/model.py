'''
Wenrui Liu
2024-4-14

CRNN Model for SVHN classification task
'''
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, num_classes = 11, imgH = 32, num_hidden = 256, num_channel = 3):
        super().__init__()
        
        # input: 32*128*3
        
        # CNN
        self.cnn1 = nn.Sequential(
            nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2) #
        )
        # 16*64*64
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # 8*32*128
        self.cnn3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        )
        # 4*32*256
        self.cnn4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        )
        # 2*32*256
        self.cnn5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        )
        # 1*32*256

        # RNN
        self.lstm = nn.LSTM(256, num_hidden, num_layers=2, bidirectional=True)

        # full connection
        self.fc = nn.Linear(num_hidden*2, num_classes)
    
    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)

        assert x.size()[2] == 1, f"the height must be 1"
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)

        x, _ = self.lstm(x)
        x = self.fc(x)

        return x




