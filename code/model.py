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
    def __init__(self, num_classes = 11, num_hidden = 256, num_channel = 3):
        super().__init__()
        
        # input: bs*3*32*128
        
        # CNN
        self.cnn1 = nn.Sequential(
            nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2) #
        )
        # bs*64*16*64
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # bs*128*8*32
        self.cnn3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        )
        # bs*256*4*33
        self.cnn4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 0))
        )
        # bs*512*2*32
        self.cnn5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        )
        # bs*512*1*16

        # RNN
        self.lstm = nn.LSTM(512, num_hidden, num_layers=2, bidirectional=True)

        # full connection
        self.fc = nn.Linear(num_hidden*2, num_classes)
        # bs*num_classes*1*16
    
    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)

        assert x.size()[2] == 1, f"the height must be 1"
        x = x.squeeze(2)
        # bs*num_classes*16
        x = x.permute(2, 0, 1)
        # 16*bs*num_classes

        x, _ = self.lstm(x)
        x = self.fc(x)
        return x




