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
            nn.BatchNorm2d(256)
        )
        # bs*256*8*32
        self.cnn4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # bs*256*4*16
        self.cnn5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        # bs*512*4*16
        self.cnn6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0, 0))
        )
        # bs*512*2*16

        # dimension change

        self.map_to_seq = nn.Linear(1024, 64)
        # 64*bs*16

        # RNN
        self.lstm = nn.LSTM(64, num_hidden, num_layers=2, bidirectional=True)

        # full connection
        self.fc = nn.Linear(num_hidden*2, num_classes)
    
    def forward(self, x):
        # print(x.size())
        x = self.cnn1(x)
        # print(x.size())
        x = self.cnn2(x)
        # print(x.size())
        x = self.cnn3(x)
        # print(x.size())
        x = self.cnn4(x)
        # print(x.size())
        x = self.cnn5(x)
        # print(x.size())
        x = self.cnn6(x)
        # print(x.size())
        # x = self.cnn7(x)
        # print(x.size())

        x = x.view(x.size()[0], x.size()[1]*x.size()[2], x.size()[3])
        # assert x.size()[2] == 1, f"the height must be 1"
        # bs*num_classes*8
        x = x.permute(2, 0, 1)
        # 8*bs*num_classes

        x = self.map_to_seq(x)

        x, _ = self.lstm(x)
        x = self.fc(x)
        return x




