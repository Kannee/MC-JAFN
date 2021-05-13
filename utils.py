import torch
import torch.nn as nn


class MLAFBlock(nn.Module):
    def __init__(self, in_channels_m, in_channels_p, inter_channels=64, ca_ratio=16):
        super(MLAFBlock, self).__init__()

        self.inter_channels = inter_channels
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels*2, out_channels=self.inter_channels // ca_ratio,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.inter_channels // ca_ratio, out_channels=self.inter_channels*2,
                      kernel_size=1, stride=1, padding=0),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_m, out_channels=self.inter_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels,
                      kernel_size=3, stride=1, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_p, out_channels=self.inter_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels,
                      kernel_size=3, stride=1, padding=1),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.MLP_1 = nn.Sequential(
            #nn.Conv2d(self.inter_channels*2, self.inter_channels // ca_ratio, 1, bias=True), #nn.ReLU(),
            nn.Conv2d(self.inter_channels * 2, self.inter_channels, 1, bias=True))
        self.MLP_2 = nn.Sequential(
            #nn.Conv2d(self.inter_channels*2, self.inter_channels // ca_ratio, 1, bias=True), #nn.ReLU(),
            nn.Conv2d(self.inter_channels * 2, self.inter_channels, 1, bias=True))

        self.sigmoid = nn.Sigmoid()

    def forward(self, m, p):
        '''
        :m : (b, c, H, W)
        :p : (b, c, H, W)
        :return:
        '''
        m = self.conv1(m)
        p = self.conv2(p)

        u = self.conv_1(torch.cat([m, p], dim=1))

        ca_m = self.sigmoid(self.MLP_1(self.avg_pool(u)))
        ca_p = self.sigmoid(self.MLP_2(self.avg_pool(u)))

        w_m = ca_m * m
        w_p = ca_p * p

        f = w_m + w_p
        return m, p, f


class MLBlock(nn.Module):
    def __init__(self, in_channels_m, in_channels_p, inter_channels=64):
        super(MLBlock, self).__init__()

        self.inter_channels = inter_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_m, out_channels=self.inter_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels,
                      kernel_size=3, stride=1, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_p, out_channels=self.inter_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels,
                      kernel_size=3, stride=1, padding=1),
        )

    def forward(self, m, p):
        '''
        :m : (b, c, H, W)
        :p : (b, c, H, W)
        :return:
        '''
        m = self.conv1(m)
        p = self.conv2(p)

        f = m + p

        return m, p, f

